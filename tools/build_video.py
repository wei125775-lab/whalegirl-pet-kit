"""豆包动画视频 -> 序列帧（抽帧 + 清水印 + 自适应抠像 + 配准 + 精灵表/GIF）。

和 build_action.py 的区别：输入是 mp4 而不是分镜表，所以多了三步。

1. **抽帧**：视频 121 帧 / 24fps / 5.04s，桌宠用不了这么长，等间隔抽到 TARGET_N 帧。
2. **清水印**：豆包在右下角烙"豆包AI生成"，帧 12 淡入、帧 96 淡出，位置固定。
   不清的话它落在绿幕上，抠像会把它当白色前景留在 alpha 里。
3. **自适应抠像**：这批视频的背景绿**远不如绿幕素材纯**（实测 e = G-max(R,B) 只有
   57~64，旧素材是 200+）。build_action.py 那套固定判据 (210-e)/170 会把整块背景判成
   不透明。改成按实测背景绿度 e_bg 比例定阈值。

配准沿用 build_action.py 的思路但简化了：视频内角色位置本来就稳定（bbox 差 2~3px），
用全部抽帧的锚点中位数当目标，不做逐帧最优化搜索。
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

_SELF = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SELF)
VIDDIR = os.path.join(ROOT, '视频素材')
OUTROOT = os.path.join(ROOT, 'v2', 'actions')

# 五个视频的角色位置实测一致（bbox 都在 x 156~559 / y 36~689），用统一裁切框。
# 右边留到 600 是为了把水印（x 580~700）大部分甩掉，再显式清一次。
CROP = (120, 8, 600, 712)      # x0 y0 x1 y1
TARGET_H = 640                 # 输出画布高。源视频里角色高 653px，这里几乎 1:1，
                               # 不浪费素材；桌宠显示大小交给 pet.json 的 scale 调。
                               # 30 帧 × 436px = 13080，仍在 petpet 的 16384 总宽上限内。
FOOT_TARGET_Y = 619            # 脚底在输出画布上的位置。各条视频实测在 613~619 之间飘
                               # （每条各自对齐到自己的锚点中位），统一成定值，
                               # 否则托盘里切动作时角色会上下跳几像素。
SUBJ_H_TARGET = 594            # 角色在输出画布上的目标身高（和分层待机一致）。
                               # 豆包生成时各条的缩放有出入——实测 wave 的角色只有 584
                               # 高、比 idle 矮 2%，宽高比一致说明是等比缩小不是形变，
                               # 所以按实测身高统一缩放到这个值，不然切换动作时忽大忽小。
TILT_TARGET = 0.0              # 直立化的目标角度。**不能转回各视频自己的中位**——豆包生成
                               # 挥手那条时角色整体就歪着，转回自己的中位等于保持歪。
                               # 实测各条中位：idle +0.18 / eat +0.36 / touchface -0.48 /
                               # heart -0.47 / wave **+1.72**，只有 wave 离谱，用户一眼看出来了。
EDGE_FLOOR = 70.0              # 轮廓外那圈半透明雾的 alpha 阈值，见 trim_faint
WM_BOX = (565, 640, 720, 720)  # 水印矩形（原始 720 帧坐标），角色最右 557 / 最下 689，不重叠

JOBS = {
    'wave': dict(video='生成挥手视频.mp4', tag='挥手', n=30, fps=12, loop=True),
    'touchface': dict(video='托头视频.mp4', tag='托头', n=30, fps=12, loop=True),
    'eat': dict(video='吃饭视频.mp4', tag='吃饭', n=30, fps=12, loop=True),
    'heart': dict(video='比心视频.mp4', tag='比心', n=30, fps=12, loop=False),
    'sway': dict(video='摇摆视频.mp4', tag='摇摆', n=30, fps=12, loop=True),
}

BG_FLAT = 128.0


def read_selected(path, n):
    """只解码抽中的帧，省得把 121 帧全留在内存里。"""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, n).round().astype(int)
    want = set(idx.tolist())
    got = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    missing = [k for k in idx.tolist() if k not in got]
    if missing:
        print('  !! 缺帧 %s' % missing)
    return [got[k] for k in idx.tolist() if k in got], idx


def bg_stats(frames):
    """背景色中位数 + 背景绿度的低分位数。

    只采上边和左右两边的上半段——水印在右下角（x 580~700 / y 665~705），采进去会
    混进一堆 e≈0 的白像素，把分位数带偏。
    """
    cols, es = [], []
    for f in frames:
        h, w = f.shape[:2]
        half = h // 2
        border = np.concatenate([
            f[:20].reshape(-1, 3),
            f[:half, :20].reshape(-1, 3),
            f[:half, w - 20:].reshape(-1, 3),
        ]).astype(np.float32)
        cols.append(np.median(border, 0))
        es.append(border[:, 1] - np.maximum(border[:, 0], border[:, 2]))
    return np.median(np.array(cols), 0), float(np.percentile(np.concatenate(es), 1))


def clear_watermark(f, bg):
    x0, y0, x1, y1 = WM_BOX
    f = f.copy()
    f[y0:min(y1, f.shape[0]), x0:min(x1, f.shape[1])] = bg.astype(np.uint8)
    return f


def chroma_key(p, e_hi, e_lo):
    """绿幕抠像 + despill。e = G - max(R,B)，越大越绿。

    **p 必须先转 float32**：p[:,:,1] - np.maximum(p[:,:,0], p[:,:,2]) 在 uint8 下
    负数会环绕成 200 多的大正数，把整个角色判成背景。旧脚本没踩到是因为源图读进来
    就 astype(np.float32) 了。

    阈值由 bg_stats 实测给出，不用固定比例——这批视频的背景绿（e≈41~55）比绿幕素材
    （e 200+）淡得多，角色自己的白围裙 e 只到 -6，两者之间的空档就是过渡带。
    """
    p = p.astype(np.float32)
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    e = g - np.maximum(r, b)
    alpha = np.clip((e_hi - e) / max(e_hi - e_lo, 1e-3) * 255.0, 0, 255)
    g2 = np.where(e > 0, np.maximum(r, b), g)
    return np.dstack([r, g2, b, alpha]).astype(np.float32)


def subject_height(rgba):
    """角色身高。取最大连通域，避开边缘零散噪点把包围盒撑高。"""
    m = rgba[:, :, 3] > 128
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    rows = np.where(m.any(axis=1))[0]
    return float(rows.max() - rows.min() + 1) if len(rows) else 0.0


def anchor(rgba):
    """(脚部中心x, 脚底y, 中轴倾角度)。

    **用脚当锚点，不用头顶。** 这批素材不是刚体：实测 eat 那条的"头脚横向偏差"在
    0.5~11.5px 之间变，而纯平移的配准不改变两点相对距离，所以这是豆包生成时的形变
    （角色某个瞬间下半身相对头位移）。钉头，这个量就全暴露在脚上——脚在画布上左右滑
    11px，肉眼就是"斜了一下"；钉脚，同样的量变成上身的自然晃动，看着像活人。

    倾角（头顶带中心 → 脚部中心的连线偏角，正 = 头偏右）另有用处：光钉脚只是把倾斜
    换个地方暴露，还得绕脚底转回竖直。取下半身 10% 高度带算脚，只含腿和鞋，不含尾巴
    也不含裙摆。
    """
    m = rgba[:, :, 3] > 128
    ys, xs = np.where(m)
    if not len(ys):
        return None
    ymin, ymax = ys.min(), ys.max()
    h = ymax - ymin + 1
    head = (ys >= ymin + h * 0.05) & (ys < ymin + h * 0.25)
    head_x, head_y = xs[head].mean(), ys[head].mean()
    foot = ys > ymin + h * 0.90
    foot_x, foot_y = xs[foot].mean(), float(ys[foot].max())
    tilt = float(np.degrees(np.arctan2(head_x - foot_x, foot_y - head_y)))
    return foot_x, foot_y, tilt


def rotate_about(rgba, deg, cx, cy):
    """绕 (cx,cy) 旋转。走预乘 alpha，免得透明区的黑渗进边缘。"""
    H, W = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(deg), 1.0)
    w = rgba[:, :, 3] / 255.0
    pr = cv2.warpAffine(rgba[:, :, :3] * w[:, :, None], M, (W, H),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ar = cv2.warpAffine(w, M, (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.zeros((H, W, 4), np.float32)
    nz = (ar > 1e-6)[:, :, None]
    out[:, :, :3] = np.where(nz, pr / np.maximum(ar[:, :, None], 1e-6), 0)
    out[:, :, 3] = ar * 255.0
    return out


def place(dst, src, ox, oy):
    sx, sy = max(0, -ox), max(0, -oy)
    dx, dy = max(0, ox), max(0, oy)
    w = min(src.shape[1] - sx, dst.shape[1] - dx)
    h = min(src.shape[0] - sy, dst.shape[0] - dy)
    if w > 0 and h > 0:
        dst[dy:dy + h, dx:dx + w] = src[sy:sy + h, sx:sx + w]
    return dst


def resize_rgba(rgba, nw, nh):
    """预乘 alpha 后缩放再解预乘。PIL 的 paste 会重复乘 alpha，不能用。"""
    al = rgba[:, :, 3] / 255.0
    prem = rgba[:, :, :3] * al[:, :, None]
    chans = [np.array(Image.fromarray(prem[:, :, c].astype(np.float32), 'F')
                      .resize((nw, nh), Image.LANCZOS)) for c in range(3)]
    ares = np.array(Image.fromarray(rgba[:, :, 3].astype(np.float32), 'F')
                    .resize((nw, nh), Image.LANCZOS))
    out = np.zeros((nh, nw, 4), np.float32)
    out[:, :, :3] = np.stack(chans, axis=2)
    out[:, :, 3] = ares
    a = out[:, :, 3:4] / 255.0
    out[:, :, :3] = np.where(a > 1e-6, out[:, :, :3] / np.maximum(a, 1e-6), 0)
    return out


def trim_faint(rgba, core_thresh=60, grow=20, gap=2):
    """四道清理，从外到内。

    1. **离主体太远的连通域**。视频压缩在绿幕上留孤立噪块；更要命的是**豆包的水印
       会漂移**——同一条视频里右下角（帧 12~96）和左上角（帧 100~120）都出现过。而且
       水印是灰绿色（实测 e≈24），正好落在抠像过渡带里，会被抠出 alpha 150~255，靠调
       阈值治不了。位置法清不干净，改用连通域法：主体是最大的一块，其余离它超过 grow
       像素的全清。水印离主体 50px 以上会掉；比心那个红心紧贴双手，落在保护区里，不受影响。
    2. **每列主体下沿之下**。治的是鞋底正下方那圈 alpha 10~40 的接地淡影（豆包生成时
       角色脚下的投影）。它离主体只有几像素、alpha 又散得开，连通域法够不着、调 floor
       又会啃到角色的抗锯齿边。脚底以下本来就没有角色内容，直接按列清，不会误伤。
    3. **轮廓外那圈雾**。H.264 在角色和绿幕之间留了一层过渡像素，抠像后是 alpha 20~150
       的边，在深色底上就是一圈白晕（蕾丝和呆毛最明显；idle 来自 PSD，没有这层）。只清零
       会在边缘留一个 0→70 的台阶，所以清完重新拉伸——代价是柔边薄一层，角色略瘦一圈。
       零星噪点也一并被这刀带掉。
    """
    core = rgba[:, :, 3] > core_thresh
    if not core.any():
        return rgba
    lab, n = ndimage.label(core)
    if n > 1:
        sizes = ndimage.sum(core, lab, range(1, n + 1))
        core = lab == (int(np.argmax(sizes)) + 1)
    keep = ndimage.binary_dilation(core, np.ones((grow * 2 + 1, grow * 2 + 1), bool))
    rgba = rgba.copy()
    rgba[~keep, 3] = 0.0

    H = rgba.shape[0]
    col = core.any(axis=0)
    bottom = H - 1 - np.argmax(core[::-1], axis=0)
    bottom[~col] = -1
    kill = np.arange(H)[:, None] > (bottom[None, :] + gap)
    rgba[:, :, 3] = np.where(kill, 0.0, rgba[:, :, 3])

    a = rgba[:, :, 3]
    rgba[:, :, 3] = np.clip((a - EDGE_FLOOR) * 255.0 / (255.0 - EDGE_FLOOR), 0, 255)
    return rgba


def flat(rgba, bg=BG_FLAT):
    a = rgba[:, :, 3:4] / 255.0
    return rgba[:, :, :3] * a + bg * (1 - a)


def build(name):
    job = JOBS[name]
    path = os.path.join(VIDDIR, job['video'])
    outdir = os.path.join(OUTROOT, name)
    os.makedirs(outdir, exist_ok=True)

    frames, idx = read_selected(path, job['n'])
    bg, e_bg_lo = bg_stats(frames)
    e_hi = e_bg_lo - 6.0        # 略低于背景绿度下限，保证背景全透明
    e_lo = e_hi * 0.25          # 角色白围裙的 e 只到 -6，落在这条线以下就完全不透明
    print('== %s（%s）抽 %d 帧，原帧号 %s' % (job['tag'], job['video'], len(frames), idx.tolist()))
    print('   背景色 RGB %s  背景绿度 p1=%.1f -> 抠像阈值 hi=%.1f lo=%.1f'
          % ([int(v) for v in bg], e_bg_lo, e_hi, e_lo))

    x0, y0, x1, y1 = CROP
    cw, ch = int(round((x1 - x0) * TARGET_H / (y1 - y0))), TARGET_H

    keyed = []
    for f in frames:
        f = clear_watermark(f, bg)
        keyed.append(chroma_key(f[y0:y1, x0:x1], e_hi, e_lo))

    # 各条视频里角色的实际身高不一致（豆包生成时缩放有出入），统一缩放到目标身高。
    # **缩放要作用在"角色相对画面"的比例上**——把整张 crop 图缩放再贴回固定画布，
    # 两个缩放会互相抵消（图变大的同时分母也变大）。所以这里只算比例，
    # 真正的缩放放在下面 resize 那一步：由 640/图高 换成 目标身高/实测身高。
    hs = [h for h in (subject_height(k) for k in keyed) if h > 10]
    h_med = float(np.median(hs)) if hs else float(y1 - y0)
    s = SUBJ_H_TARGET / h_med
    print('   角色身高 %.0f px → 统一缩放 ×%.4f（目标 %d px）' % (h_med, s, SUBJ_H_TARGET))

    # 所有帧对齐到锚点中位数（视频内角色位置本就稳定，不做逐帧搜索）
    anchors = [anchor(k) for k in keyed]
    good = [a for a in anchors if a]
    fx = float(np.median([a[0] for a in good]))
    fy = float(np.median([a[1] for a in good]))
    tilt_med = float(np.median([a[2] for a in good]))
    foot_target = FOOT_TARGET_Y     # 固定值，见文件头注释
    print('   锚点(脚) 中位 x=%.0f y=%.0f   逐帧脚部偏移 %+.1f~%+.1f px'
          % (fx, fy, min(a[0] for a in good) - fx, max(a[0] for a in good) - fx))
    print('   中轴倾角 中位 %+.2f 度   逐帧 %+.2f~%+.2f 度（统一转回 %+.1f 度）'
          % (tilt_med, min(a[2] for a in good), max(a[2] for a in good), TILT_TARGET))

    aligned = []
    for i, (k, a) in enumerate(zip(keyed, anchors)):
        if a is None:
            print('  帧%02d 无前景，跳过' % (i + 1))
            continue
        # 缩放比 s 把"角色在 crop 图里的高度"映射到目标身高，锚点坐标同样按 s 换算
        small = resize_rgba(k, max(1, round(k.shape[1] * s)), max(1, round(k.shape[0] * s)))
        canvas = np.zeros((ch, cw, 4), np.float32)
        ox = int(round(cw / 2 - a[0] * s))
        oy = int(round(foot_target - a[1] * s))
        place(canvas, small, ox, oy)
        # 直立化：绕脚底把中轴转到竖直。光钉脚只是把源素材的倾斜从脚挪到头，转一下才
        # 真的消掉。目标是统一的 TILT_TARGET 而不是本条自己的中位（见文件头注释）。
        # 旋转轴就取脚在画布上的落点，转完脚不动。
        d = a[2] - TILT_TARGET
        if abs(d) > 0.05:
            ax = ox + a[0] * s
            ay = oy + a[1] * s
            canvas = rotate_about(canvas, d, ax, ay)
        aligned.append(trim_faint(canvas))

    n = len(aligned)
    fl = [flat(f) for f in aligned]
    diffs = [np.abs(fl[i] - fl[(i + 1) % n]).mean() for i in range(n - 1)]
    print('   出 %d 帧 @ %dfps = %.1fs   相邻差异 平均%.2f 最大%.2f'
          % (n, job['fps'], n / job['fps'], np.mean(diffs), np.max(diffs)))
    print('   首尾差（闭合检查）: %.2f' % np.abs(fl[0] - fl[-1]).mean())

    sheet = Image.fromarray(np.concatenate(
        [np.clip(f, 0, 255).astype(np.uint8) for f in aligned], axis=1))
    sheet.save(os.path.join(outdir, '%s_sheet.png' % name))
    for i, f in enumerate(aligned):
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(
            os.path.join(outdir, 'frame_%02d.png' % (i + 1)))

    preview = Image.new('RGB', sheet.size, (232, 232, 236))
    preview.paste(sheet, (0, 0), sheet)
    gif = [preview.crop((i * cw, 0, (i + 1) * cw, ch)) for i in range(n)]
    gif[0].save(os.path.join(outdir, '%s_preview.gif' % name), save_all=True,
                append_images=gif[1:], duration=int(1000 / job['fps']), loop=0, disposal=2)

    cols = 6
    rows = (n + cols - 1) // cols
    check = Image.new('RGB', (cw * cols, ch * rows), (232, 232, 236))
    for i, f in enumerate(aligned):
        im = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
        check.paste(im, ((i % cols) * cw, (i // cols) * ch), im)
    check.save(os.path.join(outdir, '%s_check.png' % name))

    json.dump({"name": "whalegirl", "cellWidth": cw, "cellHeight": ch, "frames": n,
               "fps": job['fps'], "action": name, "loop": job['loop'],
               "file": '%s_sheet.png' % name},
              open(os.path.join(outdir, '%s_meta.json' % name), 'w'),
              ensure_ascii=False, indent=2)

    print('   画布 %dx%d   精灵表 %dx%d   总宽上限 16384 %s'
          % (cw, ch, sheet.size[0], sheet.size[1],
             'OK' if sheet.size[0] <= 16384 else '超了！'))
    print('   写出 %s' % outdir)


if __name__ == '__main__':
    for k in (sys.argv[1:] or JOBS.keys()):
        build(k)
