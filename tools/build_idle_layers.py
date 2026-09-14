"""用 see-through 的 22 层烘一个待机循环。

旧基准那版 idle 是"单张图整体变形"；这版分层驱动，因为分层能做两件整体变形做不到的事：

- **头发和尾巴单独飘**。合成成一张图之后，距离变换会把头发和身体看成同一坨，
  细部权重传不到发梢（旧基准的 THIN_PX 就是在这儿卡住的：调大整圈轮廓都抖，
  调小发梢不动）。按层分开算，长发外侧和尾巴才能比身体多摆。
- **能眨眼**。22 层里有独立的 eyewhite 和 irides，把它们绕眼睛中线纵向压扁就是闭眼。

呼吸和摇摆沿用旧基准那套正弦（循环天然闭合、中间态是算出来的所以必然连贯），
只是位移场改成按组给。

画布和视频动作那批对齐（436×640，角色高 594、脚底 y=619、脚部中心 x=218），
不然托盘里切动作时角色会跳位置。
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from psd_tools import PSDImage

_SELF = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(_SELF), 'v2')
PSD_PATH = os.path.join(V2, 'layers', 'seethrough_output.psd')
OUT = os.path.join(V2, 'actions', 'idle')

N = 24                 # 帧数
FPS = 12
SS = 2                 # 超采样。变形在于 PSD 原尺度上做，细节（发丝/蕾丝）约 7~14px，
                       # 位移 7px 已经到同一量级，放大一倍再降回来才不糊。
TARGET = (436, 640)    # 输出画布（与视频动作一致）
SUBJ_H_TARGET = 594    # 角色在输出画布上的高度
FOOT_FRAC_Y = 619 / 640   # 脚底在画布高度的位置
FOOT_FRAC_X = 0.5      # 脚部中心在画布宽度的位置

# 位移参数，单位是"输出画布像素"，脚本内部按尺度换算
BREATH_SCALE = 0.010   # 呼吸纵向缩放 ±1%
BREATH_CYCLES = 2      # 一个循环呼吸两次
SWAY_PX = 3.0          # 上身左右摆 ±3px
FLUTTER_PX = 4.5       # 头发/尾巴飘动 ±4.5px
FLUTTER_WAVE = 0.35    # 行波相位随高度推进（每 100px 约 0.35 圈）
BLINK_AT = 0.62        # 眨眼时刻（循环相位）
BLINK_WIDTH = 0.032    # 高斯包络宽度
BLINK_SQUASH = 0.14    # 闭眼时眼球纵向压到 14%

# 按 PSD 的绘制顺序分组，合成时也按这个顺序叠
GROUPS = [
    (['back hair'], 'hair'),
    (['tail', 'legwear', 'footwear', 'neck', 'handwear-r', 'handwear-l',
      'bottomwear', 'topwear', 'headwear', 'face', 'nose', 'mouth'], 'body'),
    (['eyewhite-r', 'eyewhite-l', 'irides-l', 'irides-r'], 'eye'),
    (['eyelash-r', 'eyelash-l', 'eyebrow-l', 'eyebrow-r'], 'body'),
    (['front hair'], 'hair'),
]


def load_groups():
    psd = PSDImage.open(PSD_PATH)
    by_name = {}
    for l in psd.descendants():
        im = l.topil()
        if im is not None:
            by_name.setdefault(l.name, []).append((im, l.offset))
    out = []
    for names, kind in GROUPS:
        c = Image.new('RGBA', psd.size, (0, 0, 0, 0))
        for n in names:
            for im, off in by_name.get(n, []):
                c.alpha_composite(im, off)
        out.append([np.array(c).astype(np.float32), kind])
    return out, psd.size


def subject_metrics(rgba):
    """(主体bbox, 脚部中心x, 脚底y)。bbox 取最大连通域，层里的低 alpha 噪点会把包围盒撑满。"""
    m = rgba[:, :, 3] > 128
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    main = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(main)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    foot = ys > y0 + (y1 - y0 + 1) * 0.90
    return (x0, y0, x1, y1), float(xs[foot].mean()), float(y1)


def flat(rgba, bg=232.0):
    a = rgba[:, :, 3:4] / 255.0
    return rgba[:, :, :3] * a + bg * (1 - a)


def over(a, b):
    """b 叠在 a 上（PIL alpha_composite 语义）。"""
    aa = a[:, :, 3:4] / 255.0
    ba = b[:, :, 3:4] / 255.0
    oa = ba + aa * (1 - ba)
    rgb = (b[:, :, :3] * ba + a[:, :, :3] * aa * (1 - ba)) / np.maximum(oa, 1e-6)
    return np.dstack([np.where(oa > 1e-6, rgb, 0), oa * 255])


def put(canvas, img, ox, oy):
    sx, sy = max(0, -ox), max(0, -oy)
    dx, dy = max(0, ox), max(0, oy)
    w = min(img.shape[1] - sx, canvas.shape[1] - dx)
    h = min(img.shape[0] - sy, canvas.shape[0] - dy)
    if w > 0 and h > 0:
        canvas[dy:dy + h, dx:dx + w] = img[sy:sy + h, sx:sx + w]
    return canvas


def warp(premult, dx, dy):
    """dx/dy 里有 Python float 参与运算，结果会升成 float64，remap 只吃 CV_32FC1，这里转回来。"""
    H, W = dx.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = (gx + dx).astype(np.float32)
    my = (gy + dy).astype(np.float32)
    out = np.empty_like(premult)
    for c in range(4):
        out[:, :, c] = cv2.remap(premult[:, :, c], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def deform(img, dx, dy):
    al = img[:, :, 3] / 255.0
    premult = np.dstack([img[:, :, :3] * al[:, :, None], img[:, :, 3]])
    w = warp(premult, dx, dy)
    a = np.clip(w[:, :, 3], 0, 255)
    rgb = np.where(a[:, :, None] > 1e-3, w[:, :, :3] / np.maximum(a[:, :, None] / 255.0, 1e-6), 0)
    return np.dstack([np.clip(rgb, 0, 255), a])


def downscale(img, out_w, out_h):
    """预乘 alpha 的降采样，否则透明区的黑会渗进边缘（旧基准踩过的坑）。"""
    al = img[:, :, 3:4] / 255.0
    pm = np.dstack([img[:, :, :3] * al, img[:, :, 3]])
    im = Image.fromarray(np.clip(pm, 0, 255).astype(np.uint8)).resize((out_w, out_h), Image.LANCZOS)
    sm = np.array(im).astype(np.float32)
    sa = sm[:, :, 3:4]
    rgb = np.where(sa > 1e-3, sm[:, :, :3] / np.maximum(sa / 255.0, 1e-6), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(sa[:, :, 0], 0, 255)])


def blink_env(t):
    d = t - BLINK_AT
    d -= round(d)
    return float(np.exp(-(d / BLINK_WIDTH) ** 2))


def build():
    groups, psd_size = load_groups()
    group_all = np.zeros((psd_size[1], psd_size[0], 4), np.float32)
    for g, _ in groups:
        group_all = over(group_all, g)
    bbox, foot_x, foot_y = subject_metrics(group_all)
    subj_h = bbox[3] - bbox[1] + 1

    scale = SUBJ_H_TARGET / subj_h
    cw = int(round(TARGET[0] / scale))
    ch = int(round(TARGET[1] / scale))
    foot_cx = cw * FOOT_FRAC_X
    foot_cy = ch * FOOT_FRAC_Y
    print('主体 bbox x%d-%d y%d-%d  高 %d  脚部中心 x%.1f 脚底 y%.0f'
          % (bbox[0], bbox[2], bbox[1], bbox[3], subj_h, foot_x, foot_y))
    print('输出 %dx%d  中间画布 %dx%d（缩放 %.4f，超采样 SS=%d）' % (
        TARGET[0], TARGET[1], cw, ch, scale, SS))

    # 贴到中间画布（超采样倍数下）
    ox = int(round(foot_cx * SS - foot_x * SS))
    oy = int(round(foot_cy * SS - foot_y * SS))
    canvas_size = (cw * SS, ch * SS)
    placed = []
    for g, kind in groups:
        big = np.array(Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
                       .resize((psd_size[0] * SS, psd_size[1] * SS), Image.LANCZOS)).astype(np.float32)
        c = np.zeros((canvas_size[1], canvas_size[0], 4), np.float32)
        placed.append([put(c, big, ox, oy), kind])
        del big

    # 位移场用的几何量（中间画布坐标）
    top = (bbox[1] * SS + oy)
    foot = foot_cy * SS
    cxm = foot_cx * SS
    halfw = (bbox[2] - bbox[0] + 1) / 2.0 * SS
    H, W = canvas_size[1], canvas_size[0]
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    up = np.clip((foot - yy) / max(foot - top, 1.0), 0, 1)      # 摇摆：越往上越大，脚底钉死
    wx = np.clip(np.abs(xx - cxm) / max(halfw, 1.0), 0, 1)      # 飘动：越靠外越大
    wy = np.clip((yy - top) / max(foot - top, 1.0), 0, 1)       # 飘动：越靠下越大
    # clip 出的边界是硬的，直接当位移场会让飘动范围出现折线，先低通一道
    sm = 4.0 * SS
    up = ndimage.gaussian_filter(np.where(yy < top, 0.0, up), sm, mode='nearest').astype(np.float32)
    hair_w = ndimage.gaussian_filter(np.where(yy < top, 0.0, wx * (0.30 + 0.70 * wy)),
                                    sm, mode='nearest').astype(np.float32)
    eye_a = placed[2][0][:, :, 3] > 32
    eye_cy = float(np.where(eye_a)[0].mean()) if eye_a.any() else foot * 0.4

    # 输出尺度 -> 中间画布尺度的位移换算
    K = SS / scale

    frames = []
    for i in range(N):
        t = i / N
        s = 1.0 + BREATH_SCALE * np.sin(2 * np.pi * BREATH_CYCLES * t)
        dx = (xx - cxm) * (-(s - 1.0) * 0.5) + SWAY_PX * K * up * np.sin(2 * np.pi * t)
        dy = (yy - foot) * (s - 1.0)
        ph = 2 * np.pi * t + (yy - top) * FLUTTER_WAVE / (100.0 * SS)
        fdx = FLUTTER_PX * K * hair_w * np.sin(ph)
        fdy = FLUTTER_PX * K * 0.5 * hair_w * np.sin(ph + np.pi / 2)
        blink = blink_env(t)
        # 眨眼：把眼睛压扁，采样步长得是原来的 1/s —— 写成 (y-cy)*(s-1) 会反过来把
        # 眼睛纵向拉长 s 倍（数值上等于 out(y)=in(cy+s*(y-cy))），实测拉出一根竖条。
        sq = 1.0 - (1.0 - BLINK_SQUASH) * blink
        ey = (yy - eye_cy) * (1.0 / max(sq, 1e-3) - 1.0)

        comp = np.zeros((H, W, 4), np.float32)
        for img, kind in placed:
            if kind == 'hair':
                comp = over(comp, deform(img, dx + fdx, dy + fdy))
            elif kind == 'eye':
                comp = over(comp, deform(img, dx, dy + ey))
            else:
                comp = over(comp, deform(img, dx, dy))
        frames.append(downscale(comp, TARGET[0], TARGET[1]))

    os.makedirs(OUT, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f.astype(np.uint8)).save(os.path.join(OUT, 'frame_%02d.png' % (i + 1)))
    sheet = Image.fromarray(np.concatenate(frames, axis=1).astype(np.uint8))
    sheet.save(os.path.join(OUT, 'idle_sheet.png'))
    preview = Image.new('RGB', sheet.size, (232, 232, 236))
    preview.paste(sheet, (0, 0), sheet)
    gif = [preview.crop((i * TARGET[0], 0, (i + 1) * TARGET[0], TARGET[1])) for i in range(N)]
    gif[0].save(os.path.join(OUT, 'idle_preview.gif'), save_all=True,
                append_images=gif[1:], duration=int(1000 / FPS), loop=0, disposal=2)

    fl = [flat(f) for f in frames]
    ds = [np.abs(fl[i] - fl[(i + 1) % N]).mean() for i in range(N)]
    feets = [np.where(f[:, :, 3] > 128)[0].max() for f in frames]
    print('出 %d 帧 @ %dfps = %.1fs   帧间差 均%.2f 最大%.2f   脚底极差 %dpx   首尾差 %.2f'
          % (N, FPS, N / FPS, np.mean(ds), np.max(ds),
             max(feets) - min(feets), np.abs(fl[0] - fl[-1]).mean()))
    print('精灵表 %dx%d   写出 %s' % (sheet.size[0], sheet.size[1], OUT))


if __name__ == '__main__':
    build()
