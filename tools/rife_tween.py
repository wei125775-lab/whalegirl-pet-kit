"""用 RIFE（rife-ncnn-vulkan）给动作帧补帧。

RIFE 只算 t=0.5 的中间帧，正好把 N 帧翻倍成 2N 帧。走目录批量模式，
一次调用跑完全部（实测 13 张输入 0.95 秒；逐对调用每对要重载模型，慢五十倍）。

输出排布是 [f1, mid(f1,f2), f2, mid(f2,f3), ...]，奇数位才是中间帧。

透明帧的处理是唯一要小心的地方：RIFE 吃 RGB、不认 alpha，直接喂透明 PNG 会把
透明区当成黑色物块。所以走三步——RGB 合成到中性底再喂、alpha 线性插值、解合成。
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import cv2
from PIL import Image

_SELF = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(os.path.dirname(_SELF), '_archive')  # 脚本已移到 tools/，旧基准数据在 ../_archive/
RIFE = r'D:\rife\rife-ncnn-vulkan-20221029-windows\rife-ncnn-vulkan.exe'
MODEL = r'D:\rife\rife-ncnn-vulkan-20221029-windows\rife-v4.6'
BG = 128.0
SHARPEN = 0.15       # 给插值帧补的锐度，见 rebuild() 说明
TMP = os.path.join(HERE, '_rife_tmp')


def flat(rgba, bg=BG):
    a = rgba[:, :, 3:4] / 255.0
    return rgba[:, :, :3] * a + bg * (1 - a)


def rebuild(mid_path, mid_a_path, sharpen=0.15, alpha_floor=0.0):
    """把 RIFE 输出的不透明中间帧还原成 RGBA。

    **alpha 也必须走 RIFE，不能线性平均**——角色整体位移时（跳跃），两帧的 alpha
    轮廓落在不同位置，平均出来就是"两个位置各一半"的双层半透明影子。wave/eat
    角色原地不动，平均等于原地平均，问题不显；跳跃一上来就露馅。

    `alpha_floor` 再压一道淡影：位移大的地方 RIFE 会插出一层 alpha 只有 1~20 的
    第二层轮廓（实测落下那几帧头顶约 1100 个这样的像素），肉眼就是"头发上有层雾"。
    低于该阈值的 alpha 直接归零、之上重新拉伸。代价是边缘略硬。

    RIFE 出来的中间帧天生比原帧软约 20%，补一点锐度进去，否则 24 帧交替播放
    就是"一清一糊"。unsharp 不能创造细节、只能把残留的高频提回去，所以量要小：
    0.15 正好补平，0.35 就开始出白边。先合成到中性底再锐化，免得透明区的黑被
    锐成光晕。
    """
    mid = np.array(Image.open(mid_path).convert('RGB')).astype(np.float32)
    am = np.array(Image.open(mid_a_path).convert('L')).astype(np.float32)
    if alpha_floor > 0:
        am = np.clip((am - alpha_floor) * 255.0 / (255.0 - alpha_floor), 0, 255)
    am = am[:, :, None]
    w = am / 255.0
    rgb = np.clip((mid - BG * (1 - w)) / np.maximum(w, 1e-3), 0, 255)
    if sharpen > 0:
        srgb = rgb * w + BG * (1 - w)
        srgb = np.clip(srgb + sharpen * (srgb - cv2.GaussianBlur(srgb, (0, 0), 1.2)), 0, 255)
        rgb = np.clip((srgb - BG * (1 - w)) / np.maximum(w, 1e-3), 0, 255)
    return np.dstack([rgb, am[:, :, 0]])


def run(name):
    d = os.path.join(HERE, 'whalegirl_%s' % name)
    meta = json.load(open(os.path.join(d, '%s_meta.json' % name)))
    files = sorted(glob.glob(os.path.join(d, 'frame_*.png')))
    frames = [np.array(Image.open(f).convert('RGBA')).astype(np.float32) for f in files]
    n = len(frames)
    loop = meta.get('loop', False)

    tin_rgb, tout_rgb = os.path.join(TMP, 'in_rgb'), os.path.join(TMP, 'out_rgb')
    tin_a, tout_a = os.path.join(TMP, 'in_a'), os.path.join(TMP, 'out_a')
    shutil.rmtree(TMP, ignore_errors=True)
    for p in (tin_rgb, tout_rgb, tin_a, tout_a):
        os.makedirs(p, exist_ok=True)

    # RGB 合成到中性底、alpha 复制三份当灰度图——两份都送 RIFE
    # 循环动作末尾补一份首帧，让 RIFE 也算出"尾 -> 首"的中间帧
    seq = list(range(n)) + ([0] if loop else [])
    for k, idx in enumerate(seq):
        f = frames[idx]
        Image.fromarray(np.clip(flat(f), 0, 255).astype(np.uint8)).save(
            os.path.join(tin_rgb, '%08d.png' % (k + 1)))
        al = f[:, :, 3]
        Image.fromarray(np.stack([al.astype(np.uint8)] * 3, axis=2)).save(
            os.path.join(tin_a, '%08d.png' % (k + 1)))

    print('== %s  %d 帧  loop=%s  送 RIFE %d 张（RGB + alpha 各一趟）'
          % (name, n, loop, len(seq)))
    r = None
    for tin, tout in ((tin_rgb, tout_rgb), (tin_a, tout_a)):
        r = subprocess.run([RIFE, '-i', tin, '-o', tout, '-m', MODEL, '-g', '0'],
                           capture_output=True, text=True)
    mids = sorted(glob.glob(os.path.join(tout_rgb, '*.png')))
    mids_a = sorted(glob.glob(os.path.join(tout_a, '*.png')))
    if len(mids) < 2 * n - 1 or len(mids_a) < 2 * n - 1:
        raise RuntimeError('RIFE 出图不足：rgb %d / alpha %d\n%s'
                           % (len(mids), len(mids_a), r.stderr[-500:]))

    # 位移大的动作，插值帧会拖出一层 alpha 1~20 的淡影，需要压掉（见 rebuild 说明）
    ALPHA_FLOOR = {'jump': 25.0}
    floor = ALPHA_FLOOR.get(name, 0.0)

    out = []
    for i in range(n):
        out.append(frames[i])
        if i < n - 1 or loop:              # 不循环的动作没有"尾->首"那一帧
            k = 2 * i + 1                  # 奇数位才是中间帧
            if k < len(mids):
                out.append(rebuild(mids[k], mids_a[k], SHARPEN, floor))

    H, W = out[0].shape[:2]
    sheet = Image.fromarray(np.concatenate(
        [np.clip(f, 0, 255).astype(np.uint8) for f in out], axis=1))
    sheet.save(os.path.join(d, '%s_rife_sheet.png' % name))
    for i, f in enumerate(out):
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(
            os.path.join(d, 'rife_%02d.png' % (i + 1)))

    preview = Image.new('RGB', sheet.size, (232, 232, 236))
    preview.paste(sheet, (0, 0), sheet)
    gif = [preview.crop((i * W, 0, (i + 1) * W, H)) for i in range(len(out))]
    fps2 = meta['fps'] * 2
    gif[0].save(os.path.join(d, '%s_rife_preview.gif' % name), save_all=True,
                append_images=gif[1:], duration=int(1000 / fps2), loop=0, disposal=2)

    # 位移大的动作，插值帧会拖出一层 alpha 1~20 的淡影，需要压掉（见 rebuild 说明）
    ALPHA_FLOOR = {'jump': 25.0}
    floor = ALPHA_FLOOR.get(name, 0.0)

    def assemble(sharpen):
        o = []
        for i in range(n):
            o.append(frames[i])
            if i < n - 1 or loop:          # 不循环的动作没有"尾->首"那一帧
                k = 2 * i + 1              # 奇数位才是中间帧
                if k < len(mids):
                    o.append(rebuild(mids[k], mids_a[k], sharpen, floor))
        return o

    def lap(f):
        al = f[:, :, 3:4] / 255.0
        g = np.array(Image.fromarray(np.clip(
            f[:, :, :3] * al + BG * (1 - al), 0, 255).astype(np.uint8)).convert('L'),
            dtype=np.float32)
        gy, gx = np.gradient(g)
        return (gx * gx + gy * gy)[f[:, :, 3] > 128].var()

    for sh in (0.0, SHARPEN):
        s = np.array([lap(f) for f in assemble(sh)])
        print('  锐化 %.2f -> 清晰度变异 %.1f%%   原帧位%.0f 插值位%.0f'
              % (sh, 100 * s.std() / s.mean(), np.median(s[0::2]), np.median(s[1::2])))
    out = assemble(SHARPEN)
    ds = [np.abs(flat(out[i]) - flat(out[i + 1])).mean() for i in range(len(out) - 1)]
    print('  出帧 %d 张 @ %dfps   相邻差异: 原 %.2f -> 补帧后 %.2f'
          % (len(out), fps2,
             np.mean([np.abs(flat(frames[i]) - flat(frames[(i + 1) % n])).mean() for i in range(n - 1)]),
             np.mean(ds)))


if __name__ == '__main__':
    try:
        for k in (sys.argv[1:] or ['wave', 'eat']):
            run(k)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
