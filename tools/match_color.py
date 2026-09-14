"""把视频动作的颜色对齐到分层待机（idle）。

两批素材的色源本来就不同：idle 来自 see-through 的 PSD（原始立绘，色彩准确），
其余几条来自豆包视频（H.264 有损压缩 + YUV 色度下采样 + 绿幕 despill）。

实测 eat 比 idle 整体暗 21~26，而且**差值随亮度增大**（暗部 7~13、亮部 31~36），
说明是对比度差异不是单纯偏移，所以按逐通道线性模型 ref ≈ a*src + b 拟合。

用最小二乘 + 迭代剔离群：挥手的共同区域里有大量姿势不重合的像素（手臂抬到别处去了），
不剔会把系数带偏。只改 RGB，alpha 不动。
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

_SELF = os.path.dirname(os.path.abspath(__file__))
ACT = os.path.join(os.path.dirname(_SELF), 'v2', 'actions')
REF = 'idle'
TARGETS = ['wave', 'touchface', 'eat', 'putaway', 'heart', 'sway']


def load(act, name='frame_01.png'):
    p = os.path.join(ACT, act, name)
    return np.array(Image.open(p).convert('RGBA')).astype(np.float32)


def fit(ref, src, iters=4):
    """逐通道最小二乘。返回 [(a, b, rmse), ...]，三个通道各一组。"""
    m = (ref[:, :, 3] > 220) & (src[:, :, 3] > 220)
    out = []
    for c in range(3):
        x = src[:, :, c][m].astype(np.float64)
        y = ref[:, :, c][m].astype(np.float64)
        keep = np.ones(len(x), bool)
        a, b = 1.0, 0.0
        for _ in range(iters):
            A = np.vstack([x[keep], np.ones(int(keep.sum()))]).T
            a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
            r = np.abs(y - (a * x + b))
            keep = r < max(6.0, 2.0 * float(np.std(r[keep])))
        rmse = float(np.sqrt(np.mean((y[keep] - (a * x[keep] + b)) ** 2)))
        out.append((float(a), float(b), rmse))
    return out


def apply_coef(rgba, coef):
    out = rgba.copy()
    for c, (a, b, _) in enumerate(coef):
        out[:, :, c] = np.clip(rgba[:, :, c] * a + b, 0, 255)
    return out


def rebuild(act, coef):
    d = os.path.join(ACT, act)
    fs = sorted(glob.glob(os.path.join(d, 'frame_*.png')))
    frames = []
    for p in fs:
        g = apply_coef(np.array(Image.open(p).convert('RGBA')).astype(np.float32), coef)
        Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save(p)
        frames.append(g)

    W, H = frames[0].shape[1], frames[0].shape[0]
    sheet = Image.fromarray(np.concatenate(frames, axis=1).astype(np.uint8))
    sheet.save(os.path.join(d, '%s_sheet.png' % act))
    preview = Image.new('RGB', sheet.size, (232, 232, 236))
    preview.paste(sheet, (0, 0), sheet)
    gif = [preview.crop((i * W, 0, (i + 1) * W, H)) for i in range(len(frames))]
    meta = json.load(open(os.path.join(d, '%s_meta.json' % act)))
    gif[0].save(os.path.join(d, '%s_preview.gif' % act), save_all=True,
                append_images=gif[1:], duration=int(1000 / meta['fps']), loop=0, disposal=2)

    cols = 6
    rows = (len(frames) + cols - 1) // cols
    check = Image.new('RGB', (W * cols, H * rows), (232, 232, 236))
    for i, f in enumerate(frames):
        im = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
        check.paste(im, ((i % cols) * W, (i // cols) * H), im)
    check.save(os.path.join(d, '%s_check.png' % act))
    return len(frames)


def main():
    want = sys.argv[1:] or TARGETS
    ref = load(REF)
    for act in want:
        src = load(act)
        coef = fit(ref, src)
        after = apply_coef(src, coef)
        m = (ref[:, :, 3] > 220) & (after[:, :, 3] > 220)
        before_abs = np.abs(ref[:, :, :3][m] - src[:, :, :3][m]).mean(0)
        after_abs = np.abs(ref[:, :, :3][m] - after[:, :, :3][m]).mean(0)
        n = rebuild(act, coef)
        print('%-10s %3d 帧  a=%s  b=%s  rmse=%.1f/%.1f/%.1f   绝对差 %.1f/%.1f/%.1f -> %.1f/%.1f/%.1f'
              % (act, n,
                 '/'.join('%.3f' % a for a, _, _ in coef),
                 '/'.join('%+.1f' % b for _, b, _ in coef),
                 coef[0][2], coef[1][2], coef[2][2],
                 before_abs[0], before_abs[1], before_abs[2],
                 after_abs[0], after_abs[1], after_abs[2]))


if __name__ == '__main__':
    main()
