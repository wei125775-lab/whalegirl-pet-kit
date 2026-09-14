"""把一张透明姿势立绘用程序化变形烘焙成微动循环。

沿用 idle 那套：呼吸（绕脚底纵向缩放）+ 摇摆（越往上越大）+ 飘动（细长部位行波抖动），
三路都是正弦所以循环天然闭合。位移场连续，不用切层。
姿势素材是静态的，所以这里不眨眼——这批图里没有每个姿势各自的闭眼版。
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

_SELF = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(os.path.dirname(_SELF), '_archive')  # 脚本已移到 tools/，旧基准数据在 ../_archive/
POSE_DIR = os.path.join(HERE, 'whalegirl_poses')
OUT_ROOT = os.path.join(HERE, 'whalegirl_actions')

SS = 3                 # 超采样：位移会把细部抹平，放大后再降回来才不糊
N_FRAMES = 24
FPS = 12

# 所有动作统一画布，脚底对齐，否则动作切换时角色会跳位置/跳大小
CANVAS_W, CANVAS_H = 248, 268
FOOT_Y = 260           # 脚底在画布中的 y
AXIS_X = CANVAS_W // 2
FOOT_FRAC = 0.15       # 取角色底部这个比例算水平锚点

DEFAULT = dict(
    breath_scale=0.013,   # 呼吸纵向缩放
    sway_x=3.4,           # 上身左右摆幅
    flutter=3.9,          # 细部飘动幅度
    flutter_wave=0.06,    # 行波相位随高度推进
    thin_px=6.0,          # 多细才算细长部位
    field_smooth=5.0,     # 位移场低通半径
    head_cut=0.42,        # 角色高度比例，这一段以上不飘
    head_fade=0.15,
    head_rigid=0.62,      # 以上算头部核心，整头当刚体平移
    head_rigid_fade=0.10,
    breath_cycles=2,      # 一个循环呼吸几次（摇摆固定 1 次）
    speed=1.0,
)

# 每个动作按自己的性格给一套参数。待机是轻呼吸慢晃；开心要晃得活泼、头发飘得欢；
# 思考几乎不动；害羞幅度小。只覆盖有差异的项，其余走 DEFAULT。
PRESETS = {
    '07_happy': dict(breath_scale=0.016, sway_x=7.0, flutter=5.6, flutter_wave=0.09),
}


def _up(a):
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.array(im.resize((a.shape[1] * SS, a.shape[0] * SS), Image.LANCZOS)).astype(np.float32)


def _down(a):
    """降回目标尺寸，按预乘 alpha 走，否则透明区的黑会渗进边缘"""
    al = a[:, :, 3:4] / 255.0
    pm = np.dstack([a[:, :, :3] * al, a[:, :, 3]])
    im = Image.fromarray(np.clip(pm, 0, 255).astype(np.uint8))
    sm = np.array(im.resize((a.shape[1] // SS, a.shape[0] // SS), Image.LANCZOS)).astype(np.float32)
    sa = sm[:, :, 3:4]
    rgb = np.where(sa > 1e-3, sm[:, :, :3] / np.maximum(sa / 255.0, 1e-6), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(sa[:, :, 0], 0, 255)])


def fields(rgba, P):
    """预计算：细部权重、高度权重、脚底、行波相位、头部刚性权重"""
    mask = rgba[:, :, 3] > 128
    dist = ndimage.distance_transform_edt(mask)
    thin = 1.0 - np.clip(dist / (P['thin_px'] * SS), 0, 1)
    ys, _ = np.where(mask)
    foot = float(ys.max())
    H = mask.shape[0]
    yy = np.arange(H)[:, None].astype(np.float32)
    height = np.clip((foot - yy) / (H * 0.62), 0, 1)
    hfrac = np.clip((foot - yy) / max(1.0, foot - ys.min()), 0, 1)
    head_damp = 1.0 - np.clip((hfrac - P['head_cut']) / P['head_fade'], 0, 1)
    weight = thin * (0.35 + 0.65 * height) * head_damp * mask
    phase = -yy * P['flutter_wave']
    rigid = np.clip((hfrac - P['head_rigid']) / P['head_rigid_fade'], 0, 1) * mask
    return (weight.astype(np.float32), height.astype(np.float32), foot,
            phase.astype(np.float32), rigid.astype(np.float32))


def warp(premult, dx, dy):
    H, W = dx.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = np.empty_like(premult)
    for c in range(4):
        out[:, :, c] = cv2.remap(premult[:, :, c], gx + dx, gy + dy, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def render(rgba, F, t, P):
    weight, height, foot, phase, rigid = F
    H, W = rgba.shape[:2]
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]

    s = 1.0 + P['breath_scale'] * np.sin(2 * np.pi * P['breath_cycles'] * t)
    dx_b = (xx - W / 2) * (-(s - 1.0) * 0.5)
    dy_b = (yy - foot) * (s - 1.0)

    dx_s = P['sway_x'] * SS * height * np.sin(2 * np.pi * t)

    ph = 2 * np.pi * t + phase
    dx_f = P['flutter'] * SS * weight * np.sin(ph)
    dy_f = P['flutter'] * SS * weight * np.sin(ph + np.pi / 2)

    dx = dx_b + dx_s + dx_f
    dy = dy_b + dy_f
    dx = ndimage.gaussian_filter(dx, P['field_smooth'] * SS, mode='nearest').astype(np.float32)
    dy = ndimage.gaussian_filter(dy, P['field_smooth'] * SS, mode='nearest').astype(np.float32)

    core = rigid > 0.999
    if core.any():
        r = rigid
        dx = dx * (1 - r) + dx[core].mean() * r
        dy = dy * (1 - r) + dy[core].mean() * r

    al = rgba[:, :, 3] / 255.0
    premult = np.dstack([rgba[:, :, :3] * al[:, :, None], rgba[:, :, 3]])
    w = warp(premult, dx, dy)
    a = np.clip(w[:, :, 3], 0, 255)
    rgb = np.where(a[:, :, None] > 1e-3, w[:, :, :3] / np.maximum(a[:, :, None] / 255.0, 1e-6), 0)
    return np.dstack([np.clip(rgb, 0, 255), a])


def flat(f, bg=232.0):
    a = f[:, :, 3:4] / 255.0
    return f[:, :, :3] * a + bg * (1 - a)


def build(stem, out_dir, P):
    os.makedirs(out_dir, exist_ok=True)
    raw = np.asarray(Image.open(os.path.join(POSE_DIR, stem + '.png')).convert('RGBA')).astype(np.float32)

    m = raw[:, :, 3] > 128
    ys, xs = np.where(m)
    fy0 = ys.max() - int((ys.max() - ys.min()) * FOOT_FRAC)
    ax = float(np.where(m[fy0:ys.max() + 1])[1].mean())      # 下部质心当水平锚点

    base = _up(raw)
    canvas = np.zeros((CANVAS_H * SS, CANVAS_W * SS, 4), np.float32)
    ox = int(round(AXIS_X * SS - ax * SS))
    oy = int(round(FOOT_Y * SS - ys.max() * SS))
    sx, sy = max(0, -ox), max(0, -oy)
    dx, dy = max(0, ox), max(0, oy)
    h = min(base.shape[0] - sy, canvas.shape[0] - dy)
    w = min(base.shape[1] - sx, canvas.shape[1] - dx)
    canvas[dy:dy + h, dx:dx + w] = base[sy:sy + h, sx:sx + w]

    F = fields(canvas, P)
    frames = []
    for i in range(N_FRAMES):
        t = (i / N_FRAMES) * P['speed'] % 1.0
        f = _down(render(canvas, F, t, P))
        frames.append(f)
    if P['speed'] != 1.0:
        frames = frames  # 相位已按 speed 重采样

    for i, f in enumerate(frames):
        Image.fromarray(f.astype(np.uint8)).save(os.path.join(out_dir, 'frame_%02d.png' % (i + 1)))
    sheet = Image.fromarray(np.concatenate(frames, axis=1).astype(np.uint8))
    sheet.save(os.path.join(out_dir, '%s_rig_sheet.png' % stem))
    gif = [Image.fromarray(flat(f).astype(np.uint8)) for f in frames]
    gif[0].save(os.path.join(out_dir, '%s_rig_preview.gif' % stem), save_all=True,
                append_images=gif[1:], duration=int(1000 / FPS), loop=0, disposal=2)

    feets = [np.where(f[:, :, 3] > 128)[0].max() for f in frames]
    ds = [np.abs(flat(frames[i]) - flat(frames[(i + 1) % N_FRAMES])).mean() for i in range(N_FRAMES)]
    print('%-14s %3dx%-3d 脚底极差 %dpx  帧间差 均%.2f 最大%.2f  %s' % (
        stem, frames[0].shape[1], frames[0].shape[0], max(feets) - min(feets),
        np.mean(ds), max(ds), os.path.basename(out_dir)))


if __name__ == '__main__':
    want = sys.argv[1:]
    done = 0
    for fn in sorted(os.listdir(POSE_DIR)):
        if not fn.endswith('.png') or fn.startswith('_'):
            continue
        stem = fn[:-4]
        if want and not any(stem.startswith(w) for w in want):
            continue
        build(stem, os.path.join(OUT_ROOT, stem.split('_', 1)[1]),
              dict(DEFAULT, **PRESETS.get(stem, {})))
        done += 1
    print('共处理 %d 个姿势 -> %s' % (done, OUT_ROOT))
