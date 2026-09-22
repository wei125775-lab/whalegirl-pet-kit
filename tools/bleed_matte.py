# -*- coding: utf-8 -*-
"""alpha bleed：把轮廓外残留的背景色换掉。

**为什么需要**：抠像只算出 alpha，RGB 三个通道原样保留着视频背景。despill 又把背景
的绿压成 max(R,B)，于是 alpha=0 的地方留下一层中性灰（实测 82~133，蓝度接近 0）。
桌面端渲染精灵表走的是非预乘插值，窗口宽不是素材基准宽的整数倍就必然重采样，这层
灰会被混进角色边缘，深底上看就是一圈灰白描边。实测 17 个动作里 16 个有这个残留，
只有 idle（PSD 烘焙，不是视频抽帧）是干净的。

**做法**：把 alpha 不透明的像素当成已知色，其余每个像素的 RGB 换成离它最近的已知
像素的颜色。插值混进来的就变成角色自己的颜色。这同时就是去污——半透明边缘像素的
RGB 换成纯前景色，正合 unpremultiplied RGBA 的语义（RGB 是纯色、alpha 是覆盖率）。

判断修好没有看**蓝度**（B - (R+G)/2）而不是"RGB 是否非零"：角色自己的深色描边也是
非零。背景残留是中性灰（蓝度 ≈ 0~10），角色是蓝紫（+25 以上）。

两种素材布局都能处理：
    <dir>/pet.json 里有带 sprite/frameWidth 的 actions  -> 按精灵表切开
    否则                                                -> 目录下每个 png 当一帧

用法：
    python bleed_matte.py <目录> --check          # 只报告
    python bleed_matte.py <目录>                  # 就地修（先备份 .bak）
    python bleed_matte.py <目录> --out <新目录>    # 写到别处
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# alpha 到这个值就算"已知色"。留 5 的余量是因为 PNG 里 255 附近常有 254 的抖动。
SOLID = 250


def bleed_block(block, far=8):
    """(h, w, 4) uint8。离已知色超过 far 像素的透明像素直接填 0。

    这刀不是省事，是两头都赚：轮廓外那一圈（欧氏距离 1~6 就到头了）照样是角色色，
    而远处填成纯黑比原来的中性灰和 bleed 后的角色色渐变都好压——实测全部素材
    89 MB -> 80 MB，比不修还小。far=0（全填）是 90 MB，效果一模一样。"""
    a = np.ascontiguousarray(block)
    known = a[:, :, 3] >= SOLID
    if not known.any():
        return a
    dist, idx = distance_transform_edt(~known, return_distances=True, return_indices=True)
    out = a.copy()
    out[:, :, :3] = a[:, :, :3][idx[0], idx[1]]
    if far > 0:
        out[dist > far] = 0
    return out


def band_stats(rgba, lo=1, hi=4):
    """轮廓外第 lo..hi 圈里 alpha==0 的像素：个数、平均 RGB、蓝度。
    只看这一圈——整片矩形离轮廓太远，缩放插值够不着，不影响观感。"""
    a = rgba.astype(np.int32)
    al = a[:, :, 3]
    solid = al >= SOLID
    if not solid.any():
        return 0, np.zeros(3), 0.0
    d = solid.copy()
    for _ in range(hi):
        n = d.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            n |= np.roll(d, sh, axis=ax)
        d = n
    band = d & ~solid & (al == 0)
    if not band.sum():
        return 0, np.zeros(3), 0.0
    rgb = a[:, :, :3][band].mean(axis=0)
    return int(band.sum()), rgb, float(rgb[2] - (rgb[0] + rgb[1]) / 2)


def plan(pet_dir):
    """-> [(相对路径, [(x, y, w, h), ...])]。"""
    jp = os.path.join(pet_dir, 'pet.json')
    j = json.load(open(jp, encoding='utf-8')) if os.path.exists(jp) else {}
    acts = j.get('actions', {})
    if any('sprite' in v and 'frameWidth' in v for v in acts.values()):
        out = []
        seen = set()
        for v in acts.values():
            if 'sprite' not in v or 'frameWidth' not in v:
                continue
            # 有多个 action 共用一张精灵表（putaway / subagent_putaway）。不去重的话
            # 第二遍会在已修好的文件上再跑一次，备份也被覆盖成修复后的版本。
            if v['sprite'] in seen:
                continue
            seen.add(v['sprite'])
            p = os.path.join(pet_dir, v['sprite'])
            if not os.path.exists(p):
                # 别抛异常：任务列表是在这里一次建完的，一个缺文件会让整轮一条都不处理
                # （而且看起来像"跑完了"）。实测踩过一次。
                print('! 缺文件，跳过：%s' % v['sprite'])
                continue
            fw, fh = int(v['frameWidth']), int(v['frameHeight'])
            with Image.open(p) as im:
                cols, rows = im.width // fw, im.height // fh
            out.append((v['sprite'], [(c * fw, r * fh, fw, fh)
                                      for r in range(rows) for c in range(cols)]))
        return out
    out = []
    for root, _, files in os.walk(pet_dir):
        for f in sorted(files):
            if not f.lower().endswith('.png'):
                continue
            p = os.path.join(root, f)
            with Image.open(p) as im:
                out.append((os.path.relpath(p, pet_dir), [(0, 0, im.width, im.height)]))
    return out


def run(pet_dir, out_dir=None, far=0, check=False):
    jobs = plan(pet_dir)
    if not jobs:
        print('没找到素材')
        return
    print('%-24s %10s  %-20s %7s   %-20s %7s' % (
        '素材', '带内像素', '修复前 RGB', '蓝度', '修复后 RGB', '蓝度'))
    bad = 0
    for rel, boxes in jobs:
        src = os.path.join(pet_dir, rel)
        arr = np.array(Image.open(src).convert('RGBA'))
        out = arr.copy()
        n0 = r0 = b0 = None
        n1 = r1 = b1 = None
        for k, (x, y, w, h) in enumerate(boxes):
            blk = arr[y:y + h, x:x + w]
            fixed = bleed_block(blk, far)
            out[y:y + h, x:x + w] = fixed
            if k == 0:      # 第一帧做报告，够代表整表
                n0, r0, b0 = band_stats(blk)
                n1, r1, b1 = band_stats(fixed)
        if b0 is not None and b0 < 15.0:
            bad += 1
        f = lambda c: '—' if c is None else '[%3.0f %3.0f %3.0f]' % tuple(c)
        print('%-24s %10d  %-20s %+7.1f   %-20s %+7.1f' % (
            rel, n0, f(r0), b0, f(r1), b1))
        if not check:
            if out_dir:
                dst = os.path.join(out_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            else:
                dst = src
                shutil.copy2(src, src + '.bak')
            Image.fromarray(out).save(dst)
    print('\n%d/%d 个素材的轮廓外残留是中性灰（蓝度 < 15）' % (bad, len(jobs)))
    if not check:
        print('已写回' if not out_dir else '已写到 ' + out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--out', default=None)
    ap.add_argument('--far', type=int, default=8,
                    help='离已知色超过这个距离的透明像素填 0（默认 8，见 bleed_block）')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    run(os.path.expanduser(a.path), a.out, a.far, a.check)


if __name__ == '__main__':
    main()
