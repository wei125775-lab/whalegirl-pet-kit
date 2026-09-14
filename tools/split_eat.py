"""把 30 帧的吃饭拆成"吃"（循环）和"收碗"（一次性）。

拆的理由：用户要"Claude 思考时一直吃、不收碗"，而这条素材是**线性推进**的
（站立→端碗→吃→碗缩小消失→放下），没有天然闭合的循环段——逐帧差实测所有片段
首尾差都 ≥7，和相邻帧差同量级，找不到能直接 loop 的段。所以"吃"走 pingpong
（pet.json 的字段，往返播放天然保证衔接），正好绕开这个问题。

切点：
  帧 10~22  端着碗吃东西。上限卡 22 是因为 23 帧起碗开始缩小（源视频的瑕疵）
  帧 22~30  碗消失 + 放下手

读一次写两次，不覆盖源帧之前先把要用的都读进内存。
"""
import json
import os

import numpy as np
from PIL import Image

_SELF = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(_SELF), 'v2', 'actions')
SRC = os.path.join(ROOT, 'eat')

EAT_RANGE = (10, 22)      # 闭区间，1-based
PUTAWAY_RANGE = (22, 30)
FPS = {'eat': 12, 'putaway': 8}   # 收碗慢一点，9 帧 @8fps = 1.1 秒

W, H = 436, 640
BG = (232, 232, 236)


def load_all():
    out = {}
    for i in range(1, 31):
        p = os.path.join(SRC, 'frame_%02d.png' % i)
        out[i] = np.array(Image.open(p).convert('RGBA'))
    return out


def write_action(name, frames, loop, fps):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    # 清掉旧的 frame_*.png，免得留下上一个长度的尾巴
    for f in os.listdir(d):
        if f.startswith('frame_') and f.endswith('.png'):
            os.remove(os.path.join(d, f))

    for i, f in enumerate(frames, 1):
        Image.fromarray(f).save(os.path.join(d, 'frame_%02d.png' % i))

    arr = np.concatenate(frames, axis=1)
    sheet = Image.fromarray(arr)
    sheet.save(os.path.join(d, '%s_sheet.png' % name))

    preview = Image.new('RGB', sheet.size, BG)
    preview.paste(sheet, (0, 0), sheet)
    gif = [preview.crop((i * W, 0, (i + 1) * W, H)) for i in range(len(frames))]
    gif[0].save(os.path.join(d, '%s_preview.gif' % name), save_all=True,
                append_images=gif[1:], duration=int(1000 / fps), loop=0, disposal=2)

    cols = 6
    rows = (len(frames) + cols - 1) // cols
    check = Image.new('RGB', (W * cols, H * rows), BG)
    for i, f in enumerate(frames):
        im = Image.fromarray(f)
        check.paste(im, ((i % cols) * W, (i // cols) * H), im)
    check.save(os.path.join(d, '%s_check.png' % name))

    json.dump({"name": "whalegirl", "cellWidth": W, "cellHeight": H,
               "frames": len(frames), "fps": fps, "action": name,
               "loop": loop, "file": '%s_sheet.png' % name},
              open(os.path.join(d, '%s_meta.json' % name), 'w'),
              ensure_ascii=False, indent=2)
    return len(frames)


def main():
    src = load_all()
    # 收碗和吃饭共用帧 22 那一帧当接点，衔接才连续
    eat = [src[i] for i in range(EAT_RANGE[0], EAT_RANGE[1] + 1)]
    putaway = [src[i] for i in range(PUTAWAY_RANGE[0], PUTAWAY_RANGE[1] + 1)]

    n1 = write_action('eat', eat, loop=True, fps=FPS['eat'])
    n2 = write_action('putaway', putaway, loop=False, fps=FPS['putaway'])

    for name, frames, fps, loop in (('eat', eat, FPS['eat'], True),
                                    ('putaway', putaway, FPS['putaway'], False)):
        fl = [f[:, :, :3].astype(np.float32) * (f[:, :, 3:4] / 255.0) + 232.0 * (1 - f[:, :, 3:4] / 255.0)
              for f in frames]
        ds = [float(np.abs(fl[i] - fl[i + 1]).mean()) for i in range(len(fl) - 1)]
        print('%-9s %2d 帧 @%2dfps = %.2fs  loop=%s  段内帧差 均%.2f 最大%.2f  首尾差 %.2f'
              % (name, len(frames), fps, len(frames) / fps, loop,
                 np.mean(ds), np.max(ds), np.abs(fl[0] - fl[-1]).mean()))
    print('源 30 帧保留在 %s（eat 覆盖为 %d 帧）' % (SRC, n1))


if __name__ == '__main__':
    main()
