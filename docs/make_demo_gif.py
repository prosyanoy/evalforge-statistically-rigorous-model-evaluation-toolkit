"""Render docs/demo.gif: a ~60 s terminal demo. The commands are really executed and their
captured output is replayed with a typing animation.

    .venv/bin/python docs/make_demo_gif.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"), 12)
BOLD = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono-Bold.ttf"), 12)
COLS, ROWS, CW, LH, PAD = 132, 25, 7, 17, 12
W, H = COLS * CW + 2 * PAD, ROWS * LH + 2 * PAD + 20
BG, FG, GREEN, DIM = "#14171c", "#d7dce2", "#7ee787", "#8b949e"
COLORS = {"SHIP": "#7ee787", "ITERATE": "#e3b341", "STOP": "#ff7b72"}

# (command shown, real argv, seconds to hold after the output is complete)
SCENES = [
    ("evalforge info", ["-m", "evalforge", "info"], 3),
    ("python examples/ab_comparison.py", ["examples/ab_comparison.py"], 11),
    ("evalforge report --out reports", ["-m", "evalforge", "report", "--out", "{tmp}"], 7),
    ("python benchmarks/bench.py", ["benchmarks/bench.py"], 9),
]
LINE_DELAY, KEY_DELAY, TARGET = 0.9, 0.06, 60.0


def run(argv):
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([sys.executable, *[a.replace("{tmp}", tmp) for a in argv]], cwd=ROOT,
                             capture_output=True, text=True, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    return out.stdout.rstrip("\n").splitlines()


def frame(lines, cursor=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 4), "evalforge demo", font=FONT, fill=DIM)
    for i, (text, is_cmd) in enumerate(lines[-ROWS:]):
        y = PAD + 16 + i * LH
        if is_cmd:
            d.text((PAD, y), "$ ", font=BOLD, fill=GREEN)
            d.text((PAD + 2 * CW, y), text, font=FONT, fill=FG)
        else:
            color = next((c for k, c in COLORS.items() if f" {k} " in f" {text} " or text.startswith("Decision:") and k in text), FG)
            d.text((PAD, y), text[:COLS], font=FONT, fill=color)
    if cursor:
        row = min(len(lines), ROWS) - 1
        last = lines[-1][0] if lines else ""
        x = PAD + CW * (len(last) + (2 if lines and lines[-1][1] else 0))
        d.rectangle([x, PAD + 16 + row * LH + 2, x + CW - 1, PAD + 16 + row * LH + LH - 2], fill=FG)
    return img


def main():
    frames, durations = [], []

    def add(lines, secs, cursor=True):
        frames.append(frame(lines, cursor))
        durations.append(secs)

    for cmd, argv, hold in SCENES:
        out = run(argv)
        screen = []
        for i in range(0, len(cmd) + 1, 2):
            add([*screen, (cmd[:i], True)], KEY_DELAY)
        screen.append((cmd, True))
        add(screen, 0.5, cursor=False)
        for line in out:
            screen.append((line, False))
            add(screen, LINE_DELAY if len(out) < 12 else 0.5, cursor=False)
        add(screen, hold, cursor=False)
    total = sum(durations)
    print(f"raw length {total:.1f}s; scaling to ~{TARGET:.0f}s")
    scale = TARGET / total
    durations = [max(0.04, d * scale) for d in durations]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    pal[0].save(ROOT / "docs/demo.gif", save_all=True, append_images=pal[1:], duration=[int(d * 1000) for d in durations],
                loop=0, optimize=True, disposal=1)
    print(f"wrote docs/demo.gif: {len(frames)} frames, {sum(durations):.1f}s, "
          f"{(ROOT / 'docs/demo.gif').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
