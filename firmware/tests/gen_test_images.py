# -*- coding: utf-8 -*-
"""生成合成表盘 PGM 测试图（供 host_test.cpp 回归验证径向扫描移植）。"""

import sys
from pathlib import Path

import cv2
import numpy as np

TESTS = Path(__file__).resolve().parents[2] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from helpers import make_gauge  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def write_pgm(path: Path, gray: np.ndarray) -> None:
    h, w = gray.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(gray.tobytes())


for deg in [45, 90, 135, 200, 300]:
    img = make_gauge(needle_deg=float(deg))
    write_pgm(OUT / f"gauge_{deg}.pgm", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# 纯白图（无指针，应返回 not ok）
blank = np.full((400, 400), 255, np.uint8)
write_pgm(OUT / "blank.pgm", blank)

print("generated:", sorted(p.name for p in OUT.glob("*.pgm")))
