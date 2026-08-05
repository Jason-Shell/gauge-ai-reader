# -*- coding: utf-8 -*-
"""benchmark.py —— 推理性能基准（延迟 / 等效 FPS）。

解决“实时性能无数据”的盲区：量化 detector 后端推理与 read_frame 全链路
（检测 + 精修 + 解算）的耗时。

用法（项目根目录）：
    python scripts/benchmark.py --source "gauge_sim_3 dataset/0.jpg" --backend onnx
    python scripts/benchmark.py --source 0 --backend tflite --iters 100

说明：
    - 视频 / 摄像头源取首帧做静态基准；真实视频吞吐还取决于采集与显示；
    - 结果同时给出 detector 原始推理耗时与全链路耗时。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reader import GaugeReader  # noqa: E402


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(str(path).strip())
    return p if p.is_absolute() else PROJECT_DIR / p


def load_frame(source) :
    """从图片 / 目录 / 视频 / 摄像头取一帧用于基准。"""
    s = str(source)
    if s.isdigit():
        cap = cv2.VideoCapture(int(s))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"无法读取摄像头: {source}")
        return frame
    src = resolve_path(s)
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not imgs:
            raise FileNotFoundError(f"目录中没有图片: {src}")
        return cv2.imread(str(imgs[0]))
    if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"无法读取视频首帧: {src}")
        return frame
    frame = cv2.imread(str(src))
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {src}")
    return frame


def _report(name: str, times_ms: List[float]) -> None:
    mean = statistics.fmean(times_ms)
    p90 = sorted(times_ms)[int(len(times_ms) * 0.9) - 1]
    print(f"  {name:<22} mean={mean:7.2f} ms  p50={statistics.median(times_ms):7.2f} ms"
          f"  p90={p90:7.2f} ms  等效FPS={1000.0 / mean:6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="推理性能基准")
    parser.add_argument("--source", default=None, help="图片/目录/视频/摄像头")
    parser.add_argument("--config", default="config/gauge.yaml")
    parser.add_argument("--backend", default=None,
                        choices=["ultralytics", "onnx", "tflite", "mock"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    if args.backend:
        config["model"]["backend"] = args.backend
    source = args.source or config.get("io", {}).get("input_image", "meter.jpg")

    reader = GaugeReader(config)
    frame = load_frame(source)
    print(f"[bench] 后端={reader.detector.backend}  输入={frame.shape[1]}x{frame.shape[0]}"
          f"  iters={args.iters} warmup={args.warmup}")

    for _ in range(args.warmup):
        reader.read_frame(frame)

    det_times: List[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        reader.detector.predict(frame)
        det_times.append((time.perf_counter() - t0) * 1000.0)

    full_times: List[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        reader.read_frame(frame)
        full_times.append((time.perf_counter() - t0) * 1000.0)

    print("--- detector 后端原始推理 ---")
    _report("predict", det_times)
    print("--- read_frame 全链路（检测+精修+解算）---")
    _report("read_frame", full_times)


if __name__ == "__main__":
    main()
