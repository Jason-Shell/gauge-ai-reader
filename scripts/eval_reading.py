# -*- coding: utf-8 -*-
"""eval_reading.py —— 端到端读数精度评估（带真值）。

解决“mAP 很高但读数精度没量化”的盲区：直接比较程序输出读数与真值。

真值来源（二选一）：
    1. 图片文件名：文件名第一个下划线前的部分为纯数字时，视为 bar 真值，
       例如 "gauge_sim_3 dataset/100.jpg" -> 100 bar；
    2. CSV 文件：header 为 path,bar[,psi]，path 相对项目根目录解析。

输出：逐样本 预测 / 真值 / 误差，以及 MAE / RMSE / 最大绝对误差；
默认与 config 一致开启传统方法精修，可用 --no-refine 对比 DL-only。

用法（项目根目录）：
    python scripts/eval_reading.py --source "gauge_sim_3 dataset" --backend onnx
    python scripts/eval_reading.py --source truth.csv --backend onnx --no-refine
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reader import GaugeReader  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_path(path: str) -> Path:
    p = Path(str(path).strip())
    return p if p.is_absolute() else PROJECT_DIR / p


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def filename_truth(path: Path) -> Optional[float]:
    """从文件名解析 bar 真值：第一个下划线前为纯数字才接受。"""
    base = path.stem.split("_")[0]
    if re.fullmatch(r"\d+(\.\d+)?", base):
        return float(base)
    return None


def load_csv(path: Path) -> List[Tuple[Path, float, Optional[float]]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = row["path"].strip()
            bar = float(row["bar"])
            psi = float(row["psi"]) if row.get("psi", "").strip() else None
            rows.append((resolve_path(p), bar, psi))
    return rows


def collect_samples(source: Path) -> List[Tuple[Path, float, Optional[float]]]:
    if source.is_dir():
        samples = []
        for p in sorted(source.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                t = filename_truth(p)
                if t is not None:
                    samples.append((p, t, None))
        return samples
    if source.suffix.lower() == ".csv":
        return load_csv(source)
    raise ValueError(
        f"--source 必须是目录或 CSV：{source}（目录要求文件名可解析真值）")


def evaluate(reader: GaugeReader, samples: List[Tuple[Path, float, Optional[float]]]):
    errors_bar, errors_psi = [], []
    refine_cnt = 0
    print(f"{'样本':<42} {'真值bar':>8} {'预测bar':>8} {'误差':>8} "
          f"{'预测psi':>9} {'精修':>5} {'conf':>5}")
    for img_path, truth_bar, truth_psi in samples:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"无法读取: {img_path}")
            continue
        readings = reader.read_frame(frame)
        r = readings[0] if readings else None
        if r is None or r.error:
            print(f"{img_path.name:<42} 解算失败: {r.error if r else '无检测'}")
            continue
        err = (r.primary_value or 0.0) - truth_bar
        errors_bar.append(abs(err))
        refine_tag = "Y" if r.refine_used else "-"
        print(f"{img_path.name:<42} {truth_bar:>8.1f} {r.primary_value:>8.1f} "
              f"{err:>+8.1f} {r.secondary_value:>9.1f} {refine_tag:>5} "
              f"{r.conf:>5.2f}")
        if r.refine_used:
            refine_cnt += 1
        if truth_psi is not None and r.secondary_value is not None:
            errors_psi.append(abs(r.secondary_value - truth_psi))

    if not errors_bar:
        print("没有可评估的样本")
        return

    n = len(errors_bar)
    mae = sum(errors_bar) / n
    rmse = math.sqrt(sum(e * e for e in errors_bar) / n)
    worst = max(errors_bar)
    print("-" * 84)
    print(f"样本数: {n}   精修采用率: {refine_cnt}/{n}")
    print(f"bar 读数: MAE={mae:.2f}  RMSE={rmse:.2f}  最大绝对误差={worst:.2f}")
    if errors_psi:
        m = len(errors_psi)
        print(f"psi 读数: MAE={sum(errors_psi) / m:.2f}  "
              f"最大绝对误差={max(errors_psi):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端读数精度评估（带真值）")
    parser.add_argument("--source", required=True,
                        help="图片目录（文件名解析真值）或 CSV（path,bar[,psi]）")
    parser.add_argument("--config", default="config/gauge.yaml")
    parser.add_argument("--backend", default=None,
                        choices=["ultralytics", "onnx", "tflite", "mock"])
    parser.add_argument("--conf", type=float, default=None,
                        help="覆盖检测置信度阈值")
    parser.add_argument("--no-refine", action="store_true",
                        help="关闭传统方法精修，对比 DL-only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.conf is not None:
        config["model"]["conf_thres"] = args.conf
    if args.no_refine:
        config["gauge"]["refinement"]["enabled"] = False
    reader = GaugeReader(config)
    samples = collect_samples(resolve_path(args.source))
    if not samples:
        print("未收集到带真值的样本（目录文件名需为数字，或用 CSV）")
        return
    evaluate(reader, samples)


if __name__ == "__main__":
    main()
