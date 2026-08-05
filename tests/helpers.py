# -*- coding: utf-8 -*-
"""测试公共工具：合成表盘图、项目配置加载。"""

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def make_gauge(size: int = 400, needle_deg: float = 45.0,
               center=(200, 200), radius: float = 180.0):
    """合成一张白底黑指针表盘图（含短刻度线），用于精修 / 端到端测试。"""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(img, center, int(radius), (0, 0, 0), 3)
    for d in range(0, 360, 15):
        a = math.radians(d)
        p1 = (int(center[0] + radius * 0.70 * math.cos(a)),
              int(center[1] + radius * 0.70 * math.sin(a)))
        p2 = (int(center[0] + radius * 0.85 * math.cos(a)),
              int(center[1] + radius * 0.85 * math.sin(a)))
        cv2.line(img, p1, p2, (0, 0, 0), 2)
    a = math.radians(needle_deg)
    tip = (int(center[0] + radius * 0.85 * math.cos(a)),
           int(center[1] + radius * 0.85 * math.sin(a)))
    cv2.line(img, center, tip, (0, 0, 0), 8)
    return img


def load_project_config(backend: str = "onnx") -> dict:
    """加载项目配置并切换后端（保留其余配置用于集成测试）。"""
    with open(PROJECT / "config" / "gauge.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["backend"] = backend
    return cfg
