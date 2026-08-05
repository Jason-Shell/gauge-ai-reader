# -*- coding: utf-8 -*-
"""reader.py —— 推理编排器：调用检测器，串联数学解算，输出双量程读数。

职责：
    1. 按配置创建检测器（ultralytics / onnx / tflite / mock）；
    2. 对单帧执行 检测 -> 关键点 -> 角度比例 -> 主/副量程数值映射；
    3. 以 Reading 数据类返回结构化结果，供 main.py 绘制与输出。

本模块不涉及任何传统图像处理方法，只做推理与数学编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from detector import GaugeDetector, create_detector
from geometry import calculate_ratio, calculate_value, round_value


@dataclass
class Reading:
    """单块表盘的完整读数结果。"""

    bbox: tuple                          # (x1, y1, x2, y2) 原图像素坐标
    keypoints: List[List[float]]         # [min, max, tip, center] 原图像素坐标
    conf: float                          # 检测框置信度
    kpt_conf: List[float]                # 各关键点置信度
    ratio: Optional[float] = None        # 量程比例 [0, 1]
    primary_value: Optional[float] = None    # 主量程数值（bar）
    secondary_value: Optional[float] = None  # 副量程数值（psi）
    error: Optional[str] = None          # 解算失败原因（None 表示成功）


class GaugeReader:
    """编排器：检测 -> 数学解算 -> 双量程读数。"""

    def __init__(self, config: Dict):
        self.config = config
        gauge_cfg = config["gauge"]
        self.sweep_direction = str(gauge_cfg.get("sweep_direction", "auto"))
        self.primary = gauge_cfg["primary"]       # 主量程：bar
        self.secondary = gauge_cfg["secondary"]   # 副量程：psi
        rounding_cfg = gauge_cfg.get("rounding", {})
        self.rounding_enabled = bool(rounding_cfg.get("enabled", False))
        self.rounding_step = float(rounding_cfg.get("step", 1.0))
        self.detector: GaugeDetector = create_detector(config)

    @property
    def primary_unit(self) -> str:
        return str(self.primary.get("unit", ""))

    @property
    def secondary_unit(self) -> str:
        return str(self.secondary.get("unit", ""))

    def read_frame(self, frame: np.ndarray) -> List[Reading]:
        """对单帧执行完整推理链路，返回该帧所有表盘的读数列表。"""
        detections = self.detector.predict(frame)
        readings: List[Reading] = []
        for det in detections:
            reading = Reading(
                bbox=det["bbox"],
                keypoints=det["keypoints"],
                conf=float(det.get("conf", 0.0)),
                kpt_conf=list(det.get("kpt_conf", [1.0] * 4)),
            )
            try:
                # 1) 角度比例（内部处理跨 360 度边界与防除零）
                ratio = calculate_ratio(det["keypoints"], self.sweep_direction)
                # 2) 主量程映射：0.0 ~ 250.0 bar
                primary = calculate_value(
                    float(self.primary["min_value"]),
                    float(self.primary["max_value"]), ratio)
                # 3) 副量程映射：0.0 ~ 3625.9 psi
                secondary = calculate_value(
                    float(self.secondary["min_value"]),
                    float(self.secondary["max_value"]), ratio)
                if self.rounding_enabled:
                    primary = round_value(primary, self.rounding_step)
                    secondary = round_value(secondary, self.rounding_step)
                reading.ratio = ratio
                reading.primary_value = primary
                reading.secondary_value = secondary
            except Exception as exc:  # 单块表盘解算失败不影响其余表盘
                reading.error = str(exc)
            readings.append(reading)
        return readings
