# -*- coding: utf-8 -*-
"""reader.py —— 推理编排器：调用检测器，可选传统方法精修，串联数学解算，输出双量程读数。

职责：
    1. 按配置创建检测器（ultralytics / onnx / tflite / mock）；
    2. 对单帧执行 检测 -> 关键点 -> （可选传统方法精修，失败回退 DL） ->
       角度比例 -> 主/副量程数值映射；
    3. 以 Reading 数据类返回结构化结果，供 main.py 绘制与输出。

传统图像方法仅出现在 refiner.py 中（径向扫描 / 椭圆拟合），
本模块只做推理、精修编排与数学解算。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import numpy as np

from detector import GaugeDetector, create_detector
from geometry import IDX_CENTER, IDX_TIP, calculate_ratio, calculate_value, round_value
from refiner import refine_detection


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
    refine_used: bool = False            # 是否采用了传统方法精修结果
    refine_conf: Optional[float] = None  # 精修置信度（径向扫描峰值得分）
    refine_error: Optional[str] = None   # 精修失败原因（None 表示未失败）


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
        refine_cfg = gauge_cfg.get("refinement", {})
        self.refine_enabled = bool(refine_cfg.get("enabled", True)) and (
            refine_cfg.get("pointer", {}).get("enabled", True)
            or refine_cfg.get("center", {}).get("enabled", False))
        self.refine_cfg = dict(refine_cfg)
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
                keypoints=[list(p) for p in det["keypoints"]],
                conf=float(det.get("conf", 0.0)),
                kpt_conf=list(det.get("kpt_conf", [1.0] * 4)),
            )

            # ---- 可选：传统方法精修（失败自动回退 DL 结果）----
            if self.refine_enabled:
                try:
                    refined = refine_detection(
                        frame, det["bbox"], det["keypoints"], self.refine_cfg)
                    if refined.get("center") is not None:
                        reading.keypoints[IDX_CENTER] = [
                            float(refined["center"][0]),
                            float(refined["center"][1])]
                    if refined.get("tip_angle") is not None:
                        c = reading.keypoints[IDX_CENTER]
                        r_tip = math.hypot(
                            det["keypoints"][IDX_TIP][0] - c[0],
                            det["keypoints"][IDX_TIP][1] - c[1])
                        rad = math.radians(float(refined["tip_angle"]))
                        reading.keypoints[IDX_TIP] = [
                            c[0] + r_tip * math.cos(rad),
                            c[1] + r_tip * math.sin(rad)]
                        reading.refine_used = True
                        reading.refine_conf = (
                            float(refined["tip_conf"])
                            if refined.get("tip_conf") is not None else None)
                except Exception as exc:  # 精修失败不阻断 DL 读数
                    reading.refine_error = str(exc)

            try:
                # 1) 角度比例（内部处理跨 360 度边界与防除零）
                ratio = calculate_ratio(reading.keypoints, self.sweep_direction)
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
