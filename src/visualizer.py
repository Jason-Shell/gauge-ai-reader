# -*- coding: utf-8 -*-
"""visualizer.py —— OpenCV 可视化：绘制关键点、连线与双量程读数。

OpenCV 职责仅限图像绘制（circle / line / rectangle / putText），
绝不使用任何轮廓分析方法（HoughCircles / findContours / Canny 等）。

绘制内容：
    - 检测框；
    - 关键点 min / max / tip / center（圆点 + 文字标签）；
    - center -> min、center -> max、center -> tip 三条连线；
    - 双量程读数（bar / psi）与量程比例。
"""

from __future__ import annotations

from typing import Dict, List

import cv2

from geometry import IDX_CENTER, IDX_MAX, IDX_MIN, IDX_TIP, KEYPOINT_NAMES
from reader import Reading

# 关键点绘制颜色（BGR），顺序 [min, max, tip, center]
KP_COLORS = [
    (255, 0, 0),      # min：蓝
    (0, 165, 255),    # max：橙
    (0, 0, 255),      # tip：红
    (0, 255, 255),    # center：黄
]

# 检测框 / 连线颜色
BOX_COLOR = (0, 255, 0)
MIN_LINE_COLOR = (255, 0, 0)
MAX_LINE_COLOR = (0, 165, 255)
TIP_LINE_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)


def _ascii_unit(unit) -> str:
    """OpenCV Hershey 字体不支持非 ASCII 字符，做简单替换避免乱码。"""
    if not unit:
        return ""
    return (str(unit)
            .replace("²", "^2").replace("³", "^3")
            .replace("μ", "u").replace("°", "deg")
            .encode("ascii", "ignore").decode("ascii"))


def _clamp(v: float, lo: float, hi: float) -> int:
    return max(int(lo), min(int(hi), int(round(v))))


def format_reading_value(value) -> str:
    """读数显示格式：整数不带小数，非整数保留 1 位小数（如 50 / 50.5）。"""
    if value is None:
        return ""
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def draw_reading(image, reading: Reading, primary_cfg: Dict,
                 secondary_cfg: Dict) -> None:
    """在 image 上原位绘制单块表盘的完整结果。"""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (_clamp(v, 0, width if i % 2 == 0 else height)
                      for i, v in enumerate(reading.bbox))

    # 1. 检测框
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # 2. center -> min / max / tip 连线（突出量程端点与指针）
    cx, cy = int(round(reading.keypoints[IDX_CENTER][0])), \
        int(round(reading.keypoints[IDX_CENTER][1]))
    for idx, color, thickness in (
            (IDX_MIN, MIN_LINE_COLOR, 2),
            (IDX_MAX, MAX_LINE_COLOR, 2),
            (IDX_TIP, TIP_LINE_COLOR, 3)):
        px, py = reading.keypoints[idx]
        cv2.line(image, (cx, cy),
                 (int(round(px)), int(round(py))), color, thickness)

    # 3. 关键点圆点 + 标签
    for i, (px, py) in enumerate(reading.keypoints):
        ix, iy = int(round(px)), int(round(py))
        cv2.circle(image, (ix, iy), 6, KP_COLORS[i], -1)
        label = f"{KEYPOINT_NAMES[i]}"
        lx = max(5, min(width - 70, ix + 10))
        ly = max(15, min(height - 5, iy + 10))
        cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, KP_COLORS[i], 2)

    # 4. 双量程读数文本（放在检测框上方，空间不足时放在框内）
    lines = []
    if reading.error:
        lines.append(f"ERR: {reading.error[:46]}")
    else:
        p_unit = _ascii_unit(primary_cfg.get("unit", ""))
        s_unit = _ascii_unit(secondary_cfg.get("unit", ""))
        lines.append(f"bar: {format_reading_value(reading.primary_value)} {p_unit}")
        lines.append(f"psi: {format_reading_value(reading.secondary_value)} {s_unit}")
        lines.append(f"Ratio: {reading.ratio * 100.0:.1f}%")

    font = cv2.FONT_HERSHEY_SIMPLEX
    if y1 >= 20 + 24 * len(lines):
        base_y = y1 - 12
        for i, line in enumerate(reversed(lines)):
            y = base_y - i * 24
            cv2.putText(image, line, (x1, y), font, 0.65, TEXT_COLOR, 2)
    else:
        base_y = y2 + 30
        for i, line in enumerate(lines):
            y = min(height - 8, base_y + i * 24)
            cv2.putText(image, line, (x1, y), font, 0.65, TEXT_COLOR, 2)


def draw_readings(image, readings: List[Reading],
                  gauge_cfg: Dict) -> None:
    """批量绘制：一帧中可能有多块表盘。"""
    primary = gauge_cfg["primary"]
    secondary = gauge_cfg["secondary"]
    for reading in readings:
        draw_reading(image, reading, primary, secondary)
