# -*- coding: utf-8 -*-
"""geometry.py —— 纯数学解算模块（角度 / 比例 / 双量程数值映射）。

本模块只做数学运算（math.atan2、取模、线性插值），不依赖 OpenCV，
不包含任何传统图像处理方法（Hough 圆、连通域、边缘检测等全部禁用）。

关键点顺序约定（与数据集标注顺序完全一致，全链路统一，绝不重排）：
    kpts[0] = min    刻度起点端（最小值端）
    kpts[1] = max    刻度终点端（最大值端）
    kpts[2] = tip    指针尖端
    kpts[3] = center 表盘中心

坐标系说明：
    图像坐标系原点在左上角，x 向右，y 向下。
    使用 math.atan2(py - cy, px - cx) 计算各点绝对角度，并归一化到 [0, 360)：
    - atan2 返回 (-pi, pi]，正值对应“顺时针”方向（因为 y 向下）；
    - 取模 360 后得到 [0, 360)，天然兼容跨越 0/360 边界的情况。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 关键点索引（严格按需求定义；表示起点 / 终点的命名一律用 min / max）
# ---------------------------------------------------------------------------
IDX_MIN = 0     # min：刻度起点端（最小值端）
IDX_MAX = 1     # max：刻度终点端（最大值端）
IDX_TIP = 2     # tip：指针尖端
IDX_CENTER = 3  # center：表盘中心（指针转轴）

KEYPOINT_NAMES: Dict[int, str] = {
    IDX_MIN: "min",
    IDX_MAX: "max",
    IDX_TIP: "tip",
    IDX_CENTER: "center",
}

Point = Tuple[float, float]
Kpts = Sequence[Point]

# 角度常量
EPS_DEG = 1e-6         # 浮点比较容差
MIN_SPAN_DEG = 5.0     # 量程弧最小跨度（度）：min / max 重合或跨度过小时视为无效，
                       # 用于防止分母接近 0 导致的除零 / 数值爆炸


def normalize_angle(angle: float) -> float:
    """把任意角度归一化到 [0, 360) 区间（自动处理负角与超过 360 的角度）。"""
    return angle % 360.0


def calculate_angle(center: Point, point: Point) -> float:
    """以 center 为原点，计算 point 的绝对角度（单位：度，范围 [0, 360)）。

    参数:
        center: 表盘中心 (cx, cy)
        point:  目标点 (px, py)，例如 min / max / tip
    返回:
        math.atan2(py - cy, px - cx) 经归一化后的角度值。
    """
    cx, cy = center
    px, py = point
    rad = math.atan2(py - cy, px - cx)   # 弧度，范围 (-pi, pi]
    return normalize_angle(math.degrees(rad))


def relative_angle(a_from: float, a_to: float) -> float:
    """计算从 a_from 沿角度增大方向到 a_to 的相对夹角（度，[0, 360)）。

    使用取模运算处理跨过 360 度边界的情况：
        例：a_from = 350，a_to = 10  ->  返回 20（而不是 -340）。
    """
    return (a_to - a_from) % 360.0


def _ellipse_to_circle(kpts: Kpts, ellipse: Tuple) -> List[Tuple[float, float]]:
    """把椭圆（斜拍表盘）仿射映射为单位圆，返回圆空间坐标。

    透视 / 斜拍下，正圆的表盘在图像中投影为椭圆；把椭圆仿射回单位圆
    后再计算角度，可近似恢复表盘真实角度关系（仿射近似，适合小角度斜拍）。

    ellipse 格式与 cv2.fitEllipse 一致：((cx, cy), (ma, mi), angle)，
    其中 ma / mi 是长短轴长度（直径），angle 为旋转角（度）。

    当 ma == mi（正对相机）时，该映射退化为“旋转 + 等比例缩放”，
    不影响角度比例，可安全使用。
    """
    (cx, cy), (ma, mi), ang_deg = ellipse
    ma, mi = max(float(ma), 1e-6), max(float(mi), 1e-6)
    rad = math.radians(float(ang_deg))
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for x, y in kpts:
        dx, dy = float(x) - cx, float(y) - cy
        rx = (dx * cos_a + dy * sin_a) / (ma / 2.0)
        ry = (-dx * sin_a + dy * cos_a) / (mi / 2.0)
        out.append((rx, ry))
    return out


def _clamped_ratio(dist: float, span: float) -> float:
    """防除零 + 限幅：ratio = dist / span，结果强制限制在 [0, 1]。

    参数:
        dist: 指针相对 min 端点的夹角（已做跨 360 度处理）
        span: 量程弧总跨度（min -> max）
    异常:
        当 span 过小（min 与 max 近乎重合）时抛出 ValueError，
        提示标注或关键点可能出错，避免静默产生异常读数。
    """
    if span <= EPS_DEG or span < MIN_SPAN_DEG:
        raise ValueError(
            f"min 与 max 的角度跨度过小（{span:.3f} 度），"
            "疑似关键点重合或标注错误，无法计算量程比例")
    return max(0.0, min(dist / span, 1.0))


def calculate_ratio(kpts: Kpts, sweep_direction: str = "auto",
                    ellipse: Optional[Tuple] = None) -> float:
    """计算指针在量程中的比例 ratio，结果限制在 [0, 1]。

    公式（需求文档）：
        ratio = (Angle_tip - Angle_min) / (Angle_max - Angle_min)

    实现要点：
        - 三个绝对角度先经 calculate_angle 归一化到 [0, 360)；
        - 分子 / 分母都使用 relative_angle 的取模差，处理跨 360 度边界；
        - 分母（量程跨度）过小时防除零；
        - 最终结果限幅到 [0, 1]。

    关于 sweep_direction（数据集同时包含两种表盘几何）：
        - "clockwise"         强制按大弧解释：min -> max 沿角度增大方向；
        - "counterclockwise"  强制按小弧解释：min -> max 沿角度减小方向；
        - "auto"              自动判定：选择“指针尖端落在量程弧内”的解释。

    几何事实：两条候选弧（顺时针 / 逆时针）恰好把圆周分割为互补的两部分，
    因此“指针在量程弧内”的解释在常态下唯一。auto 的残余局限：
    当指针真的处于量程缺口区（超出量程）时，它会被按互补弧误读为
    有量程读数——该情形在几何上无法仅凭 4 个关键点判别，
    生产环境建议按表盘型号固定 sweep_direction。

    ellipse（可选）：cv2.fitEllipse 格式的椭圆参数，用于斜拍视角的
    透视归一化（见 _ellipse_to_circle）；默认 None 表示正对相机。

    参数:
        kpts:            4 个关键点 [min, max, tip, center]（像素坐标或归一化坐标均可，
                         角度只与相对位置有关）
        sweep_direction: auto / clockwise / counterclockwise
    返回:
        ratio in [0, 1]
    异常:
        关键点不足、min/max 重合、指针不在量程弧内时抛出 ValueError。
    """
    if kpts is None or len(kpts) < 4:
        raise ValueError(
            f"关键点数量不足（需要 4 个，实际 {0 if kpts is None else len(kpts)}）")

    if ellipse is not None:
        kpts = _ellipse_to_circle(kpts, ellipse)

    center = kpts[IDX_CENTER]
    a_min = calculate_angle(center, kpts[IDX_MIN])   # min 端绝对角度
    a_max = calculate_angle(center, kpts[IDX_MAX])   # max 端绝对角度
    a_tip = calculate_angle(center, kpts[IDX_TIP])   # 指针尖端绝对角度

    # 两条候选量程弧的跨度（均做了跨 360 度边界处理）
    span_cw = relative_angle(a_min, a_max)   # 顺时针（角度增大）跨度
    span_ccw = relative_angle(a_max, a_min)  # 逆时针（角度减小）跨度

    if sweep_direction == "clockwise":
        return _clamped_ratio(relative_angle(a_min, a_tip), span_cw)

    if sweep_direction == "counterclockwise":
        return _clamped_ratio(relative_angle(a_tip, a_min), span_ccw)

    if sweep_direction != "auto":
        raise ValueError(
            f"未知 sweep_direction: {sweep_direction}"
            "（可选 auto / clockwise / counterclockwise）")

    # auto：优先按“指针尖端落在量程弧内”的几何解释
    dist_cw = relative_angle(a_min, a_tip)
    dist_ccw = relative_angle(a_tip, a_min)
    if dist_cw <= span_cw + EPS_DEG:
        return _clamped_ratio(dist_cw, span_cw)
    if dist_ccw <= span_ccw + EPS_DEG:
        return _clamped_ratio(dist_ccw, span_ccw)

    raise ValueError(
        "指针尖端不在 min~max 量程弧内，请检查关键点坐标或 sweep_direction 配置")


def calculate_value(min_value: float, max_value: float, ratio: float) -> float:
    """线性刻度数值映射：value = min_value + ratio * (max_value - min_value)。

    参数:
        min_value: 量程最小值（例如主量程 0.0 bar）
        max_value: 量程最大值（例如主量程 250.0 bar）
        ratio:     [0, 1] 的量程比例
    返回:
        当前指针读数（与 min/max 同单位）。
    """
    if max_value < min_value:
        raise ValueError(
            f"量程配置错误：max_value（{max_value}）小于 min_value（{min_value}）")
    return min_value + ratio * (max_value - min_value)


def round_value(value: float, step: float) -> float:
    """按步长凑整读数：value 近似到最近的 step 整数倍。

    参数:
        value: 原始读数（任意浮点数）
        step:  凑整步长
               step=1    -> 近似到整数（例如 101.8 -> 102）
               step=10   -> 近似到整十（例如 154.1 -> 150）
               step=0.5  -> 近似到半格（例如 3.7 -> 3.5）
               step<=0   -> 关闭凑整，原样返回
    返回:
        凑整后的读数（关闭时为原值）。
    """
    if step is None or float(step) <= 0:
        return float(value)
    s = float(step)
    return round(float(value) / s) * s
