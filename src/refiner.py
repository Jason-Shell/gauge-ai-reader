# -*- coding: utf-8 -*-
"""refiner.py —— 传统图像方法关键点精修（DL 主检，传统法辅助，失败自动回退）。

背景：
    YOLOv8-Pose 关键点已能可靠定位表盘，但 tip / center 存在像素级抖动，
    直接决定读数精度。本模块以 DL 结果为先验，用两类经典方法做精修：

    1. 指针角度精修（径向扫描，默认开启）：
       在中心附近的环形带内沿 360° 射线统计“暗色（或亮色）连续段”长度。
       指针是贯穿环带的细长结构，得分远高于短刻度线；取峰值射线角度，
       再做抛物线插值到亚度，并用 DL 的 tip 角消除 180° 指向歧义。

    2. 表盘中心精修（椭圆拟合，默认关闭）：
       在检测框 ROI 内 Canny + 轮廓 + fitEllipse，选取面积占比与形态
       合理的椭圆，其圆心作为精修中心。正对相机的表盘收益有限，
       斜拍视角下才有明显价值，故默认关闭。

边界约定：
    传统方法只允许在本模块内、以 DL 结果为先验做数值精修；
    表盘检测 / 关键点是否存在等结构性判断仍由 DL 负责。
    任何精修失败或低置信都返回 None，由 reader 回退到 DL 原结果。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from geometry import IDX_CENTER, IDX_TIP, calculate_angle


def _run_from_start(values: np.ndarray) -> float:
    """沿射线统计“从内缘开始”的连续暗段长度占比。

    指针从中心转轴伸出，在环带内缘处即为暗色并一直延伸到外缘；
    刻度线 / 数字只出现在外缘附近，内缘处为亮色，因此该判据
    能有效区分指针与刻度。
    """
    n = 0
    for v in values:
        if v:
            n += 1
        else:
            break
    return n / max(1, len(values))


def _ray_scores(gray: np.ndarray, cx: float, cy: float,
                r_in: float, r_out: float, step_deg: float,
                mask: np.ndarray) -> np.ndarray:
    """对每个角度做径向扫描，返回归一化“内缘连续暗段”得分数组。"""
    h, w = gray.shape[:2]
    n_angles = int(round(360.0 / step_deg))
    radii = np.linspace(r_in, r_out, max(8, int(r_out - r_in)))
    scores = np.zeros(n_angles, dtype=np.float64)
    for i in range(n_angles):
        theta = math.radians(i * step_deg)
        xs = np.round(cx + radii * math.cos(theta)).astype(np.int64)
        ys = np.round(cy + radii * math.sin(theta)).astype(np.int64)
        np.clip(xs, 0, w - 1, out=xs)
        np.clip(ys, 0, h - 1, out=ys)
        scores[i] = _run_from_start(mask[ys, xs])
    return scores


def _interp_peak(scores: np.ndarray, step_deg: float) -> Tuple[float, float]:
    """抛物线插值求亚度峰值，返回 (精确角度, 峰值得分)。"""
    i1 = int(np.argmax(scores))
    n = len(scores)
    i0, i2 = (i1 - 1) % n, (i1 + 1) % n
    s0, s1, s2 = float(scores[i0]), float(scores[i1]), float(scores[i2])
    denom = s0 - 2.0 * s1 + s2
    delta = 0.5 * (s0 - s2) / denom if abs(denom) > 1e-9 else 0.0
    angle = (i1 + delta) * step_deg
    return angle % 360.0, s1


def _align_to_tip(angle: float, dl_tip_angle: float) -> float:
    """用 DL tip 角消除 180° 指向歧义：取与 tip 夹角 <=90° 的那个方向。"""
    a, b = angle % 360.0, dl_tip_angle % 360.0
    diff = (a - b + 180.0) % 360.0 - 180.0
    if abs(diff) > 90.0:
        a = (a + 180.0) % 360.0
    return a


def refine_pointer_angle(gray: np.ndarray, center: Tuple[float, float],
                         radius: float, cfg: Dict) -> Tuple[Optional[float], Optional[float]]:
    """径向扫描精修指针角度。

    返回 (角度 [0,360), 峰值得分)；得分低于 min_score 或输入异常时返回 (None, score)。
    """
    inner_ratio = float(cfg.get("inner_ratio", 0.30))
    outer_ratio = float(cfg.get("outer_ratio", 0.90))
    step_deg = float(cfg.get("step_deg", 0.5))
    min_score = float(cfg.get("min_score", 0.35))
    cx, cy = float(center[0]), float(center[1])
    r_in = radius * inner_ratio
    r_out = radius * outer_ratio
    if r_out - r_in < 5.0 or radius < 8.0 or not (0 < inner_ratio < outer_ratio <= 1.1):
        return None, None

    # 环带 ROI 上的 Otsu 阈值（仅统计环带内像素，避免表盘外背景干扰阈值）
    roi_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (int(cx), int(cy)), int(r_out), 255, -1)
    cv2.circle(roi_mask, (int(cx), int(cy)), int(r_in), 0, -1)
    vals = gray[roi_mask > 0]
    if vals.size < 50:
        return None, None
    thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (gray < thr) & (roi_mask > 0)
    bright_mask = (gray >= thr) & (roi_mask > 0)

    scores_dark = _ray_scores(gray, cx, cy, r_in, r_out, step_deg, dark_mask)
    scores_bright = _ray_scores(gray, cx, cy, r_in, r_out, step_deg, bright_mask)

    # 选“更尖锐”的极性：有效得分 = 峰值 × (1 - 平坦度)。
    # 白底表盘上“亮色”整体平坦（得分 1.0 但处处相同），
    # 而指针是唯一的尖锐长条，因此暗/亮哪个是针，哪个有效得分更高。
    best = None
    for s in (scores_dark, scores_bright):
        pk = float(np.max(s))
        flat = float(np.mean(s >= 0.8 * pk)) if pk > 0 else 1.0
        eff = pk * (1.0 - flat)
        if best is None or eff > best[0]:
            best = (eff, pk, flat, s)
    eff, peak_score, flat_ratio, scores = best

    angle, _ = _interp_peak(scores, step_deg)
    # 平坦度保护：若大量角度得分都接近峰值，说明环带内结构无法判别
    # （如表盘被遮挡 / 全暗 / 全亮），此时回退 DL，避免乱选
    if eff < min_score or peak_score < min_score or flat_ratio > 0.15:
        return None, eff
    return angle, eff


def refine_dial_center(gray: np.ndarray, bbox: Tuple[float, float, float, float],
                       cfg: Dict) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """椭圆拟合精修表盘中心。

    返回 (center, radius)；未找到合格椭圆时返回 (None, None)。
    """
    min_area = float(cfg.get("min_area_ratio", 0.35))
    max_area = float(cfg.get("max_area_ratio", 0.95))
    max_aspect = float(cfg.get("max_aspect", 1.5))
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w < 24 or h < 24:
        return None, None
    m = int(0.05 * max(w, h))
    x1, y1 = max(0, x1 - m), max(0, y1 - m)
    x2, y2 = min(gray.shape[1], x2 + m), min(gray.shape[0], y2 + m)

    roi = gray[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_center, best_radius, best_score = None, None, 0.0
    bbox_area = float((x2 - x1) * (y2 - y1))
    for c in contours:
        if len(c) < 5:
            continue
        try:
            (ecx, ecy), (ma, mi), _ = cv2.fitEllipse(c)
        except cv2.error:
            continue
        if ma <= 0 or mi <= 0:
            continue
        ell_area = math.pi * ma * mi / 4.0
        area_ratio = ell_area / bbox_area
        aspect = max(ma, mi) / min(ma, mi)
        if not (min_area <= area_ratio <= max_area) or aspect > max_aspect:
            continue
        if not (x1 <= ecx <= x2 and y1 <= ecy <= y2):
            continue
        if area_ratio > best_score:
            best_score = area_ratio
            best_center = (float(ecx), float(ecy))
            best_radius = (ma + mi) / 4.0
    if best_center is None:
        return None, None
    return best_center, best_radius


def refine_detection(frame: np.ndarray, bbox: Tuple[float, float, float, float],
                     keypoints: List[List[float]], cfg: Dict) -> Dict:
    """对单块表盘执行可选的传统方法精修，返回结果字典（字段可为 None）。

    返回字段：
        center:    (cx, cy) 或 None（椭圆拟合，默认关闭）
        tip_angle: 精修指针绝对角度 [0,360) 或 None
        tip_conf:  径向扫描峰值得分或 None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    center_cfg = cfg.get("center", {})
    pointer_cfg = cfg.get("pointer", {})

    center = (float(keypoints[IDX_CENTER][0]), float(keypoints[IDX_CENTER][1]))
    dl_tip = (float(keypoints[IDX_TIP][0]), float(keypoints[IDX_TIP][1]))
    # 扫描半径：以 DL 各关键点到中心的距离估计表盘半径（刻度端点贴近表盘边缘），
    # 确保扫描带落在表盘面内，不把表盘外背景当成“暗指针”
    radius = max(
        math.hypot(keypoints[i][0] - center[0], keypoints[i][1] - center[1])
        for i in range(len(keypoints)))

    # 1) 可选：椭圆拟合精修中心
    if center_cfg.get("enabled", False):
        c, r = refine_dial_center(gray, bbox, center_cfg)
        if c is not None and r is not None and r >= 10.0:
            center = c
            radius = r

    # 2) 径向扫描精修指针角度（用精修后中心计算 DL tip 角做消歧）
    tip_angle = tip_conf = None
    if pointer_cfg.get("enabled", True):
        dl_tip_angle = calculate_angle(center, dl_tip)
        angle, score = refine_pointer_angle(gray, center, radius, pointer_cfg)
        if angle is not None and score is not None:
            angle = _align_to_tip(angle, dl_tip_angle)
            diff = (angle - dl_tip_angle + 180.0) % 360.0 - 180.0
            agree_deg = float(pointer_cfg.get("agree_deg", 2.5))
            max_disagree = float(pointer_cfg.get("max_disagree", 45.0))
            # 共识门控策略：
            #   |diff| < agree_deg      -> 两者一致，信任 DL（扫描噪声 ~1°，无需替换）；
            #   agree_deg <= |diff| <= max_disagree
            #                           -> 中度分歧，多半是 DL 关键点抖动，
            #                              取像素级更精确的扫描结果；
            #   |diff| > max_disagree   -> 严重分歧，扫描可能误检（如遮挡/全暗），保留 DL。
            if agree_deg <= abs(diff) <= max_disagree:
                tip_angle = angle
                tip_conf = score

    return {"center": center, "tip_angle": tip_angle, "tip_conf": tip_conf}
