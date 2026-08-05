# -*- coding: utf-8 -*-
"""geometry 单元测试：角度、比例、跨 360° 边界、扫掠方向、透视归一化。"""

import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geometry import (  # noqa: E402
    _ellipse_to_circle,
    calculate_angle,
    calculate_ratio,
    calculate_value,
    normalize_angle,
    relative_angle,
    round_value,
)


def polar(angle_deg: float, radius: float = 10.0):
    rad = math.radians(angle_deg)
    return (radius * math.cos(rad), radius * math.sin(rad))


def make_kpts(min_deg: float, max_deg: float, tip_deg: float,
              radius: float = 10.0):
    """center=(0,0)，min/max/tip 位于给定角度、radius 处。"""
    return [polar(min_deg, radius), polar(max_deg, radius),
            polar(tip_deg, radius), (0.0, 0.0)]


class TestAngle(unittest.TestCase):
    def test_normalize(self):
        self.assertAlmostEqual(normalize_angle(-45), 315.0)
        self.assertAlmostEqual(normalize_angle(405), 45.0)
        self.assertAlmostEqual(normalize_angle(0), 0.0)
        self.assertAlmostEqual(normalize_angle(720), 0.0)

    def test_calculate_angle_image_coords(self):
        # 图像坐标 y 向下：正 x 轴 = 0°，正 y 轴 = 90°
        self.assertAlmostEqual(calculate_angle((0.0, 0.0), (10.0, 0.0)), 0.0)
        self.assertAlmostEqual(calculate_angle((0.0, 0.0), (0.0, 10.0)), 90.0)
        self.assertAlmostEqual(calculate_angle((0.0, 0.0), (-10.0, 0.0)), 180.0)
        self.assertAlmostEqual(calculate_angle((0.0, 0.0), (0.0, -10.0)), 270.0)

    def test_relative_angle(self):
        self.assertAlmostEqual(relative_angle(350, 10), 20.0)
        self.assertAlmostEqual(relative_angle(10, 350), 340.0)
        self.assertAlmostEqual(relative_angle(0, 0), 0.0)


class TestRatio(unittest.TestCase):
    def test_clockwise(self):
        kpts = make_kpts(0, 90, 45)
        self.assertAlmostEqual(calculate_ratio(kpts, "clockwise"), 0.5, places=6)

    def test_counterclockwise(self):
        # ccw 弧 = 从 max(90°) 逆时针回到 min(0°)，跨度 270°；
        # tip 在 315° 时距 min 逆时针 45° -> 45/270
        kpts = make_kpts(0, 90, 315)
        self.assertAlmostEqual(
            calculate_ratio(kpts, "counterclockwise"), 45.0 / 270.0, places=6)

    def test_auto_prefers_containing_arc_cw(self):
        kpts = make_kpts(0, 90, 45)
        self.assertAlmostEqual(calculate_ratio(kpts, "auto"), 0.5, places=6)

    def test_auto_prefers_containing_arc_ccw(self):
        # tip=180° 不在 cw 弧(0~90°)内，自动落入 ccw 弧
        kpts = make_kpts(0, 90, 180)
        self.assertAlmostEqual(
            calculate_ratio(kpts, "auto"), 180.0 / 270.0, places=6)

    def test_auto_boundary_at_max(self):
        kpts = make_kpts(0, 90, 90)
        self.assertAlmostEqual(calculate_ratio(kpts, "auto"), 1.0, places=6)

    def test_auto_boundary_at_min(self):
        kpts = make_kpts(0, 90, 0)
        self.assertAlmostEqual(calculate_ratio(kpts, "auto"), 0.0, places=6)

    def test_cross_360(self):
        # min=350°、max=10°：cw 跨度 20°，tip=0° 距 min 10° -> 0.5
        kpts = make_kpts(350, 10, 0)
        self.assertAlmostEqual(calculate_ratio(kpts, "auto"), 0.5, places=6)

    def test_span_too_small_raises(self):
        kpts = make_kpts(0, 1, 0.5)
        with self.assertRaises(ValueError):
            calculate_ratio(kpts, "auto")

    def test_invalid_sweep_raises(self):
        kpts = make_kpts(0, 90, 45)
        with self.assertRaises(ValueError):
            calculate_ratio(kpts, "sideways")

    def test_insufficient_keypoints_raises(self):
        with self.assertRaises(ValueError):
            calculate_ratio([(0, 0), (1, 1), (2, 2)])

    def test_value_mapping(self):
        self.assertAlmostEqual(calculate_value(0.0, 250.0, 0.5), 125.0)
        self.assertAlmostEqual(calculate_value(0.0, 3625.9, 0.5), 1812.95)
        with self.assertRaises(ValueError):
            calculate_value(250.0, 0.0, 0.5)

    def test_round_value(self):
        self.assertEqual(round_value(101.8, 1), 102.0)
        self.assertEqual(round_value(154.1, 10), 150.0)
        self.assertEqual(round_value(3.7, 0.5), 3.5)
        self.assertEqual(round_value(12.3, 0), 12.3)


class TestEllipseNormalize(unittest.TestCase):
    def test_circle_maps_to_unit_circle(self):
        # 椭圆 (ma=200, mi=100)，参数角 45° 的点应归一化到 (cos45, sin45)
        ellipse = ((0.0, 0.0), (200.0, 100.0), 0.0)
        p = _ellipse_to_circle([(100 * math.cos(math.radians(45)),
                                 50 * math.sin(math.radians(45)))], ellipse)[0]
        self.assertAlmostEqual(p[0], math.cos(math.radians(45)), places=6)
        self.assertAlmostEqual(p[1], math.sin(math.radians(45)), places=6)

    def test_ratio_invariant_on_ellipse(self):
        # 表盘是正圆，斜拍后投影为椭圆；椭圆归一化后比例应恢复
        ellipse = ((0.0, 0.0), (200.0, 100.0), 0.0)

        def ellipse_kpts():
            # 椭圆上的点：x = 100*cosθ, y = 50*sinθ
            return [[100 * math.cos(math.radians(0)), 50 * math.sin(math.radians(0))],
                    [100 * math.cos(math.radians(90)), 50 * math.sin(math.radians(90))],
                    [100 * math.cos(math.radians(45)), 50 * math.sin(math.radians(45))],
                    [0.0, 0.0]]

        ratio = calculate_ratio(ellipse_kpts(), "clockwise", ellipse)
        self.assertAlmostEqual(ratio, 0.5, places=6)

    def test_circle_ellipse_no_change(self):
        # 正圆（ma==mi）加任意旋转，比例应保持不变
        ellipse = ((0.0, 0.0), (200.0, 200.0), 30.0)
        kpts = make_kpts(10, 100, 55, radius=100.0)
        r1 = calculate_ratio(kpts, "clockwise")
        r2 = calculate_ratio(kpts, "clockwise", ellipse)
        self.assertAlmostEqual(r1, r2, places=6)


if __name__ == "__main__":
    unittest.main()
