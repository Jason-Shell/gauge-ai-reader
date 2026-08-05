# -*- coding: utf-8 -*-
"""refiner 单元测试：径向扫描、椭圆拟合、共识门控、180° 消歧。"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from helpers import make_gauge  # noqa: E402
from refiner import (  # noqa: E402
    _align_to_tip,
    _run_from_start,
    apply_consensus_gate,
    refine_detection,
    refine_dial_center,
    refine_pointer_angle,
)


POINTER_CFG = {"inner_ratio": 0.30, "outer_ratio": 0.90,
               "step_deg": 0.5, "min_score": 0.35}
CENTER_CFG = {"min_area_ratio": 0.35, "max_area_ratio": 0.95, "max_aspect": 1.5}


class TestRunFromStart(unittest.TestCase):
    def test_connected_run(self):
        self.assertAlmostEqual(_run_from_start(np.array([1, 1, 1, 0, 1])), 0.6)
        self.assertEqual(_run_from_start(np.array([0, 1, 1])), 0.0)
        self.assertEqual(_run_from_start(np.array([1, 1, 1])), 1.0)


class TestAlignToTip(unittest.TestCase):
    def test_flip_when_opposite(self):
        self.assertAlmostEqual(_align_to_tip(10, 110), 190.0)

    def test_keep_when_close(self):
        self.assertAlmostEqual(_align_to_tip(10, 20), 10.0)
        self.assertAlmostEqual(_align_to_tip(10, 90), 10.0)
        self.assertAlmostEqual(_align_to_tip(350, 10), 350.0)


class TestConsensusGate(unittest.TestCase):
    def test_agree_keeps_dl(self):
        self.assertIsNone(apply_consensus_gate(100, 101, 2.5, 45.0))

    def test_moderate_disagreement_adopts_scan(self):
        self.assertAlmostEqual(apply_consensus_gate(100, 104, 2.5, 45.0), 104.0)
        self.assertAlmostEqual(apply_consensus_gate(100, 97, 2.5, 45.0), 97.0)

    def test_boundary_inclusive(self):
        self.assertAlmostEqual(apply_consensus_gate(100, 102.5, 2.5, 45.0), 102.5)

    def test_severe_disagreement_keeps_dl(self):
        self.assertIsNone(apply_consensus_gate(100, 150, 2.5, 45.0))

    def test_wrap_around(self):
        self.assertAlmostEqual(apply_consensus_gate(350, 355, 2.5, 45.0), 355.0)
        self.assertAlmostEqual(apply_consensus_gate(5, 359, 2.5, 45.0), 359.0)


class TestPointerScan(unittest.TestCase):
    def test_finds_needle_at_45(self):
        img = make_gauge(needle_deg=45.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle, score = refine_pointer_angle(gray, (200, 200), 180.0, POINTER_CFG)
        self.assertIsNotNone(angle)
        self.assertIsNotNone(score)
        self.assertLess(abs(angle - 45.0), 3.0)

    def test_uniform_rejects(self):
        gray = np.full((300, 300), 255, np.uint8)
        angle, score = refine_pointer_angle(gray, (150, 150), 120.0, POINTER_CFG)
        self.assertIsNone(angle)


class TestDialCenter(unittest.TestCase):
    def test_finds_circle_center(self):
        img = make_gauge()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center, radius, ellipse = refine_dial_center(
            gray, (20, 20, 380, 380), CENTER_CFG)
        self.assertIsNotNone(center)
        self.assertIsNotNone(ellipse)
        self.assertLess(abs(center[0] - 200), 4.0)
        self.assertLess(abs(center[1] - 200), 4.0)
        _, (ma, mi), _ = ellipse
        self.assertLess(max(ma, mi) / min(ma, mi), 1.05)

    def test_empty_returns_none(self):
        gray = np.full((200, 200), 255, np.uint8)
        center, radius, ellipse = refine_dial_center(
            gray, (30, 30, 170, 170), CENTER_CFG)
        self.assertIsNone(center)


class TestRefineDetection(unittest.TestCase):
    def _kpts(self):
        c = (200.0, 200.0)

        def pt(deg, radius):
            import math
            a = math.radians(deg)
            return [c[0] + radius * math.cos(a), c[1] + radius * math.sin(a)]

        return [pt(0, 180), pt(90, 180), pt(45, 160), list(c)]

    def test_consensus_keeps_dl_tip(self):
        frame = make_gauge(needle_deg=45.0)
        cfg = {"pointer": {**POINTER_CFG, "agree_deg": 4.0, "max_disagree": 45.0},
               "center": {**CENTER_CFG, "enabled": False}}
        result = refine_detection(frame, (20, 20, 380, 380), self._kpts(), cfg)
        # 扫描与 DL tip(45°) 高度一致 -> 共识门控保留 DL
        self.assertIsNone(result["tip_angle"])

    def test_returns_ellipse_when_requested(self):
        frame = make_gauge(needle_deg=45.0)
        cfg = {"pointer": {**POINTER_CFG, "enabled": False},
               "center": {**CENTER_CFG, "enabled": False}}
        result = refine_detection(frame, (20, 20, 380, 380), self._kpts(), cfg,
                                  need_ellipse=True)
        self.assertIsNotNone(result["ellipse"])


if __name__ == "__main__":
    unittest.main()
