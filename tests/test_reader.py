# -*- coding: utf-8 -*-
"""reader 端到端测试：mock 后端 + 合成表盘，验证读数链路与精修集成。"""

import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from helpers import load_project_config, make_gauge  # noqa: E402
from reader import GaugeReader  # noqa: E402


def _mock_kpts():
    """mock 关键点（归一化）：min 0°、max 90°、tip 45°、center (0.5,0.5)。"""
    r = 0.45

    def pt(deg):
        a = math.radians(deg)
        return [0.5 + r * math.cos(a), 0.5 + r * math.sin(a)]

    return [pt(0), pt(90), pt(45), [0.5, 0.5]]


class TestReaderEndToEnd(unittest.TestCase):
    def setUp(self):
        self.frame = make_gauge(needle_deg=45.0)
        self.kpts = _mock_kpts()

    def _reader(self, refine: bool):
        cfg = load_project_config("mock")
        cfg["gauge"]["refinement"]["enabled"] = refine
        cfg["model"]["mock"]["keypoints"] = self.kpts
        return GaugeReader(cfg)

    def test_read_125bar_with_refine(self):
        readings = self._reader(refine=True).read_frame(self.frame)
        self.assertEqual(len(readings), 1)
        r = readings[0]
        self.assertIsNone(r.error)
        self.assertAlmostEqual(r.primary_value, 125.0, delta=3.0)
        self.assertAlmostEqual(r.secondary_value, 1812.95, delta=45.0)

    def test_read_125bar_without_refine(self):
        r = self._reader(refine=False).read_frame(self.frame)[0]
        self.assertIsNone(r.error)
        self.assertAlmostEqual(r.primary_value, 125.0, delta=0.01)

    def test_offscale_tip_uses_complement_arc(self):
        # tip=135° 不在 cw 弧(0~90°)内：auto 按互补弧（跨度 270°）解释。
        # 逆时针距离 = (0-135)%360 = 225° -> ratio 225/270（已知局限，锁定行为）
        cfg = load_project_config("mock")
        cfg["gauge"]["refinement"]["enabled"] = False
        r = 0.45
        a = math.radians(135)
        cfg["model"]["mock"]["keypoints"] = [
            [0.5 + r, 0.5], [0.5, 0.5 + r],
            [0.5 + r * math.cos(a), 0.5 + r * math.sin(a)], [0.5, 0.5]]
        r_reading = GaugeReader(cfg).read_frame(self.frame)[0]
        self.assertAlmostEqual(r_reading.ratio, 225.0 / 270.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
