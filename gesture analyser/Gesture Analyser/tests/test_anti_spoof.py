"""
tests/test_anti_spoof.py
=========================
Unit tests for the MicroTremorDetector and BrightnessMonitor classes
in anti_spoof.py.

Strategy
--------
Both detectors are pure-Python (no OpenCV / MediaPipe) so we feed them
synthetic data streams and assert the expected is_suspicious state.

MicroTremorDetector
  • Receives a stream of (wrist_x, wrist_y) positions.
  • Suspicious when std-dev of recent positions < min_std.
  • Not suspicious before warmup_frames have elapsed.

BrightnessMonitor
  • Receives a stream of scalar brightness values.
  • Suspicious when std-dev of recent values < min_variance.
  • Not suspicious before 30 samples have been pushed.
"""

import unittest

from anti_spoof import MicroTremorDetector, BrightnessMonitor


class TestMicroTremorDetector(unittest.TestCase):

    # ── helpers ──────────────────────────────────────────────────────

    def _make_detector(self, buffer_size=40, min_std=0.002, warmup=10):
        return MicroTremorDetector(
            buffer_size=buffer_size,
            min_std=min_std,
            warmup_frames=warmup,
        )

    def _push_static(self, det, n: int, x=0.5, y=0.5):
        """Push `n` identical positions (perfectly static hand)."""
        for _ in range(n):
            det.push(x, y)

    def _push_jittery(self, det, n: int, amplitude=0.01):
        """Push `n` positions that alternate by ±amplitude (trembling hand)."""
        for i in range(n):
            det.push(0.5 + (amplitude if i % 2 == 0 else -amplitude), 0.5)

    # ── warmup behaviour ────────────────────────────────────────────

    def test_not_suspicious_before_warmup(self):
        """Even a perfectly static hand is not flagged before warmup."""
        det = self._make_detector(warmup=30)
        self._push_static(det, 29)
        self.assertFalse(det.is_suspicious)

    def test_suspicious_after_warmup_static_hand(self):
        """Static hand is flagged once warmup is complete."""
        det = self._make_detector(warmup=10, min_std=0.002)
        self._push_static(det, 40)
        self.assertTrue(det.is_suspicious)

    # ── live hand ────────────────────────────────────────────────────

    def test_jittery_hand_not_suspicious(self):
        """A hand with natural tremor should NOT be flagged."""
        det = self._make_detector(warmup=5, min_std=0.002)
        self._push_jittery(det, 40, amplitude=0.005)
        self.assertFalse(det.is_suspicious)

    def test_large_tremor_definitely_not_suspicious(self):
        det = self._make_detector(warmup=5, min_std=0.002)
        self._push_jittery(det, 40, amplitude=0.05)
        self.assertFalse(det.is_suspicious)

    # ── std_dev property ────────────────────────────────────────────

    def test_std_dev_returns_one_before_10_samples(self):
        """Returns sentinel 1.0 when fewer than 10 samples pushed."""
        det = self._make_detector()
        self._push_static(det, 5)
        self.assertAlmostEqual(det.std_dev, 1.0, places=6)

    def test_std_dev_near_zero_for_static(self):
        det = self._make_detector()
        self._push_static(det, 40)
        self.assertLess(det.std_dev, 1e-9)

    # ── reset ────────────────────────────────────────────────────────

    def test_reset_clears_state(self):
        det = self._make_detector(warmup=5)
        self._push_static(det, 40)
        self.assertTrue(det.is_suspicious)
        det.reset()
        # After reset: fewer than warmup frames → not suspicious
        self._push_static(det, 3)
        self.assertFalse(det.is_suspicious)


class TestBrightnessMonitor(unittest.TestCase):

    def _make_monitor(self, buffer_size=60, min_variance=0.05):
        return BrightnessMonitor(buffer_size=buffer_size, min_variance=min_variance)

    def _push_flat(self, mon, n: int, value=0.5):
        for _ in range(n):
            mon.push(value)

    def _push_varying(self, mon, n: int, amplitude=0.1):
        for i in range(n):
            mon.push(0.5 + (amplitude if i % 2 == 0 else -amplitude))

    # ── not enough data ──────────────────────────────────────────────

    def test_not_suspicious_fewer_than_30_samples(self):
        mon = self._make_monitor()
        self._push_flat(mon, 29)
        self.assertFalse(mon.is_suspicious)

    # ── static brightness (photo / frozen video) ─────────────────────

    def test_constant_brightness_is_suspicious(self):
        mon = self._make_monitor(min_variance=0.05)
        self._push_flat(mon, 60)
        self.assertTrue(mon.is_suspicious)

    # ── natural variation ─────────────────────────────────────────────

    def test_varying_brightness_not_suspicious(self):
        mon = self._make_monitor(min_variance=0.05)
        self._push_varying(mon, 60, amplitude=0.1)
        self.assertFalse(mon.is_suspicious)

    # ── std_dev sentinel ─────────────────────────────────────────────

    def test_std_dev_sentinel_before_20_samples(self):
        """Returns 999.0 when fewer than 20 samples have been pushed."""
        mon = self._make_monitor()
        self._push_flat(mon, 10)
        self.assertAlmostEqual(mon.std_dev, 999.0, places=6)

    # ── reset ────────────────────────────────────────────────────────

    def test_reset_clears_suspicious_state(self):
        mon = self._make_monitor()
        self._push_flat(mon, 60)
        self.assertTrue(mon.is_suspicious)
        mon.reset()
        self._push_flat(mon, 5)
        self.assertFalse(mon.is_suspicious)


if __name__ == "__main__":
    unittest.main()
