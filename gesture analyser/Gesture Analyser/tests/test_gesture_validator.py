"""
tests/test_gesture_validator.py
================================
Unit tests for gesture_validator.py — the pure-geometry core.

Covers:
  • hand_scale() normalisation baseline
  • finger_ratio() for each finger on open vs closed hand
  • is_finger_open() stateless classification
  • is_fist() on closed-fist and open-hand fixtures
"""

import math
import unittest

from tests.conftest import open_hand_landmarks, fist_landmarks, make_landmarks
from tests.conftest import (
    WRIST, THUMB_TIP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_PIP, RING_TIP, PINKY_MCP, PINKY_PIP, PINKY_TIP,
)

from gesture_validator import (
    finger_ratio, is_finger_open, is_fist, hand_scale, Finger,
)


class TestHandScale(unittest.TestCase):
    """hand_scale() = Dist(WRIST, MIDDLE_MCP)."""

    def test_unit_scale(self):
        lms = make_landmarks({WRIST: (0, 0), MIDDLE_MCP: (0, 1)})
        self.assertAlmostEqual(hand_scale(lms), 1.0, places=6)

    def test_arbitrary_scale(self):
        lms = make_landmarks({WRIST: (0, 0), MIDDLE_MCP: (3, 4)})
        self.assertAlmostEqual(hand_scale(lms), 5.0, places=6)

    def test_degenerate_zero(self):
        """Wrist and MiddleMCP at the same point → scale is 0."""
        lms = make_landmarks({WRIST: (0, 0), MIDDLE_MCP: (0, 0)})
        self.assertAlmostEqual(hand_scale(lms), 0.0, places=6)


class TestFingerRatio(unittest.TestCase):
    """finger_ratio() returns a scalar for each finger."""

    def _lms_for_ratio(self, pip_dist: float, tip_dist: float) -> list:
        """Build landmarks where INDEX PIP is `pip_dist` and TIP is `tip_dist` from WRIST.
        hand_scale = 1.0 (WRIST at origin, MIDDLE_MCP at y=1).
        """
        return make_landmarks({
            WRIST:      (0, 0),
            MIDDLE_MCP: (0, 1),     # hand_scale = 1.0
            INDEX_PIP:  (0, pip_dist),
            INDEX_TIP:  (0, tip_dist),
        })

    def test_open_index_positive(self):
        """TIP further than PIP → positive ratio."""
        lms = self._lms_for_ratio(pip_dist=1.0, tip_dist=1.5)
        self.assertGreater(finger_ratio(lms, Finger.INDEX), 0.0)

    def test_closed_index_negative(self):
        """TIP closer than PIP → negative ratio (curled)."""
        lms = self._lms_for_ratio(pip_dist=1.2, tip_dist=0.8)
        self.assertLess(finger_ratio(lms, Finger.INDEX), 0.0)

    def test_ratio_magnitude(self):
        """Ratio = (tip_dist - pip_dist) / hand_scale = 0.5 / 1.0."""
        lms = self._lms_for_ratio(pip_dist=1.0, tip_dist=1.5)
        self.assertAlmostEqual(finger_ratio(lms, Finger.INDEX), 0.5, places=5)

    def test_zero_scale_returns_zero(self):
        """Degenerate hand (no scale) → ratio = 0, no division error."""
        lms = make_landmarks({WRIST: (0, 0), MIDDLE_MCP: (0, 0)})
        self.assertEqual(finger_ratio(lms, Finger.INDEX), 0.0)

    def test_thumb_uses_pinky_mcp_distance(self):
        """Thumb ratio = Dist(THUMB_TIP, PINKY_MCP) / hand_scale."""
        lms = make_landmarks({
            WRIST:      (0, 0),
            MIDDLE_MCP: (0, 1),    # hand_scale = 1.0
            THUMB_TIP:  (3, 4),    # dist from origin = 5, but we measure vs PINKY_MCP
            PINKY_MCP:  (0, 4),    # Dist(ThumbTip, PinkyMCP) = 3.0
        })
        self.assertAlmostEqual(finger_ratio(lms, Finger.THUMB), 3.0, places=5)

    def test_all_fingers_return_float(self):
        """finger_ratio should return a float for every Finger enum value."""
        lms = open_hand_landmarks()
        for finger in Finger:
            result = finger_ratio(lms, finger)
            self.assertIsInstance(result, float, f"finger_ratio returned non-float for {finger}")


class TestIsFingerOpen(unittest.TestCase):
    """is_finger_open() stateless single-shot classification."""

    def test_open_hand_all_fingers_open(self):
        lms = open_hand_landmarks()
        for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
            self.assertTrue(
                is_finger_open(lms, "Right", finger),
                f"{finger.name} should be open on open-hand fixture",
            )

    def test_fist_all_fingers_closed(self):
        lms = fist_landmarks()
        for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
            self.assertFalse(
                is_finger_open(lms, "Right", finger),
                f"{finger.name} should be closed on fist fixture",
            )

    def test_custom_threshold_overrides_cfg(self):
        """Passing a very high threshold should classify any hand as closed."""
        lms = open_hand_landmarks()
        self.assertFalse(
            is_finger_open(lms, "Right", Finger.INDEX, open_threshold=99.0)
        )

    def test_custom_threshold_low_classifies_as_open(self):
        """A threshold of -1.0 means any ratio passes."""
        lms = fist_landmarks()
        self.assertTrue(
            is_finger_open(lms, "Right", Finger.INDEX, open_threshold=-1.0)
        )


class TestIsFist(unittest.TestCase):
    """is_fist() must correctly classify fist vs open hand."""

    def test_fist_fixture_detected(self):
        self.assertTrue(is_fist(fist_landmarks()))

    def test_open_hand_not_fist(self):
        self.assertFalse(is_fist(open_hand_landmarks()))

    def test_single_open_finger_breaks_fist(self):
        """If even one finger is open the hand is NOT a fist."""
        lms = fist_landmarks()
        # Force INDEX_TIP far from WRIST (open index finger)
        lms[INDEX_TIP]  = type(lms[0])(x=0.2, y=1.7, z=0.0)
        lms[INDEX_PIP]  = type(lms[0])(x=0.2, y=1.2, z=0.0)
        lms[MIDDLE_MCP] = type(lms[0])(x=0.0, y=1.0, z=0.0)  # keep hand_scale=1.0
        self.assertFalse(is_fist(lms))


if __name__ == "__main__":
    unittest.main()
