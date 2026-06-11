"""
tests/test_touch_detector.py
=============================
Unit tests for the 4-gate touch detection pipeline in finger_touch_detector.py.

Gate order tested:
  1. Fist gate  — fist landmarks must never trigger a touch command.
  2. Z-axis gate — fingers at very different depths are rejected.
  3. Distance gate — fingertips far apart are rejected.
  4. Bystander gate — all bystander fingers curled → rejected.

The public API is ``FingerTouchDetector.update(hands)`` where ``hands``
is a list of objects with ``.landmarks`` and ``.handedness`` attributes.
We supply a minimal ``FakeHand`` stub so no MediaPipe types are needed.
"""

import unittest
from dataclasses import dataclass
from typing import List

from tests.conftest import (
    make_landmarks, open_hand_landmarks, fist_landmarks,
    WRIST, THUMB_TIP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
)
from finger_touch_detector import TouchCommand, FingerTouchDetector


# ── Minimal HandResult stub ───────────────────────────────────────────

@dataclass
class FakeHand:
    """Minimal stand-in for hand_tracker.HandResult."""
    landmarks: list
    handedness: str = "Right"


def _hands(lms, handedness="Right") -> List[FakeHand]:
    return [FakeHand(landmarks=lms, handedness=handedness)]


# ── Landmark helpers ──────────────────────────────────────────────────

def _pinch_landmarks(
    thumb_x=0.0, thumb_y=0.5, thumb_z=0.0,
    index_x=0.0, index_y=0.5, index_z=0.0,
) -> list:
    """
    Build landmarks where thumb-tip and index-tip are at the given coords.
    Bystander fingers (middle, ring, pinky) are extended so Gate 4 passes.
    hand_scale = 1.0 (WRIST at origin, MIDDLE_MCP at y=1).
    """
    return make_landmarks({
        WRIST:       (0.0,  0.0,  0.0),
        MIDDLE_MCP:  (0.0,  1.0,  0.0),
        THUMB_TIP:   (thumb_x, thumb_y, thumb_z),
        INDEX_TIP:   (index_x, index_y, index_z),
        INDEX_PIP:   (0.2,  1.2,  0.0),
        # Open bystanders
        MIDDLE_PIP:  (0.0,  1.4,  0.0),
        MIDDLE_TIP:  (0.0,  1.9,  0.0),
        RING_MCP:    (-0.2, 0.9,  0.0),
        RING_PIP:    (-0.2, 1.3,  0.0),
        RING_TIP:    (-0.2, 1.8,  0.0),
        PINKY_MCP:   (-0.4, 0.8,  0.0),
        PINKY_PIP:   (-0.4, 1.1,  0.0),
        PINKY_TIP:   (-0.4, 1.5,  0.0),
    })


def _fist_with_close_fingertips() -> list:
    """
    Full fist where thumb-tip and index-tip happen to overlap in XY.
    A naïve distance check would fire; the fist gate must stop it.
    """
    lms = fist_landmarks()
    lms[THUMB_TIP] = type(lms[0])(x=0.05, y=0.8, z=0.0)
    lms[INDEX_TIP] = type(lms[0])(x=0.05, y=0.8, z=0.0)
    return lms


# ── Gate 1: Fist gate ─────────────────────────────────────────────────

class TestFistGate(unittest.TestCase):

    def setUp(self):
        self.det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX)

    def test_fist_with_overlapping_tips_never_triggers(self):
        lms = _fist_with_close_fingertips()
        for _ in range(20):
            result = self.det.update(_hands(lms))
        self.assertFalse(result)

    def test_open_hand_tips_far_apart_not_triggered(self):
        """Sanity: open hand with no pinch should not trigger."""
        result = self.det.update(_hands(open_hand_landmarks()))
        self.assertFalse(result)

    def test_empty_hands_list_not_triggered(self):
        result = self.det.update([])
        self.assertFalse(result)


# ── Gate 2: Z-axis depth gate ─────────────────────────────────────────

class TestZAxisGate(unittest.TestCase):

    def setUp(self):
        self.det = FingerTouchDetector(
            TouchCommand.THUMB_TO_INDEX,
            verify_frames=1,
        )

    def test_large_z_difference_rejected(self):
        """Same XY but 0.5 z-gap → rejected by z-axis gate."""
        lms = _pinch_landmarks(
            thumb_x=0.0, thumb_y=0.5, thumb_z=0.0,
            index_x=0.0, index_y=0.5, index_z=0.5,
        )
        result = self.det.update(_hands(lms))
        self.assertFalse(result)

    def test_same_z_touching_accepted(self):
        """Fingertips touching at same depth should pass."""
        lms = _pinch_landmarks(
            thumb_x=0.0, thumb_y=0.5, thumb_z=0.0,
            index_x=0.0, index_y=0.5, index_z=0.0,
        )
        result = self.det.update(_hands(lms))
        self.assertTrue(result)


# ── Gate 3: Distance gate ─────────────────────────────────────────────

class TestDistanceGate(unittest.TestCase):

    def setUp(self):
        self.det = FingerTouchDetector(
            TouchCommand.THUMB_TO_INDEX,
            verify_frames=1,
        )

    def test_far_apart_rejected(self):
        """Thumb at origin, index at (1, 1) → large normalised distance."""
        lms = _pinch_landmarks(
            thumb_x=0.0, thumb_y=0.0, thumb_z=0.0,
            index_x=1.0, index_y=1.0, index_z=0.0,
        )
        result = self.det.update(_hands(lms))
        self.assertFalse(result)

    def test_touching_accepted(self):
        """Both tips at the exact same point should trigger immediately."""
        lms = _pinch_landmarks(
            thumb_x=0.3, thumb_y=0.5, thumb_z=0.0,
            index_x=0.3, index_y=0.5, index_z=0.0,
        )
        result = self.det.update(_hands(lms))
        self.assertTrue(result)


# ── Gate 4: Bystander gate ────────────────────────────────────────────

class TestBystanderGate(unittest.TestCase):

    def _all_bystanders_closed_lms(self) -> list:
        """Thumb+index touching but all bystander fingers curled."""
        return make_landmarks({
            WRIST:      (0.0, 0.0, 0.0),
            MIDDLE_MCP: (0.0, 1.0, 0.0),
            THUMB_TIP:  (0.0, 0.5, 0.0),
            INDEX_TIP:  (0.0, 0.5, 0.0),
            INDEX_PIP:  (0.2, 1.1, 0.0),
            # Bystanders curled
            MIDDLE_PIP: (0.0, 1.3, 0.0),
            MIDDLE_TIP: (0.0, 1.0, 0.0),
            RING_MCP:   (-0.2, 0.9, 0.0),
            RING_PIP:   (-0.2, 1.2, 0.0),
            RING_TIP:   (-0.2, 0.9, 0.0),
            PINKY_MCP:  (-0.4, 0.8, 0.0),
            PINKY_PIP:  (-0.4, 1.0, 0.0),
            PINKY_TIP:  (-0.4, 0.8, 0.0),
        })

    def test_all_bystanders_closed_rejected(self):
        det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX, verify_frames=1)
        result = det.update(_hands(self._all_bystanders_closed_lms()))
        self.assertFalse(result)

    def test_one_bystander_open_allows_trigger(self):
        det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX, verify_frames=1)
        lms = self._all_bystanders_closed_lms()
        # Open the middle finger (a valid bystander for THUMB_TO_INDEX)
        lms[MIDDLE_PIP] = type(lms[0])(x=0.0, y=1.4, z=0.0)
        lms[MIDDLE_TIP] = type(lms[0])(x=0.0, y=1.9, z=0.0)
        result = det.update(_hands(lms))
        self.assertTrue(result)


# ── verify_frames hold requirement ────────────────────────────────────

class TestVerifyFrameHold(unittest.TestCase):

    def _touching(self) -> List[FakeHand]:
        return _hands(_pinch_landmarks(
            thumb_x=0.3, thumb_y=0.5, thumb_z=0.0,
            index_x=0.3, index_y=0.5, index_z=0.0,
        ))

    def test_requires_n_consecutive_frames(self):
        det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX, verify_frames=5)
        results = [det.update(self._touching()) for _ in range(4)]
        self.assertFalse(any(results), "Must not trigger before verify_frames")
        self.assertTrue(det.update(self._touching()), "Must trigger on 5th frame")

    def test_break_resets_counter(self):
        """Releasing the touch before verify_frames resets the hold counter."""
        det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX, verify_frames=5)
        no_touch = _hands(open_hand_landmarks())

        for _ in range(4):
            det.update(self._touching())
        det.update(no_touch)                          # break the hold

        results = [det.update(self._touching()) for _ in range(4)]
        self.assertFalse(any(results), "Counter must have reset after break")
        self.assertTrue(det.update(self._touching()), "Must trigger after new 5-frame hold")

    def test_verified_state_is_sticky(self):
        """Once verified, further frames return True without re-checking."""
        det = FingerTouchDetector(TouchCommand.THUMB_TO_INDEX, verify_frames=1)
        det.update(self._touching())
        self.assertTrue(det.update(_hands(open_hand_landmarks())))


if __name__ == "__main__":
    unittest.main()
