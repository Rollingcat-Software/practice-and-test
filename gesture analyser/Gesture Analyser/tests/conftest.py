"""
Shared fixtures and landmark helpers for the test suite.

MediaPipe landmarks are accessed as ``landmarks[i].x / .y / .z``.
We build lightweight mock objects so the pure-Python geometry modules
can be exercised without any camera or native library dependency.
"""

from __future__ import annotations
from dataclasses import dataclass


# ── Minimal landmark mock ─────────────────────────────────────────────

@dataclass
class LM:
    """Minimal stand-in for mediapipe.framework.formats.landmark_pb2.Landmark."""
    x: float
    y: float
    z: float = 0.0


def make_landmarks(positions: dict[int, tuple]) -> list[LM]:
    """Build a 21-element landmark list from a sparse position dict.

    Unspecified indices default to the origin so that only the
    landmarks relevant to each test case need to be supplied.
    """
    lms = [LM(0.0, 0.0, 0.0) for _ in range(21)]
    for idx, coords in positions.items():
        x, y = coords[0], coords[1]
        z    = coords[2] if len(coords) > 2 else 0.0
        lms[idx] = LM(x, y, z)
    return lms


# ── Landmark indices (mirrors gesture_validator._LM) ─────────────────

WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_PIP  = 6
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP   = 13
RING_PIP   = 14
RING_TIP   = 16
PINKY_MCP  = 17
PINKY_PIP  = 18
PINKY_TIP  = 20


def open_hand_landmarks() -> list[LM]:
    """
    Idealised flat open hand (palm facing camera).

    WRIST at origin. MIDDLE_MCP at y=1.0 → hand_scale = 1.0.
    All four non-thumb fingers extended so their TIPs are further
    from WRIST than their PIPs  →  finger_ratio > 0.20 (open).
    Thumb extended laterally   →  Dist(ThumbTip, PinkyMCP) > 0.75 (open).
    """
    return make_landmarks({
        WRIST:      (0.0, 0.0),
        # Thumb extended sideways
        THUMB_TIP:  (1.2, 0.5),
        # Index
        INDEX_MCP:  (0.2, 0.8),
        INDEX_PIP:  (0.2, 1.2),
        INDEX_TIP:  (0.2, 1.7),
        # Middle (sets hand_scale)
        MIDDLE_MCP: (0.0, 1.0),
        MIDDLE_PIP: (0.0, 1.4),
        MIDDLE_TIP: (0.0, 1.9),
        # Ring
        RING_MCP:   (-0.2, 0.9),
        RING_PIP:   (-0.2, 1.3),
        RING_TIP:   (-0.2, 1.8),
        # Pinky
        PINKY_MCP:  (-0.4, 0.8),
        PINKY_PIP:  (-0.4, 1.1),
        PINKY_TIP:  (-0.4, 1.5),
    })


def fist_landmarks() -> list[LM]:
    """
    Idealised closed fist.

    All fingertips are curled back towards the palm so:
      • finger_ratio for every non-thumb finger < 0.20 (closed)
      • Dist(ThumbTip, PinkyMCP) < 0.75 (thumb folded over)
    hand_scale (WRIST → MIDDLE_MCP) = 1.0 as reference.
    """
    return make_landmarks({
        WRIST:      (0.0, 0.0),
        # Thumb folded across palm, close to PinkyMCP
        THUMB_TIP:  (-0.3, 0.7),
        # Fingers curled: TIP closer to WRIST than PIP
        INDEX_MCP:  (0.2, 0.8),
        INDEX_PIP:  (0.2, 1.1),
        INDEX_TIP:  (0.1, 0.9),    # curled back
        MIDDLE_MCP: (0.0, 1.0),    # hand_scale anchor
        MIDDLE_PIP: (0.0, 1.3),
        MIDDLE_TIP: (0.0, 1.0),    # curled back
        RING_MCP:   (-0.2, 0.9),
        RING_PIP:   (-0.2, 1.2),
        RING_TIP:   (-0.2, 0.9),   # curled back
        PINKY_MCP:  (-0.4, 0.8),
        PINKY_PIP:  (-0.4, 1.0),
        PINKY_TIP:  (-0.4, 0.8),   # curled back
    })
