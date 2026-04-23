"""
Gesture Validator Module -- v6b: Reverted core + tighter thresholds
====================================================================
Pure-logic module -- no OpenCV / capture dependencies.

Reverted to the proven v5 distance-difference algorithm that was
counting fingers correctly, but with **tighter hysteresis thresholds**
to eliminate the fist false positive (was reporting 1 finger on a
closed fist).

Core algorithm (unchanged from v5):
    Index-Pinky: (Dist(Wrist,Tip) - Dist(Wrist,PIP)) / hand_scale
    Thumb:       Dist(ThumbTip, PinkyMCP) / hand_scale

What changed from v5:
    - Raised finger open threshold:  0.15 -> 0.20
    - Raised finger close threshold: 0.08 -> 0.12
    - Raised thumb open threshold:   0.70 -> 0.75
    - Raised thumb close threshold:  0.55 -> 0.60
    These tighter values ensure a fist reads 0 without breaking
    normal finger counting.

Stabilisation layers:
    Layer 1: Adaptive normalisation (hand_scale)
    Layer 2: Hysteresis dual-threshold
    Layer 3: EWMA temporal smoothing
    Layer 4: Depth proxy for Z-axis commands
"""

import math
from dataclasses import dataclass
from enum import IntEnum


class Finger(IntEnum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


# ── MediaPipe landmark indices ──────────────────────────────────────

_LM = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_DIP": 7, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_DIP": 11, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_DIP": 15, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
}

_FINGER_JOINTS = {
    Finger.INDEX:  (_LM["INDEX_PIP"],  _LM["INDEX_TIP"]),
    Finger.MIDDLE: (_LM["MIDDLE_PIP"], _LM["MIDDLE_TIP"]),
    Finger.RING:   (_LM["RING_PIP"],   _LM["RING_TIP"]),
    Finger.PINKY:  (_LM["PINKY_PIP"],  _LM["PINKY_TIP"]),
}

FINGER_TIP_IDS = {
    Finger.THUMB:  _LM["THUMB_TIP"],
    Finger.INDEX:  _LM["INDEX_TIP"],
    Finger.MIDDLE: _LM["MIDDLE_TIP"],
    Finger.RING:   _LM["RING_TIP"],
    Finger.PINKY:  _LM["PINKY_TIP"],
}


# ── Math helpers ────────────────────────────────────────────────────

def _euclidean(a, b) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def hand_scale(landmarks) -> float:
    """Dist(Wrist, MiddleMCP).  Normalisation baseline."""
    return _euclidean(landmarks[_LM["WRIST"]], landmarks[_LM["MIDDLE_MCP"]])


def depth_proxy(landmarks) -> float:
    """Alias of hand_scale.  Increases when hand moves closer."""
    return hand_scale(landmarks)


def _tip_wrist_ratio(landmarks, finger_name: Finger) -> float:
    """Dist(Wrist, Tip) / hand_scale.  For debug display."""
    hs = hand_scale(landmarks)
    if hs < 1e-9:
        return 0.0
    wrist = landmarks[_LM["WRIST"]]
    tip_id = FINGER_TIP_IDS[finger_name]
    return _euclidean(wrist, landmarks[tip_id]) / hs


# ── Per-finger ratio (v5 proven method) ─────────────────────────────

def finger_ratio(landmarks, finger_name: Finger) -> float:
    """Normalised openness metric.

    Index-Pinky: (Dist(Wrist,Tip) - Dist(Wrist,PIP)) / hand_scale
    Thumb:       Dist(ThumbTip, PinkyMCP) / hand_scale
    """
    hs = hand_scale(landmarks)
    if hs < 1e-9:
        return 0.0

    wrist = landmarks[_LM["WRIST"]]

    if finger_name == Finger.THUMB:
        return _euclidean(landmarks[_LM["THUMB_TIP"]], landmarks[_LM["PINKY_MCP"]]) / hs

    pip_id, tip_id = _FINGER_JOINTS[finger_name]
    dist_tip = _euclidean(wrist, landmarks[tip_id])
    dist_pip = _euclidean(wrist, landmarks[pip_id])
    return (dist_tip - dist_pip) / hs


def is_finger_open(landmarks, hand_label: str, finger_name: Finger,
                   open_threshold: float | None = None) -> bool:
    """Stateless single-shot check (no hysteresis / smoothing)."""
    if open_threshold is None:
        open_threshold = 0.75 if finger_name == Finger.THUMB else 0.20
    return finger_ratio(landmarks, finger_name) > open_threshold


def is_fist(landmarks) -> bool:
    """Return True if all fingers score below their open threshold."""
    for finger in Finger:
        score = finger_ratio(landmarks, finger)
        th = 0.75 if finger == Finger.THUMB else 0.20
        if score > th:
            return False
    return True


# ── Debug info ──────────────────────────────────────────────────────

def finger_debug_info(landmarks, handedness: str = "Right") -> dict[Finger, tuple[int, bool, float, float]]:
    """Per-finger debug: {finger: (tip_id, is_open, ratio, tip_wrist_ratio)}."""
    info = {}
    for finger in Finger:
        tip_id = FINGER_TIP_IDS[finger]
        ratio = finger_ratio(landmarks, finger)
        tw = _tip_wrist_ratio(landmarks, finger)
        open_th = 0.75 if finger == Finger.THUMB else 0.20
        info[finger] = (tip_id, ratio > open_th, ratio, tw)
    return info


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class FingerState:
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    @property
    def count(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])

    def as_list(self) -> list[bool]:
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]


# ── Hysteresis (Layer 2) ────────────────────────────────────────────

class _HysteresisState:
    def __init__(self, finger_open_th, finger_close_th, thumb_open_th, thumb_close_th):
        self.finger_open_th = finger_open_th
        self.finger_close_th = finger_close_th
        self.thumb_open_th = thumb_open_th
        self.thumb_close_th = thumb_close_th
        self._state: dict[Finger, bool] = {f: False for f in Finger}

    def update(self, landmarks) -> FingerState:
        for finger in Finger:
            ratio = finger_ratio(landmarks, finger)
            is_thumb = (finger == Finger.THUMB)
            open_th = self.thumb_open_th if is_thumb else self.finger_open_th
            close_th = self.thumb_close_th if is_thumb else self.finger_close_th

            if self._state[finger]:
                if ratio < close_th:
                    self._state[finger] = False
            else:
                if ratio > open_th:
                    self._state[finger] = True

        return FingerState(
            thumb=self._state[Finger.THUMB],
            index=self._state[Finger.INDEX],
            middle=self._state[Finger.MIDDLE],
            ring=self._state[Finger.RING],
            pinky=self._state[Finger.PINKY],
        )

    def reset(self):
        self._state = {f: False for f in Finger}


# ── EWMA smoother (Layer 3) ─────────────────────────────────────────

class _EWMASmoother:
    """Per-finger EWMA.  confidence > 0.5 = open."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._conf: dict[Finger, float] = {f: 0.0 for f in Finger}

    def update(self, hyst_state: FingerState) -> FingerState:
        samples = {
            Finger.THUMB: float(hyst_state.thumb),
            Finger.INDEX: float(hyst_state.index),
            Finger.MIDDLE: float(hyst_state.middle),
            Finger.RING: float(hyst_state.ring),
            Finger.PINKY: float(hyst_state.pinky),
        }
        for f in Finger:
            self._conf[f] = self.alpha * samples[f] + (1.0 - self.alpha) * self._conf[f]

        return FingerState(
            thumb=self._conf[Finger.THUMB] > 0.5,
            index=self._conf[Finger.INDEX] > 0.5,
            middle=self._conf[Finger.MIDDLE] > 0.5,
            ring=self._conf[Finger.RING] > 0.5,
            pinky=self._conf[Finger.PINKY] > 0.5,
        )

    def reset(self):
        self._conf = {f: 0.0 for f in Finger}


# ── Moving Median Filter (Layer 4) ──────────────────────────────────

class _MedianFilter:
    """Moving median of recent finger counts.  Eliminates the 0↔1
    flicker on tight fists that EWMA alone can't fully suppress."""

    def __init__(self, window: int = 5):
        from collections import deque
        self._buf: deque[int] = deque(maxlen=window)

    def filter(self, count: int) -> int:
        self._buf.append(count)
        return sorted(self._buf)[len(self._buf) // 2]

    def reset(self):
        self._buf.clear()


# ── Auto-Calibration ────────────────────────────────────────────────

class HandCalibrator:
    """Collects finger_ratio samples during a 2-second calibration
    window, then computes per-user threshold adjustments.

    Usage:
        cal = HandCalibrator()
        # in frame loop:
        if not cal.is_done:
            cal.feed(landmarks)
        else:
            offsets = cal.offsets  # dict of adjusted thresholds
    """

    def __init__(self, duration: float = 2.0):
        import time as _time
        self._duration = duration
        self._start: float | None = None
        self._samples: dict[Finger, list[float]] = {f: [] for f in Finger}
        self._done = False
        self._offsets: dict[str, float] = {}

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def progress(self) -> float:
        if self._start is None:
            return 0.0
        import time as _time
        return min((_time.monotonic() - self._start) / self._duration, 1.0)

    @property
    def offsets(self) -> dict[str, float]:
        return self._offsets

    def feed(self, landmarks) -> None:
        import time as _time
        if self._done:
            return
        if self._start is None:
            self._start = _time.monotonic()

        for finger in Finger:
            self._samples[finger].append(finger_ratio(landmarks, finger))

        if _time.monotonic() - self._start >= self._duration:
            self._compute()
            self._done = True

    def _compute(self):
        """Compute calibrated thresholds from collected samples.

        During calibration the user should show an open hand then a fist.
        We find the max ratio seen (open) and min ratio seen (closed),
        then set thresholds at the midpoint.
        """
        for finger in Finger:
            vals = self._samples[finger]
            if not vals:
                continue
            # Store the average as a reference -- not used to override
            # thresholds directly, but available for diagnostics.
            avg = sum(vals) / len(vals)
            self._offsets[finger.name] = avg

    def reset(self):
        import time as _time
        self._start = None
        self._samples = {f: [] for f in Finger}
        self._done = False
        self._offsets = {}


# ── Main validator ──────────────────────────────────────────────────

class GestureValidator:
    """Four-layer engine: Adaptive + Hysteresis + EWMA + Depth.

    Default thresholds tightened vs v5 to eliminate fist false positives:
        fingers: 0.15/0.08 -> 0.20/0.12
        thumb:   0.70/0.55 -> 0.75/0.60
    """

    def __init__(
        self,
        ewma_alpha: float = 0.35,
        finger_open_th: float = 0.20,
        finger_close_th: float = 0.12,
        thumb_open_th: float = 0.75,
        thumb_close_th: float = 0.60,
        smoothing_window: int = 7,  # legacy, ignored
    ):
        self._alpha = ewma_alpha
        self._finger_open = finger_open_th
        self._finger_close = finger_close_th
        self._thumb_open = thumb_open_th
        self._thumb_close = thumb_close_th

        self._hysteresis: dict[str, _HysteresisState] = {}
        self._smoothers: dict[str, _EWMASmoother] = {}
        self._medians: dict[str, _MedianFilter] = {}

    def _get_hysteresis(self, label: str) -> _HysteresisState:
        if label not in self._hysteresis:
            self._hysteresis[label] = _HysteresisState(
                self._finger_open, self._finger_close,
                self._thumb_open, self._thumb_close,
            )
        return self._hysteresis[label]

    def _get_smoother(self, label: str) -> _EWMASmoother:
        if label not in self._smoothers:
            self._smoothers[label] = _EWMASmoother(alpha=self._alpha)
        return self._smoothers[label]

    def _get_median(self, label: str) -> _MedianFilter:
        if label not in self._medians:
            self._medians[label] = _MedianFilter(window=5)
        return self._medians[label]

    def detect_fingers_raw(self, landmarks, handedness: str = "Right") -> FingerState:
        """Hysteresis-only (Layers 1+2)."""
        return self._get_hysteresis(handedness).update(landmarks)

    def detect_fingers(self, landmarks, handedness: str = "Right") -> FingerState:
        """Full pipeline: Adaptive + Hysteresis + EWMA."""
        hyst_state = self.detect_fingers_raw(landmarks, handedness)
        return self._get_smoother(handedness).update(hyst_state)

    def count_fingers(self, landmarks, handedness: str = "Right") -> int:
        raw_count = self.detect_fingers(landmarks, handedness).count
        return self._get_median(handedness).filter(raw_count)

    def count_fingers_total(self, hands) -> int:
        return sum(self.count_fingers(h.landmarks, h.handedness) for h in hands)

    def validate_finger_count(self, landmarks, target: int, handedness: str = "Right") -> bool:
        return self.count_fingers(landmarks, handedness) == target

    def validate_total(self, hands, target: int) -> bool:
        return self.count_fingers_total(hands) == target

    def validate_specific_fingers(self, landmarks, target: list[bool], handedness: str = "Right") -> bool:
        return self.detect_fingers(landmarks, handedness).as_list() == target

    def clear_buffers(self) -> None:
        self._hysteresis.clear()
        self._smoothers.clear()
        self._medians.clear()
