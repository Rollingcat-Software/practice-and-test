"""
Sequential Challenge Mode -- Full Suite
========================================
Tests EVERY challenge type in fixed order with a per-step timer.
Each step has a time limit; if exceeded, the step is marked TIMEOUT
and the system moves on.

Steps (19 total):
  1-8.  Finger counts: 0 (fist), 1, 2, 3, 4, 5, both open (10), both fists (0+0)
  9.    Left thumb up
  10.   Wave your hand
  11.   Move hand closer
  12.   Move hand away
  13.   Draw a circle
  14.   Draw a square
  15.   Touch thumb to index
  16.   Touch thumb to pinky
  17.   Flip your hand
  18.   Peek-a-boo (hide then show)
  19.   Left 3 + Right 2
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import (
    GestureValidator, depth_proxy, is_finger_open, Finger,
    hand_scale, _euclidean, _LM,
)
from motion_analyzer import WaveDetector, ShapeRecognizer


class SeqState(Enum):
    ACTIVE = auto()
    HOLDING = auto()
    STEP_DONE = auto()
    STEP_TIMEOUT = auto()
    COMPLETE = auto()


class StepResult(Enum):
    PENDING = auto()
    PASSED = auto()
    TIMED_OUT = auto()


@dataclass
class SeqStep:
    name: str
    step_type: str
    time_limit: float = 5.0       # seconds allowed for this step
    target: int = 0               # for finger-count steps
    gesture_reqs: dict | None = None  # for dual-hand gestures
    tap_a: int = 0                # landmark id for finger tap
    tap_b: int = 0
    tap_threshold: float = 0.3


# ── Full challenge list ─────────────────────────────────────────────

_STEPS = [
    # Finger counts (single hand)
    SeqStep("Show FIST (0 fingers)",          "fingers",    5.0, target=0),
    SeqStep("Show 1 FINGER (point)",          "fingers",    5.0, target=1),
    SeqStep("Show 2 FINGERS (peace)",         "fingers",    5.0, target=2),
    SeqStep("Show 3 FINGERS",                 "fingers",    5.0, target=3),
    SeqStep("Show 4 FINGERS",                 "fingers",    5.0, target=4),
    SeqStep("Show 5 FINGERS (open hand)",     "fingers",    5.0, target=5),
    # Dual-hand gestures
    SeqStep("OPEN BOTH HANDS (10 fingers)",   "gesture",    6.0, gesture_reqs={"Left": 5, "Right": 5}),
    SeqStep("BOTH FISTS (0+0)",              "gesture",    6.0, gesture_reqs={"Left": 0, "Right": 0}),
    SeqStep("LEFT THUMB UP (1 finger left)",  "gesture",    6.0, gesture_reqs={"Left": 1}),
    SeqStep("LEFT 3 + RIGHT 2",              "gesture",    6.0, gesture_reqs={"Left": 3, "Right": 2}),
    # Motion
    SeqStep("WAVE Your Hand",                 "wave",       6.0),
    # Spatial
    SeqStep("Move Hand CLOSER",               "closer",     5.0),
    SeqStep("Move Hand AWAY",                 "away",       5.0),
    # Drawing
    SeqStep("DRAW A CIRCLE",                  "draw_circle", 8.0),
    SeqStep("DRAW A SQUARE",                  "draw_square", 8.0),
    # Finger taps
    SeqStep("TOUCH Thumb to Index",           "finger_tap", 5.0, tap_a=4, tap_b=8, tap_threshold=0.3),
    SeqStep("TOUCH Thumb to Pinky",           "finger_tap", 5.0, tap_a=4, tap_b=20, tap_threshold=0.3),
    # Advanced
    SeqStep("FLIP Your Hand (palm then back)", "hand_flip", 6.0),
    SeqStep("PEEK-A-BOO (hide then show)",    "peek_a_boo", 6.0),
]


@dataclass
class SequentialSession:
    """Fixed-order challenge runner with per-step timers.

    Parameters
    ----------
    hold_seconds : float
        How long a static gesture must be held to confirm.
    pause_after_step : float
        Pause between steps.
    depth_threshold : float
        Required depth change for closer/away.
    smoothing_window : int
        Passed to GestureValidator.
    """

    hold_seconds: float = 1.0
    pause_after_step: float = 1.0
    depth_threshold: float = 0.20
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: SeqState = field(init=False, default=SeqState.ACTIVE)
    current_step_idx: int = field(init=False, default=0)
    step_results: list[StepResult] = field(init=False, default_factory=list)

    _hold_start: Optional[float] = field(init=False, default=None)
    _step_start: Optional[float] = field(init=False, default=None)
    _step_done_at: Optional[float] = field(init=False, default=None)
    _global_start: Optional[float] = field(init=False, default=None)

    _baseline_depth: Optional[float] = field(init=False, default=None)
    _current_depth: Optional[float] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)
    _wave: WaveDetector = field(init=False)
    _shape: ShapeRecognizer = field(init=False)
    _was_drawing: bool = field(init=False, default=False)

    # Hand flip
    _flip_baseline_z: Optional[float] = field(init=False, default=None)
    # Peek-a-boo
    _peekaboo_phase: int = field(init=False, default=0)
    _peekaboo_hidden_at: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._wave = WaveDetector(buffer_size=40, min_swing=0.10,
                                  min_total_displacement=0.20, min_reversals=2)
        self._shape = ShapeRecognizer(min_points=15)
        self.step_results = [StepResult.PENDING] * len(_STEPS)

    @property
    def steps(self) -> list[SeqStep]:
        return _STEPS

    @property
    def current_step(self) -> Optional[SeqStep]:
        if self.current_step_idx < len(_STEPS):
            return _STEPS[self.current_step_idx]
        return None

    @property
    def total_steps(self) -> int:
        return len(_STEPS)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.step_results if r == StepResult.PASSED)

    @property
    def elapsed_time(self) -> float:
        if self._global_start is None:
            return 0.0
        return time.monotonic() - self._global_start

    @property
    def step_time_remaining(self) -> float:
        step = self.current_step
        if step is None or self._step_start is None:
            return 0.0
        return max(0.0, step.time_limit - (time.monotonic() - self._step_start))

    @property
    def step_time_elapsed(self) -> float:
        if self._step_start is None:
            return 0.0
        return time.monotonic() - self._step_start

    @property
    def hold_progress(self) -> float:
        if self._hold_start is None:
            return 0.0
        return min((time.monotonic() - self._hold_start) / self.hold_seconds, 1.0)

    @property
    def depth_change_pct(self) -> Optional[float]:
        if self._baseline_depth is None or self._current_depth is None:
            return None
        return ((self._current_depth / self._baseline_depth) - 1.0) * 100.0

    @property
    def drawing_path(self) -> list[tuple[float, float]]:
        return self._shape.pixel_path

    @property
    def display_text(self) -> str:
        if self.state == SeqState.COMPLETE:
            return f"ALL DONE! ({self.passed_count}/{self.total_steps} passed)"
        if self.state == SeqState.STEP_DONE:
            return "PASSED!"
        if self.state == SeqState.STEP_TIMEOUT:
            return "TIME'S UP - skipping..."
        step = self.current_step
        return step.name if step else ""

    @property
    def status_label(self) -> str:
        if self.state == SeqState.COMPLETE:
            return "Status: All Steps Done"
        if self.state == SeqState.STEP_DONE:
            return "Status: Moving to next..."
        if self.state == SeqState.STEP_TIMEOUT:
            return "Status: Timed out"
        if self.state == SeqState.HOLDING:
            return "Status: Hold steady..."
        return "Status: Perform the gesture"

    @property
    def progress_text(self) -> str:
        return f"Task {self.current_step_idx + 1}/{self.total_steps}"

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    # -- Update -----------------------------------------------------------

    def update(self, hands: list[HandResult]) -> SeqState:
        now = time.monotonic()

        if self._global_start is None and hands:
            self._global_start = now

        if self.state == SeqState.COMPLETE:
            return self.state

        # Pause between steps (after done or timeout).
        if self.state in (SeqState.STEP_DONE, SeqState.STEP_TIMEOUT):
            if self._step_done_at and now - self._step_done_at >= self.pause_after_step:
                self._advance()
            return self.state

        step = self.current_step
        if step is None:
            self.state = SeqState.COMPLETE
            return self.state

        # Start step timer on first frame.
        if self._step_start is None:
            if step.step_type == "peek_a_boo":
                self._step_start = now  # start immediately
            elif hands:
                self._step_start = now
            else:
                return self.state

        # Per-step timeout.
        if now - self._step_start >= step.time_limit:
            self.step_results[self.current_step_idx] = StepResult.TIMED_OUT
            self.state = SeqState.STEP_TIMEOUT
            self._step_done_at = now
            return self.state

        # Check match.
        matched = self._check_step(step, hands, now)

        if matched:
            if step.step_type in ("wave", "draw_circle", "draw_square", "hand_flip", "peek_a_boo"):
                # Instant success for motion/advanced tasks.
                self.step_results[self.current_step_idx] = StepResult.PASSED
                self.state = SeqState.STEP_DONE
                self._step_done_at = now
            else:
                # Hold-based confirmation.
                if self._hold_start is None:
                    self._hold_start = now
                    self.state = SeqState.HOLDING
                elif now - self._hold_start >= self.hold_seconds:
                    self.step_results[self.current_step_idx] = StepResult.PASSED
                    self.state = SeqState.STEP_DONE
                    self._step_done_at = now
        else:
            if step.step_type not in ("wave", "draw_circle", "draw_square", "hand_flip", "peek_a_boo"):
                self._hold_start = None
                if self.state == SeqState.HOLDING:
                    self.state = SeqState.ACTIVE

        return self.state

    def _check_step(self, step: SeqStep, hands: list[HandResult], now: float) -> bool:
        st = step.step_type

        if st == "fingers":
            if not hands:
                return False
            return self._validator.count_fingers_total(hands) == step.target

        if st == "gesture":
            if not hands or not step.gesture_reqs:
                return False
            hand_map = {}
            for h in hands:
                hand_map[h.handedness] = self._validator.count_fingers(h.landmarks, h.handedness)
            return all(hand_map.get(label) == count for label, count in step.gesture_reqs.items())

        if st == "wave":
            if hands:
                self._wave.push(hands[0].landmarks[0].x)
            return self._wave.is_waving()

        if st in ("closer", "away"):
            if not hands:
                return False
            d = depth_proxy(hands[0].landmarks)
            self._current_depth = d
            if self._baseline_depth is None:
                self._baseline_depth = d
                return False
            ratio = d / self._baseline_depth
            if st == "closer":
                return ratio >= (1.0 + self.depth_threshold)
            else:
                return ratio <= (1.0 - self.depth_threshold)

        if st == "draw_circle" or st == "draw_square":
            if not hands:
                return False
            finger_open = is_finger_open(hands[0].landmarks, hands[0].handedness, Finger.INDEX)
            if finger_open:
                tip = hands[0].landmarks[8]
                self._shape.push(tip.x, tip.y)
                self._was_drawing = True
            elif self._was_drawing and len(self._shape.path) >= self._shape.min_points:
                if st == "draw_circle":
                    result = self._shape.finalize_circle()
                else:
                    result = self._shape.finalize_square()
                if result.matched:
                    return True
                self._was_drawing = False
            return False

        if st == "finger_tap":
            if not hands:
                return False
            lm = hands[0].landmarks
            hs = hand_scale(lm)
            if hs < 1e-9:
                return False
            dist = _euclidean(lm[step.tap_a], lm[step.tap_b]) / hs
            return dist < step.tap_threshold

        if st == "hand_flip":
            if not hands:
                return False
            lm = hands[0].landmarks
            avg_z = sum(lm[i].z for i in [4, 8, 12, 16, 20]) / 5.0
            if self._flip_baseline_z is None:
                self._flip_baseline_z = avg_z
                return False
            return abs(avg_z - self._flip_baseline_z) > 0.03

        if st == "peek_a_boo":
            if self._peekaboo_phase == 0:
                if not hands:
                    self._peekaboo_phase = 1
                    self._peekaboo_hidden_at = now
            elif self._peekaboo_phase == 1:
                if not hands:
                    return False
                if self._peekaboo_hidden_at and now - self._peekaboo_hidden_at > 0.5:
                    return True
            return False

        return False

    def _advance(self):
        self.current_step_idx += 1
        self._hold_start = None
        self._step_start = None
        self._step_done_at = None
        self._baseline_depth = None
        self._current_depth = None
        self._flip_baseline_z = None
        self._peekaboo_phase = 0
        self._peekaboo_hidden_at = None
        self._wave.reset()
        self._shape.reset()
        self._was_drawing = False
        self._validator.clear_buffers()

        if self.current_step_idx >= len(_STEPS):
            self.state = SeqState.COMPLETE
        else:
            self.state = SeqState.ACTIVE

    def reset(self):
        self.current_step_idx = 0
        self.state = SeqState.ACTIVE
        self.step_results = [StepResult.PENDING] * len(_STEPS)
        self._hold_start = None
        self._step_start = None
        self._step_done_at = None
        self._global_start = None
        self._baseline_depth = None
        self._current_depth = None
        self._flip_baseline_z = None
        self._peekaboo_phase = 0
        self._peekaboo_hidden_at = None
        self._was_drawing = False
        self._validator.clear_buffers()
        self._wave.reset()
        self._shape.reset()
