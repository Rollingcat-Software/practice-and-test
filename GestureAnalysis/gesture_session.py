"""
Gesture Session -- Dual-Hand State Machine
===========================================
Implements the UI/UX flow:

    PROMPTING  ->  DETECTING  ->  VALIDATED
        ^                           |
        +---------------------------+  (next challenge)

Now supports targets 0-10 by summing fingers across both hands.
Includes confidence smoothing (delegated to GestureValidator).

The session is framework-agnostic: call ``update()`` every frame with
hand results, and read back ``state`` / ``display_text`` to render however
you like (OpenCV overlay, HTML, Streamlit widget, ...).
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator


class SessionState(Enum):
    PROMPTING = auto()
    DETECTING = auto()
    VALIDATED = auto()


@dataclass
class GestureSession:
    """Drives a single challenge-response gesture loop.

    Parameters
    ----------
    stability_seconds : float
        How long the correct gesture must be held before validation.
    pause_after_success : float
        Cooldown after validation before the next challenge.
    targets : list[int] | None
        Pool of finger-count targets.  Defaults to 1-10 (dual-hand range).
    smoothing_window : int
        Frame buffer size for the validator's majority-vote smoothing.
    """

    stability_seconds: float = 1.0
    pause_after_success: float = 2.0
    targets: list[int] = field(default_factory=lambda: list(range(1, 11)))
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: SessionState = field(init=False, default=SessionState.PROMPTING)
    current_target: int = field(init=False, default=0)
    _match_start: Optional[float] = field(init=False, default=None)
    _validated_at: Optional[float] = field(init=False, default=None)
    _validator: GestureValidator = field(init=False)
    score: int = field(init=False, default=0)
    _last_total: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._pick_new_target()

    # -- public API -------------------------------------------------------

    def update(self, hands: list[HandResult]) -> SessionState:
        """Feed the latest detection results and advance the state machine."""
        now = time.monotonic()

        if self.state == SessionState.VALIDATED:
            if now - self._validated_at >= self.pause_after_success:
                self._pick_new_target()
                self.state = SessionState.DETECTING
            return self.state

        if self.state == SessionState.PROMPTING:
            self.state = SessionState.DETECTING

        # -- DETECTING ----------------------------------------------------
        if not hands:
            self._match_start = None
            self._last_total = None
            return self.state

        total = self._validator.count_fingers_total(hands)
        self._last_total = total

        if total == self.current_target:
            if self._match_start is None:
                self._match_start = now
            elif now - self._match_start >= self.stability_seconds:
                self.state = SessionState.VALIDATED
                self._validated_at = now
                self.score += 1
        else:
            self._match_start = None

        return self.state

    @property
    def hold_progress(self) -> float:
        """0.0 -> 1.0 progress toward the stability threshold."""
        if self._match_start is None:
            return 0.0
        elapsed = time.monotonic() - self._match_start
        return min(elapsed / self.stability_seconds, 1.0)

    @property
    def display_text(self) -> str:
        if self.state == SessionState.VALIDATED:
            return f"Validated!  Score: {self.score}"
        return f"Show {self.current_target} finger(s)"

    @property
    def status_label(self) -> str:
        if self.state == SessionState.VALIDATED:
            return "Status: Validated"
        if self._match_start is not None:
            return "Status: Hold steady..."
        return "Status: Waiting for Gesture"

    def detected_total(self, hands: list[HandResult]) -> Optional[int]:
        """Return the cached total finger count, or None if no hands."""
        return self._last_total

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        """Return ``{"Left": n, "Right": m}`` for the current frame."""
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    # -- internals --------------------------------------------------------

    def _pick_new_target(self):
        self.current_target = random.choice(self.targets)
        self._match_start = None
        self._validated_at = None
        self._last_total = None
        self._validator.clear_buffers()
