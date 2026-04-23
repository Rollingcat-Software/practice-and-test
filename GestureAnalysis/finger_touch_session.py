"""
Finger Touch Test Session
=========================
Dedicated test mode that presents all five FINGER_TOUCH commands
in a fixed order so every combination can be verified manually.

Command order
-------------
  1. THUMB_TO_INDEX    -- same hand, IDs 4 & 8
  2. THUMB_TO_MIDDLE   -- same hand, IDs 4 & 12
  3. THUMB_TO_RING     -- same hand, IDs 4 & 16
  4. THUMB_TO_PINKY    -- same hand, IDs 4 & 20
  5. DOUBLE_THUMB_TOUCH-- left ID 4 & right ID 4

Each command requires the touch to be held for 10 consecutive frames
(the same threshold used in the Liveness mode).  After a successful
hold the session pauses briefly and then advances to the next command.
When all five are passed the session shows a completion screen.

Press R at any point to restart from the beginning.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from finger_touch_detector import FingerTouchDetector, TouchCommand


# ── Command catalogue (fixed order for exhaustive testing) ───────────

_ALL_TOUCH_COMMANDS: list[tuple[TouchCommand, str]] = [
    (TouchCommand.THUMB_TO_INDEX,     "PINCH: THUMB + INDEX   (IDs 4 & 8)"),
    (TouchCommand.THUMB_TO_MIDDLE,    "PINCH: THUMB + MIDDLE  (IDs 4 & 12)"),
    (TouchCommand.THUMB_TO_RING,      "PINCH: THUMB + RING    (IDs 4 & 16)"),
    (TouchCommand.THUMB_TO_PINKY,     "PINCH: THUMB + PINKY   (IDs 4 & 20)"),
    (TouchCommand.DOUBLE_THUMB_TOUCH, "TOUCH BOTH THUMBS!     (L:4 & R:4)"),
]


class TouchTestState(Enum):
    ACTIVE   = auto()   # waiting for the user to hold the touch
    SUCCESS  = auto()   # touch verified, brief pause before next
    COMPLETE = auto()   # all 5 commands passed


@dataclass
class FingerTouchSession:
    """Sequential test session for all five finger-touch commands.

    Parameters
    ----------
    verify_frames : int
        Consecutive frames required to confirm each touch (default 10).
    pause_after_success : float
        Seconds to display the SUCCESS banner before advancing (default 1.5).
    touch_threshold : float
        Normalised distance threshold passed to FingerTouchDetector.
    z_max_diff : float
        Z-depth tolerance passed to FingerTouchDetector.
    """

    verify_frames:       int   = 10
    pause_after_success: float = 1.5
    touch_threshold:     float = 0.28
    z_max_diff:          float = 0.04

    # -- internal state ---------------------------------------------------
    state:           TouchTestState = field(init=False, default=TouchTestState.ACTIVE)
    current_idx:     int            = field(init=False, default=0)
    passed:          list[bool]     = field(init=False, default_factory=lambda: [False] * 5)
    _success_at:     Optional[float] = field(init=False, default=None)
    _detector:       Optional[FingerTouchDetector] = field(init=False, default=None)

    def __post_init__(self):
        self._start_command(0)

    # -- Internal helpers -------------------------------------------------

    def _start_command(self, idx: int) -> None:
        self.current_idx = idx
        self.state       = TouchTestState.ACTIVE
        self._success_at = None
        touch_cmd, _ = _ALL_TOUCH_COMMANDS[idx]
        self._detector = FingerTouchDetector(
            command         = touch_cmd,
            verify_frames   = self.verify_frames,
            touch_threshold = self.touch_threshold,
            z_max_diff      = self.z_max_diff,
        )

    # -- Public API -------------------------------------------------------

    def update(self, hands: list[HandResult]) -> TouchTestState:
        now = time.monotonic()

        if self.state == TouchTestState.COMPLETE:
            return self.state

        if self.state == TouchTestState.SUCCESS:
            if now - self._success_at >= self.pause_after_success:
                next_idx = self.current_idx + 1
                if next_idx >= len(_ALL_TOUCH_COMMANDS):
                    self.state = TouchTestState.COMPLETE
                else:
                    self._start_command(next_idx)
            return self.state

        # ACTIVE: feed the detector
        if self._detector is not None and self._detector.update(hands):
            self.passed[self.current_idx] = True
            self.state       = TouchTestState.SUCCESS
            self._success_at = now

        return self.state

    def reset(self) -> None:
        self.passed  = [False] * 5
        self._start_command(0)

    # -- Display helpers --------------------------------------------------

    @property
    def command_label(self) -> str:
        _, label = _ALL_TOUCH_COMMANDS[self.current_idx]
        return label

    @property
    def command_name(self) -> str:
        cmd, _ = _ALL_TOUCH_COMMANDS[self.current_idx]
        return cmd.name

    @property
    def progress_text(self) -> str:
        return f"Command {self.current_idx + 1} / {len(_ALL_TOUCH_COMMANDS)}"

    @property
    def frame_count(self) -> int:
        if self._detector is None:
            return 0
        return self._detector.consecutive_frames

    @property
    def hold_progress(self) -> float:
        """0.0-1.0 toward the verify_frames requirement."""
        if self._detector is None:
            return 0.0
        return self._detector.progress

    @property
    def passed_count(self) -> int:
        return sum(self.passed)

    @property
    def all_commands(self) -> list[tuple[TouchCommand, str, bool]]:
        """(command, label, passed) for every entry — used to draw the checklist."""
        return [
            (_ALL_TOUCH_COMMANDS[i][0], _ALL_TOUCH_COMMANDS[i][1], self.passed[i])
            for i in range(len(_ALL_TOUCH_COMMANDS))
        ]
