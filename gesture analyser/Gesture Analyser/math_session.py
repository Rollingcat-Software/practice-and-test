"""
Math Challenge Session
======================
Generates controlled arithmetic questions (addition, subtraction,
multiplication, division) where the answer is always in [0, 10].
The user solves each question by showing the correct number of fingers.

    WAITING  ->  HOLDING  ->  SUCCESS  ->  (next question)
       ^                         |
       +-------------------------+

The 60-second countdown starts when the first question appears.
After time runs out the session enters GAME_OVER and no more
questions are generated.

Question format: ``a OP b = ?``  (user must show the answer in fingers)

Operand constraints
-------------------
  Addition:       a + b = ?   a, b >= 0  and  a + b <= 10
  Subtraction:    a - b = ?   b >= 0     and  a - b >= 0  and  a <= 10
  Multiplication: a × b = ?   a, b >= 0  and  a × b <= 10
  Division:       a ÷ b = ?   b >= 1,    a % b == 0,  a // b <= 10

Framework-agnostic: feed ``update(hands)`` each frame, read back
``display_text`` / ``status_label`` / ``equation_text`` to render.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator


# ── Controlled question generator ───────────────────────────────────

# Pre-built table of every valid (a, b) pair for multiplication so we
# never loop at runtime — just pick randomly from this constant list.
_MUL_PAIRS: list[tuple[int, int]] = [
    (a, b)
    for a in range(0, 11)
    for b in range(0, 11)
    if a * b <= 10
]


def generate_controlled_math_challenge() -> tuple[str, int]:
    """Return a (question_string, answer) pair where answer is in [0, 10].

    One of four operations is chosen at random.  Operands are generated
    so that the result always falls within the range the user can show on
    their fingers (0 – 10).

    Constraints per operation
    -------------------------
    Addition       a + b = ?   a, b >= 0  and  a + b <= 10
    Subtraction    a - b = ?   b >= 0     and  a - b >= 0  and  a <= 10
    Multiplication a × b = ?   a, b >= 0  and  a × b <= 10
    Division       a ÷ b = ?   b >= 1,    a % b == 0,  a // b <= 10
    """
    op = random.choice(["add", "sub", "mul", "div"])

    if op == "add":
        # Pick the answer first, then split it across two non-negative terms.
        answer = random.randint(0, 10)
        a = random.randint(0, answer)
        b = answer - a
        return f"{a} + {b} = ?", answer

    if op == "sub":
        # answer = a - b  =>  a = answer + b  (keep a <= 10)
        answer = random.randint(0, 10)
        b = random.randint(0, 10 - answer)
        a = answer + b
        return f"{a} - {b} = ?", answer

    if op == "mul":
        # Choose from the pre-built table — guarantees product <= 10.
        a, b = random.choice(_MUL_PAIRS)
        return f"{a} x {b} = ?", a * b

    # div: pick answer then a random divisor; dividend = answer * divisor
    answer = random.randint(0, 10)
    b = random.randint(1, 10)           # divisor >= 1, no division by zero
    a = answer * b                      # a % b == 0 and a // b == answer
    return f"{a} / {b} = ?", answer


# ── Session state machine ────────────────────────────────────────────

class MathState(Enum):
    WAITING = auto()    # waiting for correct gesture
    HOLDING = auto()    # correct gesture detected, stability timer running
    SUCCESS = auto()    # answer confirmed, showing success briefly
    GAME_OVER = auto()  # 60-second timer expired


@dataclass
class MathSession:
    """Drives one round of Math Challenge mode.

    Parameters
    ----------
    stability_seconds : float
        How long the correct answer must be held to confirm.
    pause_after_success : float
        Seconds to show "SUCCESS!" before the next question.
    game_duration : float
        Total game length in seconds (default 60).
    smoothing_window : int
        Passed to GestureValidator for temporal smoothing.
    """

    stability_seconds: float = 2.0
    pause_after_success: float = 1.5
    game_duration: float = 60.0
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: MathState = field(init=False, default=MathState.WAITING)
    score: int = field(init=False, default=0)

    _equation: str = field(init=False, default="")
    _answer: int = field(init=False, default=0)

    _game_start: Optional[float] = field(init=False, default=None)
    _match_start: Optional[float] = field(init=False, default=None)
    _success_at: Optional[float] = field(init=False, default=None)
    _last_total: Optional[int] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._generate_question()

    # -- Question generator -----------------------------------------------

    def _generate_question(self) -> None:
        """Delegate to the module-level generator and store results."""
        self._equation, self._answer = generate_controlled_math_challenge()
        self._match_start = None
        self._success_at  = None
        self._last_total  = None
        self._validator.clear_buffers()

    # -- Public API -------------------------------------------------------

    def update(self, hands: list[HandResult]) -> MathState:
        """Feed one frame of hand results and advance the state machine."""
        now = time.monotonic()

        # Start the game clock on the very first update call.
        if self._game_start is None:
            self._game_start = now

        # Check 60-second timeout.
        if self.time_remaining <= 0 and self.state != MathState.GAME_OVER:
            self.state = MathState.GAME_OVER
            return self.state

        if self.state == MathState.GAME_OVER:
            return self.state

        # After a SUCCESS, pause briefly then generate the next question.
        if self.state == MathState.SUCCESS:
            if now - self._success_at >= self.pause_after_success:
                self._generate_question()
                self.state = MathState.WAITING
            return self.state

        # -- WAITING / HOLDING --------------------------------------------
        if not hands:
            self._match_start = None
            self._last_total = None
            self.state = MathState.WAITING
            return self.state

        total = self._validator.count_fingers_total(hands)
        self._last_total = total

        if total == self._answer:
            if self._match_start is None:
                self._match_start = now
                self.state = MathState.HOLDING
            elif now - self._match_start >= self.stability_seconds:
                self.state = MathState.SUCCESS
                self._success_at = now
                self.score += 1
                # Optional beep (works on Windows; silent fail elsewhere).
                try:
                    import winsound
                    winsound.Beep(1000, 200)
                except Exception:
                    pass
            else:
                self.state = MathState.HOLDING
        else:
            self._match_start = None
            self.state = MathState.WAITING

        return self.state

    # -- Display helpers --------------------------------------------------

    @property
    def time_remaining(self) -> float:
        """Seconds left on the game clock (0 if not started or expired)."""
        if self._game_start is None:
            return self.game_duration
        elapsed = time.monotonic() - self._game_start
        return max(0.0, self.game_duration - elapsed)

    @property
    def hold_progress(self) -> float:
        """0.0 -> 1.0 progress toward stability confirmation."""
        if self._match_start is None:
            return 0.0
        elapsed = time.monotonic() - self._match_start
        return min(elapsed / self.stability_seconds, 1.0)

    @property
    def equation_text(self) -> str:
        """The equation string to display (e.g. '5 + ? = 7')."""
        return self._equation

    @property
    def answer(self) -> int:
        return self._answer

    @property
    def display_text(self) -> str:
        if self.state == MathState.GAME_OVER:
            return f"TIME'S UP!  Final Score: {self.score}"
        if self.state == MathState.SUCCESS:
            return f"CORRECT!  Score: {self.score}"
        return f"Solve:  {self._equation}"

    @property
    def status_label(self) -> str:
        if self.state == MathState.GAME_OVER:
            return "Status: Game Over"
        if self.state == MathState.SUCCESS:
            return "Status: Correct!"
        if self.state == MathState.HOLDING:
            return "Status: Hold steady..."
        return "Status: Show your answer"

    def detected_total(self) -> Optional[int]:
        return self._last_total

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    def reset(self) -> None:
        """Reset everything for a new game."""
        self.state = MathState.WAITING
        self.score = 0
        self._game_start = None
        self._validator.clear_buffers()
        self._generate_question()
