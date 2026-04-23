"""
Math Challenge Session
======================
Generates controlled arithmetic questions (addition, subtraction,
multiplication, division) where the answer is always in [0, 10].
The user solves each question by showing the correct number of fingers.

    COUNTDOWN  ->  EVALUATING  ->  RESULT  ->  (next question)

The user is given a fixed countdown per question.  During the countdown
the system tracks finger count but does **not** reveal whether the
answer is correct — eliminating the trial-and-error "Hold" hint that
existed previously.  At T=0 the system snapshots the last 0.5 s of
detected counts (stability buffer) and makes a single pass/fail decision.

The overall game clock still limits the total session length.

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
from collections import deque
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
    COUNTDOWN  = auto()   # timer running, user shows fingers — no feedback
    EVALUATING = auto()   # timer hit zero, analysing stability buffer
    RESULT     = auto()   # showing pass/fail for this question
    GAME_OVER  = auto()   # overall game clock expired

    # Legacy aliases so existing code that checks these still compiles.
    WAITING  = COUNTDOWN
    HOLDING  = COUNTDOWN
    SUCCESS  = RESULT


# How long the stability buffer window is (seconds).
_STABILITY_WINDOW: float = 0.5
# Duration to show "Processing..." before revealing the result.
_EVAL_DURATION: float = 0.6
# Duration to show the pass/fail result before next question.
_RESULT_DISPLAY: float = 1.5


@dataclass
class MathSession:
    """Drives one round of Math Challenge mode.

    Parameters
    ----------
    question_duration : float
        How many seconds the user has per question (countdown length).
    game_duration : float
        Total game length in seconds (default 60).
    smoothing_window : int
        Passed to GestureValidator for temporal smoothing.
    """

    question_duration: float = 10.0
    game_duration: float = 60.0
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: MathState = field(init=False, default=MathState.COUNTDOWN)
    score: int = field(init=False, default=0)

    _equation: str = field(init=False, default="")
    _answer: int = field(init=False, default=0)

    _game_start: Optional[float] = field(init=False, default=None)
    _question_start: Optional[float] = field(init=False, default=None)
    _eval_start: Optional[float] = field(init=False, default=None)
    _result_start: Optional[float] = field(init=False, default=None)
    _last_total: Optional[int] = field(init=False, default=None)
    _last_result_correct: bool = field(init=False, default=False)

    # Stability buffer: stores (timestamp, finger_count) tuples.
    _finger_buffer: deque = field(init=False, default_factory=deque)

    _validator: GestureValidator = field(init=False)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._generate_question()

    # -- Question generator -----------------------------------------------

    def _generate_question(self) -> None:
        """Delegate to the module-level generator and store results."""
        self._equation, self._answer = generate_controlled_math_challenge()
        self._question_start = None
        self._eval_start     = None
        self._result_start   = None
        self._last_total     = None
        self._last_result_correct = False
        self._finger_buffer.clear()
        self._validator.clear_buffers()

    # -- Public API -------------------------------------------------------

    def update(self, hands: list[HandResult]) -> MathState:
        """Feed one frame of hand results and advance the state machine."""
        now = time.monotonic()

        # Start clocks on the very first update call.
        if self._game_start is None:
            self._game_start = now
        if self._question_start is None:
            self._question_start = now

        # Check overall game timeout.
        if self.time_remaining <= 0 and self.state != MathState.GAME_OVER:
            self.state = MathState.GAME_OVER
            return self.state

        if self.state == MathState.GAME_OVER:
            return self.state

        # -- RESULT: show pass/fail then advance to next question ----------
        if self.state == MathState.RESULT:
            if now - self._result_start >= _RESULT_DISPLAY:
                self._generate_question()
                self.state = MathState.COUNTDOWN
            return self.state

        # -- EVALUATING: brief "Processing..." state -----------------------
        if self.state == MathState.EVALUATING:
            if now - self._eval_start >= _EVAL_DURATION:
                self._last_result_correct = self._evaluate_buffer()
                if self._last_result_correct:
                    self.score += 1
                    try:
                        import winsound
                        winsound.Beep(1000, 200)
                    except Exception:
                        pass
                self._result_start = now
                self.state = MathState.RESULT
            return self.state

        # -- COUNTDOWN: timer running, collect finger data silently --------
        q_elapsed = now - self._question_start
        if q_elapsed >= self.question_duration:
            # Timer hit zero — transition to evaluation.
            self._eval_start = now
            self.state = MathState.EVALUATING
            return self.state

        # Detect fingers but give NO correctness feedback.
        if hands:
            total = self._validator.count_fingers_total(hands)
            self._last_total = total
            self._finger_buffer.append((now, total))
        else:
            self._last_total = None
            # Record None so we know hand was absent.
            self._finger_buffer.append((now, None))

        # Prune old entries outside the stability window (keep extra margin).
        cutoff = now - _STABILITY_WINDOW - 1.0
        while self._finger_buffer and self._finger_buffer[0][0] < cutoff:
            self._finger_buffer.popleft()

        return self.state

    # -- Evaluation -------------------------------------------------------

    def _evaluate_buffer(self) -> bool:
        """Analyse the last 0.5 s of finger counts and decide pass/fail."""
        if not self._finger_buffer:
            return False

        now = self._finger_buffer[-1][0]
        cutoff = now - _STABILITY_WINDOW

        recent = [cnt for ts, cnt in self._finger_buffer
                  if ts >= cutoff and cnt is not None]

        if not recent:
            return False

        # Use the most frequent (mode) count in the window.
        from collections import Counter
        counts = Counter(recent)
        mode_count, _ = counts.most_common(1)[0]
        return mode_count == self._answer

    # -- Display helpers --------------------------------------------------

    @property
    def time_remaining(self) -> float:
        """Seconds left on the overall game clock."""
        if self._game_start is None:
            return self.game_duration
        elapsed = time.monotonic() - self._game_start
        return max(0.0, self.game_duration - elapsed)

    @property
    def question_time_remaining(self) -> float:
        """Seconds left on the current question's countdown."""
        if self._question_start is None:
            return self.question_duration
        elapsed = time.monotonic() - self._question_start
        return max(0.0, self.question_duration - elapsed)

    @property
    def question_progress(self) -> float:
        """0.0 → 1.0 progress of the question countdown (1.0 = time's up)."""
        if self._question_start is None:
            return 0.0
        elapsed = time.monotonic() - self._question_start
        return min(elapsed / self.question_duration, 1.0)

    @property
    def hold_progress(self) -> float:
        """Backwards-compat alias — maps to question_progress."""
        return self.question_progress

    @property
    def equation_text(self) -> str:
        """The equation string to display (e.g. '5 + 3 = ?')."""
        return self._equation

    @property
    def answer(self) -> int:
        return self._answer

    @property
    def last_result_correct(self) -> bool:
        return self._last_result_correct

    @property
    def display_text(self) -> str:
        if self.state == MathState.GAME_OVER:
            return f"TIME'S UP!  Final Score: {self.score}"
        if self.state == MathState.RESULT:
            if self._last_result_correct:
                return f"CORRECT!  Score: {self.score}"
            return f"WRONG — answer was {self._answer}"
        if self.state == MathState.EVALUATING:
            return "Processing..."
        return f"Solve:  {self._equation}"

    @property
    def status_label(self) -> str:
        if self.state == MathState.GAME_OVER:
            return "Status: Game Over"
        if self.state == MathState.RESULT:
            return "Status: Correct!" if self._last_result_correct else "Status: Incorrect"
        if self.state == MathState.EVALUATING:
            return "Status: Processing..."
        q_rem = self.question_time_remaining
        return f"Status: Show your answer ({q_rem:.0f}s)"

    def detected_total(self) -> Optional[int]:
        return self._last_total

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    def reset(self) -> None:
        """Reset everything for a new game."""
        self.state = MathState.COUNTDOWN
        self.score = 0
        self._game_start = None
        self._validator.clear_buffers()
        self._generate_question()
