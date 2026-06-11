"""
Shape Trace Evaluation Session
================================
Dedicated testing mode for the Dynamic Shape Tracing liveness challenge.

Wraps ShapeTracerSession with three evaluation modes:

  HUMAN_TEST     -- real hand tracking, logs every verified/failed attempt
  STATIC_ATTACK  -- animates a near-static simulated path, logs the result
  RANDOM_ATTACK  -- animates a random simulated path, logs the result

Each mode feeds points into the same DTW verification pipeline so metrics
are directly comparable across human and attack populations.

The session exposes separate ``debug_template_path`` and ``debug_user_path``
properties so the HUD renderer can draw them in distinct colours
(green for template, red for user/attack path).
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from shape_tracer import (
    ShapeTracerSession, TracerState,
    generate_random_shape, ShapeTemplate,
    _resample, _centroid_normalise, dtw_normalised_cost,
    DEFAULT_DRAW_TIME, DEFAULT_DTW_THRESH, DEFAULT_RESAMPLE_N,
    DEFAULT_POS_HOLD,
)
from tracing_evaluator import (
    TracingEvaluator, AttemptLog,
    StaticAttackSimulator, RandomAttackSimulator,
)


# ── Evaluation mode ──────────────────────────────────────────────────

class EvalMode(Enum):
    HUMAN_TEST    = auto()   # live hand tracing
    STATIC_ATTACK = auto()   # frozen / printed-image simulation
    RANDOM_ATTACK = auto()   # brute-force random path simulation

    def next(self) -> "EvalMode":
        members = list(EvalMode)
        return members[(members.index(self) + 1) % len(members)]

    @property
    def label(self) -> str:
        return {
            EvalMode.HUMAN_TEST:    "HUMAN TEST",
            EvalMode.STATIC_ATTACK: "STATIC ATTACK",
            EvalMode.RANDOM_ATTACK: "RANDOM ATTACK",
        }[self]

    @property
    def color_key(self) -> str:
        """Return a colour key for the HUD: 'green', 'orange', 'red'."""
        return {
            EvalMode.HUMAN_TEST:    "green",
            EvalMode.STATIC_ATTACK: "orange",
            EvalMode.RANDOM_ATTACK: "red",
        }[self]


# ── Internal eval state ──────────────────────────────────────────────

class _EvalState(Enum):
    IDLE       = auto()   # showing last result, waiting for next round
    RUNNING    = auto()   # human or attack in progress
    RESULT     = auto()   # displaying result briefly before next round


# Points fed per frame in attack animation (≈ 2 pts/frame at 30 fps
# gives a 1-second animation for 60-point paths).
_ATTACK_FEED_RATE = 2
_RESULT_PAUSE     = 2.5   # seconds to show result before resetting


@dataclass
class ShapeTraceEvalSession:
    """Evaluation wrapper for ShapeTracerSession.

    Parameters
    ----------
    evaluator : TracingEvaluator
        Shared logger; call ``evaluator.stats`` for aggregate metrics.
    eval_mode : EvalMode
        Starting evaluation mode (cycle with ``cycle_mode()``).
    dtw_threshold : float
        Acceptance threshold forwarded to the inner ShapeTracerSession.
    auto_advance : bool
        If True the session resets automatically after ``_RESULT_PAUSE``
        seconds, allowing continuous unattended testing.
    """

    evaluator:      TracingEvaluator
    eval_mode:      EvalMode = EvalMode.HUMAN_TEST
    dtw_threshold:  float    = DEFAULT_DTW_THRESH
    auto_advance:   bool     = True

    # -- observable state -------------------------------------------------
    eval_state:   _EvalState = field(init=False, default=_EvalState.IDLE)
    latest_log:   Optional[AttemptLog] = field(init=False, default=None)

    # -- internal ---------------------------------------------------------
    _tracer:        Optional[ShapeTracerSession] = field(init=False, default=None)

    # Attack simulation bookkeeping
    _sim_full_path: list[tuple[float, float]] = field(init=False, default_factory=list)
    _sim_live_path: list[tuple[float, float]] = field(init=False, default_factory=list)
    _sim_feed_idx:  int   = field(init=False, default=0)
    _sim_start:     Optional[float] = field(init=False, default=None)
    _result_at:     Optional[float] = field(init=False, default=None)
    _frame_counter: int   = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._start_round()

    # -- Round lifecycle --------------------------------------------------

    def _start_round(self) -> None:
        """Initialise state for the next round (human or attack)."""
        self._frame_counter = 0
        self._result_at     = None

        if self.eval_mode == EvalMode.HUMAN_TEST:
            self._tracer = ShapeTracerSession(
                time_limit=DEFAULT_DRAW_TIME,
                dtw_threshold=self.dtw_threshold,
            )
            self._sim_full_path = []
            self._sim_live_path = []
            self._sim_feed_idx  = 0
            self._sim_start     = None

        else:  # attack modes
            # Pick a random target shape so attacks are tested on all types
            template = generate_random_shape()
            self._tracer = ShapeTracerSession(
                time_limit=DEFAULT_DRAW_TIME,
                dtw_threshold=self.dtw_threshold,
            )
            # Override the template with our chosen one
            self._tracer.template = template

            # Generate the full attack path up front
            if self.eval_mode == EvalMode.STATIC_ATTACK:
                sim = StaticAttackSimulator()
                self._sim_full_path = sim.generate_path(n_points=60)
            else:
                sim = RandomAttackSimulator()
                self._sim_full_path = sim.generate_path(n_points=70)

            self._sim_live_path = []
            self._sim_feed_idx  = 0
            self._sim_start     = time.time()

        self.eval_state = _EvalState.RUNNING

    def _finish_round(
        self,
        user_path: list[tuple[float, float]],
        time_taken: float,
        attack_type: str,
    ) -> None:
        """Log the result and transition to RESULT state."""
        self.latest_log = self.evaluator.record(
            user_path   = user_path,
            template    = self._tracer.template,
            time_taken  = time_taken,
            attack_type = attack_type,
        )
        self._result_at = time.time()
        self.eval_state = _EvalState.RESULT

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> _EvalState:
        """Feed one video frame.

        In HUMAN_TEST mode, ``hands`` is forwarded to the inner
        ShapeTracerSession.  In attack modes, ``hands`` is ignored and the
        pre-generated path is fed at ``_ATTACK_FEED_RATE`` points / frame.
        """
        now = time.time()
        self._frame_counter += 1

        # ── Auto-advance after result pause ──────────────────────────
        if self.eval_state == _EvalState.RESULT:
            if self.auto_advance and self._result_at and \
                    now - self._result_at >= _RESULT_PAUSE:
                self._start_round()
            return self.eval_state

        if self.eval_state == _EvalState.IDLE:
            return self.eval_state

        # ── HUMAN_TEST ───────────────────────────────────────────────
        if self.eval_mode == EvalMode.HUMAN_TEST:
            tracer_state = self._tracer.update(hands)
            if tracer_state in (TracerState.VERIFIED, TracerState.FAILED):
                self._finish_round(
                    user_path   = self._tracer.traced_path,
                    time_taken  = DEFAULT_DRAW_TIME - self._tracer.time_remaining,
                    attack_type = "HUMAN",
                )
            return self.eval_state

        # ── ATTACK SIMULATION ────────────────────────────────────────
        # Feed _ATTACK_FEED_RATE points per frame to animate the attack
        end_idx = min(self._sim_feed_idx + _ATTACK_FEED_RATE, len(self._sim_full_path))
        self._sim_live_path.extend(self._sim_full_path[self._sim_feed_idx:end_idx])
        self._sim_feed_idx = end_idx

        # All points fed → run verification
        if self._sim_feed_idx >= len(self._sim_full_path):
            time_taken = now - self._sim_start if self._sim_start else DEFAULT_DRAW_TIME
            attack_type = (
                "STATIC_ATTACK" if self.eval_mode == EvalMode.STATIC_ATTACK
                else "RANDOM_ATTACK"
            )
            self._finish_round(
                user_path   = self._sim_live_path,
                time_taken  = time_taken,
                attack_type = attack_type,
            )

        return self.eval_state

    # -- Mode control -----------------------------------------------------

    def cycle_mode(self) -> None:
        """Advance to the next EvalMode and start a fresh round."""
        self.eval_mode = self.eval_mode.next()
        self._start_round()

    def reset(self) -> None:
        """Restart the current mode from scratch."""
        self._start_round()

    # -- Debug paths for visual overlay ----------------------------------

    @property
    def debug_template_path(self) -> list[tuple[float, float]]:
        """Template waypoints for the GREEN guide overlay."""
        if self._tracer is None:
            return []
        return list(self._tracer.template.waypoints)

    @property
    def debug_user_path(self) -> list[tuple[float, float]]:
        """Live traced / simulated path for the RED overlay."""
        if self._tracer is None:
            return []
        if self.eval_mode == EvalMode.HUMAN_TEST:
            return self._tracer.traced_path
        return list(self._sim_live_path)

    @property
    def current_template(self) -> Optional[ShapeTemplate]:
        return self._tracer.template if self._tracer else None

    # -- Display helpers --------------------------------------------------

    @property
    def shape_label(self) -> str:
        t = self.current_template
        return t.label if t else "???"

    @property
    def state_label(self) -> str:
        if self.eval_state == _EvalState.RESULT and self.latest_log:
            return self.latest_log.result
        if self.eval_mode == EvalMode.HUMAN_TEST and self._tracer:
            return self._tracer.state.name
        if self.eval_mode in (EvalMode.STATIC_ATTACK, EvalMode.RANDOM_ATTACK):
            pct = min(100, int(self._sim_feed_idx / max(len(self._sim_full_path), 1) * 100))
            if self.eval_state == _EvalState.RUNNING:
                return f"SIMULATING {pct}%"
        return self.eval_state.name

    @property
    def attack_animation_progress(self) -> float:
        """0.0-1.0 animation progress for attack modes."""
        total = len(self._sim_full_path)
        return min(self._sim_feed_idx / total, 1.0) if total > 0 else 0.0

    @property
    def human_tracer_state(self) -> Optional[TracerState]:
        if self.eval_mode == EvalMode.HUMAN_TEST and self._tracer:
            return self._tracer.state
        return None

    @property
    def human_time_remaining(self) -> float:
        if self._tracer and self.eval_mode == EvalMode.HUMAN_TEST:
            return self._tracer.time_remaining
        return 0.0

    @property
    def human_position_progress(self) -> float:
        """0.0-1.0 POSITIONING countdown progress for HUMAN_TEST mode."""
        if self._tracer and self.eval_mode == EvalMode.HUMAN_TEST:
            return self._tracer.position_progress
        return 0.0

    @property
    def human_point_count(self) -> int:
        if self._tracer and self.eval_mode == EvalMode.HUMAN_TEST:
            return self._tracer.point_count
        return len(self._sim_live_path)

    @property
    def result_similarity(self) -> float:
        return self.latest_log.similarity_score if self.latest_log else 0.0

    @property
    def result_dtw_cost(self) -> float:
        return self.latest_log.dtw_cost if self.latest_log else 0.0

    @property
    def result_drift(self) -> float:
        return self.latest_log.coordinate_drift if self.latest_log else 0.0

    @property
    def result_time(self) -> float:
        return self.latest_log.time_taken if self.latest_log else 0.0
