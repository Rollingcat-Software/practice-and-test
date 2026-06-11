"""
GameManager
===========
Owns all game-mode sessions and handles mode switching, the global
hand-presence guard, and session audit logging.

Separated from main.py so the render loop contains zero game logic.
"""

import time

from app_config import cfg
from hand_tracker import HandResult
from gesture_session import GestureSession
from math_session import MathSession, MathState
from liveness_session import LivenessChallenge, LivenessState
from shape_tracer import TracerState          # noqa: F401 (re-exported for HUD)
from sequential_session import SequentialSession, SeqState
from finger_touch_session import FingerTouchSession, TouchTestState
from tracing_evaluator import TracingEvaluator
from shape_trace_eval_session import ShapeTraceEvalSession, EvalMode

MODE_NAMES = ["Normal", "Math", "Liveness", "Sequential", "Touch Test", "Shape Eval"]


class GameManager:
    """Centralised mode switching + global hand-presence guard.

    Separates all game logic from the capture/render loop.
    """

    def __init__(self):
        self.modes = MODE_NAMES
        self.current_mode = 0

        self.gesture = GestureSession(
            stability_seconds=1.0, pause_after_success=2.0,
            targets=list(range(1, 11)),
            smoothing_window=cfg.gesture.smoothing_window,
        )
        self.math = MathSession(
            stability_seconds=cfg.math.stability_seconds,
            pause_after_success=cfg.math.pause_after_success,
            game_duration=cfg.math.game_duration,
            smoothing_window=cfg.math.smoothing_window,
        )
        self.liveness = LivenessChallenge(
            time_limit=cfg.liveness.time_limit,
            debounce_seconds=cfg.liveness.debounce_seconds,
            area_change_threshold=cfg.liveness.area_change_threshold,
            pause_after_result=cfg.liveness.pause_after_result,
            smoothing_window=cfg.liveness.smoothing_window,
        )
        self.sequential = SequentialSession(
            hold_seconds=cfg.sequential.hold_seconds,
            pause_after_step=cfg.sequential.pause_after_step,
            depth_threshold=cfg.sequential.depth_threshold,
            smoothing_window=cfg.sequential.smoothing_window,
        )
        self.touch_test = FingerTouchSession(
            verify_frames=cfg.touch_test.verify_frames,
            pause_after_success=cfg.touch_test.pause_after_success,
        )
        _evaluator = TracingEvaluator(log_dir="eval_logs")
        self.shape_eval = ShapeTraceEvalSession(
            evaluator=_evaluator,
            eval_mode=EvalMode.HUMAN_TEST,
            dtw_threshold=cfg.shape_eval.dtw_threshold,
            auto_advance=cfg.shape_eval.auto_advance,
        )

        self.hands_present = False

        # Session audit logger
        from session_logger import SessionLogger
        self._logger = SessionLogger(
            log_dir=cfg.logging.log_dir,
            enabled=cfg.logging.enabled,
        )
        self._session_start: float = time.time()
        self._session_logged: bool = False
        self._liveness_spoof_blocks: int = 0
        self._last_liveness_spoof_counted: bool = False

    @property
    def mode_name(self) -> str:
        return self.modes[self.current_mode]

    @property
    def suppress_no_hand_overlay(self) -> bool:
        """True when the active challenge renders a centred shape guide that
        the generic 'No Hand Detected' overlay would otherwise cover.

        During a SHAPE_TRACE liveness challenge (and throughout Shape Eval
        mode) the shape sits in the middle of the frame; the centred no-hand
        text flickers on top of it whenever tracking briefly drops — and the
        Shape Eval attack modes run with no hand at all — so we suppress it.
        """
        if self.current_mode == 5:          # Shape Eval (always centred shape)
            return True
        return self.current_mode == 2 and self.liveness.is_shape_trace_cmd

    # ── Audit log helpers ────────────────────────────────────────────────

    def _reset_log_state(self):
        """Reset per-session audit tracking after mode switch or restart."""
        self._session_start = time.time()
        self._session_logged = False
        self._liveness_spoof_blocks = 0
        self._last_liveness_spoof_counted = False

    def _check_and_log(self, now: float) -> None:
        """Detect terminal states and write one audit record (once per session)."""
        if self._session_logged:
            return

        duration = now - self._session_start

        if self.current_mode == 1:
            if self.math.state == MathState.GAME_OVER:
                self._logger.log(
                    mode="Math",
                    result="GAME_OVER",
                    duration_s=duration,
                    metrics={"score": self.math.score},
                )
                self._session_logged = True

        elif self.current_mode == 2:
            # Track spoof-block events per challenge cycle.
            spoof_now = self.liveness.is_spoof_blocked
            if spoof_now and not self._last_liveness_spoof_counted:
                self._liveness_spoof_blocks += 1
            self._last_liveness_spoof_counted = spoof_now

            if self.liveness.state == LivenessState.VERIFIED_100:
                self._logger.log(
                    mode="Liveness",
                    result="VERIFIED_100",
                    duration_s=duration,
                    metrics={
                        "challenges_completed": cfg.liveness.num_challenges,
                        "score": 100,
                        "spoof_blocks": self._liveness_spoof_blocks,
                    },
                )
                self._session_logged = True

        elif self.current_mode == 3:
            if self.sequential.state == SeqState.COMPLETE:
                self._logger.log(
                    mode="Sequential",
                    result="COMPLETE",
                    duration_s=duration,
                    metrics={
                        "passed_count": self.sequential.passed_count,
                        "total_steps": self.sequential.total_steps,
                    },
                )
                self._session_logged = True

        elif self.current_mode == 4:
            if self.touch_test.state == TouchTestState.COMPLETE:
                self._logger.log(
                    mode="TouchTest",
                    result="COMPLETE",
                    duration_s=duration,
                    metrics={"passed_count": self.touch_test.passed_count},
                )
                self._session_logged = True

    # ── Mode control ────────────────────────────────────────────────────

    def cycle_mode(self):
        self.current_mode = (self.current_mode + 1) % len(self.modes)
        self._reset_log_state()
        if self.current_mode == 1:
            self.math.reset()
        elif self.current_mode == 2:
            self.liveness.reset()
        elif self.current_mode == 3:
            self.sequential.reset()
        elif self.current_mode == 4:
            self.touch_test.reset()
        elif self.current_mode == 5:
            self.shape_eval.reset()

    def restart(self):
        self._reset_log_state()
        if self.current_mode == 0:
            pass
        elif self.current_mode == 1:
            self.math.reset()
        elif self.current_mode == 2:
            self.liveness.reset()
        elif self.current_mode == 3:
            self.sequential.reset()
        elif self.current_mode == 4:
            self.touch_test.reset()
        elif self.current_mode == 5:
            self.shape_eval.reset()

    # ── Update (called once per frame) ──────────────────────────────────

    def update(self, hands: list[HandResult]):
        """Global hand-presence check, then delegate to active mode.

        If no hands are detected, we set hands_present=False and pass
        an empty list to the active session.  Each session already
        handles empty hands (resets timers, shows "no hand" status).
        This centralised guard ensures consistent behaviour across all
        modes and prevents false timeouts or logic crashes.
        """
        self.hands_present = len(hands) > 0

        if self.current_mode == 0:
            self.gesture.update(hands)
        elif self.current_mode == 1:
            self.math.update(hands)
        elif self.current_mode == 2:
            self.liveness.update(hands)
        elif self.current_mode == 3:
            self.sequential.update(hands)
        elif self.current_mode == 4:
            self.touch_test.update(hands)
        elif self.current_mode == 5:
            self.shape_eval.update(hands)

        self._check_and_log(time.time())
