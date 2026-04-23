"""
Desktop Demo -- v5 Hardened with GameManager
=============================================
Run with:  python main.py

Six modes, cycle with **m** key:
  1. [Normal]     Show N fingers to match the prompt (1-10).
  2. [Math]       Solve arithmetic with fingers (60s timer).
  3. [Liveness]   Fast-response single-command challenge (4s window).
  4. [Sequential] Complete a series of timed gesture steps.
  5. [Touch Test] Verify all five finger-touch combinations.
  6. [Shape Eval] Dynamic shape-tracing liveness evaluation.

Press **r** to restart the current game.
Press **q** or **Esc** to quit.

Architecture
------------
HandTracker   -- detection only (hardened: 0.8 confidence, wrist re-labeling)
GameManager   -- owns all game sessions, handles mode switching and the
                 global "hand presence" guard that protects every mode
                 from running logic on empty frames.
main loop     -- capture + render only; no game logic here.
"""

import math
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, HandResult, draw_landmarks
from gesture_session import GestureSession, SessionState
from math_session import MathSession, MathState
from liveness_session import LivenessChallenge, LivenessState, CmdType
from shape_tracer import TracerState
from sequential_session import SequentialSession, SeqState, StepResult
from finger_touch_session import FingerTouchSession, TouchTestState
from tracing_evaluator import TracingEvaluator
from shape_trace_eval_session import ShapeTraceEvalSession, EvalMode, _EvalState
from active_liveness_manager import (
    ActiveLivenessSession, ALMState, ChallengeType, NUM_CHALLENGES,
    TRANSITION_DURATION as _ALM_TRANSITION_DURATION,
    TRACE_RETRY_DISPLAY as _ALM_RETRY_DURATION,
    TRACE_MAX_ATTEMPTS,
    _MathAdapter, _SnapshotLivenessAdapter,
    _TraceWithRetryAdapter,
)


# -- colour palette -------------------------------------------------------
WHITE = (255, 255, 255)
GREEN = (0, 220, 100)
YELLOW = (0, 220, 255)
RED = (60, 60, 220)
CYAN = (220, 220, 0)
MAGENTA = (200, 50, 255)
ORANGE = (0, 160, 255)
DARK_BG = (40, 40, 40)

MODE_NAMES = ["Normal", "Math", "Liveness", "Sequential", "Touch Test", "Shape Eval", "Active Liveness"]


# ── GameManager ─────────────────────────────────────────────────────

class GameManager:
    """Centralised mode switching + global hand-presence guard.

    Separates all game logic from the capture/render loop.
    """

    def __init__(self):
        self.modes = MODE_NAMES
        self.current_mode = 0

        self.gesture = GestureSession(
            stability_seconds=1.0, pause_after_success=2.0,
            targets=list(range(1, 11)), smoothing_window=7,
        )
        self.math = MathSession(
            question_duration=10.0,
            game_duration=60.0, smoothing_window=7,
        )
        self.liveness = LivenessChallenge(
            time_limit=4.0, debounce_seconds=0.5,
            area_change_threshold=0.20, pause_after_result=1.5,
            smoothing_window=7,
        )
        self.sequential = SequentialSession(
            hold_seconds=1.0, pause_after_step=1.0,
            depth_threshold=0.20, smoothing_window=7,
        )
        self.touch_test = FingerTouchSession(
            verify_frames=10, pause_after_success=1.5,
        )
        _evaluator = TracingEvaluator(log_dir="eval_logs")
        self.shape_eval = ShapeTraceEvalSession(
            evaluator=_evaluator,
            eval_mode=EvalMode.HUMAN_TEST,
            dtw_threshold=0.25,
            auto_advance=True,
        )
        self.active_liveness = ActiveLivenessSession()

        self.hands_present = False

    @property
    def mode_name(self) -> str:
        return self.modes[self.current_mode]

    def cycle_mode(self):
        self.current_mode = (self.current_mode + 1) % len(self.modes)
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
        elif self.current_mode == 6:
            self.active_liveness.reset()

    def restart(self):
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
        elif self.current_mode == 6:
            self.active_liveness.reset()

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
        elif self.current_mode == 6:
            self.active_liveness.update(hands)


# ── Drawing utilities ───────────────────────────────────────────────

def put_text_with_bg(frame, text, org, font_scale=0.8, color=WHITE, bg=DARK_BG, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def put_text_centered(frame, text, y, font_scale=1.2, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), DARK_BG, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_progress_bar(frame, progress: float, y: int = 70, color_full=GREEN, color_fill=YELLOW):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.4)
    x_start = (w - bar_w) // 2
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 18), DARK_BG, -1)
    fill_w = int(bar_w * progress)
    color = color_full if progress >= 1.0 else color_fill
    cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + 18), color, -1)
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 18), WHITE, 1)


def draw_countdown_ring(frame, remaining, total, cx, cy, radius=60,
                        verified=False):
    """Draw a countdown ring with arc, or a green checkmark when verified."""
    cv2.circle(frame, (cx, cy), radius, DARK_BG, -1)
    cv2.circle(frame, (cx, cy), radius, WHITE, 2)

    if verified:
        # Green filled ring + checkmark glyph.
        cv2.ellipse(frame, (cx, cy), (radius - 4, radius - 4),
                    -90, 0, 360, GREEN, 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        check = "\u2713"  # ✓
        (tw, th), _ = cv2.getTextSize(check, font, 1.2, 3)
        cv2.putText(frame, check, (cx - tw // 2, cy + th // 2),
                    font, 1.2, GREEN, 3)
        return

    progress = max(0.0, remaining / total) if total > 0 else 0.0
    angle = int(360 * progress)
    color = GREEN if remaining > 2 else YELLOW if remaining > 1 else RED
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius - 4, radius - 4),
                    -90, 0, angle, color, 6)
    text = f"{remaining:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                font, 1.0, color, 2)


# ── Per-mode HUD renderers ──────────────────────────────────────────

def draw_no_hand_overlay(frame):
    """Shown when no hands are detected -- replaces mode-specific status."""
    h, w = frame.shape[:2]
    put_text_centered(frame, "No Hand Detected", h // 2, font_scale=1.0, color=RED, thickness=2)


def draw_normal_hud(frame, gm: GameManager, hands):
    ses = gm.gesture
    state = ses.state
    prompt_color = GREEN if state == SessionState.VALIDATED else WHITE
    put_text_with_bg(frame, ses.display_text, (20, 40), font_scale=1.0, color=prompt_color, thickness=2)
    draw_progress_bar(frame, ses.hold_progress)

    status_color = GREEN if state == SessionState.VALIDATED else YELLOW if ses.hold_progress > 0 else RED
    put_text_with_bg(frame, ses.status_label, (20, 110), font_scale=0.7, color=status_color)

    per_hand = ses.per_hand_counts(hands)
    total = ses.detected_total(hands)
    y_off = 145
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (20, y_off), font_scale=0.6)
            y_off += 32
    if total is not None:
        c = GREEN if total == ses.current_target else WHITE
        put_text_with_bg(frame, f"Total: {total}", (20, y_off), font_scale=0.7, color=c, thickness=2)


def draw_math_hud(frame, gm: GameManager, hands):
    ses = gm.math
    h, w = frame.shape[:2]
    ms = ses.state

    eq_color = (GREEN if ms == MathState.RESULT and ses.last_result_correct
                else RED if ms == MathState.GAME_OVER or (ms == MathState.RESULT and not ses.last_result_correct)
                else CYAN)
    put_text_centered(frame, ses.equation_text, h // 2 - 30, font_scale=1.5, color=eq_color, thickness=3)

    if ms == MathState.GAME_OVER:
        put_text_centered(frame, ses.display_text, h // 2 + 30, font_scale=1.0, color=RED, thickness=2)
        put_text_centered(frame, "Press R to restart", h // 2 + 70, font_scale=0.7, color=YELLOW)
    elif ms == MathState.RESULT:
        if ses.last_result_correct:
            put_text_centered(frame, "CORRECT!", h // 2 + 30, font_scale=1.0, color=GREEN, thickness=2)
        else:
            put_text_centered(frame, f"WRONG — answer was {ses.answer}", h // 2 + 30,
                              font_scale=0.8, color=RED, thickness=2)
    elif ms == MathState.EVALUATING:
        put_text_centered(frame, "Processing...", h // 2 + 30, font_scale=1.0, color=YELLOW, thickness=2)
    else:
        # COUNTDOWN — no correctness feedback, just show question timer.
        q_rem = ses.question_time_remaining
        put_text_with_bg(frame, f"Show your answer ({q_rem:.0f}s)", (20, 110),
                         font_scale=0.7, color=WHITE)
        draw_progress_bar(frame, 1.0 - ses.question_progress)

    remaining = ses.time_remaining
    timer_color = RED if remaining < 10 else YELLOW if remaining < 30 else WHITE
    put_text_with_bg(frame, f"Time: {remaining:.0f}s", (w - 180, 40), font_scale=0.8, color=timer_color, thickness=2)
    put_text_with_bg(frame, f"Score: {ses.score}", (20, 40), font_scale=0.8, color=GREEN, thickness=2)

    if ms == MathState.COUNTDOWN:
        per_hand = ses.per_hand_counts(hands)
        total = ses.detected_total()
        y_off = h - 80
        # Show finger count without correctness color hint.
        if total is not None:
            put_text_with_bg(frame, f"Fingers: {total}", (20, y_off - 28),
                             font_scale=0.7, color=WHITE, thickness=2)
        for label in ("Left", "Right"):
            if label in per_hand:
                put_text_with_bg(frame, f"{label}: {per_hand[label]}", (20, y_off), font_scale=0.6)
                y_off += 28
    if total is not None:
        c = GREEN if total == ses.answer else WHITE
        put_text_with_bg(frame, f"Total: {total}", (20, y_off), font_scale=0.7, color=c, thickness=2)


def _draw_air_canvas(frame, path, w, h):
    """Draw the air-drawing trajectory as a persistent line on the frame."""
    if len(path) < 2:
        return
    for i in range(1, len(path)):
        x1, y1 = int(path[i-1][0] * w), int(path[i-1][1] * h)
        x2, y2 = int(path[i][0] * w), int(path[i][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), MAGENTA, 3)
    # Draw a dot at the current tip position.
    lx, ly = int(path[-1][0] * w), int(path[-1][1] * h)
    cv2.circle(frame, (lx, ly), 7, WHITE, -1)


def _draw_verification_bar(frame, pct, y=55):
    """Draw a wide verification score bar (0-100%)."""
    h, w = frame.shape[:2]
    bar_w = int(w * 0.5)
    x_start = (w - bar_w) // 2
    # Background
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 22), DARK_BG, -1)
    # Fill
    fill_w = int(bar_w * pct / 100.0)
    bar_color = GREEN if pct >= 100 else CYAN if pct >= 60 else YELLOW if pct >= 20 else RED
    cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + 22), bar_color, -1)
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 22), WHITE, 1)
    # Label
    label = f"Verification: {pct:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    cv2.putText(frame, label, (x_start + (bar_w - tw) // 2, y + 17), font, 0.55, WHITE, 1)


def _draw_direction_arrows(frame, pts: list, color) -> None:
    """Draw small filled triangle arrows along a polyline every ~65 screen pixels.

    Arrows point in the direction of motion, giving users an immediate cue
    about which way to trace the shape.  This is the primary 'Bonus Feature'.
    """
    if len(pts) < 3:
        return
    # Arc-length accumulator in pixel space.
    arc = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        arc.append(arc[-1] + math.sqrt(dx * dx + dy * dy))
    total = arc[-1]
    if total < 80:
        return

    spacing   = 65.0          # pixels between consecutive arrows
    first_pos = spacing * 1.5  # first arrow well away from the start marker
    target    = first_pos
    j         = 0
    while target < total - spacing * 0.8:
        while j < len(arc) - 2 and arc[j + 1] < target:
            j += 1
        seg  = arc[j + 1] - arc[j]
        frac = (target - arc[j]) / seg if seg > 1e-6 else 0.0
        ax   = pts[j][0] + frac * (pts[j + 1][0] - pts[j][0])
        ay   = pts[j][1] + frac * (pts[j + 1][1] - pts[j][1])
        # Direction: look a short distance ahead in pixel space.
        ahead_idx = min(j + max(1, len(pts) // 20), len(pts) - 1)
        dx = pts[ahead_idx][0] - pts[j][0]
        dy = pts[ahead_idx][1] - pts[j][1]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1e-6:
            dx /= mag; dy /= mag
            px, py_v = -dy, dx          # perpendicular
            tip   = (int(ax + dx * 9),                    int(ay + dy * 9))
            base1 = (int(ax - dx * 5 + px   * 5),         int(ay - dy * 5 + py_v * 5))
            base2 = (int(ax - dx * 5 - px   * 5),         int(ay - dy * 5 - py_v * 5))
            cv2.fillPoly(frame,
                         [np.array([tip, base1, base2], dtype=np.int32)],
                         color)
        target += spacing


def _draw_ghost_trace(frame, waypoints: list, w: int, h: int) -> None:
    """Animate a glowing ghost dot cycling along the template path.

    Cycles once every 3 seconds, showing users the expected drawing
    direction and pace before they begin.  Draws a short comet-tail
    of fading circles behind the head.
    """
    if len(waypoints) < 2:
        return

    PERIOD     = 3.0
    TRAIL_FRAC = 0.07   # fraction of total arc shown as trail
    TRAIL_N    = 6      # number of trail dots

    # Arc-length parameterisation.
    dists = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    total = dists[-1]
    if total < 1e-9:
        return

    def _interp(frac: float) -> tuple[int, int]:
        """Return pixel position for a given arc-fraction (0-1, wrapping)."""
        target = (frac % 1.0) * total
        k = 0
        while k < len(dists) - 2 and dists[k + 1] < target:
            k += 1
        seg = dists[k + 1] - dists[k]
        s   = (target - dists[k]) / seg if seg > 1e-9 else 0.0
        x   = waypoints[k][0] + s * (waypoints[k + 1][0] - waypoints[k][0])
        y   = waypoints[k][1] + s * (waypoints[k + 1][1] - waypoints[k][1])
        return (int(x * w), int(y * h))

    now_frac = (time.monotonic() % PERIOD) / PERIOD

    # Draw comet tail first (drawn before head so head sits on top).
    for i in range(TRAIL_N, 0, -1):
        f       = now_frac - (i / TRAIL_N) * TRAIL_FRAC
        pos     = _interp(f)
        alpha   = i / TRAIL_N           # 0 = oldest (faintest), 1 = newest
        gray    = int(60 + 100 * alpha)
        radius  = int(3 + 3 * alpha)
        cv2.circle(frame, pos, radius, (gray, gray, gray + 40), -1, cv2.LINE_AA)

    # Bright ghost head.
    head = _interp(now_frac)
    cv2.circle(frame, head, 9,  (220, 240, 255), -1, cv2.LINE_AA)  # pale blue-white fill
    cv2.circle(frame, head, 9,  (255, 255, 255),  2, cv2.LINE_AA)  # white ring
    cv2.circle(frame, head, 13, (180, 200, 220),  1, cv2.LINE_AA)  # faint outer glow


def _draw_shape_template(frame, template, w: int, h: int,
                         tracer_state=None,
                         position_progress: float = 0.0) -> None:
    """Draw the target shape guide with state-adaptive visual cues.

    Layers (back → front):
      1. Ghost trace animation     (INSTRUCTING / IDLE only)
      2. Semi-transparent template overlay
      3. Solid template lines
      4. Directional arrows        (always, shows drawing direction)
      5. Start / End point markers (state-specific style)
    """
    from shape_tracer import TracerState as TS

    waypoints = list(template.waypoints)
    if len(waypoints) < 2:
        return

    pts = [(int(x * w), int(y * h)) for x, y in waypoints]

    # ── 1. Ghost trace (INSTRUCTING / IDLE) ──────────────────────────
    if tracer_state in (TS.INSTRUCTING, TS.IDLE):
        _draw_ghost_trace(frame, waypoints, w, h)

    # ── 2. Semi-transparent template overlay ─────────────────────────
    overlay = frame.copy()
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], (200, 200, 200), 3, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── 3. Solid template lines ───────────────────────────────────────
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (200, 200, 200), 2, lineType=cv2.LINE_AA)

    # ── 4. Directional arrows ─────────────────────────────────────────
    # Arrow colour: dim white in guide states, bright white during TRACING.
    arrow_col = (160, 160, 160) if tracer_state != TS.TRACING else (220, 220, 220)
    _draw_direction_arrows(frame, pts, arrow_col)

    # ── 5. Start / End point markers ─────────────────────────────────
    sx = int(template.start_point[0] * w)
    sy = int(template.start_point[1] * h)
    ex = int(template.end_point[0]   * w)
    ey = int(template.end_point[1]   * h)

    if tracer_state in (TS.INSTRUCTING, TS.IDLE):
        # Pulsing ring — faster during IDLE to draw attention.
        speed  = 3.0 if tracer_state == TS.INSTRUCTING else 5.0
        pulse  = 0.5 + 0.5 * math.sin(time.monotonic() * speed)
        r_out  = int(12 + 8 * pulse)
        cv2.circle(frame, (sx, sy), r_out, GREEN, 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),      8, GREEN, -1)
        cv2.putText(frame, "START", (sx + 14, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
        # Also show end point on open shapes so users can plan the route.
        if (ex, ey) != (sx, sy):
            cv2.circle(frame, (ex, ey), 8, RED, -1)
            cv2.putText(frame, "END", (ex + 12, ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)

    elif tracer_state == TS.POSITIONING:
        # Arc-fill ring showing hold progress.
        cv2.circle(frame, (sx, sy), 8, GREEN, -1)
        angle = int(position_progress * 360)
        cv2.ellipse(frame, (sx, sy), (20, 20), -90, 0, angle,
                    GREEN, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 20, (100, 220, 100), 1, lineType=cv2.LINE_AA)

    elif tracer_state == TS.TRACING:
        # Solid green start dot (no label — keeps canvas clean).
        cv2.circle(frame, (sx, sy), 9, GREEN, -1)
        # Orange end marker for open shapes only.
        if (ex, ey) != (sx, sy):
            cv2.circle(frame, (ex, ey), 9, ORANGE, -1)
            cv2.putText(frame, "END", (ex + 12, ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)

    else:
        # COMPLETED / result states — plain dot.
        cv2.circle(frame, (sx, sy), 9, GREEN, -1)


# ── Shape status text (fixed top-left) ───────────────────────────────────

def _draw_shape_info_panel(frame, st, label: str, w: int, h: int) -> None:
    """Render shape-tracing status as plain text lines at a fixed top-left
    position.  No overlays, no dynamic geometry — just put_text_with_bg
    calls starting at (x=20, y=130), one line per state.
    """
    from shape_tracer import TracerState as TS

    TX     = 20    # fixed left edge
    TY     = 130   # first line y — below the score row at ~y=85
    LINE_H = 24    # gap between lines

    if st.state == TS.INSTRUCTING:
        rem = st.instruct_remaining
        put_text_with_bg(frame, f"{label}  —  Trace the shape",
                         (TX, TY), font_scale=0.55, color=CYAN)
        put_text_with_bg(frame, "Hover index finger over the green START dot.",
                         (TX, TY + LINE_H), font_scale=0.55, color=WHITE)
        put_text_with_bg(frame, f"Auto-starts in {rem:.1f} s",
                         (TX, TY + LINE_H * 2), font_scale=0.55, color=YELLOW)

    elif st.state == TS.IDLE:
        put_text_with_bg(frame, f"{label}  —  Trace the shape",
                         (TX, TY), font_scale=0.55, color=CYAN)
        put_text_with_bg(frame, "Move index finger to the green START dot.",
                         (TX, TY + LINE_H), font_scale=0.55, color=WHITE)

    elif st.state == TS.POSITIONING:
        pct = int(st.position_progress * 100)
        put_text_with_bg(frame, f"{label}  —  Hold steady... {pct}%",
                         (TX, TY), font_scale=0.55, color=GREEN)
        put_text_with_bg(frame, "Keep finger on START dot to begin.",
                         (TX, TY + LINE_H), font_scale=0.55, color=WHITE)

    elif st.state == TS.TRACING:
        rem   = st.time_remaining
        t_col = RED if rem < 2 else YELLOW if rem < 4 else GREEN
        put_text_with_bg(frame, f"{label}  —  RECORDING",
                         (TX, TY), font_scale=0.55, color=RED)
        put_text_with_bg(frame, f"{st.point_count} pts  |  {rem:.1f} s  —  reach END or close fist",
                         (TX, TY + LINE_H), font_scale=0.55, color=t_col)

    elif st.state == TS.COMPLETED:
        put_text_with_bg(frame, f"{label}  —  Analysing...",
                         (TX, TY), font_scale=0.55, color=YELLOW)


def _draw_traced_path(frame, traced, w: int, h: int) -> None:
    """Draw the user's recorded trace as a colour-shifting polyline.

    Colour transitions cyan → magenta along the path so direction is
    immediately visible.  The current fingertip position is marked with a
    bright white dot.
    """
    if len(traced) < 2:
        return
    pts = [(int(x * w), int(y * h)) for x, y in traced]
    total = len(pts)
    for i in range(1, total):
        t = i / total
        r = int(200 * t)
        g = int(220 * (1 - t))
        b = 255
        cv2.line(frame, pts[i - 1], pts[i], (b, g, r), 3, lineType=cv2.LINE_AA)
    # Live fingertip dot
    cv2.circle(frame, pts[-1], 8, WHITE, -1)
    cv2.circle(frame, pts[-1], 8, CYAN,  2)


def draw_liveness_hud(frame, gm: GameManager, hands):
    lv = gm.liveness
    h, w = frame.shape[:2]
    ls = lv.state

    if lv.is_flash_red:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Shape tracing canvas (template guide + live trace, always behind text).
    if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
        st = lv.shape_tracer
        _draw_shape_template(frame, st.template, w, h,
                             tracer_state=st.state,
                             position_progress=st.position_progress)
        if st.state in (TracerState.TRACING, TracerState.VERIFIED, TracerState.FAILED):
            _draw_traced_path(frame, st.traced_path, w, h)

    # -- Verification score bar (top) -------------------------------------
    _draw_verification_bar(frame, lv.verification_pct, y=10)

    # Challenge progress + score
    put_text_with_bg(frame, lv.challenge_progress_text, (20, 55), font_scale=0.6, color=CYAN)
    put_text_with_bg(frame, f"Score: {lv.score}", (20, 85), font_scale=0.7, color=GREEN, thickness=2)

    # Countdown ring
    # For shape-trace: hide the liveness ring during pre-trace states (no timer
    # running yet); show the tracer's own draw timer only during TRACING.
    _hide_ring = False
    if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
        _st = lv.shape_tracer
        if _st.state in (TracerState.INSTRUCTING, TracerState.IDLE,
                         TracerState.POSITIONING):
            _hide_ring = True
        elif _st.state == TracerState.TRACING:
            from shape_tracer import DEFAULT_DRAW_TIME as _DDT
            draw_countdown_ring(frame, _st.time_remaining, _DDT,
                                w - 90, 90, radius=55)
            _hide_ring = True
    if not _hide_ring and ls not in (LivenessState.VERIFIED_100,):
        draw_countdown_ring(frame, lv.time_remaining, lv._effective_time_limit,
                            w - 90, 90, radius=55)

    spoof = lv.spoof_result

    # -- 100% verified state ----------------------------------------------
    if ls == LivenessState.VERIFIED_100:
        put_text_centered(frame, "ACCESS GRANTED", h // 2 - 30, font_scale=1.4, color=GREEN, thickness=3)
        put_text_centered(frame, "100% Verification Complete", h // 2 + 20, font_scale=0.8, color=WHITE)
        put_text_centered(frame, "Liveness confirmed", h // 2 + 60, font_scale=0.6, color=GREEN)
        put_text_centered(frame, "Press R to restart", h // 2 + 100, font_scale=0.7, color=YELLOW)
        return

    if ls == LivenessState.SUCCESS:
        put_text_centered(frame, "VERIFIED!", h // 2 - 20,
                          font_scale=1.4, color=GREEN, thickness=3)
        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            pct     = lv.shape_tracer.similarity_pct
            s_color = GREEN if pct >= 70 else YELLOW if pct >= 40 else RED
            # Result detail → top-left, not centre.
            put_text_with_bg(frame,
                             f"Similarity: {pct:.0f}%  DTW: {lv.shape_tracer.dtw_cost:.3f}",
                             (12, 108), font_scale=0.55, color=s_color)
    elif ls == LivenessState.FAILED:
        put_text_centered(frame, "FAILED!", h // 2 - 20,
                          font_scale=1.4, color=RED, thickness=3)
        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            pct = lv.shape_tracer.similarity_pct
            put_text_with_bg(frame,
                             f"Similarity: {pct:.0f}%  (need DTW <= 0.25)",
                             (12, 108), font_scale=0.55, color=YELLOW)
            put_text_with_bg(frame, "Next challenge incoming...",
                             (12, 132), font_scale=0.52, color=YELLOW)
        else:
            put_text_centered(frame, "Next challenge incoming...",
                              h // 2 + 55, font_scale=0.7, color=YELLOW)
    else:
        cmd_color = ORANGE if ls == LivenessState.DEBOUNCE else CYAN
        # Shape-trace: suppress the centre command label (panel handles it).
        if not lv.is_shape_trace_cmd:
            put_text_centered(frame, lv.command_label, h // 2 - 30,
                              font_scale=1.0, color=cmd_color, thickness=2)
        if ls == LivenessState.DEBOUNCE:
            draw_progress_bar(frame, lv.debounce_progress,
                              y=h // 2 + 10, color_full=GREEN, color_fill=ORANGE)
            put_text_centered(frame, "Confirming...", h // 2 + 50,
                              font_scale=0.7, color=ORANGE)

        # Wave progress counter.
        if lv.is_wave_cmd:
            rev = lv.wave_reversals
            rev_color = GREEN if rev >= 2 else YELLOW if rev >= 1 else WHITE
            put_text_with_bg(frame, f"Waves: {rev}/2", (20, h - 125),
                             font_scale=0.65, color=rev_color)

        # Finger touch hold counter + progress bar.
        if lv.is_touch_cmd:
            frames = lv.touch_frame_count
            touch_color = GREEN if frames >= 8 else YELLOW if frames >= 4 else WHITE
            put_text_with_bg(frame, f"Hold: {frames}/10 frames", (20, h - 125),
                             font_scale=0.65, color=touch_color)
            draw_progress_bar(frame, lv.touch_frame_progress, y=h - 108,
                              color_full=GREEN, color_fill=YELLOW)

        # Shape tracing: ALL text goes to the top-left panel — centre stays clear.
        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            _draw_shape_info_panel(frame, lv.shape_tracer, lv.shape_trace_label,
                                   w, h)

    status_color = GREEN if ls == LivenessState.SUCCESS else RED if ls == LivenessState.FAILED else YELLOW
    put_text_with_bg(frame, lv.status_label, (20, h - 90), font_scale=0.7, color=status_color)

    # Depth meter for spatial commands.
    if lv.current_cmd and lv.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
        pct = lv.area_change_pct
        if pct is not None:
            meter_x, meter_top, meter_bot = 50, h // 2 - 80, h // 2 + 80
            meter_h = meter_bot - meter_top
            meter_mid = (meter_top + meter_bot) // 2
            cv2.rectangle(frame, (meter_x - 15, meter_top), (meter_x + 15, meter_bot), DARK_BG, -1)
            cv2.rectangle(frame, (meter_x - 15, meter_top), (meter_x + 15, meter_bot), WHITE, 1)
            cv2.line(frame, (meter_x - 20, meter_mid), (meter_x + 20, meter_mid), WHITE, 2)
            clamped = max(-50, min(50, pct))
            fill_px = int((clamped / 50.0) * (meter_h // 2))
            fill_col = GREEN if abs(pct) >= lv.area_change_threshold * 100 else YELLOW
            if fill_px > 0:
                cv2.rectangle(frame, (meter_x - 12, meter_mid - fill_px), (meter_x + 12, meter_mid), fill_col, -1)
            elif fill_px < 0:
                cv2.rectangle(frame, (meter_x - 12, meter_mid), (meter_x + 12, meter_mid - fill_px), fill_col, -1)
            put_text_with_bg(frame, f"{pct:+.0f}%", (meter_x - 25, meter_bot + 25), font_scale=0.55, color=fill_col)

    per_hand = lv.per_hand_counts(hands)
    y_off = h - 55
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (w - 180, y_off), font_scale=0.6)
            y_off += 28


def draw_sequential_hud(frame, gm: GameManager, hands):
    from sequential_session import StepResult
    seq = gm.sequential
    h, w = frame.shape[:2]
    ss = seq.state

    # Step boxes at the top (two rows if > 10 steps)
    total = seq.total_steps
    per_row = min(total, 10)
    box_w, box_h, gap = 55, 24, 4
    rows = (total + per_row - 1) // per_row

    for row in range(rows):
        start_i = row * per_row
        end_i = min(start_i + per_row, total)
        count = end_i - start_i
        total_w = count * (box_w + gap) - gap
        x_start = (w - total_w) // 2
        y_box = 8 + row * (box_h + 4)

        for i in range(start_i, end_i):
            x = x_start + (i - start_i) * (box_w + gap)
            result = seq.step_results[i]
            if result == StepResult.PASSED:
                bg_col = GREEN
            elif result == StepResult.TIMED_OUT:
                bg_col = RED
            elif i == seq.current_step_idx:
                bg_col = YELLOW if ss in (SeqState.ACTIVE, SeqState.HOLDING) else GREEN if ss == SeqState.STEP_DONE else RED
            else:
                bg_col = (50, 50, 50)
            cv2.rectangle(frame, (x, y_box), (x + box_w, y_box + box_h), bg_col, -1)
            cv2.rectangle(frame, (x, y_box), (x + box_w, y_box + box_h), WHITE, 1)
            cv2.putText(frame, str(i + 1), (x + box_w // 2 - 5, y_box + box_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    info_y = 8 + rows * (box_h + 4) + 10

    # Progress + global timer
    put_text_with_bg(frame, seq.progress_text, (20, info_y), font_scale=0.6, color=CYAN)
    put_text_with_bg(frame, f"Passed: {seq.passed_count}/{seq.total_steps}", (180, info_y), font_scale=0.6, color=GREEN)
    if seq.elapsed_time > 0:
        put_text_with_bg(frame, f"Total: {seq.elapsed_time:.1f}s", (w - 180, info_y), font_scale=0.6, color=WHITE)

    # Per-step countdown (top-right area)
    if ss not in (SeqState.COMPLETE,):
        remaining = seq.step_time_remaining
        step = seq.current_step
        limit = step.time_limit if step else 0
        draw_countdown_ring(frame, remaining, limit, w - 70, info_y + 60, radius=40)

    # Centre: current step instruction
    if ss == SeqState.COMPLETE:
        put_text_centered(frame, "ALL TASKS COMPLETE!", h // 2 - 30, font_scale=1.2, color=GREEN, thickness=3)
        put_text_centered(frame, f"Passed: {seq.passed_count}/{seq.total_steps}  |  Time: {seq.elapsed_time:.1f}s", h // 2 + 15, font_scale=0.75, color=WHITE)
        put_text_centered(frame, "Press R to restart", h // 2 + 55, font_scale=0.7, color=YELLOW)
    elif ss == SeqState.STEP_TIMEOUT:
        put_text_centered(frame, "TIME'S UP!", h // 2 - 20, font_scale=1.2, color=RED, thickness=2)
        put_text_centered(frame, "Skipping to next...", h // 2 + 20, font_scale=0.7, color=YELLOW)
    elif ss == SeqState.STEP_DONE:
        put_text_centered(frame, "PASSED!", h // 2 - 20, font_scale=1.2, color=GREEN, thickness=2)
        put_text_centered(frame, "Next step incoming...", h // 2 + 20, font_scale=0.7, color=WHITE)
    else:
        step_color = ORANGE if ss == SeqState.HOLDING else CYAN
        put_text_centered(frame, seq.display_text, h // 2 - 20, font_scale=0.9, color=step_color, thickness=2)

        if ss == SeqState.HOLDING:
            draw_progress_bar(frame, seq.hold_progress, y=h // 2 + 15)

        # Drawing canvas
        step = seq.current_step
        if step and step.step_type in ("draw_circle", "draw_square"):
            _draw_air_canvas(frame, seq.drawing_path, w, h)
            pts = len(seq.drawing_path)
            if pts > 0:
                put_text_with_bg(frame, f"Drawing... ({pts} pts)", (20, h - 125), font_scale=0.55, color=MAGENTA)

    # Status
    status_color = GREEN if ss in (SeqState.STEP_DONE, SeqState.COMPLETE) else RED if ss == SeqState.STEP_TIMEOUT else YELLOW if ss == SeqState.HOLDING else WHITE
    put_text_with_bg(frame, seq.status_label, (20, h - 90), font_scale=0.7, color=status_color)

    # Per-hand counts
    per_hand = seq.per_hand_counts(hands)
    y_off = h - 55
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (w - 180, y_off), font_scale=0.6)
            y_off += 28


def draw_touch_test_hud(frame, gm: GameManager, hands):
    tt = gm.touch_test
    h, w = frame.shape[:2]
    ts = tt.state

    # -- Header -----------------------------------------------------------
    put_text_centered(frame, "FINGER TOUCH TEST", 32, font_scale=0.9, color=CYAN, thickness=2)

    # -- Checklist (left column) -----------------------------------------
    checklist_x = 20
    checklist_y = 60
    for i, (_, label, passed) in enumerate(tt.all_commands):
        is_current = (i == tt.current_idx) and ts != TouchTestState.COMPLETE
        if passed:
            marker = "[OK]"
            color  = GREEN
        elif is_current:
            marker = "[ > ]"
            color  = YELLOW
        else:
            marker  = "[   ]"
            color   = WHITE
        put_text_with_bg(frame, f"{marker} {label}", (checklist_x, checklist_y + i * 34),
                         font_scale=0.55, color=color)

    # -- Centre: current command + hold progress -------------------------
    mid_y = h // 2

    if ts == TouchTestState.COMPLETE:
        put_text_centered(frame, "ALL 5 COMBINATIONS PASSED!", mid_y - 30,
                          font_scale=1.1, color=GREEN, thickness=3)
        put_text_centered(frame, f"Passed: {tt.passed_count}/5", mid_y + 20,
                          font_scale=0.8, color=WHITE)
        put_text_centered(frame, "Press R to restart", mid_y + 60,
                          font_scale=0.7, color=YELLOW)
        return

    if ts == TouchTestState.SUCCESS:
        put_text_centered(frame, "VERIFIED!", mid_y - 20,
                          font_scale=1.3, color=GREEN, thickness=3)
        put_text_centered(frame, "Next command incoming...", mid_y + 25,
                          font_scale=0.7, color=YELLOW)
    else:
        # Active: show current command and hold counter
        put_text_centered(frame, tt.progress_text, mid_y - 65,
                          font_scale=0.7, color=CYAN)
        put_text_centered(frame, tt.command_label, mid_y - 30,
                          font_scale=0.85, color=ORANGE, thickness=2)

        frames    = tt.frame_count
        progress  = tt.hold_progress
        f_color   = GREEN if frames >= 8 else YELLOW if frames >= 4 else WHITE
        put_text_centered(frame, f"Hold: {frames} / {tt.verify_frames} frames",
                          mid_y + 20, font_scale=0.75, color=f_color, thickness=2)
        draw_progress_bar(frame, progress, y=mid_y + 38, color_full=GREEN, color_fill=YELLOW)

        # Hint for double-thumb command
        if "BOTH THUMBS" in tt.command_label:
            put_text_centered(frame, "(show both hands, bring thumbs together)",
                              mid_y + 75, font_scale=0.55, color=WHITE)
        else:
            put_text_centered(frame, "(use any hand)", mid_y + 75,
                              font_scale=0.55, color=WHITE)

    # -- Summary bar (bottom) --------------------------------------------
    put_text_with_bg(frame, f"Passed: {tt.passed_count}/5", (20, h - 90),
                     font_scale=0.7, color=GREEN, thickness=2)
    put_text_with_bg(frame, "R = restart", (w - 170, h - 90),
                     font_scale=0.6, color=YELLOW)


def draw_shape_eval_hud(frame, gm: GameManager, hands):
    """Visual Debugger HUD for the Shape Tracing Evaluation mode.

    Layout
    ------
    Top bar     : eval mode badge + session stats (attempts, FAR, FRR)
    Centre      : GREEN target shape + RED user/simulated path + state text
    Bottom-left : latest attempt metrics (DTW, similarity, drift, time)
    Bottom-right: running averages
    Key hints   : E = cycle eval mode, R = restart round
    """
    ev = gm.shape_eval
    h, w = frame.shape[:2]
    stats = ev.evaluator.stats

    # ── Mode colour ───────────────────────────────────────────────────
    MODE_COLORS = {
        EvalMode.HUMAN_TEST:    GREEN,
        EvalMode.STATIC_ATTACK: ORANGE,
        EvalMode.RANDOM_ATTACK: RED,
    }
    mode_col = MODE_COLORS.get(ev.eval_mode, WHITE)

    # ── Visual Debugger: draw guide shapes ────────────────────────────
    # GREEN  = target shape template (what the user should trace)
    # RED    = user's real-time path or simulated attack path

    tpl = ev.debug_template_path
    usr = ev.debug_user_path

    # Draw template in GREEN with ghost trace + directional arrows
    _eval_tpl = ev.current_template
    _eval_ts  = ev.human_tracer_state  # None in attack modes
    if _eval_tpl is not None and len(tpl) >= 2:
        # Ghost trace during INSTRUCTING / IDLE.
        if _eval_ts in (None,) or True:   # always show ghost in eval mode
            from shape_tracer import TracerState as _ETS
            if _eval_ts in (_ETS.INSTRUCTING, _ETS.IDLE, None):
                _draw_ghost_trace(frame, list(_eval_tpl.waypoints), w, h)

        # Green semi-transparent overlay.
        pts = [(int(x * w), int(y * h)) for x, y in tpl]
        overlay = frame.copy()
        for i in range(1, len(pts)):
            cv2.line(overlay, pts[i-1], pts[i], (0, 200, 80), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0, 220, 100), 2, cv2.LINE_AA)

        # Directional arrows in dark green.
        _draw_direction_arrows(frame, pts, (0, 160, 60))

        # START marker using template's explicit start_point.
        _sx = int(_eval_tpl.start_point[0] * w)
        _sy = int(_eval_tpl.start_point[1] * h)
        cv2.circle(frame, (_sx, _sy), 10, (0, 255, 200), -1)
        cv2.putText(frame, "START", (_sx + 12, _sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
        # END marker (only when start != end).
        _ex = int(_eval_tpl.end_point[0] * w)
        _ey = int(_eval_tpl.end_point[1] * h)
        if (_ex, _ey) != (_sx, _sy):
            cv2.circle(frame, (_ex, _ey), 10, ORANGE, -1)
            cv2.putText(frame, "END", (_ex + 12, _ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1, cv2.LINE_AA)

    # Draw user / attack path in RED
    if len(usr) >= 2:
        pts_u = [(int(x * w), int(y * h)) for x, y in usr]
        for i in range(1, len(pts_u)):
            t = i / len(pts_u)
            # Gradient: red at start, orange-red at end
            cv2.line(frame, pts_u[i-1], pts_u[i],
                     (40, int(80*t), 220), 3, cv2.LINE_AA)
        cv2.circle(frame, pts_u[-1], 7, (0, 60, 255), -1)
        cv2.circle(frame, pts_u[-1], 7, WHITE, 1)

    # ── Top bar: eval mode badge + session stats ──────────────────────
    bar_h = 44
    cv2.rectangle(frame, (0, 0), (w, bar_h), DARK_BG, -1)

    # Mode badge
    badge_label = f" {ev.eval_mode.label} "
    cv2.rectangle(frame, (8, 5), (8 + len(badge_label)*10, 38), mode_col, -1)
    cv2.putText(frame, badge_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Stats: Attempts | Humans | Attacks | FAR | FRR
    stat_txt = (f"Att:{stats['total']}  "
                f"H:{stats['human_total']}  "
                f"Atk:{stats['attack_total']}  |  "
                f"FAR:{stats['far']*100:.0f}%  "
                f"FRR:{stats['frr']*100:.0f}%")
    put_text_with_bg(frame, stat_txt, (220, 30), font_scale=0.55, color=WHITE)

    # Shape label + state
    put_text_centered(frame, f"Shape: {ev.shape_label}",
                      h // 2 - 55, font_scale=0.8, color=CYAN, thickness=2)

    # ── Centre state / instruction ────────────────────────────────────
    state_label = ev.state_label
    if ev.eval_state == _EvalState.RESULT and ev.latest_log:
        res_col = GREEN if ev.latest_log.result == "VERIFIED" else RED
        put_text_centered(frame, ev.latest_log.result,
                          h // 2 - 20, font_scale=1.3, color=res_col, thickness=3)
        sim_col = GREEN if ev.result_similarity >= 70 else YELLOW if ev.result_similarity >= 40 else RED
        put_text_centered(frame,
                          f"Similarity: {ev.result_similarity:.1f}%  "
                          f"DTW: {ev.result_dtw_cost:.3f}",
                          h // 2 + 20, font_scale=0.72, color=sim_col)
        put_text_centered(frame,
                          f"Drift: {ev.result_drift:.3f}  "
                          f"Time: {ev.result_time:.1f}s  "
                          f"Points: {ev.latest_log.point_count}",
                          h // 2 + 48, font_scale=0.6, color=WHITE)
        put_text_centered(frame, "Next round starting...",
                          h // 2 + 78, font_scale=0.55, color=YELLOW)

    elif ev.eval_mode == EvalMode.HUMAN_TEST:
        ts = ev.human_tracer_state
        if ts is not None:
            from shape_tracer import TracerState as TS
            if ts == TS.INSTRUCTING:
                rem = ev._tracer.instruct_remaining if ev._tracer else 0.0
                put_text_centered(frame,
                                  f"Read instructions...  {rem:.1f}s",
                                  h // 2 - 20, font_scale=0.65, color=YELLOW)
            elif ts == TS.IDLE:
                put_text_centered(frame, "Move index finger to the START point",
                                  h // 2 - 20, font_scale=0.65, color=YELLOW)
            elif ts == TS.POSITIONING:
                pct = int(ev.human_position_progress * 100)
                put_text_centered(frame, f"Hold at START...  {pct}%",
                                  h // 2 - 20, font_scale=0.65, color=GREEN)
            elif ts == TS.TRACING:
                t_rem = ev.human_time_remaining
                t_col = RED if t_rem < 2 else YELLOW if t_rem < 4 else WHITE
                put_text_centered(frame,
                                  f"Tracing  {ev.human_point_count} pts  |  {t_rem:.1f}s left",
                                  h // 2 - 20, font_scale=0.7, color=t_col, thickness=2)
                put_text_centered(frame, "Reach END point or close fist",
                                  h // 2 + 12, font_scale=0.58, color=ORANGE)
            elif ts == TS.COMPLETED:
                put_text_centered(frame, "Computing DTW...",
                                  h // 2 - 20, font_scale=0.8, color=CYAN)
    else:
        # Attack mode: show animation progress
        pct = int(ev.attack_animation_progress * 100)
        put_text_centered(frame, f"Simulating attack...  {pct}%",
                          h // 2 - 20, font_scale=0.75, color=mode_col, thickness=2)
        draw_progress_bar(frame, ev.attack_animation_progress,
                          y=h // 2 + 5, color_full=mode_col, color_fill=mode_col)

    # ── Legend (bottom-left) ─────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 62), (w, h), DARK_BG, -1)

    # Colour legend
    cv2.line(frame, (10, h-46), (35, h-46), (0, 220, 100), 3)
    cv2.putText(frame, "Target shape", (40, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (10, h-26), (35, h-26), (0, 60, 255), 3)
    cv2.putText(frame, "User / Attack path", (40, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 60, 255), 1, cv2.LINE_AA)

    # Running averages (right side)
    if stats["human_total"] > 0:
        avg_txt = (f"Avg Sim: {stats['avg_similarity']:.1f}%  "
                   f"Avg DTW: {stats['avg_dtw_cost']:.3f}  "
                   f"Avg Time: {stats['avg_time']:.1f}s")
        put_text_with_bg(frame, avg_txt, (w // 2 + 20, h - 40),
                         font_scale=0.48, color=CYAN)

    # Key hints
    put_text_with_bg(frame, "E=cycle mode  R=restart",
                     (w - 230, h - 20), font_scale=0.48, color=YELLOW)

    # Log file path tip (top-right, small)
    log_info = f"Log: {stats.get('log_path', 'eval_logs/attempts.csv')}"
    put_text_with_bg(frame, log_info, (w - 420, bar_h + 12),
                     font_scale=0.38, color=WHITE, bg=(20, 20, 20))


def draw_active_liveness_hud(frame, gm: GameManager, hands):
    """HUD for the unified Active Liveness mode (mode 6).

    Layout (top-left corner, consistent with existing liveness HUD)
    ---------------------------------------------------------------
    TRANSITION   "Next Task" countdown + upcoming challenge badge
    CHALLENGE    Challenge-type badge + task-specific sub-HUD
    COMPLETE     Score gauge + per-challenge breakdown + verdict
    """
    alm = gm.active_liveness
    h, w = frame.shape[:2]

    # ── Colour badges per challenge type ─────────────────────────────
    TYPE_COLORS = {
        ChallengeType.MATH:    CYAN,
        ChallengeType.GESTURE: YELLOW,
        ChallengeType.TOUCH:   ORANGE,
        ChallengeType.TRACE:   MAGENTA,
    }

    def _badge(ct: ChallengeType, x: int, y: int) -> None:
        col = TYPE_COLORS.get(ct, WHITE)
        put_text_with_bg(frame, f"[ {ct.label} ]", (x, y),
                         font_scale=0.6, color=col, thickness=2)

    # ── TRANSITION screen ─────────────────────────────────────────────
    if alm.state == ALMState.TRANSITION:
        ut = alm.upcoming_type
        rem = alm.transition_remaining

        # Semi-dark overlay to signal the break.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        put_text_centered(frame, "NEXT TASK", h // 2 - 55,
                          font_scale=1.3, color=WHITE, thickness=3)
        if ut is not None:
            col = TYPE_COLORS.get(ut, WHITE)
            put_text_centered(frame, f"[ {ut.label} ]", h // 2,
                              font_scale=1.0, color=col, thickness=2)
            put_text_centered(frame, ut.description, h // 2 + 45,
                              font_scale=0.55, color=WHITE)

        # Countdown bar at the bottom.
        prog = 1.0 - (rem / max(_ALM_TRANSITION_DURATION, 0.001))
        draw_progress_bar(frame, prog, y=h - 30,
                          color_full=GREEN, color_fill=YELLOW)
        put_text_with_bg(frame, f"Starting in {rem:.1f}s", (20, h - 50),
                         font_scale=0.55, color=YELLOW)

        # Completed challenges so far (small top-left panel).
        put_text_with_bg(frame, alm.progress_text, (20, 30),
                         font_scale=0.6, color=CYAN)
        for i, line in enumerate(alm.results_text):
            put_text_with_bg(frame, line, (20, 60 + i * 24),
                             font_scale=0.5, color=WHITE)
        return

    # ── COMPLETE screen ───────────────────────────────────────────────
    if alm.state == ALMState.COMPLETE:
        score      = alm.active_liveness_score
        verified   = alm.is_verified
        score_pct  = int(score * 100)
        score_col  = GREEN if score_pct >= 80 else YELLOW if score_pct >= 50 else RED
        verdict    = "LIVENESS VERIFIED" if verified else "NOT VERIFIED"
        v_col      = GREEN if verified else RED

        # Score ring (centre).
        cx, cy, rad = w // 2, h // 2 - 30, 70
        cv2.circle(frame, (cx, cy), rad, DARK_BG, -1)
        cv2.circle(frame, (cx, cy), rad, WHITE, 2)
        angle = int(360 * score)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (rad - 4, rad - 4),
                        -90, 0, angle, score_col, 8)
        put_text_centered(frame, f"{score_pct}%", cy,
                          font_scale=1.1, color=score_col, thickness=3)

        put_text_centered(frame, verdict, h // 2 + 55,
                          font_scale=1.0, color=v_col, thickness=3)
        put_text_centered(frame, f"Score: {score:.4f}", h // 2 + 88,
                          font_scale=0.6, color=WHITE)
        put_text_centered(frame, f"Session: {alm.session_duration_s:.1f}s",
                          h // 2 + 110, font_scale=0.55, color=WHITE)

        # Per-challenge breakdown (top-left).
        put_text_with_bg(frame, "Results:", (20, 30),
                         font_scale=0.6, color=CYAN, thickness=2)
        for i, r in enumerate(alm.results):
            icon  = "PASS" if r.passed else "FAIL"
            i_col = GREEN  if r.passed else RED
            q_str = f" ({r.quality*100:.0f}%)" if r.challenge_type == ChallengeType.TRACE else ""
            a_str = f" x{r.attempts}" if r.attempts > 1 else ""
            put_text_with_bg(frame,
                             f"{icon}  {r.challenge_type.label}{q_str}{a_str}  {r.duration_s:.1f}s",
                             (20, 58 + i * 24),
                             font_scale=0.50, color=i_col)

        put_text_centered(frame, "Press R to restart", h - 30,
                          font_scale=0.6, color=YELLOW)
        return

    # ── CHALLENGE screen ──────────────────────────────────────────────
    ct  = alm.current_type
    cur = alm._current

    # Progress + type badge (always top-left).
    put_text_with_bg(frame, alm.progress_text, (20, 30),
                     font_scale=0.6, color=CYAN)
    _badge(ct, 20, 58)

    # Completed challenges so far.
    for i, line in enumerate(alm.results_text):
        put_text_with_bg(frame, line, (20, 86 + i * 22),
                         font_scale=0.48, color=WHITE)

    # ── MATH sub-HUD ─────────────────────────────────────────────────
    if ct == ChallengeType.MATH and isinstance(cur, _MathAdapter):
        ms = cur.math_state

        eq_col = (GREEN if ms == MathState.RESULT and cur.last_result_correct
                  else RED if ms in (MathState.GAME_OVER,) or (ms == MathState.RESULT and not cur.last_result_correct)
                  else CYAN)
        put_text_centered(frame, cur.label, h // 2 - 30,
                          font_scale=1.4, color=eq_col, thickness=3)

        if ms == MathState.RESULT:
            if cur.last_result_correct:
                put_text_centered(frame, "CORRECT!", h // 2 + 20,
                                  font_scale=1.0, color=GREEN, thickness=2)
            else:
                put_text_centered(frame, f"WRONG — answer was {cur.answer}",
                                  h // 2 + 20,
                                  font_scale=0.8, color=RED, thickness=2)
        elif ms == MathState.EVALUATING:
            put_text_centered(frame, "Processing...", h // 2 + 20,
                              font_scale=1.0, color=YELLOW, thickness=2)
        elif ms == MathState.GAME_OVER:
            put_text_centered(frame, "TIME'S UP!", h // 2 + 20,
                              font_scale=1.0, color=RED, thickness=2)
        else:
            # COUNTDOWN — show remaining question time, no correctness hint.
            q_rem = cur.question_time_remaining
            put_text_with_bg(frame,
                             f"Show your answer — {q_rem:.0f}s",
                             (20, h - 90), font_scale=0.65, color=WHITE)
            # Progress bar: fills as countdown elapses (inverted so it drains).
            draw_progress_bar(frame, 1.0 - cur.question_progress, y=h - 70,
                              color_full=GREEN, color_fill=YELLOW)

        # Unified countdown ring (top-right).
        if ms in (MathState.RESULT, MathState.EVALUATING):
            draw_countdown_ring(frame, 0, 1, w - 95, 95, radius=65,
                                verified=True)
        elif ms == MathState.GAME_OVER:
            draw_countdown_ring(frame, 0, cur.question_duration,
                                w - 95, 95, radius=65)
        else:
            draw_countdown_ring(frame, cur.question_time_remaining,
                                cur.question_duration, w - 95, 95, radius=65)

        # Show detected fingers (count only, NO color hint about correctness).
        total = cur.detected_total
        if total is not None and ms == MathState.COUNTDOWN:
            put_text_with_bg(frame, f"Fingers: {total}", (20, h - 115),
                             font_scale=0.7, color=WHITE, thickness=2)

        if ms == MathState.COUNTDOWN:
            per_hand = cur.per_hand_counts(hands)
            y_off = h - 55
            for side in ("Left", "Right"):
                if side in per_hand:
                    put_text_with_bg(frame, f"{side}: {per_hand[side]}",
                                     (w - 180, y_off), font_scale=0.6)
                    y_off += 28

    # ── GESTURE / TOUCH sub-HUD (unified blind countdown verification) ─
    elif ct in (ChallengeType.GESTURE, ChallengeType.TOUCH) and isinstance(cur, _SnapshotLivenessAdapter):
        snap_rem  = cur.snapshot_remaining
        snap_prog = cur.snapshot_progress
        buf_ratio = cur.buffer_ratio
        result    = cur.snapshot_result        # None / True / False
        buf_frames = cur.buffer_frame_count
        cd_total  = cur.snapshot_countdown_total

        # ── After verdict: show PASS / FAILED splash ──────────────────
        if result is not None:
            splash_col = GREEN if result else RED
            splash_txt = "PASS!" if result else "FAILED!"

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h),
                          (0, 100, 0) if result else (0, 0, 120), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Verified checkmark ring (top-right).
            draw_countdown_ring(frame, 0, 1, w - 95, 95, radius=65,
                                verified=result)

            put_text_centered(frame, splash_txt, h // 2 - 25,
                              font_scale=1.8, color=splash_col, thickness=4)
            put_text_centered(frame, cur.label, h // 2 + 30,
                              font_scale=0.75, color=WHITE)
            # Buffer consistency indicator.
            b_col = GREEN if buf_ratio >= 0.8 else YELLOW if buf_ratio >= 0.5 else RED
            put_text_centered(frame,
                              f"Pose consistency: {buf_ratio * 100:.0f}%",
                              h // 2 + 62, font_scale=0.55, color=b_col)
            return

        # ── Evaluating phase: "Captured — evaluating..." ─────────────
        if cur.is_evaluating:
            # Brief flash overlay to signal the snapshot moment.
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Stopped ring (0 remaining).
            draw_countdown_ring(frame, 0, cd_total, w - 95, 95, radius=65)

            put_text_centered(frame, cur.label, h // 2 - 50,
                              font_scale=1.0, color=WHITE, thickness=2)
            put_text_centered(frame, "Captured — evaluating...",
                              h // 2 + 10, font_scale=1.0, color=CYAN, thickness=2)
            return

        # ── Fallback: snapshot_taken but not yet evaluating (rare) ────
        if cur.snapshot_taken:
            draw_countdown_ring(frame, 0, cd_total, w - 95, 95, radius=65)
            put_text_centered(frame, cur.label, h // 2 - 50,
                              font_scale=1.0, color=WHITE, thickness=2)
            put_text_centered(frame, "EVALUATING...",
                              h // 2 + 10, font_scale=1.1, color=CYAN, thickness=2)
            return

        # ── Countdown still running — blind (no correctness hints) ────

        # Command label in neutral white.
        put_text_centered(frame, cur.label, h // 2 - 55,
                          font_scale=1.0, color=WHITE, thickness=2)

        # Prominent countdown ring (top-right).
        draw_countdown_ring(frame, snap_rem, cd_total,
                            w - 95, 95, radius=65)

        # Neutral instruction — no detection feedback.
        put_text_centered(frame, "Perform the gesture and hold",
                          h // 2 - 10, font_scale=0.75, color=WHITE)

        # Countdown drain bar (remaining time visualisation).
        draw_progress_bar(frame, 1.0 - snap_prog, y=h // 2 + 20,
                          color_full=GREEN, color_fill=YELLOW)

        # Countdown text (bottom-left).
        t_col = RED if snap_rem < 1.0 else YELLOW if snap_rem < 2.0 else WHITE
        put_text_with_bg(frame, f"Snapshot in {snap_rem:.1f}s",
                         (20, h - 90), font_scale=0.7, color=t_col, thickness=2)

        # Instruction hint (bottom of frame).
        put_text_centered(frame, "Hold your pose — captured at 0:00",
                          h - 55, font_scale=0.55, color=WHITE)

    # ── TRACE sub-HUD (with retry support) ───────────────────────────
    elif ct == ChallengeType.TRACE and isinstance(cur, _TraceWithRetryAdapter):

        # -- Retry screen (between attempt 1 fail and attempt 2 start) ----
        if cur.in_retry_screen:
            rem  = cur.retry_remaining
            prog = 1.0 - (rem / max(_ALM_RETRY_DURATION, 0.001))

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 40), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            put_text_centered(frame, "RETRY", h // 2 - 55,
                              font_scale=1.4, color=ORANGE, thickness=3)
            put_text_centered(frame, "One more attempt — trace more carefully",
                              h // 2, font_scale=0.6, color=WHITE)
            if cur.attempt_qualities:
                prev_q = cur.attempt_qualities[-1]
                q_col  = GREEN if prev_q >= 0.7 else YELLOW if prev_q >= 0.4 else RED
                put_text_centered(frame,
                                  f"Attempt 1 similarity: {prev_q*100:.0f}%",
                                  h // 2 + 35, font_scale=0.6, color=q_col)
            draw_progress_bar(frame, prog, y=h - 30,
                              color_full=ORANGE, color_fill=YELLOW)
            put_text_with_bg(frame, f"Retrying in {rem:.1f}s",
                             (20, h - 50), font_scale=0.55, color=YELLOW)
            put_text_with_bg(frame,
                             f"Attempt {cur.attempt_num} / {TRACE_MAX_ATTEMPTS}",
                             (w - 220, 30), font_scale=0.6, color=ORANGE)
            return

        # -- Active trace attempt -----------------------------------------
        lv = cur.inner
        ls = lv.state

        # Attempt badge (below the countdown ring).
        a_col = ORANGE if cur.attempt_num > 1 else WHITE
        put_text_with_bg(frame,
                         f"Attempt {cur.attempt_num}/{TRACE_MAX_ATTEMPTS}",
                         (w - 165, 170), font_scale=0.6, color=a_col)

        if lv.is_flash_red:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Template + live trace (drawn behind all text).
        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            st = lv.shape_tracer
            _draw_shape_template(frame, st.template, w, h,
                                 tracer_state=st.state,
                                 position_progress=st.position_progress)
            if st.state in (TracerState.TRACING, TracerState.VERIFIED,
                            TracerState.FAILED):
                _draw_traced_path(frame, st.traced_path, w, h)

        # Unified countdown ring (top-right, same position as other challenges).
        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            _st = lv.shape_tracer
            from shape_tracer import DEFAULT_DRAW_TIME as _DDT
            if _st.state == TracerState.TRACING:
                # Active tracing — show the 12-second tracer timer.
                draw_countdown_ring(frame, _st.time_remaining, _DDT,
                                    w - 95, 95, radius=65)
            elif _st.state == TracerState.VERIFIED:
                # Trace succeeded — show verified checkmark, timer frozen.
                draw_countdown_ring(frame, 0, 1, w - 95, 95, radius=65,
                                    verified=True)
            elif _st.state == TracerState.FAILED:
                # Trace failed — show empty (expired) ring.
                draw_countdown_ring(frame, 0, _DDT, w - 95, 95, radius=65)
            # Pre-trace states (INSTRUCTING, IDLE, POSITIONING): no ring.

        if ls == LivenessState.SUCCESS:
            put_text_centered(frame, "TRACED!", h // 2 - 20,
                              font_scale=1.4, color=GREEN, thickness=3)
            if lv.shape_tracer is not None:
                pct   = lv.shape_tracer.similarity_pct
                p_col = GREEN if pct >= 70 else YELLOW if pct >= 40 else RED
                put_text_with_bg(frame,
                                 f"Similarity: {pct:.0f}%  DTW: {lv.shape_tracer.dtw_cost:.3f}",
                                 (20, h - 115), font_scale=0.55, color=p_col)
                # Show averaged score if this was a retry.
                if cur.attempt_num > 1:
                    avg = cur.quality
                    put_text_with_bg(frame,
                                     f"Averaged score: {avg*100:.0f}%  (both attempts)",
                                     (20, h - 138), font_scale=0.5, color=ORANGE)
        elif ls == LivenessState.FAILED:
            put_text_centered(frame, "FAILED!", h // 2 - 20,
                              font_scale=1.4, color=RED, thickness=3)
            if lv.shape_tracer is not None:
                pct = lv.shape_tracer.similarity_pct
                put_text_with_bg(frame,
                                 f"Similarity: {pct:.0f}%  (need DTW <= 0.25)",
                                 (20, h - 115), font_scale=0.55, color=YELLOW)
            # If this was the last attempt, say so.
            if cur.attempt_num >= TRACE_MAX_ATTEMPTS:
                put_text_centered(frame, "No retries remaining",
                                  h // 2 + 30, font_scale=0.65, color=RED)
        else:
            if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
                _draw_shape_info_panel(frame, lv.shape_tracer,
                                       lv.shape_trace_label, w, h)

        status_col = GREEN if ls == LivenessState.SUCCESS else RED if ls == LivenessState.FAILED else YELLOW
        put_text_with_bg(frame, lv.status_label, (20, h - 90),
                         font_scale=0.7, color=status_col)


_HUD_DRAWERS = [draw_normal_hud, draw_math_hud,
                draw_liveness_hud, draw_sequential_hud, draw_touch_test_hud,
                draw_shape_eval_hud, draw_active_liveness_hud]


# ── Main loop ───────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_lost_frames=2,
    )
    gm = GameManager()

    print("Gesture Validator -- 'M' cycle modes, 'R' restart, 'Q' quit.")

    # FPS counter state
    _fps_prev_time = time.monotonic()
    _fps_value = 0.0

    with tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # -- FPS calculation ------------------------------------------
            _now = time.monotonic()
            dt = _now - _fps_prev_time
            if dt > 0:
                _fps_value = 0.7 * (1.0 / dt) + 0.3 * _fps_value
            _fps_prev_time = _now

            # -- Detection (separated from game logic) --------------------
            hands = tracker.process(frame)

            # -- Draw landmarks -------------------------------------------
            for hand in hands:
                draw_landmarks(frame, hand)

            # -- Game logic (delegated to GameManager) --------------------
            gm.update(hands)

            # -- HUD rendering --------------------------------------------
            if not gm.hands_present:
                draw_no_hand_overlay(frame)
            _HUD_DRAWERS[gm.current_mode](frame, gm, hands)

            # Mode indicator (bottom-right) + FPS counter (top-right corner)
            fh, fw = frame.shape[:2]
            put_text_with_bg(frame, f"[M] {gm.mode_name}", (fw - 220, fh - 20), font_scale=0.55, color=CYAN)
            fps_color = GREEN if _fps_value >= 25 else YELLOW if _fps_value >= 15 else RED
            put_text_with_bg(frame, f"FPS: {_fps_value:.0f}", (10, fh - 20), font_scale=0.5, color=fps_color)

            cv2.imshow("Gesture Validator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                gm.cycle_mode()
                print(f"Mode: {gm.mode_name}")
            elif key == ord("r"):
                gm.restart()
                print(f"{gm.mode_name} restarted!")
            elif key == ord("e") and gm.current_mode == 6:
                gm.shape_eval.cycle_mode()
                print(f"Eval mode: {gm.shape_eval.eval_mode.label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
