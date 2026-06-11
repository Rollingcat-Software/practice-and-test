"""
HUD Renderer
============
All OpenCV drawing utilities and per-mode HUD functions.

Imported by main.py; never imports from main.py (no circular deps).
GameManager is referenced via TYPE_CHECKING only, so the type annotation
is purely informational at runtime.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gesture_session import SessionState
from math_session import MathState
from liveness_session import LivenessState, CmdType
from shape_tracer import TracerState
from sequential_session import SeqState, StepResult
from finger_touch_session import TouchTestState
from shape_trace_eval_session import EvalMode, _EvalState

if TYPE_CHECKING:
    from game_manager import GameManager

# ── Colour palette ────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (0, 220, 100)
YELLOW  = (0, 220, 255)
RED     = (60, 60, 220)
CYAN    = (220, 220, 0)
MAGENTA = (200, 50, 255)
ORANGE  = (0, 160, 255)
DARK_BG = (40, 40, 40)


# ── Generic drawing utilities ─────────────────────────────────────────

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


def draw_countdown_ring(frame, remaining, total, cx, cy, radius=60):
    progress = max(0.0, remaining / total)
    angle = int(360 * progress)
    cv2.circle(frame, (cx, cy), radius, DARK_BG, -1)
    cv2.circle(frame, (cx, cy), radius, WHITE, 2)
    color = GREEN if remaining > 2 else YELLOW if remaining > 1 else RED
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius - 4, radius - 4), -90, 0, angle, color, 6)
    text = f"{remaining:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), font, 1.0, color, 2)


# ── Shared internal helpers ───────────────────────────────────────────

def _draw_air_canvas(frame, path, w, h):
    """Draw the air-drawing trajectory as a persistent line on the frame."""
    if len(path) < 2:
        return
    for i in range(1, len(path)):
        x1, y1 = int(path[i-1][0] * w), int(path[i-1][1] * h)
        x2, y2 = int(path[i][0] * w), int(path[i][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), MAGENTA, 3)
    lx, ly = int(path[-1][0] * w), int(path[-1][1] * h)
    cv2.circle(frame, (lx, ly), 7, WHITE, -1)


def _draw_verification_bar(frame, pct, y=55):
    """Draw a wide verification score bar (0-100%)."""
    h, w = frame.shape[:2]
    bar_w = int(w * 0.5)
    x_start = (w - bar_w) // 2
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 22), DARK_BG, -1)
    fill_w = int(bar_w * pct / 100.0)
    bar_color = GREEN if pct >= 100 else CYAN if pct >= 60 else YELLOW if pct >= 20 else RED
    cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + 22), bar_color, -1)
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 22), WHITE, 1)
    label = f"Verification: {pct:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    cv2.putText(frame, label, (x_start + (bar_w - tw) // 2, y + 17), font, 0.55, WHITE, 1)


def _draw_direction_arrows(frame, pts: list, color) -> None:
    """Draw small filled triangle arrows along a polyline every ~65 screen pixels."""
    if len(pts) < 3:
        return
    arc = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        arc.append(arc[-1] + math.sqrt(dx * dx + dy * dy))
    total = arc[-1]
    if total < 80:
        return

    spacing   = 65.0
    first_pos = spacing * 1.5
    target    = first_pos
    j         = 0
    while target < total - spacing * 0.8:
        while j < len(arc) - 2 and arc[j + 1] < target:
            j += 1
        seg  = arc[j + 1] - arc[j]
        frac = (target - arc[j]) / seg if seg > 1e-6 else 0.0
        ax   = pts[j][0] + frac * (pts[j + 1][0] - pts[j][0])
        ay   = pts[j][1] + frac * (pts[j + 1][1] - pts[j][1])
        ahead_idx = min(j + max(1, len(pts) // 20), len(pts) - 1)
        dx = pts[ahead_idx][0] - pts[j][0]
        dy = pts[ahead_idx][1] - pts[j][1]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1e-6:
            dx /= mag; dy /= mag
            px, py_v = -dy, dx
            tip   = (int(ax + dx * 9),              int(ay + dy * 9))
            base1 = (int(ax - dx * 5 + px   * 5),   int(ay - dy * 5 + py_v * 5))
            base2 = (int(ax - dx * 5 - px   * 5),   int(ay - dy * 5 - py_v * 5))
            cv2.fillPoly(frame,
                         [np.array([tip, base1, base2], dtype=np.int32)],
                         color)
        target += spacing


def _draw_ghost_trace(frame, waypoints: list, w: int, h: int) -> None:
    """Animate a glowing ghost dot cycling along the template path (3s cycle)."""
    if len(waypoints) < 2:
        return

    PERIOD     = 3.0
    TRAIL_FRAC = 0.07
    TRAIL_N    = 6

    dists = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    total = dists[-1]
    if total < 1e-9:
        return

    def _interp(frac: float) -> tuple[int, int]:
        target = (frac % 1.0) * total
        k = 0
        while k < len(dists) - 2 and dists[k + 1] < target:
            k += 1
        seg = dists[k + 1] - dists[k]
        s   = (target - dists[k]) / seg if seg > 1e-9 else 0.0
        x   = waypoints[k][0] + s * (waypoints[k + 1][0] - waypoints[k][0])
        y   = waypoints[k][1] + s * (waypoints[k + 1][1] - waypoints[k][1])
        return (int(x * w), int(y * h))

    now_frac = (time.time() % PERIOD) / PERIOD

    for i in range(TRAIL_N, 0, -1):
        f      = now_frac - (i / TRAIL_N) * TRAIL_FRAC
        pos    = _interp(f)
        alpha  = i / TRAIL_N
        gray   = int(60 + 100 * alpha)
        radius = int(3 + 3 * alpha)
        cv2.circle(frame, pos, radius, (gray, gray, gray + 40), -1, cv2.LINE_AA)

    head = _interp(now_frac)
    cv2.circle(frame, head, 9,  (220, 240, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, head, 9,  (255, 255, 255),  2, cv2.LINE_AA)
    cv2.circle(frame, head, 13, (180, 200, 220),  1, cv2.LINE_AA)


def _draw_shape_template(frame, template, w: int, h: int,
                         tracer_state=None,
                         position_progress: float = 0.0) -> None:
    """Draw the target shape guide with state-adaptive visual cues."""
    TS = TracerState

    waypoints = list(template.waypoints)
    if len(waypoints) < 2:
        return

    pts = [(int(x * w), int(y * h)) for x, y in waypoints]

    if tracer_state in (TS.INSTRUCTING, TS.IDLE):
        _draw_ghost_trace(frame, waypoints, w, h)

    overlay = frame.copy()
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], (200, 200, 200), 3, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (200, 200, 200), 2, lineType=cv2.LINE_AA)

    arrow_col = (160, 160, 160) if tracer_state != TS.TRACING else (220, 220, 220)
    _draw_direction_arrows(frame, pts, arrow_col)

    sx = int(template.start_point[0] * w)
    sy = int(template.start_point[1] * h)
    ex = int(template.end_point[0]   * w)
    ey = int(template.end_point[1]   * h)

    if tracer_state in (TS.INSTRUCTING, TS.IDLE):
        speed  = 3.0 if tracer_state == TS.INSTRUCTING else 5.0
        pulse  = 0.5 + 0.5 * math.sin(time.time() * speed)
        r_out  = int(12 + 8 * pulse)
        cv2.circle(frame, (sx, sy), r_out, GREEN, 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),      8, GREEN, -1)
        cv2.putText(frame, "START", (sx + 14, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
        if (ex, ey) != (sx, sy):
            cv2.circle(frame, (ex, ey), 8, RED, -1)
            cv2.putText(frame, "END", (ex + 12, ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)

    elif tracer_state == TS.POSITIONING:
        cv2.circle(frame, (sx, sy), 8, GREEN, -1)
        angle = int(position_progress * 360)
        cv2.ellipse(frame, (sx, sy), (20, 20), -90, 0, angle,
                    GREEN, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 20, (100, 220, 100), 1, lineType=cv2.LINE_AA)

    elif tracer_state == TS.TRACING:
        cv2.circle(frame, (sx, sy), 9, GREEN, -1)
        if (ex, ey) != (sx, sy):
            cv2.circle(frame, (ex, ey), 9, ORANGE, -1)
            cv2.putText(frame, "END", (ex + 12, ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)

    else:
        cv2.circle(frame, (sx, sy), 9, GREEN, -1)


def _draw_shape_info_panel(frame, st, label: str, w: int, h: int) -> None:
    """Render shape-tracing status as a single compact line.

    Kept to ONE line placed in the clear band just above the centred shape
    (below the score line, above the shape's top edge) so the guide and the
    user's trace in the centre of the frame are never covered by text boxes.
    """
    TS = TracerState
    TX = 20
    TY = 110

    if st.state == TS.INSTRUCTING:
        rem = st.instruct_remaining
        put_text_with_bg(frame, f"{label}  —  hover finger on green START  (starts {rem:.1f}s)",
                         (TX, TY), font_scale=0.55, color=CYAN)

    elif st.state == TS.IDLE:
        put_text_with_bg(frame, f"{label}  —  move finger to the green START dot",
                         (TX, TY), font_scale=0.55, color=CYAN)

    elif st.state == TS.POSITIONING:
        pct = int(st.position_progress * 100)
        put_text_with_bg(frame, f"{label}  —  hold steady on START... {pct}%",
                         (TX, TY), font_scale=0.55, color=GREEN)

    elif st.state == TS.TRACING:
        rem   = st.time_remaining
        t_col = RED if rem < 2 else YELLOW if rem < 4 else GREEN
        put_text_with_bg(frame, f"{label}  —  REC  {st.point_count} pts | {rem:.0f}s  (reach END / close fist)",
                         (TX, TY), font_scale=0.55, color=t_col)

    elif st.state == TS.COMPLETED:
        put_text_with_bg(frame, f"{label}  —  Analysing...",
                         (TX, TY), font_scale=0.55, color=YELLOW)


def _draw_traced_path(frame, traced, w: int, h: int) -> None:
    """Draw the user's recorded trace as a colour-shifting polyline."""
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
    cv2.circle(frame, pts[-1], 8, WHITE, -1)
    cv2.circle(frame, pts[-1], 8, CYAN,  2)


def _draw_fingertip_spinner(frame, fx: int, fy: int, progress: float) -> None:
    """Animated loading-spinner ring drawn around the index fingertip while
    the user holds on the START point (POSITIONING).

    Two opposing arcs rotate with wall-clock time to read as a spinner; a
    thinner inner arc fills clockwise to show how much of the required hold
    has elapsed.
    """
    r = 28
    # Faint static guide ring.
    cv2.circle(frame, (fx, fy), r, (80, 120, 80), 1, lineType=cv2.LINE_AA)
    # Rotating spinner: two opposing 80° arcs that sweep with time.
    base = int(time.time() * 240) % 360
    cv2.ellipse(frame, (fx, fy), (r, r), base,       0, 80, GREEN, 3, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (fx, fy), (r, r), base + 180, 0, 80, GREEN, 3, lineType=cv2.LINE_AA)
    # Inner progress arc (clockwise from the top) for the hold countdown.
    cv2.ellipse(frame, (fx, fy), (r - 8, r - 8), -90, 0, int(progress * 360),
                CYAN, 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (fx, fy), 3, WHITE, -1)


# ── No-hand overlay ──────────────────────────────────────────────────

def draw_no_hand_overlay(frame):
    """Shown when no hands are detected — replaces mode-specific status."""
    h, w = frame.shape[:2]
    put_text_centered(frame, "No Hand Detected", h // 2, font_scale=1.0, color=RED, thickness=2)


# ── Per-mode HUD renderers ────────────────────────────────────────────

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

    eq_color = RED if ms == MathState.GAME_OVER else GREEN if ms == MathState.SUCCESS else CYAN
    put_text_centered(frame, ses.equation_text, h // 2 - 30, font_scale=1.5, color=eq_color, thickness=3)

    if ms == MathState.GAME_OVER:
        put_text_centered(frame, ses.display_text, h // 2 + 30, font_scale=1.0, color=RED, thickness=2)
        put_text_centered(frame, "Press R to restart", h // 2 + 70, font_scale=0.7, color=YELLOW)
    elif ms == MathState.SUCCESS:
        put_text_centered(frame, "CORRECT!", h // 2 + 30, font_scale=1.0, color=GREEN, thickness=2)
    else:
        status_color = YELLOW if ms == MathState.HOLDING else WHITE
        put_text_with_bg(frame, ses.status_label, (20, 110), font_scale=0.7, color=status_color)
        draw_progress_bar(frame, ses.hold_progress)

    remaining = ses.time_remaining
    timer_color = RED if remaining < 10 else YELLOW if remaining < 30 else WHITE
    put_text_with_bg(frame, f"Time: {remaining:.0f}s", (w - 180, 40), font_scale=0.8, color=timer_color, thickness=2)
    put_text_with_bg(frame, f"Score: {ses.score}", (20, 40), font_scale=0.8, color=GREEN, thickness=2)

    per_hand = ses.per_hand_counts(hands)
    total = ses.detected_total()
    y_off = h - 80
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (20, y_off), font_scale=0.6)
            y_off += 28
    if total is not None:
        c = GREEN if total == ses.answer else WHITE
        put_text_with_bg(frame, f"Total: {total}", (20, y_off), font_scale=0.7, color=c, thickness=2)


def draw_liveness_hud(frame, gm: GameManager, hands):
    lv = gm.liveness
    h, w = frame.shape[:2]
    ls = lv.state

    if lv.is_flash_red:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
        st = lv.shape_tracer
        _draw_shape_template(frame, st.template, w, h,
                             tracer_state=st.state,
                             position_progress=st.position_progress)
        if st.state in (TracerState.TRACING, TracerState.VERIFIED, TracerState.FAILED):
            _draw_traced_path(frame, st.traced_path, w, h)
        # Spinner around the fingertip while holding on the START point.
        if st.state == TracerState.POSITIONING and hands:
            tip = hands[0].landmarks[8]   # INDEX_TIP
            _draw_fingertip_spinner(frame, int(tip.x * w), int(tip.y * h),
                                    st.position_progress)

    _draw_verification_bar(frame, lv.verification_pct, y=10)

    put_text_with_bg(frame, lv.challenge_progress_text, (20, 55), font_scale=0.6, color=CYAN)
    put_text_with_bg(frame, f"Score: {lv.score}", (20, 85), font_scale=0.7, color=GREEN, thickness=2)

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

    # Anti-spoof warning bar
    warn_pct = lv.spoof_warning_progress
    if warn_pct > 0 and ls in (LivenessState.ACTIVE, LivenessState.DEBOUNCE):
        bar_w = int(w * 0.22)
        bar_x = w - bar_w - 10
        bar_y = 10
        bar_h = 14
        fill_w = int(bar_w * warn_pct)
        bar_col = RED if warn_pct >= 1.0 else ORANGE
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), DARK_BG, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), ORANGE, 1)
        put_text_with_bg(frame, "ANTI-SPOOF", (bar_x, bar_y + bar_h + 14),
                         font_scale=0.38, color=ORANGE, bg=DARK_BG, thickness=1)

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
            put_text_with_bg(frame,
                             f"Similarity: {pct:.0f}%  DTW: {lv.shape_tracer.dtw_cost:.3f}",
                             (12, 108), font_scale=0.55, color=s_color)
    elif ls == LivenessState.FAILED:
        if lv.is_spoof_blocked:
            put_text_centered(frame, "SPOOF DETECTED!", h // 2 - 40,
                              font_scale=1.3, color=ORANGE, thickness=3)
            put_text_centered(frame, "Static image / frozen video blocked.",
                              h // 2 + 5, font_scale=0.65, color=ORANGE)
            put_text_centered(frame, "Next challenge incoming...",
                              h // 2 + 45, font_scale=0.65, color=YELLOW)
        else:
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
        if not lv.is_shape_trace_cmd:
            put_text_centered(frame, lv.command_label, h // 2 - 30,
                              font_scale=1.0, color=cmd_color, thickness=2)
        if ls == LivenessState.DEBOUNCE:
            draw_progress_bar(frame, lv.debounce_progress,
                              y=h // 2 + 10, color_full=GREEN, color_fill=ORANGE)
            put_text_centered(frame, "Confirming...", h // 2 + 50,
                              font_scale=0.7, color=ORANGE)

        if lv.is_wave_cmd:
            rev = lv.wave_reversals
            rev_color = GREEN if rev >= 2 else YELLOW if rev >= 1 else WHITE
            put_text_with_bg(frame, f"Waves: {rev}/2", (20, h - 125),
                             font_scale=0.65, color=rev_color)

        if lv.is_touch_cmd:
            frames = lv.touch_frame_count
            touch_color = GREEN if frames >= 8 else YELLOW if frames >= 4 else WHITE
            put_text_with_bg(frame, f"Hold: {frames}/10 frames", (20, h - 125),
                             font_scale=0.65, color=touch_color)
            draw_progress_bar(frame, lv.touch_frame_progress, y=h - 108,
                              color_full=GREEN, color_fill=YELLOW)

        if lv.is_shape_trace_cmd and lv.shape_tracer is not None:
            _draw_shape_info_panel(frame, lv.shape_tracer, lv.shape_trace_label,
                                   w, h)

    status_color = GREEN if ls == LivenessState.SUCCESS else RED if ls == LivenessState.FAILED else YELLOW
    put_text_with_bg(frame, lv.status_label, (20, h - 90), font_scale=0.7, color=status_color)

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
    seq = gm.sequential
    h, w = frame.shape[:2]
    ss = seq.state

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

    put_text_with_bg(frame, seq.progress_text, (20, info_y), font_scale=0.6, color=CYAN)
    put_text_with_bg(frame, f"Passed: {seq.passed_count}/{seq.total_steps}", (180, info_y), font_scale=0.6, color=GREEN)
    if seq.elapsed_time > 0:
        put_text_with_bg(frame, f"Total: {seq.elapsed_time:.1f}s", (w - 180, info_y), font_scale=0.6, color=WHITE)

    if ss not in (SeqState.COMPLETE,):
        remaining = seq.step_time_remaining
        step = seq.current_step
        limit = step.time_limit if step else 0
        draw_countdown_ring(frame, remaining, limit, w - 70, info_y + 60, radius=40)

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

        step = seq.current_step
        if step and step.step_type in ("draw_circle", "draw_square"):
            _draw_air_canvas(frame, seq.drawing_path, w, h)
            pts = len(seq.drawing_path)
            if pts > 0:
                put_text_with_bg(frame, f"Drawing... ({pts} pts)", (20, h - 125), font_scale=0.55, color=MAGENTA)

    status_color = GREEN if ss in (SeqState.STEP_DONE, SeqState.COMPLETE) else RED if ss == SeqState.STEP_TIMEOUT else YELLOW if ss == SeqState.HOLDING else WHITE
    put_text_with_bg(frame, seq.status_label, (20, h - 90), font_scale=0.7, color=status_color)

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

    put_text_centered(frame, "FINGER TOUCH TEST", 32, font_scale=0.9, color=CYAN, thickness=2)

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

        if "BOTH THUMBS" in tt.command_label:
            put_text_centered(frame, "(show both hands, bring thumbs together)",
                              mid_y + 75, font_scale=0.55, color=WHITE)
        else:
            put_text_centered(frame, "(use any hand)", mid_y + 75,
                              font_scale=0.55, color=WHITE)

    put_text_with_bg(frame, f"Passed: {tt.passed_count}/5", (20, h - 90),
                     font_scale=0.7, color=GREEN, thickness=2)
    put_text_with_bg(frame, "R = restart", (w - 170, h - 90),
                     font_scale=0.6, color=YELLOW)


def draw_shape_eval_hud(frame, gm: GameManager, hands):
    """Visual Debugger HUD for the Shape Tracing Evaluation mode."""
    ev = gm.shape_eval
    h, w = frame.shape[:2]
    stats = ev.evaluator.stats

    MODE_COLORS = {
        EvalMode.HUMAN_TEST:    GREEN,
        EvalMode.STATIC_ATTACK: ORANGE,
        EvalMode.RANDOM_ATTACK: RED,
    }
    mode_col = MODE_COLORS.get(ev.eval_mode, WHITE)

    tpl = ev.debug_template_path
    usr = ev.debug_user_path

    _eval_tpl = ev.current_template
    _eval_ts  = ev.human_tracer_state
    if _eval_tpl is not None and len(tpl) >= 2:
        if _eval_ts in (TracerState.INSTRUCTING, TracerState.IDLE, None):
            _draw_ghost_trace(frame, list(_eval_tpl.waypoints), w, h)

        pts = [(int(x * w), int(y * h)) for x, y in tpl]
        overlay = frame.copy()
        for i in range(1, len(pts)):
            cv2.line(overlay, pts[i-1], pts[i], (0, 200, 80), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0, 220, 100), 2, cv2.LINE_AA)

        _draw_direction_arrows(frame, pts, (0, 160, 60))

        _sx = int(_eval_tpl.start_point[0] * w)
        _sy = int(_eval_tpl.start_point[1] * h)
        cv2.circle(frame, (_sx, _sy), 10, (0, 255, 200), -1)
        cv2.putText(frame, "START", (_sx + 12, _sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
        _ex = int(_eval_tpl.end_point[0] * w)
        _ey = int(_eval_tpl.end_point[1] * h)
        if (_ex, _ey) != (_sx, _sy):
            cv2.circle(frame, (_ex, _ey), 10, ORANGE, -1)
            cv2.putText(frame, "END", (_ex + 12, _ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1, cv2.LINE_AA)

    if len(usr) >= 2:
        pts_u = [(int(x * w), int(y * h)) for x, y in usr]
        for i in range(1, len(pts_u)):
            t = i / len(pts_u)
            cv2.line(frame, pts_u[i-1], pts_u[i],
                     (40, int(80*t), 220), 3, cv2.LINE_AA)
        cv2.circle(frame, pts_u[-1], 7, (0, 60, 255), -1)
        cv2.circle(frame, pts_u[-1], 7, WHITE, 1)

    bar_h = 44
    cv2.rectangle(frame, (0, 0), (w, bar_h), DARK_BG, -1)

    badge_label = f" {ev.eval_mode.label} "
    cv2.rectangle(frame, (8, 5), (8 + len(badge_label)*10, 38), mode_col, -1)
    cv2.putText(frame, badge_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    stat_txt = (f"Att:{stats['total']}  "
                f"H:{stats['human_total']}  "
                f"Atk:{stats['attack_total']}  |  "
                f"FAR:{stats['far']*100:.0f}%  "
                f"FRR:{stats['frr']*100:.0f}%")
    put_text_with_bg(frame, stat_txt, (220, 30), font_scale=0.55, color=WHITE)

    put_text_with_bg(frame, f"Shape: {ev.shape_label}",
                     (12, 66), font_scale=0.6, color=CYAN, thickness=2)

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
            # Instructions kept short and corner-anchored (top-left, below the
            # stats bar) so they never cover the centred shape / trace.
            IX, IY = 12, 92
            if ts == TracerState.INSTRUCTING:
                rem = ev._tracer.instruct_remaining if ev._tracer else 0.0
                put_text_with_bg(frame, f"Read instructions...  {rem:.0f}s",
                                 (IX, IY), font_scale=0.55, color=YELLOW)
            elif ts == TracerState.IDLE:
                put_text_with_bg(frame, "Move finger to START dot",
                                 (IX, IY), font_scale=0.55, color=YELLOW)
            elif ts == TracerState.POSITIONING:
                pct = int(ev.human_position_progress * 100)
                put_text_with_bg(frame, f"Hold at START...  {pct}%",
                                 (IX, IY), font_scale=0.55, color=GREEN)
                if hands:
                    tip = hands[0].landmarks[8]   # INDEX_TIP
                    _draw_fingertip_spinner(frame, int(tip.x * w), int(tip.y * h),
                                            ev.human_position_progress)
            elif ts == TracerState.TRACING:
                t_rem = ev.human_time_remaining
                t_col = RED if t_rem < 2 else YELLOW if t_rem < 4 else WHITE
                put_text_with_bg(frame,
                                 f"REC  {ev.human_point_count}p  {t_rem:.0f}s",
                                 (IX, IY), font_scale=0.55, color=t_col)
            elif ts == TracerState.COMPLETED:
                put_text_with_bg(frame, "Computing DTW...",
                                 (IX, IY), font_scale=0.55, color=CYAN)
    else:
        pct = int(ev.attack_animation_progress * 100)
        put_text_centered(frame, f"Simulating attack...  {pct}%",
                          h // 2 - 20, font_scale=0.75, color=mode_col, thickness=2)
        draw_progress_bar(frame, ev.attack_animation_progress,
                          y=h // 2 + 5, color_full=mode_col, color_fill=mode_col)

    cv2.rectangle(frame, (0, h - 62), (w, h), DARK_BG, -1)

    cv2.line(frame, (10, h-46), (35, h-46), (0, 220, 100), 3)
    cv2.putText(frame, "Target shape", (40, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (10, h-26), (35, h-26), (0, 60, 255), 3)
    cv2.putText(frame, "User / Attack path", (40, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 60, 255), 1, cv2.LINE_AA)

    if stats["human_total"] > 0:
        avg_txt = (f"Avg Sim: {stats['avg_similarity']:.1f}%  "
                   f"Avg DTW: {stats['avg_dtw_cost']:.3f}  "
                   f"Avg Time: {stats['avg_time']:.1f}s")
        put_text_with_bg(frame, avg_txt, (w // 2 + 20, h - 40),
                         font_scale=0.48, color=CYAN)

    put_text_with_bg(frame, "E=cycle mode  R=restart",
                     (w - 230, h - 20), font_scale=0.48, color=YELLOW)

    log_info = f"Log: {stats.get('log_path', 'eval_logs/attempts.csv')}"
    put_text_with_bg(frame, log_info, (w - 420, bar_h + 12),
                     font_scale=0.38, color=WHITE, bg=(20, 20, 20))


# ── HUD dispatcher (indexed by mode number) ───────────────────────────
HUD_DRAWERS = [
    draw_normal_hud,
    draw_math_hud,
    draw_liveness_hud,
    draw_sequential_hud,
    draw_touch_test_hud,
    draw_shape_eval_hud,
]
