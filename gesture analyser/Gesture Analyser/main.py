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
hud_renderer  -- all OpenCV drawing utilities and per-mode HUD functions.
main loop     -- capture + render only; no game logic here.
"""

import logging
import time

import cv2

from app_config import cfg

_log = logging.getLogger(__name__)

# Maximum consecutive frame errors before the loop exits cleanly.
_MAX_CONSECUTIVE_ERRORS = 10
from hand_tracker import HandTracker, draw_landmarks
from game_manager import GameManager
from hud_renderer import (
    HUD_DRAWERS,
    draw_no_hand_overlay,
    put_text_with_bg,
    GREEN, YELLOW, RED, CYAN,
)


def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=cfg.tracker.detection_confidence,
        min_tracking_confidence=cfg.tracker.tracking_confidence,
        max_lost_frames=cfg.tracker.max_lost_frames,
    )
    gm = GameManager()

    print("Gesture Validator -- 'M' cycle modes, 'R' restart, 'Q' quit.")

    _fps_prev_time = time.time()
    _fps_value = 0.0
    _consecutive_errors = 0

    with tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                _log.warning("cap.read() returned False — camera disconnected?")
                break

            frame = cv2.flip(frame, 1)

            # FPS calculation
            _now = time.time()
            dt = _now - _fps_prev_time
            if dt > 0:
                _fps_value = 0.7 * (1.0 / dt) + 0.3 * _fps_value
            _fps_prev_time = _now

            try:
                # Detection (separated from game logic)
                hands = tracker.process(frame)

                # Draw landmarks
                for hand in hands:
                    draw_landmarks(frame, hand)

                # Game logic (delegated to GameManager)
                gm.update(hands)

                # HUD rendering
                if not gm.hands_present and not gm.suppress_no_hand_overlay:
                    draw_no_hand_overlay(frame)
                HUD_DRAWERS[gm.current_mode](frame, gm, hands)

                # Mode indicator (bottom-right) + FPS counter (bottom-left)
                fh, fw = frame.shape[:2]
                put_text_with_bg(frame, f"[M] {gm.mode_name}", (fw - 220, fh - 20),
                                 font_scale=0.55, color=CYAN)
                fps_color = GREEN if _fps_value >= 25 else YELLOW if _fps_value >= 15 else RED
                put_text_with_bg(frame, f"FPS: {_fps_value:.0f}", (10, fh - 20),
                                 font_scale=0.5, color=fps_color)

                _consecutive_errors = 0  # reset on successful frame

            except Exception as exc:  # noqa: BLE001
                _consecutive_errors += 1
                _log.error(
                    "Frame error #%d/%d: %s",
                    _consecutive_errors, _MAX_CONSECUTIVE_ERRORS, exc,
                    exc_info=True,
                )
                if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    _log.critical(
                        "Reached %d consecutive errors — shutting down.",
                        _MAX_CONSECUTIVE_ERRORS,
                    )
                    break
                # Show a brief error banner so the user knows something is wrong.
                try:
                    fh, fw = frame.shape[:2]
                    put_text_with_bg(
                        frame,
                        f"ERROR ({_consecutive_errors}/{_MAX_CONSECUTIVE_ERRORS}) — see logs",
                        (10, fh // 2),
                        font_scale=0.6, color=RED,
                    )
                except Exception:  # noqa: BLE001
                    pass  # if even drawing fails, just continue

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
            elif key == ord("e") and gm.current_mode == 5:
                gm.shape_eval.cycle_mode()
                print(f"Eval mode: {gm.shape_eval.eval_mode.label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
