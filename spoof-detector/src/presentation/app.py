"""Main application loop with keyboard controls."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.presentation.camera import ThreadedCamera
from src.presentation.overlay import Overlay
from src.application.pipeline import SpoofDetectionPipeline
from src.application.session_engine import SessionEngine
from src.application.liveness_prover import ChallengeState
from src.infrastructure.logging.structured_logger import StructuredLogger
from src.domain.models import FrameAnalysis
from src.domain.session import SessionVerdict

logger = logging.getLogger(__name__)

WINDOW_NAME = "FIVUCSAS Spoof Detector"

HELP_TEXT = """
FIVUCSAS Spoof Detector — Keyboard Controls
═══════════════════════════════════════════
  q / ESC  — Quit
  d        — Toggle detail panel
  l        — Toggle logging
  s        — Save current frame + analysis
  p        — Toggle profiler
  h        — Toggle this help
═══════════════════════════════════════════
""".strip().split("\n")


class SpoofDetectorApp:
    """Main desktop application.

    Captures frames from camera, runs the spoof detection pipeline,
    and renders results via OpenCV overlay.
    """

    def __init__(
        self,
        pipeline: SpoofDetectionPipeline,
        camera: ThreadedCamera,
        struct_logger: StructuredLogger,
        show_detail: bool = True,
        show_profiler: bool = False,
    ):
        self._pipeline = pipeline
        self._camera = camera
        self._logger = struct_logger
        self._overlay = Overlay(show_detail=show_detail, show_profiler=show_profiler)
        self._session = SessionEngine()
        self._show_help = False
        self._save_counter = 0
        self._running = False
        self._last_analysis: FrameAnalysis | None = None

    def run(self):
        """Main loop: capture → process → render → display."""
        logger.info("Starting Spoof Detector application")
        self._running = True
        self._camera.start()
        self._logger.start()
        self._session.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get screen resolution for content scaling
        screen_w, screen_h = self._get_screen_size()

        try:
            while self._running:
                grabbed, frame = self._camera.read_copy()
                if not grabbed or frame is None:
                    logger.warning("No frame from camera")
                    time.sleep(0.01)
                    continue

                # Process frame through pipeline
                analysis = self._pipeline.process(frame)
                self._last_analysis = analysis

                # Feed into session engine
                self._session.ingest(analysis, frame)
                verdict = self._session.get_verdict()

                # Log
                self._logger.log(analysis)

                # Render overlay
                self._overlay.render(frame, analysis)

                # Render session verdict bar
                self._draw_session_bar(frame, verdict)

                # Render active challenge
                self._draw_challenge(frame)

                # Render liveness score
                self._draw_liveness_score(frame)

                # Help overlay
                if self._show_help:
                    self._draw_help(frame)

                # Scale frame to fill screen
                if screen_w > 0 and screen_h > 0:
                    frame = cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(WINDOW_NAME, frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame, analysis):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    def _handle_key(self, key: int, frame: np.ndarray, analysis: FrameAnalysis) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == ord("q") or key == 27:  # q or ESC
            return False
        elif key == ord("d"):
            self._overlay.toggle_detail()
        elif key == ord("l"):
            self._logger.toggle()
        elif key == ord("p"):
            self._overlay.toggle_profiler()
        elif key == ord("h"):
            self._show_help = not self._show_help
        elif key == ord("s"):
            self._save_frame(frame, analysis)
        return True

    def _save_frame(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Save current frame to data/captures/."""
        import json
        from pathlib import Path

        captures_dir = Path("data/captures")
        captures_dir.mkdir(parents=True, exist_ok=True)

        self._save_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"capture_{ts}_{self._save_counter:04d}"

        # Save image
        img_path = captures_dir / f"{base}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Save metadata
        meta = {
            "frame_id": analysis.frame_id,
            "face_count": len(analysis.faces),
            "classifications": {},
        }
        for fid, cls in analysis.classifications.items():
            meta["classifications"][str(fid)] = {
                cat.value: round(prob, 4)
                for cat, prob in cls.probabilities.items()
            }

        meta_path = captures_dir / f"{base}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {img_path}")

    def _draw_help(self, frame: np.ndarray):
        """Draw help overlay."""
        h, w = frame.shape[:2]
        panel_h = len(HELP_TEXT) * 22 + 20
        panel_w = 400
        x = (w - panel_w) // 2
        y = (h - panel_h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        ty = y + 25
        for line in HELP_TEXT:
            cv2.putText(frame, line, (x + 15, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            ty += 22

    @staticmethod
    def _get_screen_size() -> tuple[int, int]:
        """Get screen resolution for content scaling."""
        try:
            import ctypes
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            return 0, 0  # Fallback: no scaling

    def _draw_challenge(self, frame: np.ndarray):
        """Draw active challenge prompt in center of screen."""
        challenge = self._session.prover.get_active_challenge()
        if challenge is None or challenge.state != ChallengeState.PROMPTED:
            return

        h, w = frame.shape[:2]
        text = challenge.display_text
        elapsed = self._session.elapsed_sec - challenge.prompted_at
        remaining = max(0, challenge.timeout_sec - elapsed)

        # Flashing background for urgency
        alpha = 0.7 if int(elapsed * 3) % 2 == 0 else 0.5
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 4, h // 3 - 40), (3 * w // 4, h // 3 + 50), (0, 0, 180), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Challenge text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        tx = (w - tw) // 2
        cv2.putText(frame, text, (tx, h // 3 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # Timer
        timer_text = f"{remaining:.1f}s"
        cv2.putText(frame, timer_text, (w // 2 - 30, h // 3 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_liveness_score(self, frame: np.ndarray):
        """Draw liveness proof score on the right side."""
        score = self._session.prover.get_score()
        h, w = frame.shape[:2]

        # Panel position (right side)
        px = w - 220
        py = 120
        panel_h = 130

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (w - 10, py + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        status = "PROVEN LIVE" if score.is_proven_live else "PROVING..."
        color = (0, 200, 0) if score.is_proven_live else (0, 200, 200)
        cv2.putText(frame, f"Liveness: {score.total:.0f}/100 [{status}]", (px + 5, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Score breakdown
        items = [
            (f"Blinks: {score.blink_points:.0f}/20", score.blink_points / 20),
            (f"Motion: {score.landmark_points:.0f}/15", score.landmark_points / 15),
            (f"Rotation: {score.rotation_points:.0f}/15", score.rotation_points / 15),
            (f"Expression: {score.expression_points:.0f}/10", score.expression_points / 10),
            (f"Challenges: {score.challenges_passed}/{score.challenges_passed + score.challenges_failed}",
             score.challenge_points / 40),
        ]

        y = py + 38
        for label, ratio in items:
            bar_color = (0, 200, 0) if ratio > 0.5 else (0, 200, 200) if ratio > 0 else (80, 80, 80)
            cv2.putText(frame, label, (px + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
            # Mini bar
            bar_x = px + 150
            bar_w = 55
            fill = int(bar_w * min(1.0, ratio))
            cv2.rectangle(frame, (bar_x, y - 8), (bar_x + bar_w, y), (40, 40, 40), -1)
            if fill > 0:
                cv2.rectangle(frame, (bar_x, y - 8), (bar_x + fill, y), bar_color, -1)
            y += 20

    def _draw_session_bar(self, frame: np.ndarray, verdict: SessionVerdict):
        """Draw session verdict bar at the top of the frame."""
        h, w = frame.shape[:2]

        # Bar dimensions
        bar_h = 32
        bar_y = h - bar_h - 5

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, bar_y), (w - 5, bar_y + bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Verdict color and text
        if verdict.is_live:
            color = (0, 200, 0)
            label = "SESSION: LIVE"
        else:
            color = (0, 0, 220)
            threat = verdict.dominant_threat.value if verdict.dominant_threat else "spoof"
            label = f"SESSION: SPOOF ({threat})"

        conf_pct = verdict.confidence * 100
        elapsed = verdict.session_duration_sec

        cv2.putText(frame, label, (15, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Stats
        stats = (
            f"conf={conf_pct:.0f}% | {elapsed:.0f}s | "
            f"frames={verdict.frames_analyzed} | "
            f"incidents={len(verdict.incidents)}"
        )
        cv2.putText(frame, stats, (350, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Confidence bar
        bar_x = w - 160
        bar_w = 145
        fill = int(bar_w * verdict.confidence)
        cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + bar_w, bar_y + 24),
                      (40, 40, 40), -1)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + fill, bar_y + 24),
                          color, -1)

    def _cleanup(self):
        self._running = False

        # Session report
        verdict = self._session.conclude()
        liveness = self._session.prover.get_score()
        print(f"\n{'=' * 60}")
        print(f"  SESSION REPORT")
        print(f"{'=' * 60}")
        print(f"  Verdict: {verdict.summary}")
        print(f"  Face detected: {verdict.face_detected_ratio:.0%} of frames")
        print(f"\n  Liveness Proof: {liveness.total:.0f}/100 {'PROVEN' if liveness.is_proven_live else 'NOT PROVEN'}")
        print(f"    Blinks:     {liveness.blink_points:.0f}/20")
        print(f"    Motion:     {liveness.landmark_points:.0f}/15")
        print(f"    Rotation:   {liveness.rotation_points:.0f}/15")
        print(f"    Expression: {liveness.expression_points:.0f}/10")
        print(f"    Challenges: {liveness.challenge_points:.0f}/40 ({liveness.challenges_passed} passed, {liveness.challenges_failed} failed)")
        challenges = self._session.prover.get_challenge_history()
        if challenges:
            print(f"\n  Challenge History:")
            for c in challenges:
                print(f"    {c['type']:>12s}: {c['state']:>10s} ({c['latency_ms']:.0f}ms)")
        if verdict.incidents:
            print(f"\n  Incident Timeline:")
            for item in self._session.get_timeline():
                print(f"    [{item['time_sec']:6.1f}s] {item['severity']:>8s} | {item['category']:>15s} | {item['description']}")
        print(f"{'=' * 60}\n")

        self._camera.stop()
        self._logger.stop()
        cv2.destroyAllWindows()
        logger.info("Application stopped")
