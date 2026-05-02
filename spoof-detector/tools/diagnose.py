#!/usr/bin/env python3
"""
Live Diagnostic Tool
====================

Runs each analyzer independently on live camera feed and displays
per-analyzer timing + scores in a diagnostic dashboard.

This helps identify:
- Which analyzer is the bottleneck
- Which analyzer scores are most/least stable
- How scores change with different presentation attacks

Usage:
    python tools/diagnose.py              # Live camera diagnostic
    python tools/diagnose.py --image X    # Single image diagnostic
    python tools/diagnose.py --log-csv    # Log to CSV for analysis
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np

from src.domain.models import FaceROI, BBox
from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
from src.application.face_tracker import FaceTracker


class DiagnosticDashboard:
    """Real-time diagnostic overlay showing per-analyzer breakdown."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self._histories: dict[str, deque] = {}

    def record(self, name: str, score: float, ms: float):
        if name not in self._histories:
            self._histories[name] = deque(maxlen=60)
        self._histories[name].append((score, ms))

    def draw(self, frame: np.ndarray, face_count: int, pipeline_ms: float):
        h, w = frame.shape[:2]

        # Right panel
        panel_w = 320
        panel_x = w - panel_w - 10
        panel_h = len(self._histories) * 45 + 80
        panel_y = 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 5, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Title
        cv2.putText(frame, "DIAGNOSTIC DASHBOARD", (panel_x + 10, panel_y + 22),
                    self.FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {face_count}  Pipeline: {pipeline_ms:.1f}ms",
                    (panel_x + 10, panel_y + 42),
                    self.FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        y = panel_y + 65
        total_ms = 0
        for name, history in sorted(self._histories.items()):
            scores = [h[0] for h in history]
            times = [h[1] for h in history]

            avg_score = sum(scores) / len(scores) if scores else 0
            avg_ms = sum(times) / len(times) if times else 0
            latest_score = scores[-1] if scores else 0
            total_ms += avg_ms

            # Score color
            if latest_score >= 70:
                color = (0, 200, 0)  # Green
            elif latest_score >= 40:
                color = (0, 200, 200)  # Yellow
            else:
                color = (0, 0, 200)  # Red

            # Timing color
            t_color = (0, 200, 0) if avg_ms < 10 else (0, 200, 200) if avg_ms < 25 else (0, 0, 200)

            # Name + score
            cv2.putText(frame, f"{name:>14s}", (panel_x + 10, y),
                        self.FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{latest_score:5.1f}", (panel_x + 130, y),
                        self.FONT, 0.42, color, 1, cv2.LINE_AA)

            # Score bar
            bar_x = panel_x + 180
            bar_w = 80
            fill = int(bar_w * latest_score / 100)
            cv2.rectangle(frame, (bar_x, y - 10), (bar_x + bar_w, y), (40, 40, 40), -1)
            if fill > 0:
                cv2.rectangle(frame, (bar_x, y - 10), (bar_x + fill, y), color, -1)

            # Timing
            cv2.putText(frame, f"{avg_ms:5.1f}ms", (panel_x + 265, y),
                        self.FONT, 0.35, t_color, 1, cv2.LINE_AA)

            # Sparkline (last 30 scores)
            if len(scores) > 2:
                spark_x = panel_x + 10
                spark_y = y + 5
                spark_w = 300
                spark_h = 15
                recent = list(scores)[-30:]
                for i in range(1, len(recent)):
                    x1 = spark_x + int((i - 1) / 30 * spark_w)
                    x2 = spark_x + int(i / 30 * spark_w)
                    y1 = spark_y + spark_h - int(recent[i - 1] / 100 * spark_h)
                    y2 = spark_y + spark_h - int(recent[i] / 100 * spark_h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            y += 45

        # Total
        cv2.putText(frame, f"Total analysis: {total_ms:.1f}ms",
                    (panel_x + 10, y + 5),
                    self.FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def analyze_single_image(image_path: str):
    """Run all analyzers on a single image and print report."""
    print(f"\n{'=' * 60}")
    print(f"  Analyzing: {image_path}")
    print(f"{'=' * 60}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read {image_path}")
        return

    h, w = img.shape[:2]
    print(f"  Size: {w}x{h}")

    # Detect faces
    detector = MediaPipeFaceDetector(min_confidence=0.4)
    faces = detector.detect(img)
    print(f"  Faces detected: {len(faces)}")

    if not faces:
        print("  No faces found. Running analyzers on full image as crop.")
        faces = [FaceROI(face_id=1, bbox=BBox(0, 0, w, h), confidence=0.0, crop=img.copy())]

    analyzers = [
        ("minifasnet", MiniFASNetAnalyzer()),
        ("texture", TextureAnalyzer()),
        ("moire", MoireAnalyzer()),
        ("screen_replay", ScreenReplayAnalyzer()),
    ]

    for face in faces:
        crop = face.crop if face.crop is not None else img[face.bbox.y1:face.bbox.y2, face.bbox.x1:face.bbox.x2]
        ch, cw = crop.shape[:2]
        print(f"\n  Face #{face.face_id} ({cw}x{ch}, conf={face.confidence:.2f})")
        print(f"  {'Analyzer':>15s}  {'Score':>8s}  {'Time':>10s}  Details")
        print(f"  {'─' * 60}")

        for name, analyzer in analyzers:
            if name == "screen_replay":
                result = analyzer.analyze(img)  # Full frame
            else:
                result = analyzer.analyze(crop, face)

            detail_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in list(result.details.items())[:4])
            status = "LIVE" if result.score > 60 else "SPOOF" if result.score < 30 else "UNSURE"
            print(f"  {name:>15s}  {result.score:6.1f}  [{status:>5s}]  {result.elapsed_ms:7.2f}ms  {detail_str}")

    detector.close()


def run_live_diagnostic(log_csv: bool = False):
    """Run live camera diagnostic with per-analyzer dashboard."""
    print("Starting live diagnostic — press 'q' to quit")

    detector = MediaPipeFaceDetector(min_confidence=0.5)
    tracker = FaceTracker()
    dashboard = DiagnosticDashboard()

    analyzers = [
        ("minifasnet", MiniFASNetAnalyzer()),
        ("texture", TextureAnalyzer()),
        ("moire", MoireAnalyzer()),
        ("temporal", TemporalAnalyzer()),
    ]
    frame_analyzers = [
        ("screen_replay", ScreenReplayAnalyzer()),
    ]

    csv_file = None
    csv_writer = None
    if log_csv:
        csv_path = Path("logs") / f"diag_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path.parent.mkdir(exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        fields = ["frame", "face_id", "pipeline_ms"] + [n for n, _ in analyzers + frame_analyzers]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields + [f"{n}_ms" for n, _ in analyzers + frame_analyzers])
        csv_writer.writeheader()
        print(f"  Logging to {csv_path}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.perf_counter()

            # Detect + track
            raw_faces = detector.detect(frame)
            tracked = tracker.update(raw_faces)

            # Ensure crops
            for face in tracked:
                if face.crop is None:
                    b = face.bbox
                    face.crop = frame[b.y1:b.y2, b.x1:b.x2].copy()

            # Per-face analyzers
            for face in tracked:
                if face.crop is None or face.crop.size == 0:
                    continue
                row = {"frame": frame_count, "face_id": face.face_id}
                for name, analyzer in analyzers:
                    r = analyzer.analyze(face.crop, face)
                    dashboard.record(name, r.score, r.elapsed_ms)
                    row[name] = round(r.score, 1)
                    row[f"{name}_ms"] = round(r.elapsed_ms, 2)

                # Frame analyzers
                for name, analyzer in frame_analyzers:
                    r = analyzer.analyze(frame)
                    dashboard.record(name, r.score, r.elapsed_ms)
                    row[name] = round(r.score, 1)
                    row[f"{name}_ms"] = round(r.elapsed_ms, 2)

                pipeline_ms = (time.perf_counter() - t0) * 1000
                row["pipeline_ms"] = round(pipeline_ms, 1)

                if csv_writer:
                    csv_writer.writerow(row)

            pipeline_ms = (time.perf_counter() - t0) * 1000

            # Draw bounding boxes
            for face in tracked:
                b = face.bbox
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{face.face_id}", (b.x1, b.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw dashboard
            dashboard.draw(frame, len(tracked), pipeline_ms)

            window_name = "Spoof Detector Diagnostic"
            if frame_count == 1:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        if csv_file:
            csv_file.close()
            print(f"  CSV log saved")


def main():
    parser = argparse.ArgumentParser(description="Spoof Detector Diagnostic")
    parser.add_argument("--image", type=str, help="Analyze single image")
    parser.add_argument("--log-csv", action="store_true", help="Log diagnostics to CSV")
    args = parser.parse_args()

    if args.image:
        analyze_single_image(args.image)
    else:
        run_live_diagnostic(log_csv=args.log_csv)


if __name__ == "__main__":
    main()
