#!/usr/bin/env python3
"""
AR Filter Dataset Collection Tool
===================================

Captures labeled face crops for training the AR filter detector.
Guides the user through recording sessions with and without filters.

Usage:
    python tools/collect_ar_dataset.py --label real       # Record real faces
    python tools/collect_ar_dataset.py --label ar_filter   # Record with AR filter active
    python tools/collect_ar_dataset.py --label snapchat    # Specific filter source
    python tools/collect_ar_dataset.py --list              # Show dataset stats
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np


DATASET_DIR = Path(__file__).parent.parent / "data" / "ar_dataset"
LABELS = ["real", "ar_filter", "snapchat", "instagram", "tiktok", "faceapp", "obs", "other"]


def collect_samples(label: str, n_samples: int = 50, interval_ms: int = 200):
    """Capture labeled face crops from webcam."""
    output_dir = DATASET_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_dir.glob("*.jpg")))
    print(f"\nCollecting '{label}' samples (existing: {existing})")
    print(f"Target: {n_samples} new samples, {interval_ms}ms interval")

    if label == "real":
        print("\nINSTRUCTION: Show your REAL face. Move naturally, blink, look around.")
    else:
        print(f"\nINSTRUCTION: Apply {label.upper()} filter and face the camera.")
        print("Make sure the filter is clearly visible and active.")

    print("\nPress SPACE to start recording, Q to quit.")

    # Setup face detection
    import mediapipe as mp
    detector = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite")
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = False
    captured = 0
    last_capture = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Detect faces
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            # Draw UI
            status = "RECORDING" if recording else "READY"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"[{status}] Label: {label} | Captured: {captured}/{n_samples}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for det in result.detections:
                bb = det.bounding_box
                cv2.rectangle(frame, (bb.origin_x, bb.origin_y),
                              (bb.origin_x + bb.width, bb.origin_y + bb.height), color, 2)

            cv2.namedWindow("AR Dataset Collector", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("AR Dataset Collector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("AR Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                if recording:
                    print("  Recording started...")

            # Capture faces at interval
            if recording and result.detections:
                now = time.time() * 1000
                if now - last_capture >= interval_ms:
                    for det in result.detections:
                        bb = det.bounding_box
                        x1 = max(0, bb.origin_x)
                        y1 = max(0, bb.origin_y)
                        x2 = min(w, bb.origin_x + bb.width)
                        y2 = min(h, bb.origin_y + bb.height)
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        # Resize to standard size
                        crop_resized = cv2.resize(crop, (224, 224))

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        img_path = output_dir / f"{label}_{ts}.jpg"
                        cv2.imwrite(str(img_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        captured += 1
                        last_capture = now

                        if captured % 10 == 0:
                            print(f"  Captured {captured}/{n_samples}")

                    if captured >= n_samples:
                        print(f"\n  Done! {captured} samples saved to {output_dir}")
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    # Save metadata
    meta = {
        "label": label,
        "samples_collected": captured,
        "timestamp": datetime.now().isoformat(),
        "total_in_dir": len(list(output_dir.glob("*.jpg"))),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def show_dataset_stats():
    """Show current dataset statistics."""
    print("\nAR Filter Dataset Statistics")
    print("=" * 50)

    if not DATASET_DIR.exists():
        print("  No dataset directory found.")
        return

    total = 0
    for label_dir in sorted(DATASET_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        count = len(list(label_dir.glob("*.jpg")))
        total += count
        print(f"  {label_dir.name:>15s}: {count:>5d} samples")

    print(f"  {'TOTAL':>15s}: {total:>5d} samples")


def main():
    parser = argparse.ArgumentParser(description="AR Filter Dataset Collector")
    parser.add_argument("--label", type=str, choices=LABELS, help="Label for collected samples")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to collect")
    parser.add_argument("--interval", type=int, default=200, help="Capture interval in ms")
    parser.add_argument("--list", action="store_true", help="Show dataset stats")
    args = parser.parse_args()

    if args.list:
        show_dataset_stats()
    elif args.label:
        collect_samples(args.label, args.samples, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
