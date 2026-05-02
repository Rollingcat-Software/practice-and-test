#!/usr/bin/env python3
"""
Analyze existing captures with ground truth labels.

Re-runs all analyzers on saved captures and generates accuracy report.

Usage:
    python tools/analyze_captures.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2

from src.domain.models import FaceROI, BBox, SpoofCategory, CATEGORY_LABELS
from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser


def main():
    captures_dir = Path("data/captures")
    images = sorted(captures_dir.glob("*.jpg"))

    if not images:
        print("No captures found in data/captures/")
        return

    # Known ground truth from visual inspection
    # Update these labels based on what you actually showed the camera
    GROUND_TRUTH = {
        "capture_20260502_144006_0001.jpg": "real",           # Run 1: your face
        "capture_20260502_144242_0001.jpg": "real",           # Run 2: your face
        "capture_20260502_144257_0002.jpg": "static_image",   # Run 2: photo on screen
        "capture_20260502_145529_0001.jpg": "real",           # Run 3: your live face
        "capture_20260502_145619_0002.jpg": "static_image",   # Run 3: printed photo of children
        "capture_20260502_145746_0003.jpg": "static_image",   # Run 3: phone screen with photos
        "capture_20260502_145752_0004.jpg": "static_image",   # Run 3: phone screen with photos (3 faces)
    }

    print("=" * 70)
    print("  Capture Analysis with Ground Truth Labels")
    print("=" * 70)

    detector = MediaPipeFaceDetector(min_confidence=0.4)
    analyzers = {
        "minifasnet": MiniFASNetAnalyzer(),
        "texture": TextureAnalyzer(),
        "moire": MoireAnalyzer(),
        "screen_replay": ScreenReplayAnalyzer(),
        "device_boundary": DeviceBoundaryAnalyzer(),
    }
    fuser = MultiClassFuser()

    results = []
    for img_path in images:
        fname = img_path.name
        gt = GROUND_TRUTH.get(fname, "unknown")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        faces = detector.detect(img)

        print(f"\n  {fname}")
        print(f"  Ground Truth: {gt}")
        print(f"  Faces found: {len(faces)}")

        if not faces:
            print(f"  (no face detected)")
            continue

        for face in faces:
            crop = face.crop if face.crop is not None else img[face.bbox.y1:face.bbox.y2, face.bbox.x1:face.bbox.x2]
            if crop.size == 0:
                continue

            # Run all analyzers
            analyzer_results = {}
            for name, analyzer in analyzers.items():
                # Pass frame context to analyzers that need it
                if hasattr(analyzer, "set_frame"):
                    analyzer.set_frame(img)
                if name == "screen_replay":
                    r = analyzer.analyze(img)
                else:
                    r = analyzer.analyze(crop, face)
                analyzer_results[name] = r

            cls = fuser.fuse(face.face_id, analyzer_results)

            correct = (gt == "real" and cls.dominant_category == SpoofCategory.REAL) or \
                      (gt != "real" and cls.dominant_category != SpoofCategory.REAL)

            mark = "CORRECT" if correct else "WRONG"
            print(f"    Face #{face.face_id}: {cls.dominant_category.value} {cls.confidence*100:.0f}% [{mark}]")
            print(f"      Analyzers: ", end="")
            for aname, ar in sorted(analyzer_results.items()):
                print(f"{aname}={ar.score:.0f} ", end="")
            print()
            print(f"      Probabilities: ", end="")
            for cat in sorted(cls.probabilities, key=cls.probabilities.get, reverse=True)[:4]:
                print(f"{CATEGORY_LABELS[cat]}={cls.probabilities[cat]*100:.1f}% ", end="")
            print()

            results.append({
                "file": fname,
                "ground_truth": gt,
                "predicted": cls.dominant_category.value,
                "confidence": cls.confidence,
                "correct": correct,
                "p_real": cls.probabilities.get(SpoofCategory.REAL, 0),
                "analyzer_scores": {n: round(r.score, 1) for n, r in analyzer_results.items()},
            })

    # Summary
    if results:
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY: {correct}/{total} correct ({correct/total*100:.0f}%)")
        print(f"{'=' * 70}")

        # Real vs spoof breakdown
        real_r = [r for r in results if r["ground_truth"] == "real"]
        spoof_r = [r for r in results if r["ground_truth"] != "real"]

        if real_r:
            rc = sum(1 for r in real_r if r["correct"])
            avg_p = sum(r["p_real"] for r in real_r) / len(real_r)
            print(f"  Real faces:  {rc}/{len(real_r)} correct, avg P(Real)={avg_p*100:.1f}%")

        if spoof_r:
            sc = sum(1 for r in spoof_r if r["correct"])
            avg_p = sum(r["p_real"] for r in spoof_r) / len(spoof_r)
            print(f"  Spoof faces: {sc}/{len(spoof_r)} correct, avg P(Real)={avg_p*100:.1f}%")

        # Per-analyzer discrimination power
        if real_r and spoof_r:
            print(f"\n  Analyzer Discrimination (Real avg vs Spoof avg):")
            all_analyzers = set()
            for r in results:
                all_analyzers.update(r["analyzer_scores"].keys())

            for aname in sorted(all_analyzers):
                real_avg = sum(r["analyzer_scores"].get(aname, 50) for r in real_r) / len(real_r)
                spoof_avg = sum(r["analyzer_scores"].get(aname, 50) for r in spoof_r) / len(spoof_r)
                gap = real_avg - spoof_avg
                useful = "GOOD" if gap > 15 else "WEAK" if gap > 5 else "USELESS"
                print(f"    {aname:>15s}: real={real_avg:5.1f} spoof={spoof_avg:5.1f} gap={gap:+5.1f} [{useful}]")

    detector.close()


if __name__ == "__main__":
    main()
