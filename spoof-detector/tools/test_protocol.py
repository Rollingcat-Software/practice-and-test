#!/usr/bin/env python3
"""
Labeled Test Protocol Tool
===========================

Guided testing protocol that walks you through each attack type,
captures labeled samples, and generates an accuracy report.

The tool tells you what to show the camera, captures frames with
ground truth labels, then evaluates how the detector performed.

Usage:
    python tools/test_protocol.py              # Full protocol (all 5 scenarios)
    python tools/test_protocol.py --scenario 1 # Single scenario
    python tools/test_protocol.py --report     # Re-analyze existing captures

Scenarios:
    1. REAL         - Your live face, natural movement
    2. STATIC_PRINT - Printed photo held in front of camera
    3. STATIC_SCREEN- Photo displayed on phone/tablet screen
    4. VIDEO_REPLAY - Video playing on phone/tablet screen
    5. MULTI_FACE   - Multiple scenarios in one frame
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np

from src.domain.models import SpoofCategory, CATEGORY_LABELS, CATEGORY_COLORS


# ---- Scenario Definitions ----

SCENARIOS = {
    1: {
        "name": "REAL",
        "label": SpoofCategory.REAL,
        "instruction": "Show your REAL face to the camera. Look naturally, blink, move slightly.",
        "duration_sec": 5,
        "captures": 10,
    },
    2: {
        "name": "STATIC_PRINT",
        "label": SpoofCategory.STATIC_IMAGE,
        "instruction": "Hold a PRINTED PHOTO of a face in front of the camera.",
        "duration_sec": 5,
        "captures": 10,
    },
    3: {
        "name": "STATIC_SCREEN",
        "label": SpoofCategory.STATIC_IMAGE,
        "instruction": "Show a PHOTO on your PHONE/TABLET screen to the camera.",
        "duration_sec": 5,
        "captures": 10,
    },
    4: {
        "name": "VIDEO_REPLAY",
        "label": SpoofCategory.VIDEO_REPLAY,
        "instruction": "Play a VIDEO of a face on your phone/tablet and show it to the camera.",
        "duration_sec": 5,
        "captures": 10,
    },
    5: {
        "name": "MIXED",
        "label": SpoofCategory.STATIC_IMAGE,
        "instruction": "Show your phone screen with photos AND your real face together.",
        "duration_sec": 5,
        "captures": 5,
    },
}


def build_pipeline():
    """Build the full detection pipeline."""
    from src.application.pipeline import SpoofDetectionPipeline
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
    from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
    from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
    from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
    from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser

    return SpoofDetectionPipeline(
        detector=MediaPipeFaceDetector(min_confidence=0.4),
        tracker=FaceTracker(),
        face_analyzers=[
            MiniFASNetAnalyzer(),
            TextureAnalyzer(),
            MoireAnalyzer(),
            TemporalAnalyzer(),
        ],
        frame_analyzers=[ScreenReplayAnalyzer()],
        fuser=MultiClassFuser(),
    )


def run_scenario(scenario_id: int, pipeline, cap, output_dir: Path):
    """Run a single test scenario with guided capture."""
    scenario = SCENARIOS[scenario_id]
    name = scenario["name"]
    label = scenario["label"]
    duration = scenario["duration_sec"]
    n_captures = scenario["captures"]

    print(f"\n{'=' * 60}")
    print(f"  SCENARIO {scenario_id}: {name}")
    print(f"  Ground truth: {label.value}")
    print(f"{'=' * 60}")
    print(f"\n  INSTRUCTION: {scenario['instruction']}")
    print(f"  Duration: {duration}s ({n_captures} captures)")
    print(f"\n  Press SPACE to start, 'q' to skip...")

    # Wait for space
    while True:
        ret, frame = cap.read()
        if not ret:
            return []
        cv2.putText(frame, f"SCENARIO {scenario_id}: {name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, scenario["instruction"], (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press SPACE to start, Q to skip", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        window_name = "Test Protocol"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return []

    # Capture phase
    results = []
    interval = max(1, int(duration * 30 / n_captures))  # frames between captures
    frame_count = 0
    captured = 0
    start_time = time.time()

    print(f"\n  Capturing...")
    while captured < n_captures and (time.time() - start_time) < duration + 2:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run pipeline
        analysis = pipeline.process(frame)

        # Draw countdown
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        cv2.putText(frame, f"RECORDING: {name} [{remaining:.0f}s]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Captured: {captured}/{n_captures}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw boxes
        for face in analysis.faces:
            b = face.bbox
            cls = analysis.classifications.get(face.face_id)
            if cls:
                color = CATEGORY_COLORS.get(cls.dominant_category, (0, 255, 0))
                lbl = f"#{face.face_id} {CATEGORY_LABELS[cls.dominant_category]} {cls.confidence*100:.0f}%"
            else:
                color = (0, 255, 0)
                lbl = f"#{face.face_id}"
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            cv2.putText(frame, lbl, (b.x1, b.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

        # Capture at intervals
        if frame_count % interval == 0 and analysis.faces:
            captured += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = f"proto_{name}_{ts}_{captured:03d}"

            # Save image
            img_path = output_dir / f"{base}.jpg"
            cv2.imwrite(str(img_path), frame)

            # Save metadata with ground truth
            for face in analysis.faces:
                cls = analysis.classifications.get(face.face_id)
                if cls:
                    entry = {
                        "file": str(img_path.name),
                        "scenario": name,
                        "scenario_id": scenario_id,
                        "ground_truth": label.value,
                        "face_id": face.face_id,
                        "predicted_dominant": cls.dominant_category.value,
                        "predicted_confidence": round(cls.confidence, 4),
                        "probabilities": {
                            cat.value: round(prob, 4)
                            for cat, prob in cls.probabilities.items()
                        },
                        "analyzer_scores": {
                            aname: round(ar.score, 2)
                            for aname, ar in cls.analyzer_results.items()
                        },
                        "correct": _is_correct(label, cls.dominant_category, cls.probabilities),
                    }
                    results.append(entry)

            print(f"    [{captured}/{n_captures}] faces={len(analysis.faces)}", end="")
            for face in analysis.faces:
                cls = analysis.classifications.get(face.face_id)
                if cls:
                    mark = "v" if _is_correct(label, cls.dominant_category, cls.probabilities) else "X"
                    print(f"  [{mark}] {cls.dominant_category.value}={cls.confidence*100:.0f}%", end="")
            print()

    print(f"  Done: {captured} captures, {len(results)} face classifications")
    return results


def _is_correct(ground_truth: SpoofCategory, predicted: SpoofCategory,
                probs: dict) -> bool:
    """Check if prediction matches ground truth."""
    if ground_truth == SpoofCategory.REAL:
        return predicted == SpoofCategory.REAL and probs.get(SpoofCategory.REAL, 0) > 0.5
    else:
        # Any spoof category is acceptable for spoof ground truth
        return predicted != SpoofCategory.REAL


def generate_report(results: list, output_dir: Path):
    """Generate accuracy report from labeled results."""
    print(f"\n{'=' * 60}")
    print(f"  ACCURACY REPORT")
    print(f"{'=' * 60}")

    if not results:
        print("  No results to report.")
        return

    # Overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\n  Overall: {correct}/{total} ({correct/total*100:.0f}%)")

    # Per-scenario breakdown
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    print(f"\n  {'Scenario':>15s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>9s}  {'Avg P(GT)':>10s}")
    print(f"  {'-'*55}")

    for scenario_name in sorted(by_scenario.keys()):
        entries = by_scenario[scenario_name]
        n = len(entries)
        c = sum(1 for e in entries if e["correct"])
        gt_label = entries[0]["ground_truth"]

        # Average probability of ground truth category
        avg_gt_prob = sum(
            e["probabilities"].get(gt_label, 0) for e in entries
        ) / n

        print(f"  {scenario_name:>15s}  {c:>5d}/{n:<3d}  {n:>5d}  {c/n*100:7.1f}%  {avg_gt_prob*100:8.1f}%")

    # Per-analyzer score comparison (real vs spoof)
    real_entries = [r for r in results if r["ground_truth"] == "real"]
    spoof_entries = [r for r in results if r["ground_truth"] != "real"]

    if real_entries and spoof_entries:
        print(f"\n  Per-Analyzer Score: Real vs Spoof")
        print(f"  {'Analyzer':>15s}  {'Real Avg':>9s}  {'Spoof Avg':>10s}  {'Gap':>6s}  {'Useful':>7s}")
        print(f"  {'-'*55}")

        all_analyzers = set()
        for r in results:
            all_analyzers.update(r.get("analyzer_scores", {}).keys())

        for analyzer in sorted(all_analyzers):
            real_scores = [r["analyzer_scores"].get(analyzer, 50) for r in real_entries]
            spoof_scores = [r["analyzer_scores"].get(analyzer, 50) for r in spoof_entries]
            real_avg = sum(real_scores) / len(real_scores)
            spoof_avg = sum(spoof_scores) / len(spoof_scores)
            gap = real_avg - spoof_avg
            useful = "YES" if gap > 10 else "WEAK" if gap > 3 else "NO"
            print(f"  {analyzer:>15s}  {real_avg:7.1f}  {spoof_avg:8.1f}  {gap:+5.1f}  {useful:>6s}")

    # Confusion matrix
    print(f"\n  Predicted Distribution:")
    pred_counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        pred_counts[r["ground_truth"]][r["predicted_dominant"]] += 1

    all_cats = sorted(set(r["ground_truth"] for r in results) | set(r["predicted_dominant"] for r in results))
    gt_pred_label = "GT \\ Pred"
    header = f"  {gt_pred_label:>15s}" + "".join(f"  {c:>12s}" for c in all_cats)
    print(header)
    print(f"  {'-' * len(header)}")
    for gt in sorted(pred_counts.keys()):
        row = f"  {gt:>15s}"
        for pred in all_cats:
            count = pred_counts[gt].get(pred, 0)
            row += f"  {count:>12d}"
        print(row)

    # Save report
    report_path = output_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "results": results,
        }, f, indent=2)
    print(f"\n  Report saved: {report_path}")


def report_from_existing(output_dir: Path):
    """Re-analyze existing protocol captures."""
    json_files = sorted(output_dir.glob("report_*.json"))
    if not json_files:
        print("No existing reports found. Run the protocol first.")
        return

    latest = json_files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        data = json.load(f)
    generate_report(data["results"], output_dir)


def main():
    parser = argparse.ArgumentParser(description="Labeled Test Protocol")
    parser.add_argument("--scenario", type=int, help="Run single scenario (1-5)")
    parser.add_argument("--report", action="store_true", help="Re-analyze existing captures")
    args = parser.parse_args()

    output_dir = Path("data/protocol")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report:
        report_from_existing(output_dir)
        return

    print("=" * 60)
    print("  FIVUCSAS Spoof Detector - Labeled Test Protocol")
    print("=" * 60)
    print("\n  This tool guides you through testing each attack type")
    print("  with ground truth labels for accuracy measurement.\n")

    # Build pipeline
    print("  Loading pipeline...")
    pipeline = build_pipeline()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    all_results = []
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    try:
        for sid in scenarios:
            results = run_scenario(sid, pipeline, cap, output_dir)
            all_results.extend(results)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if all_results:
        generate_report(all_results, output_dir)


if __name__ == "__main__":
    main()
