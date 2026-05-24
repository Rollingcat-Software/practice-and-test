#!/usr/bin/env python3
"""
Labeled Test Protocol Tool
===========================

Guided testing protocol that walks you through each attack type,
captures labeled samples, and generates an accuracy report with
ISO 30107-3 compliant metrics and automated grading.

The tool tells you what to show the camera, captures frames with
ground truth labels, then evaluates how the detector performed.

Usage:
    python tools/test_protocol.py                  # Full protocol (all 5 scenarios)
    python tools/test_protocol.py --scenario 1     # Single scenario
    python tools/test_protocol.py --report         # Re-analyze existing captures
    python tools/test_protocol.py --evaluate-only  # Re-run pipeline on saved images

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
    """Build the full detection pipeline with all enabled analyzers.

    Reads config.yaml and mirrors main.py's build_pipeline() registration
    logic, but skips the threaded camera (the protocol drives its own capture).
    """
    import yaml

    from src.application.pipeline import SpoofDetectionPipeline
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
    from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
    from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
    from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
    from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
    from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
    from src.infrastructure.analyzers.blink_analyzer import BlinkAnalyzer
    from src.infrastructure.analyzers.rppg_analyzer import RPPGAnalyzer
    from src.infrastructure.analyzers.ar_filter_analyzer import ARFilterAnalyzer
    from src.infrastructure.analyzers.landmark_variance_analyzer import LandmarkVarianceAnalyzer
    from src.infrastructure.analyzers.screen_flicker_analyzer import ScreenFlickerAnalyzer
    from src.infrastructure.analyzers.micro_tremor_analyzer import MicroTremorAnalyzer
    from src.infrastructure.analyzers.background_grid_analyzer import BackgroundGridAnalyzer
    from src.infrastructure.analyzers.blink_rhythm_analyzer import BlinkRhythmAnalyzer
    from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
    from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ana_cfg = config.get("analyzers", {})
    fus_cfg = config.get("fusion", {})

    # image_quality runs FIRST to gate unreliable analyzers in poor conditions
    face_analyzers = [ImageQualityAnalyzer()]
    if ana_cfg.get("minifasnet", {}).get("enabled", True):
        face_analyzers.append(MiniFASNetAnalyzer())
    if ana_cfg.get("texture", {}).get("enabled", True):
        tex_cfg = ana_cfg.get("texture", {})
        face_analyzers.append(TextureAnalyzer(
            texture_threshold=tex_cfg.get("laplacian_threshold", 100.0),
            fft_downsample=tuple(tex_cfg.get("fft_downsample", [192, 108])),
        ))
    if ana_cfg.get("moire", {}).get("enabled", True):
        moire_cfg = ana_cfg.get("moire", {})
        face_analyzers.append(MoireAnalyzer(
            response_std_threshold=moire_cfg.get("response_std_threshold", 30.0),
        ))
    if ana_cfg.get("temporal", {}).get("enabled", True):
        tmp_cfg = ana_cfg.get("temporal", {})
        face_analyzers.append(TemporalAnalyzer(
            buffer_size=tmp_cfg.get("buffer_size", 30),
            min_motion_std=tmp_cfg.get("min_motion_std", 0.0003),
        ))
    if ana_cfg.get("device_boundary", {}).get("enabled", True):
        db_cfg = ana_cfg.get("device_boundary", {})
        face_analyzers.append(DeviceBoundaryAnalyzer(
            padding_ratio=db_cfg.get("padding_ratio", 0.55),
            spoof_threshold=db_cfg.get("spoof_threshold", 0.50),
        ))
    if ana_cfg.get("blink", {}).get("enabled", True):
        face_analyzers.append(BlinkAnalyzer())
    if ana_cfg.get("rppg", {}).get("enabled", True):
        face_analyzers.append(RPPGAnalyzer())
    if ana_cfg.get("ar_filter", {}).get("enabled", True):
        ar_cfg = ana_cfg.get("ar_filter", {})
        face_analyzers.append(ARFilterAnalyzer(model_path=ar_cfg.get("model_path")))
    if ana_cfg.get("landmark_variance", {}).get("enabled", True):
        face_analyzers.append(LandmarkVarianceAnalyzer())
    if ana_cfg.get("screen_flicker", {}).get("enabled", True):
        face_analyzers.append(ScreenFlickerAnalyzer())
    if ana_cfg.get("micro_tremor", {}).get("enabled", True):
        face_analyzers.append(MicroTremorAnalyzer())
    if ana_cfg.get("background_grid", {}).get("enabled", True):
        face_analyzers.append(BackgroundGridAnalyzer())
    if ana_cfg.get("blink_rhythm", {}).get("enabled", True):
        blink_rhythm = BlinkRhythmAnalyzer()
        for a in face_analyzers:
            if isinstance(a, BlinkAnalyzer):
                blink_rhythm.set_blink_analyzer(a)
                break
        face_analyzers.append(blink_rhythm)

    frame_analyzers = []
    if ana_cfg.get("screen_replay", {}).get("enabled", True):
        frame_analyzers.append(ScreenReplayAnalyzer())

    return SpoofDetectionPipeline(
        detector=MediaPipeFaceDetector(min_confidence=0.4),
        tracker=FaceTracker(),
        face_analyzers=face_analyzers,
        frame_analyzers=frame_analyzers,
        fuser=MultiClassFuser(
            analyzer_weights=fus_cfg.get("weights") if fus_cfg.get("weights") else None,
        ),
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
        try:
            import ctypes
            _u32 = ctypes.windll.user32
            _u32.SetProcessDPIAware()
            _sw, _sh = _u32.GetSystemMetrics(0), _u32.GetSystemMetrics(1)
        except Exception:
            _sw, _sh = 0, 0
        if _sw > 0:
            frame = cv2.resize(frame, (_sw, _sh), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return []

    # Capture phase
    results = []
    interval = max(1, int(duration * 15 / n_captures))  # adjusted for ~15fps
    frame_count = 0
    captured = 0
    start_time = time.time()
    last_analysis = None

    print(f"\n  Capturing...")
    while captured < n_captures and (time.time() - start_time) < duration + 2:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run pipeline every 2nd frame to maintain UI responsiveness
        if frame_count % 2 == 0 or last_analysis is None:
            analysis = pipeline.process(frame)
            last_analysis = analysis
        else:
            analysis = last_analysis

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

        if _sw > 0:
            display_frame = cv2.resize(frame, (_sw, _sh), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return results

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
                        "analyzer_details": {
                            aname: ar.details
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


def _pai_species(entry: dict) -> str:
    """Map a result entry to its PAI species for ISO 30107-3 APCER grouping.

    PAI species (Presentation Attack Instrument):
      - print  : printed photo (STATIC_PRINT scenario)
      - screen : digital still on phone/tablet (STATIC_SCREEN scenario)
      - video  : pre-recorded video replay (VIDEO_REPLAY scenario)
      - mixed  : combined real+spoof in frame (MIXED scenario)
    Real entries return 'real' (not a PAI species, filtered out for APCER).
    """
    scenario = entry.get("scenario", "")
    if scenario == "REAL":
        return "real"
    if scenario == "STATIC_PRINT":
        return "print"
    if scenario == "STATIC_SCREEN":
        return "screen"
    if scenario == "VIDEO_REPLAY":
        return "video"
    if scenario == "MIXED":
        return "mixed"
    # Fallback: infer from ground truth
    gt = entry.get("ground_truth", "")
    if gt == "real":
        return "real"
    return "unknown"


def _compute_grade(acer: float) -> tuple[str, str]:
    """Return (grade_letter, grade_label) from ACER percentage.

    Grade A: ACER < 2%  (research-grade)
    Grade B: ACER < 8%  (commercial-grade)
    Grade C: ACER < 20% (baseline)
    Grade D: ACER < 35% (weak)
    Grade F: ACER >= 35% (fail)
    """
    if acer < 2.0:
        return "A", "research-grade"
    if acer < 8.0:
        return "B", "commercial-grade"
    if acer < 20.0:
        return "C", "baseline"
    if acer < 35.0:
        return "D", "weak"
    return "F", "fail"


def _compute_iso_metrics(results: list) -> dict:
    """Compute ISO 30107-3 compliant metrics from labeled results.

    Returns a dict with all metrics for both reporting and JSON export.
    """
    real_entries = [r for r in results if r["ground_truth"] == "real"]
    spoof_entries = [r for r in results if r["ground_truth"] != "real"]

    # ---- BPCER: Bona-fide Presentation Classification Error Rate ----
    # False rejects of genuine users
    total_real = len(real_entries)
    false_rejects = sum(1 for r in real_entries if r["predicted_dominant"] != "real")
    bpcer = (false_rejects / total_real * 100) if total_real > 0 else 0.0

    # ---- APCER per PAI species ----
    # False accepts of attack presentations (predicted as real when ground truth is spoof)
    by_pai = defaultdict(list)
    for r in spoof_entries:
        pai = _pai_species(r)
        by_pai[pai].append(r)

    apcer_per_pai = {}
    for pai, entries in sorted(by_pai.items()):
        total_attacks = len(entries)
        false_accepts = sum(1 for e in entries if e["predicted_dominant"] == "real")
        apcer_val = (false_accepts / total_attacks * 100) if total_attacks > 0 else 0.0
        apcer_per_pai[pai] = {
            "apcer": round(apcer_val, 2),
            "false_accepts": false_accepts,
            "total_attempts": total_attacks,
        }

    # ---- ACER: Average Classification Error Rate ----
    # Uses worst-case APCER (max across PAI species)
    worst_apcer = max((v["apcer"] for v in apcer_per_pai.values()), default=0.0)
    acer = (worst_apcer + bpcer) / 2.0

    grade_letter, grade_label = _compute_grade(acer)

    return {
        "bpcer": round(bpcer, 2),
        "bpcer_detail": {
            "false_rejects": false_rejects,
            "total_real": total_real,
        },
        "apcer_per_pai": apcer_per_pai,
        "worst_apcer": round(worst_apcer, 2),
        "acer": round(acer, 2),
        "grade": grade_letter,
        "grade_label": grade_label,
    }


def _build_confusion_matrix(results: list) -> dict:
    """Build confusion matrix: ground_truth rows x predicted columns."""
    pred_counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        pred_counts[r["ground_truth"]][r["predicted_dominant"]] += 1
    # Convert to plain dict for JSON
    return {gt: dict(preds) for gt, preds in pred_counts.items()}


def _build_analyzer_table(results: list) -> list[dict]:
    """Build per-analyzer discrimination table (real_avg vs spoof_avg, gap, rank)."""
    real_entries = [r for r in results if r["ground_truth"] == "real"]
    spoof_entries = [r for r in results if r["ground_truth"] != "real"]
    if not real_entries or not spoof_entries:
        return []

    all_analyzers = set()
    for r in results:
        all_analyzers.update(r.get("analyzer_scores", {}).keys())

    rows = []
    for analyzer in sorted(all_analyzers):
        real_scores = [r["analyzer_scores"].get(analyzer, 50) for r in real_entries]
        spoof_scores = [r["analyzer_scores"].get(analyzer, 50) for r in spoof_entries]
        real_avg = sum(real_scores) / len(real_scores)
        spoof_avg = sum(spoof_scores) / len(spoof_scores)
        gap = real_avg - spoof_avg
        usefulness = "YES" if gap > 10 else "WEAK" if gap > 3 else "NO"
        rows.append({
            "analyzer": analyzer,
            "real_avg": round(real_avg, 2),
            "spoof_avg": round(spoof_avg, 2),
            "gap": round(gap, 2),
            "useful": usefulness,
        })

    # Rank by absolute gap descending
    rows.sort(key=lambda x: abs(x["gap"]), reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def _build_scenario_breakdown(results: list) -> list[dict]:
    """Per-scenario accuracy breakdown."""
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    rows = []
    for scenario_name in sorted(by_scenario.keys()):
        entries = by_scenario[scenario_name]
        n = len(entries)
        c = sum(1 for e in entries if e["correct"])
        gt_label = entries[0]["ground_truth"]
        avg_gt_prob = sum(e["probabilities"].get(gt_label, 0) for e in entries) / n
        rows.append({
            "scenario": scenario_name,
            "correct": c,
            "total": n,
            "accuracy": round(c / n, 4),
            "avg_gt_prob": round(avg_gt_prob, 4),
            "ground_truth": gt_label,
        })
    return rows


def _build_liveness_stats(results: list) -> list[dict]:
    """Liveness prover statistics: avg P(real) per scenario."""
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    rows = []
    for scenario_name in sorted(by_scenario.keys()):
        entries = by_scenario[scenario_name]
        real_probs = [e["probabilities"].get("real", 0) for e in entries]
        confidences = [e["predicted_confidence"] for e in entries]
        rows.append({
            "scenario": scenario_name,
            "count": len(entries),
            "avg_p_real": round(sum(real_probs) / len(real_probs), 4),
            "min_p_real": round(min(real_probs), 4),
            "max_p_real": round(max(real_probs), 4),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
        })
    return rows


def _build_incident_timeline(results: list) -> list[dict]:
    """List misclassified frames with details."""
    incidents = []
    for r in results:
        if not r["correct"]:
            incidents.append({
                "file": r["file"],
                "scenario": r["scenario"],
                "ground_truth": r["ground_truth"],
                "predicted": r["predicted_dominant"],
                "confidence": r["predicted_confidence"],
                "p_real": r["probabilities"].get("real", 0),
            })
    return incidents


def generate_report(results: list, output_dir: Path):
    """Generate ISO 30107-3 compliant accuracy report from labeled results."""
    print(f"\n{'=' * 70}")
    print(f"  ISO 30107-3 ACCURACY REPORT")
    print(f"{'=' * 70}")

    if not results:
        print("  No results to report.")
        return

    # ---- Overall accuracy ----
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\n  Overall: {correct}/{total} ({correct/total*100:.1f}%)")

    # ---- Image quality summary ----
    quality_scores = [
        r.get("analyzer_scores", {}).get("image_quality", None)
        for r in results
    ]
    quality_scores = [q for q in quality_scores if q is not None]
    if quality_scores:
        avg_q = sum(quality_scores) / len(quality_scores)
        min_q = min(quality_scores)
        max_q = max(quality_scores)
        # Get quality details from first result that has them
        sample_details = None
        for r in results:
            ar = r.get("analyzer_details", {}).get("image_quality")
            if ar:
                sample_details = ar
                break
        grade = "A" if avg_q >= 80 else "B" if avg_q >= 60 else "C" if avg_q >= 40 else "D" if avg_q >= 20 else "F"
        print(f"\n  {'-' * 50}")
        print(f"  IMAGE QUALITY")
        print(f"  {'-' * 50}")
        print(f"  Avg Quality Score:  {avg_q:5.1f}/100  (Grade {grade})")
        print(f"  Range:              {min_q:.1f} - {max_q:.1f}")
        if sample_details:
            print(f"  Brightness:         {sample_details.get('brightness', '?')}")
            print(f"  Contrast:           {sample_details.get('contrast', '?')}")
            print(f"  Sharpness:          {sample_details.get('sharpness', '?')}")
        if avg_q < 40:
            print(f"\n  *** WARNING: Poor image quality (grade {grade}). ***")
            print(f"  *** Results unreliable — improve lighting/focus before testing. ***")
        elif avg_q < 60:
            print(f"\n  * Note: Moderate quality (grade {grade}). Some analyzers may be degraded.")

    # ---- ISO 30107-3 Metrics ----
    iso = _compute_iso_metrics(results)

    print(f"\n  {'-' * 50}")
    print(f"  ISO 30107-3 METRICS")
    print(f"  {'-' * 50}")
    print(f"  BPCER (False Reject Rate):  {iso['bpcer']:6.2f}%  "
          f"({iso['bpcer_detail']['false_rejects']}/{iso['bpcer_detail']['total_real']})")
    print(f"  {'-' * 50}")
    print(f"  APCER per PAI Species:")
    print(f"    {'PAI Species':>12s}  {'APCER':>7s}  {'False Accept':>13s}  {'Total':>6s}")
    for pai, data in sorted(iso["apcer_per_pai"].items()):
        print(f"    {pai:>12s}  {data['apcer']:6.2f}%  {data['false_accepts']:>10d}     {data['total_attempts']:>5d}")
    print(f"  {'-' * 50}")
    print(f"  Worst APCER:                {iso['worst_apcer']:6.2f}%")
    print(f"  ACER (worst + BPCER) / 2:   {iso['acer']:6.2f}%")
    print(f"  {'-' * 50}")
    grade_bar = "=" * max(1, int(50 * (1 - iso['acer'] / 50)))
    print(f"  GRADE: {iso['grade']}  ({iso['grade_label']})")
    print(f"  [{grade_bar:50s}]  ACER={iso['acer']:.2f}%")

    # Grade thresholds reference
    print(f"\n  Grade thresholds: A < 2% | B < 8% | C < 20% | D < 35% | F >= 35%")

    # ---- Per-scenario breakdown ----
    scenario_rows = _build_scenario_breakdown(results)
    print(f"\n  {'-' * 50}")
    print(f"  PER-SCENARIO BREAKDOWN")
    print(f"  {'-' * 50}")
    print(f"  {'Scenario':>15s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>9s}  {'Avg P(GT)':>10s}")
    print(f"  {'-' * 55}")
    for row in scenario_rows:
        c, n = row["correct"], row["total"]
        print(f"  {row['scenario']:>15s}  {c:>5d}/{n:<3d}  {n:>5d}  "
              f"{row['accuracy']*100:7.1f}%  {row['avg_gt_prob']*100:8.1f}%")

    # ---- Per-analyzer discrimination table ----
    analyzer_rows = _build_analyzer_table(results)
    if analyzer_rows:
        print(f"\n  {'-' * 50}")
        print(f"  PER-ANALYZER DISCRIMINATION (ranked by |gap|)")
        print(f"  {'-' * 50}")
        print(f"  {'#':>3s}  {'Analyzer':>15s}  {'Real Avg':>9s}  {'Spoof Avg':>10s}  {'Gap':>7s}  {'Useful':>7s}")
        print(f"  {'-' * 58}")
        for row in analyzer_rows:
            print(f"  {row['rank']:>3d}  {row['analyzer']:>15s}  {row['real_avg']:7.1f}  "
                  f"{row['spoof_avg']:8.1f}  {row['gap']:+6.1f}  {row['useful']:>6s}")

    # ---- Confusion matrix ----
    confusion = _build_confusion_matrix(results)
    all_cats = sorted(
        set(r["ground_truth"] for r in results) | set(r["predicted_dominant"] for r in results)
    )
    print(f"\n  {'-' * 50}")
    print(f"  CONFUSION MATRIX (rows=GT, cols=Predicted)")
    print(f"  {'-' * 50}")
    gt_pred_label = "GT \\ Pred"
    header = f"  {gt_pred_label:>15s}" + "".join(f"  {c:>12s}" for c in all_cats)
    print(header)
    print(f"  {'-' * len(header)}")
    for gt in sorted(confusion.keys()):
        row_str = f"  {gt:>15s}"
        for pred in all_cats:
            count = confusion[gt].get(pred, 0)
            row_str += f"  {count:>12d}"
        print(row_str)

    # ---- Liveness prover statistics ----
    liveness_rows = _build_liveness_stats(results)
    print(f"\n  {'-' * 50}")
    print(f"  LIVENESS PROVER STATISTICS")
    print(f"  {'-' * 50}")
    print(f"  {'Scenario':>15s}  {'Count':>6s}  {'Avg P(R)':>9s}  {'Min P(R)':>9s}  "
          f"{'Max P(R)':>9s}  {'Avg Conf':>9s}")
    print(f"  {'-' * 65}")
    for row in liveness_rows:
        print(f"  {row['scenario']:>15s}  {row['count']:>5d}  "
              f"{row['avg_p_real']*100:7.1f}%  {row['min_p_real']*100:7.1f}%  "
              f"{row['max_p_real']*100:7.1f}%  {row['avg_confidence']*100:7.1f}%")

    # ---- Incident timeline (misclassifications) ----
    incidents = _build_incident_timeline(results)
    if incidents:
        print(f"\n  {'-' * 50}")
        print(f"  INCIDENT TIMELINE ({len(incidents)} misclassification(s))")
        print(f"  {'-' * 50}")
        for inc in incidents:
            print(f"  [{inc['scenario']:>14s}] {inc['file']}")
            print(f"    GT={inc['ground_truth']:<14s} Pred={inc['predicted']:<14s} "
                  f"Conf={inc['confidence']*100:.0f}%  P(real)={inc['p_real']*100:.1f}%")

    # ---- Build full JSON report ----
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "protocol_version": "2.0-ISO30107",
        "total_frames": total,
        "correct_frames": correct,
        "accuracy": round(correct / total, 4),
        "iso_30107_3": {
            "bpcer_pct": iso["bpcer"],
            "bpcer_false_rejects": iso["bpcer_detail"]["false_rejects"],
            "bpcer_total_real": iso["bpcer_detail"]["total_real"],
            "apcer_per_pai": iso["apcer_per_pai"],
            "worst_apcer_pct": iso["worst_apcer"],
            "acer_pct": iso["acer"],
        },
        "grade": iso["grade"],
        "grade_label": iso["grade_label"],
        "scenario_breakdown": scenario_rows,
        "analyzer_discrimination": analyzer_rows,
        "confusion_matrix": confusion,
        "liveness_stats": liveness_rows,
        "incidents": incidents,
        "results": results,
    }

    report_path = output_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # ---- Summary line ----
    print(f"\n{'=' * 70}")
    print(f"  GRADE {iso['grade']} | ACER {iso['acer']:.2f}% | "
          f"BPCER {iso['bpcer']:.2f}% | Worst APCER {iso['worst_apcer']:.2f}% | "
          f"{iso['grade_label']}")
    print(f"{'=' * 70}")


def report_from_existing(output_dir: Path):
    """Re-analyze existing protocol captures using saved JSON report."""
    json_files = sorted(output_dir.glob("report_*.json"))
    if not json_files:
        print("No existing reports found. Run the protocol first.")
        return

    latest = json_files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        data = json.load(f)
    generate_report(data["results"], output_dir)


def evaluate_existing_captures(output_dir: Path):
    """Re-run the detection pipeline on saved images (no camera needed).

    Reads all proto_*.jpg files, matches them to scenarios by filename,
    re-runs the pipeline, and generates a fresh report.
    """
    print("=" * 70)
    print("  EVALUATE-ONLY: Re-analyzing saved captures")
    print("=" * 70)

    image_files = sorted(output_dir.glob("proto_*.jpg"))
    if not image_files:
        print("  No saved captures found in", output_dir)
        return

    print(f"  Found {len(image_files)} saved images")
    print("  Loading pipeline...")
    pipeline = build_pipeline()

    # Map filename prefix to scenario
    prefix_to_scenario = {}
    for sid, scenario in SCENARIOS.items():
        prefix_to_scenario[scenario["name"]] = (sid, scenario)

    all_results = []
    processed = 0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARN: Cannot read {img_path.name}, skipping")
            continue

        # Parse scenario from filename: proto_{SCENARIO}_{timestamp}_{seq}.jpg
        parts = img_path.stem.split("_")
        # Find scenario name (may contain underscores like STATIC_PRINT)
        scenario_name = None
        for prefix in prefix_to_scenario:
            if f"proto_{prefix}_" in img_path.stem:
                scenario_name = prefix
                break

        if scenario_name is None:
            print(f"  WARN: Cannot identify scenario for {img_path.name}, skipping")
            continue

        sid, scenario = prefix_to_scenario[scenario_name]
        label = scenario["label"]

        # Run pipeline
        analysis = pipeline.process(frame)
        processed += 1

        for face in analysis.faces:
            cls = analysis.classifications.get(face.face_id)
            if cls:
                entry = {
                    "file": str(img_path.name),
                    "scenario": scenario_name,
                    "scenario_id": sid,
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
                    "analyzer_details": {
                        aname: ar.details
                        for aname, ar in cls.analyzer_results.items()
                    },
                    "correct": _is_correct(label, cls.dominant_category, cls.probabilities),
                }
                all_results.append(entry)

        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(image_files)} images...")

    print(f"  Processed {processed} images, {len(all_results)} face classifications")

    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("  No face detections found in saved images.")


def main():
    parser = argparse.ArgumentParser(description="Labeled Test Protocol")
    parser.add_argument("--scenario", type=int, help="Run single scenario (1-5)")
    parser.add_argument("--report", action="store_true",
                        help="Re-analyze existing report JSON (no pipeline, no camera)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Re-run pipeline on saved images (no camera needed)")
    args = parser.parse_args()

    output_dir = Path("data/protocol")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report:
        report_from_existing(output_dir)
        return

    if args.evaluate_only:
        evaluate_existing_captures(output_dir)
        return

    print("=" * 70)
    print("  FIVUCSAS Spoof Detector - Labeled Test Protocol")
    print("=" * 70)
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
