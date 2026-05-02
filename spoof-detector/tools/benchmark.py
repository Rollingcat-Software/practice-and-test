#!/usr/bin/env python3
"""
Comprehensive Benchmark & Diagnostic Tool
==========================================

Tests every analyzer individually and the full pipeline against:
1. Synthetic images (solid, noisy, gradient, moire patterns)
2. Real face images from DeepFacePractice1 dataset
3. Your live camera capture (saved frame from first run)

Produces a detailed performance and accuracy report.

Usage:
    python tools/benchmark.py                    # Full benchmark
    python tools/benchmark.py --quick            # Quick (skip slow tests)
    python tools/benchmark.py --analyzer moire   # Single analyzer
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np

from src.domain.models import FaceROI, BBox, SpoofCategory, AnalyzerResult


# --- Test Image Generators ------------------------------------

def make_solid(w=200, h=200, color=(128, 128, 128)):
    """Solid color — should flag as suspicious (no texture)."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_noise(w=200, h=200, seed=42):
    """Random noise — natural texture profile."""
    return np.random.RandomState(seed).randint(50, 200, (h, w, 3), dtype=np.uint8)


def make_gradient(w=200, h=200):
    """Horizontal gradient — moderate texture."""
    g = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.merge([g, g, g])


def make_moire(w=200, h=200, freq=0.3):
    """Strong periodic moire pattern — should flag screen replay."""
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    p = ((np.sin(freq * x) * np.sin(freq * y) + 1) * 127).astype(np.uint8)
    return cv2.merge([p, p, p])


def make_skin_tone(w=200, h=200):
    """Skin-tone colored image — simulates face color without texture."""
    # Average Caucasian skin in BGR
    return np.full((h, w, 3), (140, 170, 210), dtype=np.uint8)


def make_high_specular(w=200, h=200):
    """Image with bright specular highlights — simulates screen glare."""
    img = make_noise(w, h)
    # Add bright spots
    for _ in range(20):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
    return img


# --- Benchmark Infrastructure ---------------------------------

@dataclass
class BenchmarkResult:
    test_name: str
    image_desc: str
    analyzer_name: str
    score: float
    elapsed_ms: float
    details: dict = field(default_factory=dict)
    expected_behavior: str = ""
    passed: bool = True


def make_roi(w=200, h=200):
    return FaceROI(face_id=1, bbox=BBox(0, 0, w, h), confidence=0.95)


def run_analyzer_suite(analyzer, name: str, is_frame_analyzer: bool = False, verbose: bool = True) -> list[BenchmarkResult]:
    """Run a single analyzer against all synthetic test images."""
    results = []

    test_cases = [
        ("solid_gray", make_solid(), "Low score expected (no texture)"),
        ("solid_black", make_solid(color=(0, 0, 0)), "Low score expected (no content)"),
        ("solid_white", make_solid(color=(255, 255, 255)), "Low score expected (no content)"),
        ("random_noise", make_noise(), "Moderate-high score (natural texture)"),
        ("gradient", make_gradient(), "Moderate score"),
        ("moire_low", make_moire(freq=0.1), "Moire: should detect periodic pattern"),
        ("moire_high", make_moire(freq=0.5), "Moire: strong periodic pattern"),
        ("skin_tone", make_skin_tone(), "Low score (no texture, skin color)"),
        ("specular", make_high_specular(), "Specular highlights (screen-like)"),
    ]

    for test_name, img, expected in test_cases:
        h, w = img.shape[:2]
        roi = make_roi(w, h)

        # Run 3 times, take median for stable timing
        times = []
        result = None
        for _ in range(3):
            if is_frame_analyzer:
                r = analyzer.analyze(img)
            else:
                r = analyzer.analyze(img, roi)
            times.append(r.elapsed_ms)
            result = r

        median_ms = sorted(times)[1]
        br = BenchmarkResult(
            test_name=test_name,
            image_desc=f"{w}x{h}",
            analyzer_name=name,
            score=result.score,
            elapsed_ms=median_ms,
            details=result.details,
            expected_behavior=expected,
        )
        results.append(br)

        if verbose:
            status = "OK" if median_ms < 30 else "SLOW"
            print(f"  {test_name:>18s}  score={result.score:6.1f}  {median_ms:6.2f}ms  [{status}]  {expected}")

    return results


def run_on_real_images(analyzer, name: str, images_dir: Path, is_frame_analyzer: bool = False, verbose: bool = True) -> list[BenchmarkResult]:
    """Run analyzer on real face images from DeepFacePractice1."""
    results = []
    image_files = sorted(images_dir.rglob("*.jpg"))[:12]  # Max 12 images

    if not image_files:
        if verbose:
            print(f"  No images found in {images_dir}")
        return results

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        # Resize to reasonable face crop size
        if max(h, w) > 300:
            scale = 300 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        roi = make_roi(w, h)
        if is_frame_analyzer:
            result = analyzer.analyze(img)
        else:
            result = analyzer.analyze(img, roi)

        br = BenchmarkResult(
            test_name=img_path.name,
            image_desc=f"{w}x{h} real",
            analyzer_name=name,
            score=result.score,
            elapsed_ms=result.elapsed_ms,
            details=result.details,
            expected_behavior="Real face → high score expected",
        )
        results.append(br)

        if verbose:
            print(f"  {img_path.parent.name}/{img_path.name:>12s}  score={result.score:6.1f}  {result.elapsed_ms:6.2f}ms")

    return results


def run_size_scaling(analyzer, name: str, is_frame_analyzer: bool = False, verbose: bool = True) -> list[BenchmarkResult]:
    """Test how performance scales with image size."""
    results = []
    sizes = [(100, 100), (200, 200), (300, 300), (480, 480), (640, 640)]

    for w, h in sizes:
        img = make_noise(w, h)
        roi = make_roi(w, h)

        times = []
        for _ in range(5):
            if is_frame_analyzer:
                r = analyzer.analyze(img)
            else:
                r = analyzer.analyze(img, roi)
            times.append(r.elapsed_ms)
        median_ms = sorted(times)[2]

        br = BenchmarkResult(
            test_name=f"scale_{w}x{h}",
            image_desc=f"{w}x{h}",
            analyzer_name=name,
            score=r.score,
            elapsed_ms=median_ms,
        )
        results.append(br)

        if verbose:
            print(f"  {w:4d}x{h:<4d}  {median_ms:6.2f}ms")

    return results


# --- Main -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spoof Detector Benchmark")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--analyzer", type=str, help="Test single analyzer (minifasnet,texture,moire,screen_replay,temporal)")
    parser.add_argument("--no-real", action="store_true", help="Skip real image tests")
    args = parser.parse_args()

    print("=" * 70)
    print("  FIVUCSAS Spoof Detector — Comprehensive Benchmark")
    print("=" * 70)

    # Find real images
    images_dir = Path(__file__).parent.parent.parent / "DeepFacePractice1" / "images"
    capture_dir = Path(__file__).parent.parent / "data" / "captures"

    all_results: list[BenchmarkResult] = []

    # Build analyzers
    analyzers = {}
    target = args.analyzer

    if not target or target == "texture":
        from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
        analyzers["texture"] = TextureAnalyzer()

    if not target or target == "moire":
        from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
        analyzers["moire"] = MoireAnalyzer()

    if not target or target == "screen_replay":
        from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
        analyzers["screen_replay"] = ScreenReplayAnalyzer()

    if not target or target == "temporal":
        from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
        analyzers["temporal"] = TemporalAnalyzer()

    if (not target or target == "minifasnet") and not args.quick:
        from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
        analyzers["minifasnet"] = MiniFASNetAnalyzer()

    # Track which analyzers are frame-level (not per-face)
    frame_analyzers = {"screen_replay"}

    for name, analyzer in analyzers.items():
        is_frame = name in frame_analyzers
        print(f"\n{'-' * 70}")
        print(f"  Analyzer: {name} {'(frame-level)' if is_frame else '(per-face)'}")
        print(f"{'-' * 70}")

        # Synthetic tests
        print(f"\n  [Synthetic Images]")
        results = run_analyzer_suite(analyzer, name, is_frame_analyzer=is_frame)
        all_results.extend(results)

        # Size scaling
        print(f"\n  [Size Scaling]")
        results = run_size_scaling(analyzer, name, is_frame_analyzer=is_frame)
        all_results.extend(results)

        # Real images
        if not args.no_real and images_dir.exists():
            print(f"\n  [Real Face Images — DeepFacePractice1]")
            results = run_on_real_images(analyzer, name, images_dir, is_frame_analyzer=is_frame)
            all_results.extend(results)

        # Saved capture
        if not args.no_real:
            captures = sorted(capture_dir.glob("*.jpg"))
            if captures:
                print(f"\n  [Saved Captures]")
                for cap in captures[:3]:
                    img = cv2.imread(str(cap))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    roi = make_roi(w, h)
                    if is_frame:
                        r = analyzer.analyze(img)
                    else:
                        r = analyzer.analyze(img, roi)
                    print(f"  {cap.name}  score={r.score:6.1f}  {r.elapsed_ms:6.2f}ms")

    # --- Summary ----------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    # Per-analyzer timing summary
    by_analyzer: dict[str, list[float]] = {}
    for r in all_results:
        by_analyzer.setdefault(r.analyzer_name, []).append(r.elapsed_ms)

    print(f"\n  {'Analyzer':>15s}  {'Min':>8s}  {'Avg':>8s}  {'Max':>8s}  {'P95':>8s}  Tests")
    print(f"  {'-' * 65}")
    for name, times in sorted(by_analyzer.items()):
        times_sorted = sorted(times)
        p95 = times_sorted[int(len(times_sorted) * 0.95)] if len(times_sorted) > 1 else times_sorted[0]
        avg = sum(times) / len(times)
        print(f"  {name:>15s}  {min(times):7.2f}ms  {avg:7.2f}ms  {max(times):7.2f}ms  {p95:7.2f}ms  {len(times):>4d}")

    total_avg = sum(sum(t) for t in by_analyzer.values()) / max(sum(len(t) for t in by_analyzer.values()), 1)
    print(f"\n  Estimated per-frame (1 face): ~{total_avg:.1f}ms per analyzer")
    print(f"  Total analyzers active: {len(by_analyzer)}")
    print(f"  Estimated pipeline: ~{total_avg * len(by_analyzer):.0f}ms + detection overhead")

    # Score distribution for real images
    real_scores: dict[str, list[float]] = {}
    for r in all_results:
        if "real" in r.image_desc:
            real_scores.setdefault(r.analyzer_name, []).append(r.score)

    if real_scores:
        print(f"\n  Score Distribution on Real Faces:")
        print(f"  {'Analyzer':>15s}  {'Min':>6s}  {'Avg':>6s}  {'Max':>6s}")
        print(f"  {'-' * 45}")
        for name, scores in sorted(real_scores.items()):
            avg = sum(scores) / len(scores)
            print(f"  {name:>15s}  {min(scores):5.1f}  {avg:5.1f}  {max(scores):5.1f}")

    print(f"\n{'=' * 70}")
    print("  Benchmark complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
