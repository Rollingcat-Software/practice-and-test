#!/usr/bin/env python3
"""
ISO 30107-3 Evaluation Metrics
================================

Computes APCER, BPCER, ACER from test protocol reports.
These are the standard metrics for face presentation attack detection.

Definitions (ISO 30107-3):
  APCER: Attack Presentation Classification Error Rate
         = proportion of attack presentations incorrectly classified as real
  BPCER: Bona Fide Presentation Classification Error Rate
         = proportion of real presentations incorrectly classified as attack
  ACER:  Average Classification Error Rate = (APCER + BPCER) / 2

Usage:
    python tools/evaluate.py                                    # Latest report
    python tools/evaluate.py --report data/protocol/report.json # Specific report
    python tools/evaluate.py --all                              # All reports
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_metrics(results: list[dict]) -> dict:
    """Compute APCER, BPCER, ACER from labeled results."""
    # Separate real (bona fide) and attack presentations
    bona_fide = [r for r in results if r["ground_truth"] == "real"]
    attacks = [r for r in results if r["ground_truth"] != "real"]

    if not bona_fide or not attacks:
        return {"error": "Need both real and attack samples"}

    # BPCER: real classified as attack
    bpcer_errors = sum(1 for r in bona_fide if not r["correct"])
    bpcer = bpcer_errors / len(bona_fide)

    # APCER: attack classified as real (per attack type for ISO compliance)
    attack_types = defaultdict(list)
    for r in attacks:
        attack_types[r["ground_truth"]].append(r)

    apcer_per_type = {}
    for attack_type, samples in attack_types.items():
        errors = sum(1 for r in samples if not r["correct"])
        apcer_per_type[attack_type] = errors / len(samples)

    # Overall APCER = max over attack types (ISO 30107-3 definition)
    apcer = max(apcer_per_type.values()) if apcer_per_type else 0.0

    # ACER
    acer = (apcer + bpcer) / 2.0

    return {
        "n_bona_fide": len(bona_fide),
        "n_attacks": len(attacks),
        "BPCER": round(bpcer, 4),
        "APCER": round(apcer, 4),
        "ACER": round(acer, 4),
        "APCER_per_type": {k: round(v, 4) for k, v in apcer_per_type.items()},
        "bpcer_errors": bpcer_errors,
        "apcer_errors_total": sum(1 for r in attacks if not r["correct"]),
    }


def analyze_report(report_path: Path):
    """Analyze a single report file."""
    with open(report_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"  No results in {report_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"  ISO 30107-3 Evaluation Report")
    print(f"  Source: {report_path.name}")
    print(f"  Date: {data.get('timestamp', 'unknown')}")
    print(f"{'=' * 60}")

    metrics = compute_metrics(results)

    if "error" in metrics:
        print(f"  Error: {metrics['error']}")
        return

    print(f"\n  Samples: {metrics['n_bona_fide']} bona fide + {metrics['n_attacks']} attacks")
    print(f"\n  STANDARD METRICS:")
    print(f"  {'BPCER':>10s}: {metrics['BPCER']:.2%}  ({metrics['bpcer_errors']} real misclassified as attack)")
    print(f"  {'APCER':>10s}: {metrics['APCER']:.2%}  (worst attack type pass-through rate)")
    print(f"  {'ACER':>10s}: {metrics['ACER']:.2%}  (average of APCER + BPCER)")

    print(f"\n  APCER per attack type:")
    for attack_type, rate in sorted(metrics["APCER_per_type"].items()):
        status = "PASS" if rate < 0.10 else "WARN" if rate < 0.30 else "FAIL"
        print(f"    {attack_type:>15s}: {rate:.2%}  [{status}]")

    # Grade
    acer = metrics["ACER"]
    if acer < 0.05:
        grade = "A (Excellent)"
    elif acer < 0.10:
        grade = "B (Good)"
    elif acer < 0.20:
        grade = "C (Acceptable)"
    elif acer < 0.30:
        grade = "D (Poor)"
    else:
        grade = "F (Failing)"

    print(f"\n  Overall Grade: {grade}")
    print(f"  (ACER < 5%: A, < 10%: B, < 20%: C, < 30%: D, >= 30%: F)")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="ISO 30107-3 Evaluation")
    parser.add_argument("--report", type=str, help="Specific report file")
    parser.add_argument("--all", action="store_true", help="Evaluate all reports")
    args = parser.parse_args()

    protocol_dir = Path("data/protocol")

    if args.report:
        analyze_report(Path(args.report))
    elif args.all:
        reports = sorted(protocol_dir.glob("report_*.json"))
        if not reports:
            print("No reports found. Run: python tools/test_protocol.py")
            return
        for rp in reports:
            analyze_report(rp)
    else:
        # Latest report
        reports = sorted(protocol_dir.glob("report_*.json"))
        if not reports:
            print("No reports found. Run: python tools/test_protocol.py")
            return
        analyze_report(reports[-1])


if __name__ == "__main__":
    main()
