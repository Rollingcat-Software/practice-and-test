"""
Threshold Optimizer -- Standalone CLI Utility
=============================================
Reads the evaluation log CSV produced by TracingEvaluator and suggests
an optimal DTW acceptance threshold that balances:

  FAR  (False Acceptance Rate) -- attacks wrongly verified
  FRR  (False Rejection Rate)  -- legitimate users wrongly rejected

Usage
-----
  python threshold_optimizer.py                          # default log path
  python threshold_optimizer.py eval_logs/attempts.csv  # explicit path
  python threshold_optimizer.py --last 30               # analyse last N rows
  python threshold_optimizer.py --plot                  # show FAR/FRR curves (requires matplotlib)

No mediapipe or OpenCV dependency -- pure CSV analysis.
"""

import argparse
import csv
import sys
from pathlib import Path


# ── CSV reader (no mediapipe) ─────────────────────────────────────────

def _load_logs(csv_path: Path, last_n: int = 0) -> list[dict]:
    """Load attempt rows from the evaluator CSV as plain dicts."""
    if not csv_path.exists():
        print(f"[ERROR] Log file not found: {csv_path}")
        sys.exit(1)

    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "attack_type":    row["attack_type"],
                    "result":         row["result"],
                    "dtw_cost":       float(row["dtw_cost"]),
                    "similarity_score": float(row["similarity_score"]),
                    "coordinate_drift": float(row["coordinate_drift"]),
                    "time_taken":     float(row["time_taken"]),
                    "target_shape":   row["target_shape"],
                })
            except (KeyError, ValueError):
                continue  # skip malformed rows

    if last_n > 0:
        rows = rows[-last_n:]
    return rows


# ── Grid-search optimiser ─────────────────────────────────────────────

def _optimise(rows: list[dict], th_min=0.05, th_max=0.80, th_step=0.01) -> dict:
    """Grid-search over thresholds; returns analysis dict."""
    import numpy as np

    humans  = [r for r in rows if r["attack_type"] == "HUMAN"]
    attacks = [r for r in rows if r["attack_type"] != "HUMAN"]

    if not humans and not attacks:
        return {"error": "No rows to analyse."}

    thresholds = list(_frange(th_min, th_max, th_step))
    far_curve, frr_curve, ter_curve = [], [], []

    for th in thresholds:
        fa  = sum(1 for r in attacks if r["dtw_cost"] <= th)
        fr  = sum(1 for r in humans  if r["dtw_cost"] >  th)
        far = fa / max(len(attacks), 1)
        frr = fr / max(len(humans),  1)
        far_curve.append(far)
        frr_curve.append(frr)
        ter_curve.append(far + frr)

    best_idx = ter_curve.index(min(ter_curve))
    eer_idx  = min(range(len(thresholds)),
                   key=lambda i: abs(far_curve[i] - frr_curve[i]))

    # Per-shape breakdown for humans
    shape_stats: dict[str, dict] = {}
    for r in humans:
        sh = r["target_shape"]
        if sh not in shape_stats:
            shape_stats[sh] = {"total": 0, "pass": 0, "dtw_costs": []}
        shape_stats[sh]["total"] += 1
        if r["result"] == "VERIFIED":
            shape_stats[sh]["pass"] += 1
        shape_stats[sh]["dtw_costs"].append(r["dtw_cost"])

    return {
        "thresholds":        thresholds,
        "far_curve":         far_curve,
        "frr_curve":         frr_curve,
        "ter_curve":         ter_curve,
        "optimal_threshold": thresholds[best_idx],
        "far_at_optimal":    far_curve[best_idx],
        "frr_at_optimal":    frr_curve[best_idx],
        "eer_threshold":     thresholds[eer_idx],
        "human_count":       len(humans),
        "attack_count":      len(attacks),
        "shape_stats":       shape_stats,
    }


def _frange(start, stop, step):
    x = start
    while x <= stop + step / 2:
        yield round(x, 6)
        x += step


# ── Report printer ────────────────────────────────────────────────────

def _print_report(rep: dict, current_threshold: float = 0.25) -> None:
    if "error" in rep:
        print(f"[ThresholdOptimizer] {rep['error']}")
        return

    W = 62
    print("\n" + "=" * W)
    print("  THRESHOLD OPTIMISER REPORT")
    print("=" * W)
    print(f"  Analysed : {rep['human_count']} human attempts  "
          f"+ {rep['attack_count']} attack attempts\n")

    # Table header
    print(f"  {'Threshold':>10} | {'FAR %':>6} | {'FRR %':>6} | "
          f"{'Total %':>8} | Note")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*12}")

    shown = set()
    for i, th in enumerate(rep["thresholds"]):
        th_r = round(th, 2)
        if th_r not in shown and abs(th % 0.05) < 0.006:
            shown.add(th_r)
            notes = []
            if abs(th - rep["optimal_threshold"]) < 0.006:
                notes.append("OPTIMAL")
            if abs(th - rep["eer_threshold"]) < 0.006:
                notes.append("EER")
            if abs(th - current_threshold) < 0.006:
                notes.append("current")
            far = rep["far_curve"][i] * 100
            frr = rep["frr_curve"][i] * 100
            ter = rep["ter_curve"][i] * 100
            print(f"  {th:>10.3f} | {far:>6.1f} | {frr:>6.1f} | "
                  f"{ter:>8.1f} | {', '.join(notes)}")

    print()
    print(f"  Optimal threshold  : {rep['optimal_threshold']:.3f}")
    print(f"    FAR at optimal   : {rep['far_at_optimal']*100:.1f}%  "
          f"(attacks falsely accepted)")
    print(f"    FRR at optimal   : {rep['frr_at_optimal']*100:.1f}%  "
          f"(humans falsely rejected)")
    print(f"  Equal Error Rate   : threshold = {rep['eer_threshold']:.3f}")

    if rep.get("shape_stats"):
        print(f"\n  Per-shape breakdown (human attempts):")
        print(f"  {'Shape':>10} | {'Total':>6} | {'Pass':>5} | "
              f"{'Pass %':>7} | {'Avg DTW':>8}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}")
        for sh, st in sorted(rep["shape_stats"].items()):
            avg_dtw = (sum(st["dtw_costs"]) / len(st["dtw_costs"])
                       if st["dtw_costs"] else 0)
            pct = st["pass"] / st["total"] * 100 if st["total"] else 0
            print(f"  {sh:>10} | {st['total']:>6} | {st['pass']:>5} | "
                  f"{pct:>7.1f} | {avg_dtw:>8.4f}")

    print("=" * W + "\n")

    print(f"  RECOMMENDATION: set dtw_threshold = {rep['optimal_threshold']:.2f}")
    print(f"  (update ShapeTracerSession(dtw_threshold={rep['optimal_threshold']:.2f}))")
    print()


# ── Optional matplotlib plot ─────────────────────────────────────────

def _plot(rep: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ThresholdOptimizer] matplotlib not installed; skipping plot.")
        return

    ths = rep["thresholds"]
    plt.figure(figsize=(9, 5))
    plt.plot(ths, [v * 100 for v in rep["far_curve"]], "r-",  label="FAR %", linewidth=2)
    plt.plot(ths, [v * 100 for v in rep["frr_curve"]], "b-",  label="FRR %", linewidth=2)
    plt.plot(ths, [v * 100 for v in rep["ter_curve"]], "g--", label="Total Error %", linewidth=1.5)

    plt.axvline(rep["optimal_threshold"], color="green",  linestyle=":", label=f"Optimal ({rep['optimal_threshold']:.3f})")
    plt.axvline(rep["eer_threshold"],     color="purple", linestyle=":", label=f"EER ({rep['eer_threshold']:.3f})")

    plt.xlabel("DTW Threshold")
    plt.ylabel("Error Rate (%)")
    plt.title("FAR / FRR vs DTW Threshold\n(Shape Tracing Evaluator)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── CLI entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest an optimal DTW threshold from evaluation logs."
    )
    parser.add_argument(
        "csv_path", nargs="?",
        default="eval_logs/attempts.csv",
        help="Path to the attempts CSV (default: eval_logs/attempts.csv)",
    )
    parser.add_argument(
        "--last", type=int, default=0,
        metavar="N",
        help="Analyse only the last N rows (0 = all)",
    )
    parser.add_argument(
        "--current-threshold", type=float, default=0.25,
        metavar="T",
        help="Current threshold to highlight in the table (default: 0.25)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show FAR/FRR curves with matplotlib (if installed)",
    )
    args = parser.parse_args()

    rows = _load_logs(Path(args.csv_path), last_n=args.last)
    if not rows:
        print("[ThresholdOptimizer] No rows loaded. Exiting.")
        sys.exit(1)

    print(f"[ThresholdOptimizer] Loaded {len(rows)} rows from '{args.csv_path}'.")
    rep = _optimise(rows)
    _print_report(rep, current_threshold=args.current_threshold)

    if args.plot:
        _plot(rep)


if __name__ == "__main__":
    main()
