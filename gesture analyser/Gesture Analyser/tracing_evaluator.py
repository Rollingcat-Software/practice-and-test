"""
Tracing Evaluator -- Testing & Evaluation Module
=================================================
Provides structured logging, attack simulation, and threshold optimisation
for the Dynamic Shape Tracing liveness challenge.

Components
----------
AttemptLog          -- structured record of one tracing attempt
compute_attempt_metrics() -- derives all numeric metrics from a raw path pair
TracingEvaluator    -- append-only logger; CSV + JSON persistence; live stats
StaticAttackSimulator  -- generates near-static paths (frozen-hand / photo attack)
RandomAttackSimulator  -- generates random paths (brute-force attack)
ThresholdOptimizer  -- analyses logged attempts, computes FAR/FRR, suggests EER
"""

import csv
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from shape_tracer import (
    _resample, _centroid_normalise, dtw_normalised_cost,
    ShapeTemplate, DEFAULT_RESAMPLE_N, DEFAULT_DTW_THRESH,
)


# ── Attempt record ───────────────────────────────────────────────────

@dataclass
class AttemptLog:
    """One complete tracing attempt, fully self-describing."""

    session_id:       str    # UUID for the evaluation session
    timestamp:        str    # ISO-8601 wall-clock time
    target_shape:     str    # "CIRCLE" / "SQUARE" / "TRIANGLE" / "S_CURVE"
    attack_type:      str    # "HUMAN" / "STATIC_ATTACK" / "RANDOM_ATTACK"
    result:           str    # "VERIFIED" / "FAILED"

    # Core metrics
    similarity_score:   float  # 0-100 % (100 = perfect match)
    dtw_cost:           float  # raw normalised per-point DTW cost
    coordinate_drift:   float  # mean point-to-point dist after resampling
    time_taken:         float  # seconds from first point to verification

    # Auxiliary info
    point_count:        int    # number of points in the user/simulated path
    dtw_threshold_used: float  # threshold at time of this attempt


# ── Metrics helper ───────────────────────────────────────────────────

def compute_attempt_metrics(
    user_path:       list[tuple[float, float]],
    template:        ShapeTemplate,
    time_taken:      float,
    dtw_threshold:   float = DEFAULT_DTW_THRESH,
    resample_n:      int   = DEFAULT_RESAMPLE_N,
) -> dict:
    """Compute all numeric metrics for one (user_path, template) pair.

    Returns a dict with keys matching ``AttemptLog`` fields:
    ``dtw_cost``, ``similarity_score``, ``coordinate_drift``, ``result``.

    Parameters
    ----------
    user_path : list of (x, y)
        Raw recorded or simulated path in normalised screen coords.
    template : ShapeTemplate
        The target shape the user was supposed to trace.
    time_taken : float
        Elapsed drawing time in seconds.
    dtw_threshold : float
        Acceptance threshold for this attempt.
    resample_n : int
        Number of resampled points for DTW comparison.
    """
    if len(user_path) < 2:
        return {
            "dtw_cost":          float("inf"),
            "similarity_score":  0.0,
            "coordinate_drift":  float("inf"),
            "result":            "FAILED",
        }

    t_rs = _resample(list(template.waypoints), resample_n)
    u_rs = _resample(user_path,                resample_n)

    # ── DTW in centroid-normalised space ────────────────────────────
    t_norm = _centroid_normalise(t_rs)
    u_norm = _centroid_normalise(u_rs)
    dtw_cost   = dtw_normalised_cost(t_norm, u_norm)
    similarity = max(0.0, min(1.0, 1.0 - dtw_cost / dtw_threshold))

    # ── Coordinate drift: fixed-point alignment (no warping) ────────
    # Mean Euclidean distance between corresponding resampled points
    # in centroid-normalised space so the metric is scale-invariant.
    drift = float(np.mean([
        math.sqrt((t_norm[i][0] - u_norm[i][0]) ** 2 +
                  (t_norm[i][1] - u_norm[i][1]) ** 2)
        for i in range(resample_n)
    ]))

    return {
        "dtw_cost":          dtw_cost,
        "similarity_score":  similarity * 100.0,
        "coordinate_drift":  drift,
        "result":            "VERIFIED" if dtw_cost <= dtw_threshold else "FAILED",
    }


# ── Logger ───────────────────────────────────────────────────────────

class TracingEvaluator:
    """Append-only evaluator that logs every attempt and tracks live stats.

    Data is flushed to ``<log_dir>/attempts.csv`` and
    ``<log_dir>/attempts.json`` after each recorded attempt, so no data is
    lost even if the process crashes.

    Parameters
    ----------
    log_dir : str | Path
        Directory where log files are written (created if absent).
    session_id : str | None
        Fixed session UUID.  Auto-generated if None.
    dtw_threshold : float
        Threshold used to classify VERIFIED / FAILED.
    """

    def __init__(
        self,
        log_dir:       str | Path = "eval_logs",
        session_id:    Optional[str] = None,
        dtw_threshold: float = DEFAULT_DTW_THRESH,
    ):
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id    = session_id or str(uuid.uuid4())[:8]
        self.dtw_threshold = dtw_threshold

        self._logs: list[AttemptLog] = []

        self._csv_path  = self.log_dir / "attempts.csv"
        self._json_path = self.log_dir / "attempts.json"

        # Write CSV header if file is new
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                writer.writeheader()

    # -- Logging ---------------------------------------------------------

    def record(
        self,
        user_path:    list[tuple[float, float]],
        template:     ShapeTemplate,
        time_taken:   float,
        attack_type:  str = "HUMAN",
    ) -> AttemptLog:
        """Compute metrics, build a log entry, persist, and return it."""
        metrics = compute_attempt_metrics(
            user_path, template, time_taken, self.dtw_threshold,
        )
        log = AttemptLog(
            session_id       = self.session_id,
            timestamp        = time.strftime("%Y-%m-%dT%H:%M:%S"),
            target_shape     = template.shape_type.name,
            attack_type      = attack_type,
            result           = metrics["result"],
            similarity_score = round(metrics["similarity_score"], 2),
            dtw_cost         = round(metrics["dtw_cost"], 4),
            coordinate_drift = round(metrics["coordinate_drift"], 4),
            time_taken       = round(time_taken, 2),
            point_count      = len(user_path),
            dtw_threshold_used = self.dtw_threshold,
        )
        self._logs.append(log)
        self._flush(log)
        return log

    def _flush(self, log: AttemptLog) -> None:
        """Append one row to CSV and rewrite JSON atomically."""
        # CSV: append single row
        with self._csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(asdict(log))
        # JSON: rewrite full list (small enough for typical eval sessions)
        with self._json_path.open("w") as f:
            json.dump([asdict(lg) for lg in self._logs], f, indent=2)

    @staticmethod
    def _fieldnames() -> list[str]:
        return list(AttemptLog.__dataclass_fields__.keys())

    # -- Query helpers ---------------------------------------------------

    def get_last_n(self, n: int = 50) -> list[AttemptLog]:
        return self._logs[-n:]

    @property
    def all_logs(self) -> list[AttemptLog]:
        return list(self._logs)

    # -- Live stats ------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Aggregate statistics over the entire session."""
        logs       = self._logs
        humans     = [lg for lg in logs if lg.attack_type == "HUMAN"]
        attacks    = [lg for lg in logs if lg.attack_type != "HUMAN"]
        h_pass     = [lg for lg in humans  if lg.result == "VERIFIED"]
        a_pass     = [lg for lg in attacks if lg.result == "VERIFIED"]  # false accepts
        h_fail     = [lg for lg in humans  if lg.result == "FAILED"]    # false rejects

        def _avg(seq, attr):
            vals = [getattr(x, attr) for x in seq]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "total":           len(logs),
            "human_total":     len(humans),
            "attack_total":    len(attacks),
            "human_pass":      len(h_pass),
            "human_fail":      len(h_fail),
            "attack_pass":     len(a_pass),        # FAR numerator
            "far":             len(a_pass) / max(len(attacks), 1),
            "frr":             len(h_fail) / max(len(humans), 1),
            "avg_similarity":  _avg(humans, "similarity_score"),
            "avg_dtw_cost":    _avg(humans, "dtw_cost"),
            "avg_drift":       _avg(humans, "coordinate_drift"),
            "avg_time":        _avg(humans, "time_taken"),
            "log_path":        str(self._csv_path.resolve()),
        }


# ── Attack simulators ────────────────────────────────────────────────

class StaticAttackSimulator:
    """Generates a near-static path to simulate a frozen-hand / photo attack.

    The attacker places their (printed / static) hand at a single point and
    barely moves.  After centroid-normalisation, all points collapse to ≈ 0,
    giving a very high DTW cost against any meaningful shape.

    Parameters
    ----------
    center_x, center_y : float
        Approximate screen position (default: frame centre).
    jitter : float
        Gaussian std-dev of random noise added to each point.
        Default 0.003 (~0.3 % of frame width) — plausible for a
        slightly shaky static image.
    """

    def __init__(
        self,
        center_x: float = 0.50,
        center_y: float = 0.50,
        jitter:   float = 0.003,
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.jitter   = jitter

    def generate_path(self, n_points: int = 60) -> list[tuple[float, float]]:
        """Return a near-static path of *n_points* with Gaussian jitter."""
        return [
            (self.center_x + random.gauss(0, self.jitter),
             self.center_y + random.gauss(0, self.jitter))
            for _ in range(n_points)
        ]


class RandomAttackSimulator:
    """Generates a uniformly random path to simulate a brute-force attack.

    Random motion across the draw region has no correlation with any target
    shape, so DTW cost is typically very high.  In rare cases a random path
    may partially match a simple shape — these become the FAR data-points.

    Parameters
    ----------
    x_range, y_range : tuple(float, float)
        Screen region sampled uniformly for each point.
    smooth_steps : int
        Number of exponential-moving-average smoothing steps applied to
        avoid teleporting jumps that would never occur in a real hand.
    """

    def __init__(
        self,
        x_range:      tuple[float, float] = (0.25, 0.75),
        y_range:      tuple[float, float] = (0.20, 0.80),
        smooth_steps: int = 3,
    ):
        self.x_range      = x_range
        self.y_range      = y_range
        self.smooth_steps = smooth_steps

    def generate_path(self, n_points: int = 60) -> list[tuple[float, float]]:
        """Return a smoothed random path of *n_points*."""
        raw = [
            (random.uniform(*self.x_range), random.uniform(*self.y_range))
            for _ in range(n_points)
        ]
        # Light smoothing: running EMA so consecutive points are closer
        alpha  = 0.35
        smooth = [raw[0]]
        for i in range(1, len(raw)):
            sx = alpha * raw[i][0] + (1 - alpha) * smooth[-1][0]
            sy = alpha * raw[i][1] + (1 - alpha) * smooth[-1][1]
            smooth.append((sx, sy))
        return smooth


# ── Threshold optimiser ──────────────────────────────────────────────

class ThresholdOptimizer:
    """Analyse logged attempts and find the optimal DTW acceptance threshold.

    Uses a grid-search over candidate thresholds to minimise the sum
    ``FAR + FRR`` (total error rate), and also reports the Equal Error
    Rate (EER) point where FAR ≈ FRR.

    Parameters
    ----------
    threshold_min, threshold_max, threshold_step : float
        Grid of candidate thresholds to evaluate.
    """

    def __init__(
        self,
        threshold_min:  float = 0.05,
        threshold_max:  float = 0.80,
        threshold_step: float = 0.01,
    ):
        self.threshold_min  = threshold_min
        self.threshold_max  = threshold_max
        self.threshold_step = threshold_step

    def analyse(self, logs: list[AttemptLog]) -> dict:
        """Return a full analysis report over the given log entries.

        The report dict contains:
          thresholds      -- array of evaluated thresholds
          far_curve       -- FAR at each threshold
          frr_curve       -- FRR at each threshold
          total_err_curve -- FAR + FRR at each threshold
          optimal_threshold -- threshold minimising total error
          eer_threshold     -- threshold closest to FAR == FRR
          far_at_optimal    -- FAR at the optimal threshold
          frr_at_optimal    -- FRR at the optimal threshold
          human_count       -- number of human attempts analysed
          attack_count      -- number of attack attempts analysed
          table             -- list of dicts for tabular display
        """
        humans  = [lg for lg in logs if lg.attack_type == "HUMAN"]
        attacks = [lg for lg in logs if lg.attack_type != "HUMAN"]

        if not humans and not attacks:
            return {"error": "No log entries to analyse."}

        thresholds = np.arange(
            self.threshold_min, self.threshold_max + self.threshold_step / 2,
            self.threshold_step,
        )
        far_curve = []
        frr_curve = []
        ter_curve = []

        for th in thresholds:
            # False Accept: attack cost ≤ threshold (would be wrongly VERIFIED)
            fa  = sum(1 for lg in attacks if lg.dtw_cost <= th)
            far = fa / max(len(attacks), 1)
            # False Reject: human cost > threshold (would be wrongly FAILED)
            fr  = sum(1 for lg in humans if lg.dtw_cost > th)
            frr = fr / max(len(humans), 1)
            far_curve.append(far)
            frr_curve.append(frr)
            ter_curve.append(far + frr)

        # Best threshold: minimum total error
        best_idx          = int(np.argmin(ter_curve))
        optimal_threshold = float(thresholds[best_idx])
        far_at_opt        = far_curve[best_idx]
        frr_at_opt        = frr_curve[best_idx]

        # EER: smallest |FAR - FRR|
        eer_idx = int(np.argmin(np.abs(np.array(far_curve) - np.array(frr_curve))))
        eer_th  = float(thresholds[eer_idx])

        # Build human-readable table (every 0.05 step)
        table = []
        for i, th in enumerate(thresholds):
            if abs(th % 0.05) < self.threshold_step / 2:
                table.append({
                    "threshold": round(float(th), 3),
                    "far_pct":   round(far_curve[i] * 100, 1),
                    "frr_pct":   round(frr_curve[i] * 100, 1),
                    "total_err": round(ter_curve[i] * 100, 1),
                    "is_optimal": (i == best_idx),
                })

        return {
            "thresholds":         thresholds.tolist(),
            "far_curve":          far_curve,
            "frr_curve":          frr_curve,
            "total_err_curve":    ter_curve,
            "optimal_threshold":  optimal_threshold,
            "eer_threshold":      eer_th,
            "far_at_optimal":     far_at_opt,
            "frr_at_optimal":     frr_at_opt,
            "human_count":        len(humans),
            "attack_count":       len(attacks),
            "table":              table,
        }

    def print_report(self, logs: list[AttemptLog]) -> None:
        """Print a formatted threshold-optimisation report to stdout."""
        rep = self.analyse(logs)
        if "error" in rep:
            print(f"[ThresholdOptimizer] {rep['error']}")
            return

        print("\n" + "=" * 60)
        print("  THRESHOLD OPTIMISER REPORT")
        print("=" * 60)
        print(f"  Analysed {rep['human_count']} human attempts "
              f"+ {rep['attack_count']} attack attempts\n")
        print(f"  {'Threshold':>10} | {'FAR %':>6} | {'FRR %':>6} | {'Total %':>8}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
        for row in rep["table"]:
            marker = "  <-- OPTIMAL" if row["is_optimal"] else ""
            print(f"  {row['threshold']:>10.3f} | {row['far_pct']:>6.1f} | "
                  f"{row['frr_pct']:>6.1f} | {row['total_err']:>8.1f}{marker}")
        print()
        print(f"  Optimal threshold : {rep['optimal_threshold']:.3f}")
        print(f"    FAR at optimal  : {rep['far_at_optimal']*100:.1f}%")
        print(f"    FRR at optimal  : {rep['frr_at_optimal']*100:.1f}%")
        print(f"  EER threshold     : {rep['eer_threshold']:.3f}")
        print("=" * 60 + "\n")
