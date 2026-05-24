"""Multi-class spoof fusion engine.

Combines per-analyzer scores into a probability distribution
over the 7-category spoof taxonomy.

Evidence-based weighting: analyzers with proven discrimination
power get higher weights. Analyzers that are anti-correlated
(score spoofs higher than real) get near-zero weight.

Quality-aware: when image_quality analyzer reports poor conditions
(darkness, blur, low contrast), analyzer weights are scaled down
by quality_weight. This prevents garbage-in-garbage-out — e.g.
MiniFASNet drops from 95→1 in darkness, producing false SPOOF.

Calibration data (from analyze_captures.py ground truth test):
  MiniFASNet:       real=99.9  spoof=5.1   gap=+94.7  GOOD
  screen_replay:    real=46.7  spoof=37.1  gap=+9.6   WEAK
  device_boundary:  (new, untested — expected HIGH)
  moire:            real=39.1  spoof=44.1  gap=-5.0   ANTI-CORRELATED
  texture:          real=72.1  spoof=78.4  gap=-6.3   ANTI-CORRELATED
  temporal:         real=90.0  (single-frame only)     NEUTRAL
"""

from __future__ import annotations

import logging

from src.domain.models import SpoofCategory, SpoofClassification, AnalyzerResult
from src.domain.taxonomy import SPOOF_SIGNAL_MAP

logger = logging.getLogger(__name__)

# Calibrated weights based on measured discrimination power.
# Higher weight = analyzer score has more influence on final classification.
DEFAULT_ANALYZER_WEIGHTS: dict[str, float] = {
    "minifasnet": 5.0,          # PROVEN: +94.7 gap
    "screen_flicker": 3.0,     # NEW: 50/60Hz temporal detection — catches ANY screen
    "device_boundary": 2.5,    # GOOD: physical bezel detection
    "micro_tremor": 2.5,       # NEW: 8-12Hz oscillation — catches video replay
    "landmark_variance": 2.0,  # STRONG: zero variance = photo
    "background_grid": 1.5,    # NEW: background stability for proctoring
    "rppg": 1.5,               # RE-ENABLED: notch filters for 50/60Hz (needs field test)
    "blink": 0.5,              # MODERATE: blink count
    "blink_rhythm": 1.0,       # NEW: video loop detection via blink periodicity
    "screen_replay": 0.5,      # WEAK: +9.6 gap
    "ar_filter": 0.3,          # MODERATE: heuristic mode
    "temporal": 0.3,           # NEUTRAL: micro-motion
    "texture": 0.1,            # ANTI-CORRELATED: suppressed
    "moire": 0.1,              # ANTI-CORRELATED: suppressed
}

# Analyzers that are robust to poor lighting / quality.
# These keep their full weight even when quality is low.
# All others get scaled by quality_weight.
QUALITY_ROBUST_ANALYZERS = frozenset({
    "image_quality",       # The quality analyzer itself
    "device_boundary",     # Edge/contour detection still works in dim light
    "blink_rhythm",        # Statistical — independent of image quality
})


class MultiClassFuser:
    """Fuses analyzer scores into per-category probabilities.

    For each analyzer:
    - A HIGH score (close to 100) means "live-like" → increases P(REAL)
    - A LOW score (close to 0) means "spoof-like" → distributes evidence
      across spoof categories based on SPOOF_SIGNAL_MAP weights

    Quality-aware: when image_quality reports poor conditions,
    non-robust analyzers have their weights scaled down. This
    prevents darkness/blur from producing false SPOOF verdicts.

    Analyzer weights are calibrated from ground-truth testing.
    """

    def __init__(self, analyzer_weights: dict[str, float] | None = None):
        self._weights = analyzer_weights or DEFAULT_ANALYZER_WEIGHTS

    def fuse(
        self,
        face_id: int,
        results: dict[str, AnalyzerResult],
    ) -> SpoofClassification:
        evidence: dict[SpoofCategory, float] = {cat: 0.0 for cat in SpoofCategory}
        total_weight = 0.0

        # Extract quality weight if image_quality analyzer ran
        quality_weight = 1.0
        quality_result = results.get("image_quality")
        if quality_result is not None:
            quality_weight = quality_result.details.get("quality_weight", 1.0)
            quality_grade = quality_result.details.get("quality_grade", "?")
            if quality_weight < 0.5:
                logger.info(
                    f"Low image quality (grade={quality_grade}, weight={quality_weight:.2f}) "
                    f"— scaling down non-robust analyzers"
                )

        for analyzer_name, result in results.items():
            # image_quality is informational, not a spoof signal
            if analyzer_name == "image_quality":
                continue

            weight = self._weights.get(analyzer_name, 0.5)
            if weight <= 0:
                continue

            # Apply quality scaling to non-robust analyzers
            if analyzer_name not in QUALITY_ROBUST_ANALYZERS:
                effective_weight = weight * quality_weight
            else:
                effective_weight = weight

            if effective_weight <= 0:
                continue

            total_weight += effective_weight

            score = result.score  # 0-100, higher = more live
            spoof_strength = (100.0 - score) / 100.0  # 0-1, higher = more spoof

            # High score → evidence for REAL
            evidence[SpoofCategory.REAL] += effective_weight * (score / 100.0)

            # Low score → distribute across spoof categories
            if analyzer_name in SPOOF_SIGNAL_MAP:
                category_map = SPOOF_SIGNAL_MAP[analyzer_name]
                for category, cat_weight in category_map.items():
                    evidence[category] += effective_weight * spoof_strength * cat_weight

        # Normalize to probabilities
        if total_weight > 0:
            for cat in evidence:
                evidence[cat] /= total_weight

        return SpoofClassification.from_probabilities(
            face_id=face_id,
            probs=evidence,
            analyzer_results=results,
        )
