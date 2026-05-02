"""Multi-class spoof fusion engine.

Combines per-analyzer scores into a probability distribution
over the 7-category spoof taxonomy.

Evidence-based weighting: analyzers with proven discrimination
power get higher weights. Analyzers that are anti-correlated
(score spoofs higher than real) get near-zero weight.

Calibration data (from analyze_captures.py ground truth test):
  MiniFASNet:       real=99.9  spoof=5.1   gap=+94.7  GOOD
  screen_replay:    real=46.7  spoof=37.1  gap=+9.6   WEAK
  device_boundary:  (new, untested — expected HIGH)
  moire:            real=39.1  spoof=44.1  gap=-5.0   ANTI-CORRELATED
  texture:          real=72.1  spoof=78.4  gap=-6.3   ANTI-CORRELATED
  temporal:         real=90.0  (single-frame only)     NEUTRAL
"""

from __future__ import annotations

from src.domain.models import SpoofCategory, SpoofClassification, AnalyzerResult
from src.domain.taxonomy import SPOOF_SIGNAL_MAP

# Calibrated weights based on measured discrimination power.
# Higher weight = analyzer score has more influence on final classification.
DEFAULT_ANALYZER_WEIGHTS: dict[str, float] = {
    "minifasnet": 5.0,        # PROVEN: +94.7 gap — must dominate fusion
    "device_boundary": 2.5,   # GOOD: physical bezel detection
    "blink": 0.5,             # UNVALIDATED: false blinks on spoofs, 0 on real — needs calibration
    "rppg": 0.0,              # ANTI-CORRELATED: detects screen flicker as pulse — disabled
    "ar_filter": 0.3,         # MODERATE: heuristic mode
    "screen_replay": 0.5,     # WEAK: +9.6 gap
    "temporal": 0.3,          # NEUTRAL: micro-motion naturalness
    "texture": 0.1,           # ANTI-CORRELATED: suppressed
    "moire": 0.1,             # ANTI-CORRELATED: suppressed
    "landmark_variance": 2.0, # STRONG: zero variance = photo, natural = live
}


class MultiClassFuser:
    """Fuses analyzer scores into per-category probabilities.

    For each analyzer:
    - A HIGH score (close to 100) means "live-like" → increases P(REAL)
    - A LOW score (close to 0) means "spoof-like" → distributes evidence
      across spoof categories based on SPOOF_SIGNAL_MAP weights

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

        for analyzer_name, result in results.items():
            weight = self._weights.get(analyzer_name, 0.5)
            if weight <= 0:
                continue
            total_weight += weight

            score = result.score  # 0-100, higher = more live
            spoof_strength = (100.0 - score) / 100.0  # 0-1, higher = more spoof

            # High score → evidence for REAL
            evidence[SpoofCategory.REAL] += weight * (score / 100.0)

            # Low score → distribute across spoof categories
            if analyzer_name in SPOOF_SIGNAL_MAP:
                category_map = SPOOF_SIGNAL_MAP[analyzer_name]
                for category, cat_weight in category_map.items():
                    evidence[category] += weight * spoof_strength * cat_weight

        # Normalize to probabilities
        if total_weight > 0:
            for cat in evidence:
                evidence[cat] /= total_weight

        return SpoofClassification.from_probabilities(
            face_id=face_id,
            probs=evidence,
            analyzer_results=results,
        )
