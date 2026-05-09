"""Hybrid fusion evaluator for liveness and spoof signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class FusionWeights:
    """Weights for combining model and heuristic spoof signals."""

    pretrained_model: float = 0.30
    flash_response: float = 0.30
    moire_pattern: float = 0.20
    device_replay: float = 0.20

    def __post_init__(self) -> None:
        total = sum(
            (
                self.pretrained_model,
                self.flash_response,
                self.moire_pattern,
                self.device_replay,
            )
        )
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class FusionResult:
    """Hybrid fusion decision output."""

    is_spoof: bool
    confidence: float
    spoof_score: float
    breakdown: dict[str, float]
    reasoning: str


class HybridFusionEvaluator:
    """Fuse pretrained liveness output with replay- and physiology-based signals."""

    def __init__(self, weights: Optional[FusionWeights] = None, threshold: float = 0.45) -> None:
        self.weights = weights or FusionWeights()
        self.threshold = float(threshold)

    def evaluate(
        self,
        pretrained_spoof_score: float,
        custom_signals: dict[str, Any],
    ) -> FusionResult:
        """Combine all signals into a final spoof decision."""
        pretrained_score = self._clamp01(pretrained_spoof_score)
        flicker_score = self._resolve_numeric_signal(
            custom_signals.get("flicker_score"),
            neutral=0.0,
        )
        device_replay_score = self._resolve_numeric_signal(
            custom_signals.get("device_replay_score"),
            neutral=0.0,
        )
        if flicker_score > 0.85 or (flicker_score > 0.75 and device_replay_score > 0.55):
            reasoning = (
                f"High flicker ({flicker_score:.2f}) + device replay ({device_replay_score:.2f})"
                if flicker_score <= 0.85
                else f"Very high flicker detected ({flicker_score:.2f})"
            )
            return FusionResult(
                is_spoof=True,
                confidence=0.90,
                spoof_score=0.90,
                breakdown={
                    "pretrained": pretrained_score,
                    "flicker": flicker_score,
                    "device_replay": device_replay_score,
                    **custom_signals,
                },
                reasoning=reasoning,
            )

        signal_scores = self._compute_signal_scores(custom_signals)
        final_spoof_score = self._clamp01(
            self.weights.pretrained_model * pretrained_score
            + self.weights.flash_response * signal_scores["flash"]
            + self.weights.moire_pattern * signal_scores["moire"]
            + self.weights.device_replay * signal_scores["device"]
        )
        is_spoof = final_spoof_score > self.threshold
        confidence = self._decision_confidence(final_spoof_score)
        breakdown = {
            "pretrained": pretrained_score,
            **signal_scores,
        }
        reasoning = self._generate_reasoning(
            is_spoof=is_spoof,
            score=final_spoof_score,
            breakdown=breakdown,
        )
        return FusionResult(
            is_spoof=is_spoof,
            confidence=confidence,
            spoof_score=final_spoof_score,
            breakdown=breakdown,
            reasoning=reasoning,
        )

    def _compute_signal_scores(self, signals: dict[str, Any]) -> dict[str, float]:
        flash_score = self._resolve_flash_score(signals)
        moire_score = self._resolve_numeric_signal(
            signals.get("moire_score", signals.get("moire_risk")),
            neutral=0.5,
        )
        device_score = self._resolve_numeric_signal(
            signals.get("device_replay_score", signals.get("device_replay_risk")),
            neutral=0.5,
        )
        return {
            "flash": flash_score,
            "moire": moire_score,
            "device": device_score,
        }

    def _resolve_flash_score(self, signals: dict[str, Any]) -> float:
        flash_samples = self._coerce_float(
            signals.get("flash_response_samples", signals.get("flash_response_sample_count"))
        )
        if flash_samples is not None and flash_samples < 1.0:
            return 0.5

        flash_response_score = self._coerce_float(signals.get("flash_response_score"))
        if flash_response_score is not None:
            return self._clamp01(1.0 - flash_response_score)

        flash_response = self._coerce_float(signals.get("flash_response"))
        if flash_response is None:
            return 0.5
        return self._normalize_flash_score(flash_response)

    def _resolve_numeric_signal(self, value: Any, *, neutral: float) -> float:
        numeric = self._coerce_float(value)
        if numeric is None:
            return neutral
        return self._clamp01(numeric)

    def _normalize_flash_score(self, flash_response: float) -> float:
        if flash_response >= 0.15:
            return 0.0
        if flash_response <= 0.02:
            return 1.0
        return self._clamp01(1.0 - (flash_response - 0.02) / (0.15 - 0.02))

    def _generate_reasoning(
        self,
        *,
        is_spoof: bool,
        score: float,
        breakdown: dict[str, float],
    ) -> str:
        top_signal = max(breakdown.items(), key=lambda item: item[1])
        if is_spoof:
            return (
                f"SPOOF detected (score={score:.2f}). "
                f"Primary indicator: {top_signal[0]} ({top_signal[1]:.2f})"
            )
        return (
            f"LIVE verified (score={score:.2f}). "
            f"Strongest remaining spoof cue: {top_signal[0]} ({top_signal[1]:.2f})"
        )

    def _decision_confidence(self, spoof_score: float) -> float:
        decision_margin = abs(spoof_score - self.threshold)
        max_margin = max(self.threshold, 1.0 - self.threshold, 1e-6)
        return self._clamp01(decision_margin / max_margin)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
