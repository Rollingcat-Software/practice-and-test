"""Hybrid liveness detector that combines enhanced heuristics with UniFace."""

from __future__ import annotations

import logging

import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.infrastructure.ml.liveness.enhanced_liveness_detector import EnhancedLivenessDetector
from app.infrastructure.ml.liveness.uniface_liveness_detector import UniFaceLivenessDetector

logger = logging.getLogger(__name__)


class HybridLivenessDetector(ILivenessDetector):
    """Use enhanced heuristics first and UniFace as a second opinion."""

    def __init__(
        self,
        *,
        liveness_threshold: float = 70.0,
        enhanced_detector: EnhancedLivenessDetector | None = None,
        uniface_detector: UniFaceLivenessDetector | None = None,
        min_uniface_live_score: float = 65.0,
        min_uniface_soft_accept_score: float = 55.0,
        min_enhanced_soft_accept_score: float = 82.0,
        min_enhanced_soft_accept_confidence: float = 0.72,
    ) -> None:
        self._liveness_threshold = liveness_threshold
        self._min_uniface_live_score = min_uniface_live_score
        self._min_uniface_soft_accept_score = min_uniface_soft_accept_score
        self._min_enhanced_soft_accept_score = min_enhanced_soft_accept_score
        self._min_enhanced_soft_accept_confidence = min_enhanced_soft_accept_confidence
        self._enhanced = enhanced_detector or EnhancedLivenessDetector(
            texture_threshold=100.0,
            liveness_threshold=liveness_threshold,
            enable_blink_detection=True,
            enable_smile_detection=True,
            blink_frames_required=2,
        )
        self._uniface = uniface_detector or UniFaceLivenessDetector(
            liveness_threshold=liveness_threshold,
        )

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        enhanced_result = await self._enhanced.check_liveness(image)
        if enhanced_result.details.get("screen_replay_hard_veto"):
            return enhanced_result

        uniface_result = await self._uniface.check_liveness(image)
        if uniface_result.details.get("indeterminate"):
            return LivenessResult(
                is_live=enhanced_result.is_live,
                score=enhanced_result.score,
                challenge="enhanced+uniface_fallback",
                challenge_completed=enhanced_result.challenge_completed,
                confidence=enhanced_result.confidence,
                details={
                    **enhanced_result.details,
                    "hybrid_fallback_to_enhanced": True,
                    "uniface_score": uniface_result.score,
                    "uniface_confidence": uniface_result.confidence,
                    **{f"uniface_{key}": value for key, value in uniface_result.details.items()},
                },
            )

        combined_score = min(enhanced_result.score, uniface_result.score + 10.0)
        combined_confidence = min(
            1.0,
            max(
                0.0,
                0.55 * enhanced_result.confidence + 0.45 * uniface_result.confidence,
            ),
        )
        strict_accept = (
            enhanced_result.is_live
            and combined_score >= self._liveness_threshold
            and uniface_result.is_live
            and uniface_result.score >= self._min_uniface_live_score
        )
        soft_accept = (
            enhanced_result.is_live
            and enhanced_result.score >= self._min_enhanced_soft_accept_score
            and enhanced_result.confidence >= self._min_enhanced_soft_accept_confidence
            and uniface_result.score >= self._min_uniface_soft_accept_score
        )
        is_live = strict_accept or soft_accept
        return LivenessResult(
            is_live=is_live,
            score=combined_score,
            challenge="enhanced+uniface",
            challenge_completed=enhanced_result.challenge_completed and uniface_result.is_live,
            confidence=combined_confidence,
            details={
                **enhanced_result.details,
                "enhanced_score": enhanced_result.score,
                "enhanced_confidence": enhanced_result.confidence,
                "uniface_score": uniface_result.score,
                "uniface_confidence": uniface_result.confidence,
                "hybrid_combined_score": combined_score,
                "hybrid_min_uniface_live_score": self._min_uniface_live_score,
                "hybrid_min_uniface_soft_accept_score": self._min_uniface_soft_accept_score,
                "hybrid_min_enhanced_soft_accept_score": self._min_enhanced_soft_accept_score,
                "hybrid_min_enhanced_soft_accept_confidence": self._min_enhanced_soft_accept_confidence,
                "hybrid_strict_accept": strict_accept,
                "hybrid_soft_accept": soft_accept,
                **{f"uniface_{key}": value for key, value in uniface_result.details.items()},
            },
        )
