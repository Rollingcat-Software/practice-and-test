"""Unit tests for hybrid liveness fusion."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.api.schemas.live_analysis import AnalysisMode
from app.application.services.hybrid_fusion_evaluator import (
    HybridFusionEvaluator,
    FusionWeights,
)
from app.application.use_cases.live_camera_analysis import LiveCameraAnalysisUseCase
from app.core.config import Settings
from app.domain.entities.face_detection import FaceDetectionResult
from app.domain.entities.liveness_result import LivenessResult as DomainLivenessResult


def test_hybrid_fusion_evaluator_detects_clear_spoof_case() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.95,
        custom_signals={
            "flash_response_score": 0.0,
            "flash_response_samples": 2,
            "rppg_live_signal": False,
            "rppg_available": True,
            "moire_score": 0.9,
            "device_replay_score": 0.8,
        },
    )

    assert result.is_spoof is True
    assert result.spoof_score > 0.8
    assert "SPOOF detected" in result.reasoning


def test_hybrid_fusion_evaluator_detects_clear_live_case() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "rppg_live_signal": True,
            "rppg_available": True,
            "moire_score": 0.1,
            "device_replay_score": 0.15,
        },
    )

    assert result.is_spoof is False
    assert result.spoof_score < 0.3
    assert "LIVE verified" in result.reasoning


def test_hybrid_fusion_evaluator_flicker_override_forces_spoof() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flicker_score": 0.91,
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "moire_score": 0.05,
            "device_replay_score": 0.10,
        },
    )

    assert result.is_spoof is True
    assert result.spoof_score == pytest.approx(0.90)
    assert result.confidence == pytest.approx(0.90)
    assert result.breakdown["flicker"] == pytest.approx(0.91)
    assert result.reasoning == "High flicker detected (0.91)"


def test_hybrid_fusion_evaluator_flicker_below_override_uses_normal_fusion() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flicker_score": 0.65,
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "moire_score": 0.10,
            "device_replay_score": 0.15,
        },
    )

    assert result.is_spoof is False
    assert result.spoof_score < 0.30
    assert "LIVE verified" in result.reasoning


def test_hybrid_fusion_evaluator_weak_minifasnet_neutral_signals_should_remain_live() -> None:
    """Critical edge case: Live face with weak MiniFASNet but missing custom signals.

    This tests the scenario where:
    - MiniFASNet is uncertain (0.65) due to poor lighting/angle
    - Flash response timed out (no samples collected)
    - rPPG analysis failed (insufficient frames)
    - Moire/Device scores are low but not negligible

    Expected: Should NOT flag as SPOOF just because MiniFASNet is weak
    when other signals are neutral/unavailable.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.65,  # Uncertain (not clearly spoof)
        custom_signals={
            "flash_response_samples": 0,  # Timeout - no samples
            "flash_response_score": None,
            "rppg_available": False,  # Insufficient frames
            "rppg_live_signal": None,
            "moire_score": 0.30,  # Low but present
            "device_replay_score": 0.25,  # Low but present
        },
    )

    # CRITICAL: Must NOT be flagged as spoof
    assert result.is_spoof is False, (
        f"Live face incorrectly flagged as SPOOF when signals are neutral. "
        f"Score: {result.spoof_score:.3f}, Reasoning: {result.reasoning}"
    )
    # Score should be in "uncertain" zone but below threshold
    assert result.spoof_score < 0.55, f"Score {result.spoof_score:.3f} should be < 0.55"
    assert "LIVE" in result.reasoning


def test_hybrid_fusion_evaluator_near_threshold_boundary_low() -> None:
    """Boundary case: Close to threshold on LIVE side (0.45-0.55).

    Mixed signals should not cause spurious flipping.
    Score = 0.25*0.50 + 0.25*0.40 + 0.20*0.0 + 0.15*0.40 + 0.15*0.35 = 0.3375
    """
    evaluator = HybridFusionEvaluator(threshold=0.55)

    result = evaluator.evaluate(
        pretrained_spoof_score=0.50,  # Neutral
        custom_signals={
            "flash_response_score": 0.60,  # Some response (normalized → 0.40)
            "flash_response_samples": 2,
            "rppg_live_signal": True,  # Pulse detected → 0.0
            "rppg_available": True,
            "moire_score": 0.40,  # Mild screen presence
            "device_replay_score": 0.35,
        },
    )

    # Score should be below threshold even with mild device indicators
    assert result.is_spoof is False
    assert result.spoof_score < 0.40  # Actual score is ~0.3375
    assert result.confidence > 0.30  # Confidence margin / max_margin = 0.2125 / 0.55 ≈ 0.386


def test_hybrid_fusion_evaluator_near_threshold_boundary_high() -> None:
    """Boundary case: Close to threshold on SPOOF side (0.50-0.70).

    When approaching threshold, multiple weak signals matter.
    Score = 0.25*0.55 + 0.25*0.70 + 0.20*1.0 + 0.15*0.65 + 0.15*0.60 = 0.70
    """
    evaluator = HybridFusionEvaluator(threshold=0.55)

    result = evaluator.evaluate(
        pretrained_spoof_score=0.55,  # Leaning spoof
        custom_signals={
            "flash_response_score": 0.30,  # Weak response (normalized → 0.70)
            "flash_response_samples": 2,
            "rppg_live_signal": False,  # No pulse → 1.0
            "rppg_available": True,
            "moire_score": 0.65,  # Moderate screen
            "device_replay_score": 0.60,  # Moderate replay
        },
    )

    # Score should exceed threshold - multiple spoof indicators
    assert result.is_spoof is True
    assert result.spoof_score > 0.65
    assert "SPOOF detected" in result.reasoning


def test_hybrid_fusion_evaluator_spoof_with_one_strong_live_signal() -> None:
    """Adversarial case: Spoof attack with one strong contradictory signal.

    If a spoof video manages to trigger a false rPPG signal, fusion should
    still catch it via other indicators.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.90,  # Clear spoof signature
        custom_signals={
            "flash_response_score": 0.05,  # No flash response
            "flash_response_samples": 2,
            "rppg_live_signal": True,  # FALSE POSITIVE from rPPG (video artifact)
            "rppg_available": True,
            "moire_score": 0.88,  # Clear moire from screen
            "device_replay_score": 0.85,  # Clear replay from screen
        },
    )

    # Multiple strong spoof indicators should overcome single rPPG false positive
    assert result.is_spoof is True, (
        f"Should detect spoof despite rPPG false positive. "
        f"Score: {result.spoof_score:.3f}"
    )
    assert result.spoof_score > 0.65


def test_hybrid_fusion_evaluator_missing_all_custom_signals() -> None:
    """Fallback case: What if all custom signals are unavailable?

    Should fall back to MiniFASNet with neutral padding.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.80,  # Strong spoof from MiniFASNet
        custom_signals={
            # Everything missing/null
            "flash_response_score": None,
            "flash_response_samples": None,
            "rppg_available": False,
            "rppg_live_signal": None,
            "moire_score": None,
            "device_replay_score": None,
        },
    )

    # With all custom signals neutral (0.5), should still detect spoof from MiniFASNet
    # Score = 0.25*0.80 + 0.75*0.5 = 0.20 + 0.375 = 0.575 > 0.55
    assert result.is_spoof is True
    assert result.spoof_score > 0.55
    assert "backend_score" in result.breakdown or "pretrained" in result.breakdown


def test_hybrid_fusion_evaluator_weights_normalize_correctly() -> None:
    """Verify that FusionWeights properly normalize."""
    weights = FusionWeights(
        pretrained_model=0.30,
        flash_response=0.30,
        moire_pattern=0.20,
        device_replay=0.20,
    )

    assert abs(sum([
        weights.pretrained_model,
        weights.flash_response,
        weights.moire_pattern,
        weights.device_replay,
    ]) - 1.0) < 1e-6


def test_hybrid_fusion_evaluator_invalid_weights_raise() -> None:
    """Verify that invalid weights are caught."""
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        FusionWeights(
            pretrained_model=0.5,  # Sum will be > 1.0
            flash_response=0.5,
            moire_pattern=0.1,
            device_replay=0.1,
        )


@pytest.mark.asyncio
async def test_live_camera_analysis_applies_hybrid_fusion_when_enabled() -> None:
    image = np.full((120, 120, 3), 120, dtype=np.uint8)

    detector = Mock()
    detector.detect = AsyncMock(
        return_value=FaceDetectionResult(
            found=True,
            bounding_box=(10, 10, 80, 80),
            landmarks=None,
            confidence=0.97,
        )
    )
    quality_assessor = Mock()
    liveness_detector = Mock()
    liveness_detector.check_liveness = AsyncMock(
        return_value=DomainLivenessResult(
            is_live=True,
            score=92.0,
            challenge="uniface_minifasnet",
            challenge_completed=True,
            confidence=0.92,
            details={"backend_score": 0.92},
        )
    )
    liveness_detector.get_liveness_threshold = Mock(return_value=70.0)

    hybrid_fusion_evaluator = Mock()
    hybrid_fusion_evaluator.evaluate.return_value = Mock(
        is_spoof=True,
        confidence=0.82,
        spoof_score=0.84,
        breakdown={"pretrained": 0.08, "flash": 1.0, "rppg": 1.0, "moire": 0.9, "device": 0.85},
        reasoning="SPOOF detected (score=0.84). Primary indicator: flash (1.00)",
    )

    device_spoof_risk_evaluator = Mock()
    device_spoof_risk_evaluator.evaluate.return_value = Mock(
        to_dict=Mock(
            return_value={
                "moire_risk": 0.9,
                "flash_response_score": 0.0,
                "device_replay_risk": 0.85,
            }
        ),
        details={
            "flash_response_sample_count": 2.0,
            "reflection_compact_highlight_score": 0.6,
        },
    )

    settings = Settings(
        _env_file=None,
        JWT_ENABLED=False,
        LIVENESS_FUSION_ENABLED=True,
    )
    use_case = LiveCameraAnalysisUseCase(
        detector=detector,
        quality_assessor=quality_assessor,
        liveness_detector=liveness_detector,
        settings=settings,
        device_spoof_risk_evaluator=device_spoof_risk_evaluator,
        hybrid_fusion_evaluator=hybrid_fusion_evaluator,
    )

    response = await use_case.analyze_frame(image=image, mode=AnalysisMode.LIVENESS)

    assert response.liveness is not None
    assert response.liveness.is_live is False
    assert response.liveness.confidence == pytest.approx(0.82)
    assert response.liveness.scores["hybrid_fusion_spoof_score"] == pytest.approx(84.0)
    assert response.liveness.scores["liveness_score"] == pytest.approx(16.0)
    assert response.liveness.checks["hybrid_fusion_enabled"] is True
    assert response.liveness.checks["hybrid_fusion_is_spoof"] is True
    assert "SPOOF detected" in response.liveness.metadata["hybrid_fusion_reasoning"]
