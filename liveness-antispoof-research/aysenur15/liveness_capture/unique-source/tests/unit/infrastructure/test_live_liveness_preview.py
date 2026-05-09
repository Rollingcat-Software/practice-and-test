"""Tests for the developer live liveness preview utility."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.application.services.background_active_reaction_evaluator import (
    BackgroundActiveReactionEvaluator,
    ReactionSignalFrame,
)
from app.application.services.cutout_anomaly_detector import CutoutAnomalyDetector
from app.application.services.device_spoof_risk_evaluator import DeviceSpoofRiskEvaluator
from app.application.services.flash_spoof_analyzer import FlashSpoofAnalyzer
from app.application.services.light_challenge_service import LightChallengeService
from app.application.services.preview_biometric_puzzle import (
    PreviewBiometricPuzzleController,
    PreviewPuzzleSummary,
)
from app.core.config import Settings
from app.domain.entities.puzzle import Puzzle, PuzzleDifficulty, PuzzleStep
from app.infrastructure.ml.liveness.moire_pattern_analysis import analyze_moire_pattern
from app.tools.live_liveness_preview import (
    FrameMetrics,
    TemporalLivenessAggregator,
    _is_device_replay_spoof_detected,
)


def _device_spoof(
    *,
    moire_risk: float = 0.0,
    reflection_risk: float = 0.0,
    flicker_risk: float = 0.0,
    flash_response_score: float = 0.0,
    flash_response_strength: float = 0.0,
    flash_response_consistency: float = 0.0,
    flash_replay_risk: float = 0.0,
    hole_cutout_risk: float = 0.0,
    focal_blur_anomaly_risk: float = 0.0,
    cutout_spoof_support: float = 0.0,
    screen_frame_risk: float = 0.0,
    device_replay_risk: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        moire_risk=moire_risk,
        reflection_risk=reflection_risk,
        flicker_risk=flicker_risk,
        flash_response_score=flash_response_score,
        flash_response_strength=flash_response_strength,
        flash_response_consistency=flash_response_consistency,
        flash_replay_risk=flash_replay_risk,
        hole_cutout_risk=hole_cutout_risk,
        focal_blur_anomaly_risk=focal_blur_anomaly_risk,
        cutout_spoof_support=cutout_spoof_support,
        screen_frame_risk=screen_frame_risk,
        device_replay_risk=device_replay_risk,
    )


def _frame_metrics(*, raw_score: float, confidence: float, is_live: bool = True, timestamp: float = 1.0) -> FrameMetrics:
    return FrameMetrics(
        timestamp=timestamp,
        face_detected=True,
        is_live=is_live,
        raw_score=raw_score,
        confidence=confidence,
        passive_score=raw_score - 5.0,
        active_score=raw_score - 10.0,
        active_evidence=0.5,
        directional_agreement=0.8,
        face_quality=0.7,
        face_size_ratio=0.16,
        blur_score=120.0,
        brightness=110.0,
        ear_current=0.24,
        mar_current=0.41,
        yaw_current=5.0,
        pitch_current=1.0,
        roll_current=0.0,
        landmark_model="test-model",
        background_active_mode="blink_and_smile",
        background_active_detected=True,
        details={},
    )


def test_temporal_liveness_aggregator_updates_ema_and_variance():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    first = aggregator.add(_frame_metrics(raw_score=80.0, confidence=0.90))
    second = aggregator.add(_frame_metrics(raw_score=60.0, confidence=0.50, is_live=False, timestamp=2.0))
    third = aggregator.add(_frame_metrics(raw_score=100.0, confidence=1.00, timestamp=3.0))

    assert first.ema_score == pytest.approx(80.0)
    assert first.smoothed_score == pytest.approx(80.0)
    assert second.ema_score == pytest.approx(70.0)
    assert second.smoothed_score == pytest.approx(70.0)
    assert third.ema_score == pytest.approx(85.0)
    assert third.smoothed_score == pytest.approx(85.0)
    assert third.window_confidence > 0.0
    assert third.window_confidence < 1.0
    assert third.decision_confidence == pytest.approx(third.window_confidence)
    assert third.frame_confidence_mean == pytest.approx((0.90 + 0.50 + 1.00) / 3)
    assert third.score_mean == pytest.approx(80.0)
    assert third.face_size_adequacy > 0.0
    assert third.window_seconds == pytest.approx(2.0)
    assert third.score_variance == pytest.approx(266.6666666666667)
    assert third.min_score == 60.0
    assert third.max_score == 100.0
    assert third.stable_live_ratio == pytest.approx(2 / 3)


def test_device_spoof_risk_evaluator_returns_normalized_scores():
    evaluator = DeviceSpoofRiskEvaluator(history_size=6)
    frame = np.full((160, 160, 3), 120, dtype=np.uint8)
    frame[:, ::6] = 230
    frame[40:80, 40:80] = 255

    assessment = None
    for shift in [0, 18, -16, 22, -20, 12]:
        shifted = np.clip(frame.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        assessment = evaluator.evaluate(frame_bgr=shifted, face_region_bgr=shifted[20:140, 20:140])

    assert assessment is not None
    assert 0.0 <= assessment.moire_risk <= 1.0
    assert 0.0 <= assessment.reflection_risk <= 1.0
    assert 0.0 <= assessment.flicker_risk <= 1.0
    assert 0.0 <= assessment.flash_response_score <= 1.0
    assert 0.0 <= assessment.flash_replay_risk <= 1.0
    assert 0.0 <= assessment.hole_cutout_risk <= 1.0
    assert 0.0 <= assessment.focal_blur_anomaly_risk <= 1.0
    assert 0.0 <= assessment.cutout_spoof_support <= 1.0
    assert 0.0 <= assessment.device_replay_risk <= 1.0


def test_device_spoof_moire_risk_reuses_shared_moire_analysis():
    evaluator = DeviceSpoofRiskEvaluator(history_size=6)
    frame = np.full((160, 160, 3), 120, dtype=np.uint8)
    for i in range(0, 160, 6):
        frame[:, i : i + 2] = 240

    assessment = evaluator.evaluate(frame_bgr=frame, face_region_bgr=frame[16:144, 16:144])
    shared = analyze_moire_pattern(frame[16:144, 16:144, 0])

    assert assessment.moire_risk == pytest.approx(shared["moire_risk"])
    assert assessment.details["moire_score"] == pytest.approx(shared["moire_score"])


def test_device_spoof_reflection_risk_rises_for_compact_glossy_highlights():
    evaluator = DeviceSpoofRiskEvaluator(history_size=6)
    baseline = np.full((160, 160, 3), (150, 165, 180), dtype=np.uint8)
    glossy = baseline.copy()
    glossy[48:88, 60:100] = (250, 250, 250)
    glossy[58:78, 68:92] = (255, 255, 255)

    baseline_assessment = evaluator.evaluate(frame_bgr=baseline, face_region_bgr=baseline[16:144, 16:144])
    glossy_assessment = evaluator.evaluate(frame_bgr=glossy, face_region_bgr=glossy[16:144, 16:144])

    assert glossy_assessment.reflection_risk > baseline_assessment.reflection_risk
    assert glossy_assessment.details["reflection_glossy_patch_ratio"] > 0.0
    assert glossy_assessment.details["reflection_compact_highlight_score"] > 0.0


def test_device_spoof_flicker_risk_uses_temporal_window_signals():
    evaluator = DeviceSpoofRiskEvaluator(history_size=6)
    stable_samples = []
    oscillating_samples = []

    for brightness in [132, 133, 132, 133, 132, 133]:
        stable = np.full((96, 96, 3), brightness, dtype=np.uint8)
        stable_assessment = evaluator.evaluate(frame_bgr=stable, face_region_bgr=stable)
        stable_samples.append(evaluator.temporal_signal_sample_from_details(stable_assessment.details))

    for index, brightness in enumerate([110, 160, 108, 165, 112, 170]):
        flicker = np.full((96, 96, 3), brightness, dtype=np.uint8)
        if index % 2 == 0:
            flicker[:, ::4] = np.clip(brightness + 40, 0, 255)
        else:
            flicker[::4, :] = np.clip(brightness - 35, 0, 255)
        flicker_assessment = evaluator.evaluate(frame_bgr=flicker, face_region_bgr=flicker)
        oscillating_samples.append(evaluator.temporal_signal_sample_from_details(flicker_assessment.details))

    stable_updated = evaluator.update_with_temporal_history(stable_assessment, [s for s in stable_samples if s is not None])
    oscillating_updated = evaluator.update_with_temporal_history(
        flicker_assessment,
        [s for s in oscillating_samples if s is not None],
    )

    assert oscillating_updated.flicker_risk > stable_updated.flicker_risk
    assert oscillating_updated.details["flicker_luma_std"] > stable_updated.details["flicker_luma_std"]
    assert oscillating_updated.details["flicker_delta_std"] > stable_updated.details["flicker_delta_std"]


def test_device_replay_risk_uses_configured_weighted_fusion():
    combined = DeviceSpoofRiskEvaluator._combine_risks(
        moire_risk=0.8,
        reflection_risk=0.5,
        flicker_risk=0.4,
        flash_replay_risk=0.6,
    )

    expected = (
        DeviceSpoofRiskEvaluator.DEVICE_REPLAY_MOIRE_WEIGHT * 0.8
        + DeviceSpoofRiskEvaluator.DEVICE_REPLAY_REFLECTION_WEIGHT * 0.5
        + DeviceSpoofRiskEvaluator.DEVICE_REPLAY_FLICKER_WEIGHT * 0.4
        + DeviceSpoofRiskEvaluator.DEVICE_REPLAY_FLASH_WEIGHT * 0.6
    )

    assert combined == pytest.approx(expected)


def test_device_spoof_flash_replay_risk_drops_when_face_roi_tracks_flash_response(monkeypatch):
    evaluator = DeviceSpoofRiskEvaluator(
        history_size=6,
        enable_flash_replay=True,
        flash_interval_seconds=10.0,
        replay_fusion_weights={
            "moire": 0.0,
            "reflection": 0.0,
            "flicker": 0.0,
            "flash": 1.0,
            "screen_frame": 0.0,
        },
    )
    monkeypatch.setattr(
        evaluator._light_challenge_service,
        "generate_challenge",
        lambda: {
            "color": "red",
            "issued_at": 0.0,
            "expires_at": 0.5,
            "duration_ms": 150,
            "expected_response_window_ms": 500,
            "minimum_delay_ms": 50,
            "baseline_required": True,
            "ready_for_flash": True,
        },
    )

    baseline = np.full((32, 32, 3), (45, 45, 45), dtype=np.uint8)
    response = np.full((32, 32, 3), (45, 45, 98), dtype=np.uint8)

    evaluator.evaluate(frame_bgr=baseline, face_region_bgr=baseline, frame_timestamp=1.00)
    assessment = evaluator.evaluate(frame_bgr=response, face_region_bgr=response, frame_timestamp=1.12)

    assert assessment.flash_response_strength > 0.0
    assert assessment.flash_response_score > 0.0
    assert assessment.flash_replay_risk < 0.7
    assert assessment.device_replay_risk == pytest.approx(assessment.flash_replay_risk)


def test_device_spoof_flash_replay_risk_rises_when_face_roi_does_not_track_flash(monkeypatch):
    evaluator = DeviceSpoofRiskEvaluator(
        history_size=6,
        enable_flash_replay=True,
        flash_interval_seconds=10.0,
        replay_fusion_weights={
            "moire": 0.0,
            "reflection": 0.0,
            "flicker": 0.0,
            "flash": 1.0,
            "screen_frame": 0.0,
        },
    )
    monkeypatch.setattr(
        evaluator._light_challenge_service,
        "generate_challenge",
        lambda: {
            "color": "green",
            "issued_at": 0.0,
            "expires_at": 0.5,
            "duration_ms": 150,
            "expected_response_window_ms": 500,
            "minimum_delay_ms": 50,
            "baseline_required": True,
            "ready_for_flash": True,
        },
    )

    baseline = np.full((32, 32, 3), (52, 52, 52), dtype=np.uint8)
    no_response = np.full((32, 32, 3), (52, 52, 52), dtype=np.uint8)

    evaluator.evaluate(frame_bgr=baseline, face_region_bgr=baseline, frame_timestamp=2.00)
    assessment = evaluator.evaluate(frame_bgr=no_response, face_region_bgr=no_response, frame_timestamp=2.14)

    assert assessment.flash_response_score < 0.2
    assert assessment.flash_replay_risk > 0.8
    assert assessment.device_replay_risk == pytest.approx(assessment.flash_replay_risk)


def test_light_challenge_service_detects_subtle_red_chroma_shift_from_baseline():
    service = LightChallengeService(min_color_shift=0.02, colors=("red",))
    baseline = np.full((32, 32, 3), (84, 96, 108), dtype=np.uint8)
    response = np.full((32, 32, 3), (88, 97, 123), dtype=np.uint8)

    verification = service.verify_response(
        frame=response,
        expected_color="red",
        flash_timestamp=1.0,
        frame_timestamp=1.12,
        baseline_bgr=baseline.mean(axis=(0, 1)).tolist(),
    )

    assert verification["passed"] is True
    assert verification["color_shift"] > 0.02


def test_flash_spoof_analyzer_color_match_survives_small_roi_translation():
    analyzer = FlashSpoofAnalyzer()
    pre = np.full((64, 64, 3), (82, 96, 108), dtype=np.uint8)
    flash = pre.copy()
    flash[16:48, 16:48, 2] = np.clip(flash[16:48, 16:48, 2] + 18, 0, 255)
    flash = np.roll(flash, shift=2, axis=1)

    analysis = analyzer.analyze(
        pre_flash_bgr=pre,
        flash_bgr=flash,
        expected_color="red",
    )

    assert analysis.flash_color_match_score > 0.10
    assert analysis.flash_response_strength > 0.10


def test_temporal_liveness_aggregator_ignores_device_spoof_values_for_scores():
    aggregator_without_spoof = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)
    aggregator_with_spoof = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    result_without_spoof = aggregator_without_spoof.add(_frame_metrics(raw_score=84.0, confidence=0.82, timestamp=1.0))
    spoofed_frame = replace(
        _frame_metrics(raw_score=84.0, confidence=0.82, timestamp=1.0),
        device_spoof=_device_spoof(
            moire_risk=0.95,
            reflection_risk=0.90,
            flicker_risk=0.85,
            device_replay_risk=0.92,
        ),
    )
    result_with_spoof = aggregator_with_spoof.add(spoofed_frame)

    assert result_without_spoof.smoothed_score == pytest.approx(result_with_spoof.smoothed_score)
    assert result_without_spoof.window_confidence == pytest.approx(result_with_spoof.window_confidence)
    assert result_without_spoof.decision_state == result_with_spoof.decision_state


def test_temporal_liveness_aggregator_replay_veto_changes_only_decision_state():
    control = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)
    veto = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    control_result = None
    veto_result = None
    for index in range(7):
        timestamp = 1.0 + index * 0.12
        base_frame = replace(
            _frame_metrics(raw_score=92.0, confidence=0.84, timestamp=timestamp),
            details={"smile": 48.0},
        )
        control_result = control.add(base_frame)
        veto_result = veto.add(
            replace(
                base_frame,
                device_spoof=_device_spoof(
                    moire_risk=0.62,
                    reflection_risk=0.79,
                    flicker_risk=0.66,
                    screen_frame_risk=0.78,
                    device_replay_risk=0.86,
                ),
            )
        )

    assert control_result is not None
    assert veto_result is not None
    assert control_result.decision_state == "LIKELY_LIVE"
    assert veto_result.decision_state == "LIKELY_SPOOF"
    assert veto_result.smoothed_score == pytest.approx(control_result.smoothed_score)
    assert veto_result.window_confidence == pytest.approx(control_result.window_confidence)


def test_temporal_liveness_aggregator_does_not_veto_live_for_uncorroborated_replay_signal():
    control = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)
    candidate = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    control_result = None
    candidate_result = None
    for index in range(7):
        timestamp = 1.0 + index * 0.12
        base_frame = replace(
            _frame_metrics(raw_score=92.0, confidence=0.84, timestamp=timestamp),
            details={"smile": 48.0},
        )
        control_result = control.add(base_frame)
        candidate_result = candidate.add(
            replace(
                base_frame,
                device_spoof=_device_spoof(
                    moire_risk=0.18,
                    reflection_risk=0.34,
                    flicker_risk=0.22,
                    device_replay_risk=0.79,
                ),
            )
        )

    assert control_result is not None
    assert candidate_result is not None
    assert control_result.decision_state == "LIKELY_LIVE"
    assert candidate_result.decision_state == "LIKELY_LIVE"
    assert candidate_result.smoothed_score == pytest.approx(control_result.smoothed_score)
    assert candidate_result.window_confidence == pytest.approx(control_result.window_confidence)


def test_temporal_liveness_aggregator_replay_veto_overrides_insufficient_evidence_when_screen_signals_are_strong():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=77.3, confidence=0.81, timestamp=1.0),
            device_spoof=_device_spoof(
                moire_risk=1.00,
                reflection_risk=0.79,
                flicker_risk=0.96,
                screen_frame_risk=0.80,
                device_replay_risk=0.93,
            ),
            details={
                "moire_fft_risk": 0.72,
                "screen_frame_risk": 0.80,
                "screen_frame_face_center_inside": 0.66,
                "flicker_risk": 0.96,
            },
        )
    )

    assert result.smoothed_score == pytest.approx(77.3)
    assert result.window_confidence > 0.0
    assert result.decision_state == "LIKELY_SPOOF"


def test_temporal_liveness_aggregator_replay_veto_accepts_high_flicker_low_reflection_screen_pattern():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=84.8, confidence=0.84, timestamp=1.0),
            device_spoof=_device_spoof(
                moire_risk=1.00,
                reflection_risk=0.35,
                flicker_risk=0.93,
                screen_frame_risk=0.44,
                device_replay_risk=0.79,
            ),
            details={
                "moire_fft_risk": 0.75,
                "screen_frame_risk": 0.44,
                "screen_frame_face_center_inside": 0.51,
                "flicker_risk": 0.93,
            },
        )
    )

    assert result.smoothed_score == pytest.approx(84.8)
    assert result.window_confidence > 0.0
    assert result.decision_state == "LIKELY_SPOOF"


def test_temporal_liveness_aggregator_replay_veto_accepts_glossy_static_screen_pattern():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=90.5, confidence=0.82, timestamp=1.0),
            device_spoof=_device_spoof(
                moire_risk=1.00,
                reflection_risk=0.97,
                flicker_risk=0.29,
                device_replay_risk=0.81,
            ),
            details={
                "moire_fft_risk": 0.76,
                "reflection_compact_highlight_score": 0.54,
            },
        )
    )

    assert result.decision_state != "LIKELY_SPOOF"

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=90.5, confidence=0.82, timestamp=2.0),
            device_spoof=_device_spoof(
                moire_risk=1.00,
                reflection_risk=0.97,
                flicker_risk=0.29,
                device_replay_risk=0.89,
            ),
            details={
                "moire_fft_risk": 0.76,
                "reflection_compact_highlight_score": 0.54,
            },
        )
    )

    assert result.decision_state == "LIKELY_SPOOF"


def test_temporal_liveness_aggregator_replay_veto_overrides_low_quality_when_screen_signals_are_strong():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(7):
        result = aggregator.add(
            replace(
                _frame_metrics(raw_score=87.1, confidence=0.85, timestamp=1.0 + index * 0.12),
                face_size_ratio=0.06,
                blur_score=193.98,
                brightness=147.7,
                details={
                    "smile": 45.0,
                    "moire_fft_risk": 0.70,
                    "screen_frame_risk": 0.79,
                    "screen_frame_face_center_inside": 0.71,
                    "flicker_risk": 0.78,
                },
                device_spoof=_device_spoof(
                    moire_risk=1.00,
                    reflection_risk=0.74,
                    flicker_risk=0.78,
                    screen_frame_risk=0.79,
                    device_replay_risk=0.87,
                ),
            )
        )

    assert result is not None
    assert result.smoothed_score > 80.0
    assert result.decision_state == "LIKELY_SPOOF"


def test_temporal_liveness_aggregator_keeps_live_for_reflection_only_head_motion():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(7):
        result = aggregator.add(
            replace(
                _frame_metrics(raw_score=89.5, confidence=0.61, timestamp=1.0 + index * 0.12),
                active_score=60.4,
                active_evidence=0.36,
                face_size_ratio=0.10,
                blur_score=372.04,
                brightness=115.2,
                yaw_current=4.4,
                pitch_current=-1.4,
                roll_current=-0.6,
                details={
                    "smile": 45.0,
                    "reflection_risk": 0.88,
                    "screen_frame_risk": 0.41,
                    "reflection_clipped_ratio": 0.17,
                    "reflection_compact_highlight_score": 0.84,
                    "reflection_glossy_patch_ratio": 0.08,
                    "flicker_risk": 0.80,
                    "moire_fft_risk": 0.05,
                    "moire_orientation_selectivity": 0.34,
                    "screen_frame_face_center_inside": 0.41,
                },
                device_spoof=_device_spoof(
                    moire_risk=0.52,
                    reflection_risk=0.88,
                    flicker_risk=0.80,
                    screen_frame_risk=0.41,
                    device_replay_risk=0.62,
                ),
            )
        )

    assert result is not None
    assert result.smoothed_score > 80.0
    assert result.decision_state == "LIKELY_LIVE"


def test_temporal_liveness_aggregator_evicts_entries_outside_time_window():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.3)

    aggregator.add(_frame_metrics(raw_score=30.0, confidence=0.30, is_live=False, timestamp=1.0))
    aggregator.add(_frame_metrics(raw_score=70.0, confidence=0.70, timestamp=2.0))
    result = aggregator.add(_frame_metrics(raw_score=90.0, confidence=0.90, timestamp=4.2))

    assert result.sample_count == 2
    assert result.min_score == 70.0
    assert result.max_score == 90.0
    assert result.stable_live_ratio == 1.0
    assert [entry.raw_score for entry in aggregator.get_recent_entries(now=4.2)] == [70.0, 90.0]


def test_smoothed_score_depends_on_score_history_not_confidence_history():
    low_confidence = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)
    high_confidence = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    low_result = None
    high_result = None
    for index, raw_score in enumerate([62.0, 78.0, 91.0], start=1):
        low_result = low_confidence.add(
            _frame_metrics(raw_score=raw_score, confidence=0.20, timestamp=float(index))
        )
        high_result = high_confidence.add(
            _frame_metrics(raw_score=raw_score, confidence=0.95, timestamp=float(index))
        )

    assert low_result is not None
    assert high_result is not None
    assert low_result.smoothed_score == pytest.approx(high_result.smoothed_score)
    assert low_result.ema_score == pytest.approx(high_result.ema_score)


def test_window_confidence_uses_recent_frame_reliability():
    low_confidence = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)
    high_confidence = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)

    low_result = None
    high_result = None
    for index, frame_confidence in enumerate([0.20, 0.25, 0.30, 0.35], start=1):
        low_result = low_confidence.add(
            _frame_metrics(raw_score=82.0, confidence=frame_confidence, timestamp=float(index))
        )
    for index, frame_confidence in enumerate([0.85, 0.88, 0.92, 0.95], start=1):
        high_result = high_confidence.add(
            _frame_metrics(raw_score=82.0, confidence=frame_confidence, timestamp=float(index))
        )

    assert low_result is not None
    assert high_result is not None
    assert low_result.smoothed_score == pytest.approx(high_result.smoothed_score)
    assert low_result.window_confidence < high_result.window_confidence


def test_background_active_evidence_is_supportive_not_dominant():
    evaluator = BackgroundActiveReactionEvaluator()
    frames = [
        ReactionSignalFrame(
            timestamp=1.0 + index * 0.1,
            face_detected=True,
            active_score=40.0,
            active_evidence=0.12,
            ear_current=0.30,
            mar_current=0.30,
            yaw_current=1.0 if index < 2 else -1.0,
            ear_baseline=0.31,
            mar_baseline=0.29,
            yaw_baseline=0.0,
        )
        for index in range(5)
    ]

    result = evaluator.evaluate(frames, passive_window_score=84.0)

    assert result.supported_score >= 84.0
    assert result.supported_score <= 90.0
    assert result.active_weight < 0.22



def test_preview_biometric_puzzle_controller_preserves_ordered_steps(monkeypatch):
    controller = PreviewBiometricPuzzleController()

    async def _fake_execute(**kwargs):
        return Puzzle(
            steps=(
                PuzzleStep(action="blink", duration_seconds=5.0, order=0),
                PuzzleStep(action="smile", duration_seconds=5.0, order=1),
            ),
            difficulty=PuzzleDifficulty.STANDARD,
        )

    monkeypatch.setattr(controller._generate_use_case, "execute", _fake_execute)

    started = controller.start_session()
    assert started.status == "running"
    assert started.current_step == "blink"

    first = controller.evaluate(
        frame_timestamp=1.0,
        current_frame_details={},
        temporal_signal_summary={"blink_evidence": 0.85, "smile_evidence": 0.10},
    )
    assert first.status == "running"
    assert first.current_step == "smile"
    assert first.completed_steps == 1
    assert first.progress == pytest.approx(0.5)

    second = controller.evaluate(
        frame_timestamp=1.4,
        current_frame_details={},
        temporal_signal_summary={"blink_evidence": 0.05, "smile_evidence": 0.92},
    )
    assert second.status == "completed"
    assert second.success is True
    assert second.active_evidence == pytest.approx(1.0)


def test_temporal_liveness_aggregator_keeps_final_active_equal_to_background_without_puzzle(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(_env_file=None, JWT_ENABLED=False, DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED=True),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)
    monkeypatch.setattr(aggregator, "_evaluate_puzzle", lambda metrics, summary: PreviewPuzzleSummary(
        status="idle",
        current_step="-",
        progress=0.0,
        completed_steps=0,
        total_steps=0,
        active_evidence=0.0,
        confidence=0.0,
        success=False,
        fusion_active=False,
        sequence_label="-",
    ))

    result = aggregator.add(_frame_metrics(raw_score=84.0, confidence=0.82, timestamp=1.0))

    assert result.final_active_evidence == pytest.approx(result.background_active_evidence)
    assert result.final_active_score == pytest.approx(result.background_active_score)
    assert result.puzzle_status == "idle"


def test_temporal_liveness_aggregator_keeps_background_active_score_when_puzzle_is_running(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(
            _env_file=None,
            JWT_ENABLED=False,
            DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED=True,
            DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_BACKGROUND_WEIGHT=0.40,
            DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_PUZZLE_WEIGHT=0.60,
        ),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.5)
    monkeypatch.setattr(aggregator, "_evaluate_puzzle", lambda metrics, summary: PreviewPuzzleSummary(
        status="running",
        current_step="smile",
        progress=0.5,
        completed_steps=1,
        total_steps=2,
        active_evidence=0.90,
        confidence=0.80,
        success=False,
        fusion_active=True,
        sequence_label="blink -> smile",
    ))

    result = aggregator.add(_frame_metrics(raw_score=88.0, confidence=0.85, timestamp=1.0))
    assert result.puzzle_fusion_active is True
    assert result.puzzle_current_step == "smile"
    assert result.puzzle_progress == pytest.approx(0.5)
    assert result.puzzle_active_evidence == pytest.approx(0.90)
    assert result.final_active_evidence == pytest.approx(result.background_active_evidence)
    assert result.final_active_score == pytest.approx(result.background_active_score)
    assert result.final_supported_score == pytest.approx(result.supported_score)


def test_preview_debug_decision_layer_applies_mid_replay_penalty():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(7):
        result = aggregator.add(
            replace(
                _frame_metrics(raw_score=90.0, confidence=0.84, timestamp=1.0 + index * 0.1),
                details={"smile": 49.0},
                device_spoof=_device_spoof(device_replay_risk=0.50),
            )
        )

    assert result is not None
    assert result.replay_veto is False
    assert result.adjusted_score == pytest.approx(result.smoothed_score)
    assert "HIGH_REPLAY_RISK" in result.suspicion_reasons


def test_support_based_spoof_gate_triggers_with_support_count_and_compact_reflection():
    frame = replace(
        _frame_metrics(raw_score=88.0, confidence=0.84, timestamp=1.0),
        details={
            "reflection_risk": 0.70,
            "reflection_compact_highlight_score": 0.73,
            "preview_spoof_support_streak": 8.0,
            "cutout_spoof_support": 0.68,
            "hole_cutout_risk": 0.64,
            "focal_blur_anomaly_risk": 0.60,
            "screen_frame_face_center_inside": 0.18,
            "uniface_score": 40.0,
        },
        device_spoof=_device_spoof(
            reflection_risk=0.70,
            cutout_spoof_support=0.68,
            hole_cutout_risk=0.64,
            focal_blur_anomaly_risk=0.60,
            device_replay_risk=0.58,
        ),
    )

    assert _is_device_replay_spoof_detected(frame) is True


def test_support_based_spoof_gate_does_not_trigger_below_compact_reflection_threshold():
    frame = replace(
        _frame_metrics(raw_score=88.0, confidence=0.84, timestamp=1.0),
        details={
            "reflection_risk": 0.70,
            "reflection_compact_highlight_score": 0.62,
            "preview_spoof_support_streak": 8.0,
            "screen_frame_risk": 0.72,
        },
        device_spoof=_device_spoof(
            reflection_risk=0.70,
            screen_frame_risk=0.72,
            device_replay_risk=0.58,
        ),
    )

    assert _is_device_replay_spoof_detected(frame) is False


def test_preview_debug_decision_layer_replay_veto_forces_likely_spoof():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(7):
        result = aggregator.add(
            replace(
                _frame_metrics(raw_score=92.0, confidence=0.86, timestamp=1.0 + index * 0.1),
                details={
                    "smile": 49.0,
                    "moire_risk": 0.78,
                    "moire_fft_risk": 0.62,
                    "moire_orientation_selectivity": 0.38,
                },
                device_spoof=_device_spoof(
                    moire_risk=0.78,
                    device_replay_risk=0.84,
                ),
            )
        )

    assert result is not None
    assert result.replay_veto is True
    assert result.decision_state == "LIKELY_SPOOF"
    assert "HIGH_REPLAY_RISK" in result.suspicion_reasons


def test_preview_debug_decision_layer_high_reflection_alone_does_not_trigger_replay_veto():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=72.0, confidence=0.82, timestamp=1.0),
            details={
                "reflection_risk": 0.88,
                "reflection_compact_highlight_score": 0.84,
            },
            device_spoof=_device_spoof(
                reflection_risk=0.88,
                device_replay_risk=0.77,
            ),
        )
    )

    assert result.replay_veto is False


def test_preview_debug_decision_layer_uses_standard_active_thresholds(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(_env_file=None, JWT_ENABLED=False),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)
    monkeypatch.setattr(
        aggregator,
        "_evaluate_puzzle",
        lambda metrics, summary: PreviewPuzzleSummary(
            status="idle",
            current_step="-",
            progress=0.0,
            completed_steps=0,
            total_steps=0,
            active_evidence=0.0,
            confidence=0.0,
            success=False,
            fusion_active=False,
            sequence_label="-",
        ),
    )

    result = None
    for index in range(7):
        result = aggregator.add(
            replace(
                _frame_metrics(raw_score=91.0, confidence=0.88, timestamp=1.0 + index * 0.1),
                details={"smile": 49.0},
            )
        )

    assert result is not None
    assert result.final_active_evidence > 0.25
    assert result.debug_active_score < 20.0
    assert result.decision_state == "LIKELY_SPOOF"


def test_positive_flash_response_is_exposed_in_preview_decision(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(_env_file=None, JWT_ENABLED=False),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=86.0, confidence=0.84, timestamp=1.0),
            details={
                "flash_response_sample_count": 2.0,
            },
            device_spoof=_device_spoof(
                flash_response_score=0.72,
                flash_replay_risk=0.22,
                device_replay_risk=0.41,
            ),
        )
    )

    assert result.preview_flash_live_response is True
    assert result.preview_flash_replay_support is False
    assert "FLASH_LIVE_RESPONSE" in result.suspicion_reasons


def test_flash_replay_risk_with_compact_reflection_forces_likely_spoof():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=86.0, confidence=0.84, timestamp=1.0),
            details={
                "flash_response_sample_count": 2.0,
                "reflection_compact_highlight_score": 0.72,
                "smile": 49.0,
            },
            device_spoof=_device_spoof(
                flash_response_score=0.18,
                flash_replay_risk=0.82,
                device_replay_risk=0.41,
            ),
        )
    )

    assert "FLASH_REPLAY_RISK" in result.suspicion_reasons
    assert result.decision_state == "LIKELY_SPOOF"


def test_preview_debug_decision_layer_triggers_puzzle_and_failed_puzzle_forces_spoof(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(_env_file=None, JWT_ENABLED=False, DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED=True),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)
    monkeypatch.setattr(
        aggregator,
        "start_puzzle_session",
        lambda: PreviewPuzzleSummary(
            status="running",
            current_step="blink",
            progress=0.0,
            completed_steps=0,
            total_steps=2,
            active_evidence=0.0,
            confidence=0.0,
            success=False,
            fusion_active=True,
            sequence_label="blink -> smile",
        ),
    )
    monkeypatch.setattr(
        aggregator,
        "_evaluate_puzzle",
        lambda metrics, summary: PreviewPuzzleSummary(
            status="failed",
            current_step="smile",
            progress=0.5,
            completed_steps=1,
            total_steps=2,
            active_evidence=0.40,
            confidence=0.45,
            success=False,
            fusion_active=True,
            sequence_label="blink -> smile",
        ),
    )

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=88.0, confidence=0.82, timestamp=1.0),
            details={"smile": 49.0},
            device_spoof=_device_spoof(device_replay_risk=0.84),
        )
    )

    assert result.puzzle_required is True
    assert result.puzzle_result == pytest.approx(0.40)
    assert result.debug_active_score == pytest.approx(result.final_active_score)
    assert result.decision_state == "LIKELY_SPOOF"
    assert "PUZZLE_FAILED" in result.suspicion_reasons


def test_preview_debug_decision_layer_failed_puzzle_without_strong_replay_does_not_force_spoof(monkeypatch):
    monkeypatch.setattr(
        "app.tools.live_liveness_preview.get_settings",
        lambda: Settings(_env_file=None, JWT_ENABLED=False, DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED=True),
    )
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)
    monkeypatch.setattr(
        aggregator,
        "_evaluate_puzzle",
        lambda metrics, summary: PreviewPuzzleSummary(
            status="failed",
            current_step="smile",
            progress=0.5,
            completed_steps=1,
            total_steps=2,
            active_evidence=0.40,
            confidence=0.45,
            success=False,
            fusion_active=True,
            sequence_label="blink -> smile",
        ),
    )

    result = aggregator.add(
        replace(
            _frame_metrics(raw_score=88.0, confidence=0.82, timestamp=1.0),
            details={"smile": 49.0},
            device_spoof=_device_spoof(device_replay_risk=0.45),
        )
    )

    assert result.puzzle_result == pytest.approx(0.40)
    assert "PUZZLE_FAILED" in result.suspicion_reasons
    assert result.decision_state != "LIKELY_SPOOF"


def test_preview_debug_decision_layer_adds_post_no_face_cooldown():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    no_face_result = aggregator.add(
        replace(
            _frame_metrics(raw_score=0.0, confidence=0.0, is_live=False, timestamp=1.0),
            face_detected=False,
            details={},
        )
    )
    live_result = aggregator.add(
        replace(
            _frame_metrics(raw_score=92.0, confidence=0.86, timestamp=1.1),
            details={"smile": 49.0},
        )
    )

    assert no_face_result.decision_state == "NO_FACE"
    assert live_result.decision_state == "INSUFFICIENT_EVIDENCE"
    assert live_result.no_face_cooldown_active is True
    assert "POST_NO_FACE_COOLDOWN" in live_result.suspicion_reasons



def test_cutout_anomaly_detector_rises_for_dark_flat_eye_mouth_cutouts():
    detector = CutoutAnomalyDetector()

    natural = np.full((160, 160, 3), 165, dtype=np.uint8)
    natural[30:65, 30:130] = 150
    natural[95:125, 45:115] = 145
    natural = cv2.GaussianBlur(natural, (7, 7), 0)

    cutout = natural.copy()
    cv2.rectangle(cutout, (28, 34), (68, 60), (18, 18, 18), -1)
    cv2.rectangle(cutout, (92, 34), (132, 60), (18, 18, 18), -1)
    cv2.rectangle(cutout, (48, 98), (112, 124), (12, 12, 12), -1)
    cv2.rectangle(cutout, (26, 32), (70, 62), (245, 245, 245), 2)
    cv2.rectangle(cutout, (90, 32), (134, 62), (245, 245, 245), 2)
    cv2.rectangle(cutout, (46, 96), (114, 126), (245, 245, 245), 2)

    natural_assessment = detector.analyze(natural)
    cutout_assessment = detector.analyze(cutout)

    assert cutout_assessment.hole_cutout_risk > natural_assessment.hole_cutout_risk
    assert cutout_assessment.focal_blur_anomaly_risk > natural_assessment.focal_blur_anomaly_risk
    assert cutout_assessment.cutout_spoof_support > natural_assessment.cutout_spoof_support



def test_background_active_score_is_low_without_meaningful_evidence():
    evaluator = BackgroundActiveReactionEvaluator()
    frames = [
        ReactionSignalFrame(
            timestamp=1.0 + index * 0.1,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=0.31,
            mar_current=0.29,
            yaw_current=0.5,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.31,
            mar_baseline=0.29,
            yaw_baseline=0.0,
        )
        for index in range(5)
    ]

    result = evaluator.evaluate(frames, passive_window_score=90.0)

    assert result.combined_active_score < 25.0
    assert result.combined_active_evidence < 0.25


def test_background_active_score_can_rise_above_80_with_strong_events():
    evaluator = BackgroundActiveReactionEvaluator()
    frames = []
    for index, values in enumerate(
        [
            (0.31, 0.29, 0.0),
            (0.17, 0.30, 0.0),
            (0.30, 0.30, 14.0),
            (0.31, 0.50, 0.0),
            (0.31, 0.29, 0.0),
        ]
    ):
        ear, mar, yaw = values
        frames.append(
            ReactionSignalFrame(
                timestamp=1.0 + index * 0.12,
                face_detected=True,
                active_score=0.0,
                active_evidence=0.0,
                ear_current=ear,
                mar_current=mar,
                yaw_current=yaw,
                face_quality=0.9,
                face_size_ratio=0.18,
                ear_baseline=0.31,
                mar_baseline=0.29,
                yaw_baseline=0.0,
            )
        )

    result = evaluator.evaluate(frames, passive_window_score=90.0)

    assert result.combined_active_evidence > 0.64
    assert result.combined_active_score > 80.0


def test_background_active_fusion_boosts_blink_over_mouth_open_when_similar():
    evaluator = BackgroundActiveReactionEvaluator()
    frames = [
        ReactionSignalFrame(
            timestamp=1.0,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=0.34,
            mar_current=0.30,
            yaw_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.34,
            mar_baseline=0.30,
            yaw_baseline=0.0,
        ),
        ReactionSignalFrame(
            timestamp=1.12,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=0.22,
            mar_current=0.42,
            yaw_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.34,
            mar_baseline=0.30,
            yaw_baseline=0.0,
        ),
        ReactionSignalFrame(
            timestamp=1.24,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=0.31,
            mar_current=0.39,
            yaw_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.34,
            mar_baseline=0.30,
            yaw_baseline=0.0,
        ),
        ReactionSignalFrame(
            timestamp=1.36,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=0.34,
            mar_current=0.31,
            yaw_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.34,
            mar_baseline=0.30,
            yaw_baseline=0.0,
        ),
    ]

    result = evaluator.evaluate(frames, passive_window_score=88.0)

    assert result.blink_evidence is not None
    assert result.mouth_open_evidence is not None
    assert result.blink_evidence >= 0.45
    assert result.mouth_open_evidence >= 0.35
    assert result.primary_event == pytest.approx(
        min(1.0, result.blink_evidence * BackgroundActiveReactionEvaluator.BLINK_EVENT_WEIGHT)
    )


def test_background_active_fusion_prioritizes_strongest_recent_events():
    evaluator = BackgroundActiveReactionEvaluator()
    frames = [
        ReactionSignalFrame(
            timestamp=1.0 + index * 0.12,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=ear,
            mar_current=mar,
            yaw_current=yaw,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.31,
            mar_baseline=0.29,
            yaw_baseline=0.0,
            smile_score=smile,
            smile_baseline=46.0,
        )
        for index, (ear, mar, yaw, smile) in enumerate(
            [
                (0.31, 0.29, 0.0, 46.0),
                (0.17, 0.30, 0.0, 46.0),
                (0.30, 0.30, 12.5, 46.0),
                (0.31, 0.31, 0.0, 47.0),
                (0.31, 0.29, 0.0, 46.0),
            ]
        )
    ]

    result = evaluator.evaluate(frames, passive_window_score=88.0)

    assert result.primary_event >= result.secondary_event
    assert result.raw_reaction_evidence == pytest.approx(
        0.75 * result.primary_event + 0.25 * result.secondary_event
    )
    assert result.effective_trust >= 0.65
    assert result.trusted_reaction_evidence == pytest.approx(
        result.raw_reaction_evidence * result.effective_trust
    )
    assert result.persisted_reaction_evidence == pytest.approx(
        0.75 * result.persisted_primary + 0.25 * result.persisted_secondary
    )
    assert result.combined_active_evidence == pytest.approx(
        BackgroundActiveReactionEvaluator.CURRENT_REACTION_BLEND_WEIGHT * result.trusted_reaction_evidence
        + BackgroundActiveReactionEvaluator.PERSISTED_REACTION_BLEND_WEIGHT * result.persisted_reaction_evidence
    )
    assert result.combined_active_score == pytest.approx(
        100.0 * (result.combined_active_evidence ** 0.5)
    )


def test_background_active_evidence_decays_not_drops_immediately():
    evaluator = BackgroundActiveReactionEvaluator(decay_seconds=1.25)
    strong_frames = [
        ReactionSignalFrame(
            timestamp=1.0 + index * 0.12,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=ear,
            mar_current=0.29,
            yaw_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.31,
            mar_baseline=0.29,
            yaw_baseline=0.0,
        )
        for index, ear in enumerate([0.31, 0.18, 0.17, 0.30, 0.31])
    ]
    strong_result = evaluator.evaluate(strong_frames, passive_window_score=88.0)

    no_face_result = evaluator.evaluate(
        [
            ReactionSignalFrame(
                timestamp=1.8,
                face_detected=False,
                active_score=0.0,
                active_evidence=0.0,
                ear_current=None,
                mar_current=None,
                yaw_current=None,
            )
        ],
        passive_window_score=88.0,
    )

    assert strong_result.combined_active_evidence > 0.4
    assert no_face_result.combined_active_evidence > 0.0
    assert no_face_result.combined_active_evidence < strong_result.combined_active_evidence


def test_background_active_trust_penalty_detects_unrealistic_blink_frequency():
    evaluator = BackgroundActiveReactionEvaluator()
    strong_blink_frames = [
        ReactionSignalFrame(
            timestamp=1.00 + index * 0.04,
            face_detected=True,
            active_score=0.0,
            active_evidence=0.0,
            ear_current=ear,
            mar_current=0.30,
            yaw_current=0.0,
            pitch_current=0.0,
            roll_current=0.0,
            face_quality=0.9,
            face_size_ratio=0.18,
            ear_baseline=0.34,
            mar_baseline=0.30,
            yaw_baseline=0.0,
        )
        for index, ear in enumerate([0.34, 0.18, 0.33, 0.17, 0.34, 0.18, 0.34])
    ]
    first = evaluator.evaluate(strong_blink_frames[:4], passive_window_score=88.0)
    second = evaluator.evaluate(strong_blink_frames, passive_window_score=88.0)

    assert first.blink_evidence is not None and first.blink_evidence > 0.4
    assert second.blink_anomaly_score > 0.0
    assert second.trust_penalty > 0.0
    assert second.effective_trust < second.base_active_trust + 0.35


def test_temporal_liveness_aggregator_respects_max_entries_within_time_window():
    aggregator = TemporalLivenessAggregator(window_seconds=3.0, max_entries=2, ema_alpha=0.3)

    aggregator.add(_frame_metrics(raw_score=30.0, confidence=0.30, is_live=False, timestamp=1.0))
    aggregator.add(_frame_metrics(raw_score=70.0, confidence=0.70, timestamp=1.5))
    result = aggregator.add(_frame_metrics(raw_score=90.0, confidence=0.90, timestamp=2.0))

    assert result.sample_count == 2
    assert [entry.raw_score for entry in aggregator.get_recent_entries(now=2.0)] == [70.0, 90.0]


def test_temporal_liveness_aggregator_computes_signal_summaries():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, max_entries=10, ema_alpha=0.3)

    aggregator.add(
        FrameMetrics(
            **{
                **_frame_metrics(raw_score=55.0, confidence=0.6, timestamp=1.0).__dict__,
                "ear_current": 0.32,
                "mar_current": 0.28,
                "yaw_current": -12.0,
                "pitch_current": 1.5,
                "roll_current": -1.0,
                "details": {
                    "ear_baseline": 0.32,
                    "mar_baseline": 0.28,
                    "yaw_baseline": 0.0,
                    "pitch_baseline": 0.0,
                    "roll_baseline": 0.0,
                },
            }
        )
    )
    result = aggregator.add(
        FrameMetrics(
            **{
                **_frame_metrics(raw_score=82.0, confidence=0.9, timestamp=2.0).__dict__,
                "ear_current": 0.18,
                "mar_current": 0.52,
                "yaw_current": 18.0,
                "pitch_current": 2.5,
                "roll_current": 3.0,
                "details": {
                    "ear_baseline": 0.32,
                    "mar_baseline": 0.28,
                },
            }
        )
    )

    assert result.ear_mean == pytest.approx(0.25)
    assert result.ear_min == pytest.approx(0.18)
    assert result.ear_max == pytest.approx(0.32)
    assert result.ear_drop == pytest.approx(0.14)
    assert result.ear_drop_ratio == pytest.approx(0.4375)
    assert result.mar_mean == pytest.approx(0.4)
    assert result.mar_max == pytest.approx(0.52)
    assert result.mar_rise == pytest.approx(0.24)
    assert result.mar_rise_ratio == pytest.approx(0.8571428571)
    assert result.yaw_mean == pytest.approx(3.0)
    assert result.yaw_left_peak == pytest.approx(12.0)
    assert result.yaw_right_peak == pytest.approx(18.0)
    assert result.pitch_mean == pytest.approx(2.0)
    assert result.roll_mean == pytest.approx(1.0)
    assert result.blink_evidence is not None and result.blink_evidence > 0.5
    assert result.smile_evidence is not None and result.smile_evidence > 0.2
    assert result.mouth_open_evidence is not None and result.mouth_open_evidence > 0.6
    assert result.head_turn_left_evidence is not None and result.head_turn_left_evidence > 0.4
    assert result.head_turn_right_evidence is not None and result.head_turn_right_evidence > 0.7
    assert result.combined_active_evidence > 0.45
    assert result.combined_active_score >= result.active_frame_score_mean
    assert result.smoothed_score >= result.ema_score
    assert result.window_confidence > 0.0


def test_temporal_liveness_aggregator_detects_temporal_blink_event_with_recovery():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    for index, ear in enumerate([0.31, 0.31, 0.18, 0.17, 0.29, 0.31]):
        result = aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=74.0, confidence=0.82, timestamp=1.0 + index * 0.12).__dict__,
                    "ear_current": ear,
                    "mar_current": 0.29,
                    "yaw_current": 0.5,
                    "details": {"smile": 46.0},
                }
            )
        )

    assert result is not None
    assert result.blink_evidence is not None and result.blink_evidence > 0.7


def test_temporal_liveness_aggregator_calibrates_session_baseline_from_stable_frames():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.6, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(7):
        result = aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=72.0, confidence=0.8, timestamp=1.0 + index * 0.12).__dict__,
                    "ear_current": 0.31 + (0.002 if index % 2 == 0 else -0.002),
                    "mar_current": 0.29 + (0.003 if index % 2 == 0 else -0.003),
                    "yaw_current": 1.5 + (0.6 if index % 2 == 0 else -0.6),
                    "pitch_current": 0.8,
                    "roll_current": 0.4,
                    "details": {"smile": 48.0 + index * 0.3},
                }
            )
        )

    assert result is not None
    assert result.baseline_ready is True
    assert result.baseline_sample_count >= 6
    assert result.baseline_duration_seconds >= 0.6
    assert result.ear_baseline == pytest.approx(0.312, abs=0.01)
    assert result.mar_baseline == pytest.approx(0.293, abs=0.01)
    assert result.smile_baseline == pytest.approx(48.9, abs=1.0)
    assert result.yaw_baseline == pytest.approx(1.5, abs=1.0)
    assert result.window_confidence > 0.0


def test_preview_state_starts_as_insufficient_evidence_until_buffer_is_ready():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.75, max_entries=20, ema_alpha=0.3)

    result = aggregator.add(_frame_metrics(raw_score=78.0, confidence=0.75, timestamp=1.0))

    assert result.decision_state == "INSUFFICIENT_EVIDENCE"


def test_preview_state_transitions_to_no_face_after_consecutive_missing_frames():
    aggregator = TemporalLivenessAggregator(
        window_seconds=2.0,
        baseline_seconds=0.5,
        max_entries=20,
        ema_alpha=0.3,
        no_face_consecutive_threshold=3,
    )

    for index in range(6):
        aggregator.add(_frame_metrics(raw_score=82.0, confidence=0.8, timestamp=1.0 + index * 0.12))

    result = None
    for index in range(3):
        result = aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=0.0, confidence=0.0, is_live=False, timestamp=2.0 + index * 0.1).__dict__,
                    "face_detected": False,
                    "background_active_detected": False,
                    "background_active_mode": "no_face",
                }
            )
        )

    assert result is not None
    assert result.decision_state == "NO_FACE"
    assert result.consecutive_no_face_frames == 3


def test_temporal_liveness_aggregator_warm_recovers_after_brief_face_loss():
    aggregator = TemporalLivenessAggregator(
        window_seconds=2.0,
        baseline_seconds=0.5,
        max_entries=20,
        ema_alpha=0.3,
        no_face_consecutive_threshold=4,
        face_return_grace_seconds=1.0,
    )

    result = None
    for index in range(7):
        result = aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=93.0, confidence=0.84, timestamp=1.0 + index * 0.12).__dict__,
                    "details": {"smile": 48.0},
                }
            )
        )

    for index in range(2):
        aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=0.0, confidence=0.0, is_live=False, timestamp=2.0 + index * 0.1).__dict__,
                    "face_detected": False,
                    "background_active_detected": False,
                    "background_active_mode": "no_face",
                }
            )
        )

    result = aggregator.add(
        FrameMetrics(
            **{
                **_frame_metrics(raw_score=94.0, confidence=0.84, timestamp=2.35).__dict__,
                "details": {"smile": 48.0},
            }
        )
    )

    assert result is not None
    assert result.face_present_ratio > 0.7
    assert result.decision_state != "NO_FACE"


def test_preview_state_marks_low_quality_before_scoring_states():
    aggregator = TemporalLivenessAggregator(window_seconds=2.0, baseline_seconds=0.5, max_entries=20, ema_alpha=0.3)

    result = None
    for index in range(6):
        result = aggregator.add(
            FrameMetrics(
                **{
                    **_frame_metrics(raw_score=88.0, confidence=0.9, timestamp=1.0 + index * 0.12).__dict__,
                    "face_quality": 0.30,
                    "blur_score": 10.0,
                    "details": {"smile": 49.0},
                }
            )
        )

    assert result is not None
    assert result.low_quality is True
    assert result.decision_state == "LOW_QUALITY"


def test_settings_gate_dev_liveness_preview_to_development_only():
    dev_settings = Settings(
        _env_file=None,
        JWT_ENABLED=False,
        ENVIRONMENT="development",
        DEV_LIVENESS_PREVIEW=True,
        DEV_LIVENESS_PREVIEW_BASELINE_SECONDS=0.75,
    )
    prod_settings = Settings(
        _env_file=None,
        JWT_ENABLED=False,
        ENVIRONMENT="production",
        DEV_LIVENESS_PREVIEW=True,
        DEV_LIVENESS_PREVIEW_BASELINE_SECONDS=0.75,
    )

    assert dev_settings.should_run_dev_liveness_preview() is True
    assert prod_settings.should_run_dev_liveness_preview() is False


def test_start_dev_liveness_preview_if_enabled(monkeypatch):
    import app.main as main_module
    import app.tools.live_liveness_preview as preview_module

    preview = Mock()
    monkeypatch.setattr(main_module.settings, "should_run_dev_liveness_preview", lambda: True)
    monkeypatch.setattr(main_module, "get_face_detector", lambda: "face-detector")
    monkeypatch.setattr(main_module, "get_liveness_detector", lambda: "liveness-detector")
    monkeypatch.setattr(main_module, "get_landmark_detector", lambda: "landmark-detector")
    monkeypatch.setattr(preview_module, "create_dev_liveness_preview", lambda **kwargs: preview)

    started = main_module._start_dev_liveness_preview_if_enabled()

    assert started is preview
    preview.start.assert_called_once_with()


def test_start_dev_liveness_preview_if_disabled(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module.settings, "should_run_dev_liveness_preview", lambda: False)

    assert main_module._start_dev_liveness_preview_if_enabled() is None
