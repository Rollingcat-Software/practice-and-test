"""Developer-only webcam preview for live liveness calibration."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields, replace
from statistics import pvariance
from typing import Any, Optional

import cv2
import numpy as np

from app.application.services.background_active_reaction_evaluator import (
    BackgroundActiveReactionEvaluator,
    ReactionSignalFrame,
)
from app.application.services.device_spoof_risk_evaluator import (
    DeviceSpoofRiskAssessment,
    DeviceSpoofRiskEvaluator,
)
from app.application.services.face_signal_metrics import extract_face_signal_metrics
from app.application.services.live_session_baseline_calibrator import (
    BaselineCalibrationFrame,
    LiveSessionBaselineCalibrator,
    SessionBaseline,
)
from app.application.services.preview_biometric_puzzle import (
    PreviewBiometricPuzzleController,
    PreviewPuzzleSummary,
)
from app.core.config import Settings, get_settings
from app.domain.entities.liveness_result import LivenessResult
from app.domain.exceptions.face_errors import FaceNotDetectedError
from app.domain.exceptions.liveness_errors import LivenessCheckError
from app.domain.interfaces.face_detector import IFaceDetector
from app.domain.interfaces.landmark_detector import ILandmarkDetector
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.infrastructure.ml.liveness.face_usability_gate import FaceUsabilityGate

logger = logging.getLogger(__name__)
PREVIEW_SECURITY_PROFILE = "standard"
# Occlusion state-machine thresholds (frame counts at ~25 fps)
_OCCLUSION_INSUFFICIENT_FRAMES = 2   # frames before INSUFFICIENT_EVIDENCE
_OCCLUSION_VISIBILITY_THRESHOLD = 0.50  # landmark_visibility_score below this Ã¢â€ â€™ occluded
_OCCLUSION_QUALITY_THRESHOLD = 35.0    # quality_occlusion (0Ã¢â‚¬â€œ100) below this Ã¢â€ â€™ occluded
_OCCLUSION_LOWER_TEXTURE_RATIO = 0.55  # lower-face Laplacian variance / overall blur below this Ã¢â€ â€™ covered


@dataclass(frozen=True)
class FrameMetrics:
    """Per-frame metrics extracted from the current liveness pipeline."""

    timestamp: float
    face_detected: bool
    is_live: bool
    raw_score: float
    confidence: float
    passive_score: float
    active_score: float
    active_evidence: Optional[float]
    directional_agreement: Optional[float]
    face_quality: Optional[float]
    face_size_ratio: Optional[float]
    blur_score: Optional[float]
    brightness: float
    ear_current: Optional[float]
    mar_current: Optional[float]
    yaw_current: Optional[float]
    pitch_current: Optional[float]
    roll_current: Optional[float]
    landmark_model: Optional[str]
    background_active_mode: str
    background_active_detected: bool
    details: dict[str, Any]
    device_spoof: Optional[DeviceSpoofRiskAssessment] = None
    error: Optional[str] = None
    profiling: dict[str, float] = field(default_factory=dict)
    reused_face_detection: bool = False
    reused_landmarks: bool = False
    reused_liveness: bool = False
    held_from_previous: bool = False
    inference_scale: float = 1.0

    @property
    def frame_confidence(self) -> float:
        """Explicit alias for the detector's per-frame confidence."""
        return self.confidence


@dataclass(frozen=True)
class AggregatedMetrics:
    """Temporally smoothed metrics across the recent frame buffer."""

    sample_count: int
    window_seconds: float
    decision_state: str
    ema_score: float
    score_mean: float
    supported_score: float
    smoothed_score: float
    window_confidence: float
    score_variance: float
    min_score: float
    max_score: float
    stable_live_ratio: float
    face_present_ratio: float
    consecutive_no_face_frames: int
    consecutive_occluded_frames: int
    critical_occ: bool
    critical_occ_score: float
    critical_occ_regions: tuple[str, ...]
    critical_region_visibility: dict[str, float]
    critical_occ_streak: int
    critical_clear_streak: int
    critical_occ_state: str
    critical_occ_reason: str
    face_usable: bool
    face_usability_reason: str
    face_usability_state: str
    face_usability_blocked: bool
    original_status_before_occ_gate: str
    final_status_after_occ_gate: str
    liveness_skipped_due_to_face_usability: bool
    warm_recovery: bool
    low_quality: bool
    sufficient_evidence: bool
    temporal_consistency: float
    frame_confidence_mean: float
    directional_agreement_mean: float
    face_quality_mean: float
    face_size_mean: float
    face_size_adequacy: float
    blur_adequacy: float
    brightness_adequacy: float
    ear_mean: Optional[float]
    ear_min: Optional[float]
    ear_max: Optional[float]
    ear_drop: Optional[float]
    ear_drop_ratio: Optional[float]
    mar_mean: Optional[float]
    mar_max: Optional[float]
    mar_rise: Optional[float]
    mar_rise_ratio: Optional[float]
    yaw_mean: Optional[float]
    yaw_left_peak: Optional[float]
    yaw_right_peak: Optional[float]
    pitch_mean: Optional[float]
    roll_mean: Optional[float]
    baseline_ready: bool
    baseline_sample_count: int
    baseline_duration_seconds: float
    ear_baseline: Optional[float]
    mar_baseline: Optional[float]
    smile_baseline: Optional[float]
    yaw_baseline: Optional[float]
    pitch_baseline: Optional[float]
    roll_baseline: Optional[float]
    blink_evidence: Optional[float]
    smile_evidence: Optional[float]
    mouth_open_evidence: Optional[float]
    head_turn_left_evidence: Optional[float]
    head_turn_right_evidence: Optional[float]
    primary_event: float
    secondary_event: float
    raw_reaction_evidence: float
    effective_trust: float
    trusted_reaction_evidence: float
    base_active_trust: float
    trust_penalty: float
    blink_anomaly_score: float
    motion_anomaly_score: float
    signal_inconsistency_score: float
    spoof_support_score: float
    persisted_primary: float
    persisted_secondary: float
    persisted_reaction_evidence: float
    raw_active_evidence: float
    background_active_evidence: float
    background_active_score: float
    combined_active_evidence: float
    combined_active_score: float
    active_score_mapping_mode: str
    active_score_standard_mapping: float
    active_score_strict_mapping: float
    puzzle_active_evidence: float
    puzzle_progress: float
    puzzle_current_step: str
    puzzle_completed_steps: int
    puzzle_total_steps: int
    puzzle_status: str
    puzzle_confidence: float
    puzzle_success: bool
    puzzle_fusion_active: bool
    puzzle_sequence_label: str
    final_active_evidence: float
    final_active_score: float
    final_active_score_mapping_mode: str
    final_active_score_standard_mapping: float
    final_active_score_strict_mapping: float
    final_supported_score: float
    strict_decision_score: float
    strict_replay_penalty: float
    strict_spoof_support_penalty: float
    strict_challenge_penalty: float
    strict_hard_block: bool
    strict_hard_replay_cues: float
    replay_veto: bool
    adjusted_score: float
    debug_active_score: float
    puzzle_required: bool
    puzzle_result: Optional[float]
    puzzle_trigger_reasons: tuple[str, ...]
    preview_flash_live_response: bool
    preview_flash_replay_support: bool
    spoof_reason_explicit: bool
    strong_spoof_evidence: bool
    unstable_non_spoof: bool
    decision_guard_reason: str
<<<<<<< Updated upstream
    motion_spoof: bool
    flash_replay_strong: bool
    recovery_after_low_quality: bool
    recovery_frames_left: int
    spoof_streak_frozen: bool
    high_replay_risk_blocked: bool
=======
    strong_screen_signal: bool
    effective_unstable: bool
    screen_replay_confirmed: bool
    screen_live_suppressed: bool
    screen_replay_recovery_override: bool
    screen_replay_confirm_reason: str
    screen_replay_soft: bool
    screen_replay_soft_votes: int
    screen_replay_soft_recent_ratio: float
    screen_replay_recent_valid_count: int
    screen_replay_recent_soft_count: int
    screen_replay_history_updated_before_decision: bool
    screen_replay_recent_values: str
    screen_replay_soft_off_streak: int
    screen_replay_streak_decay_reason: str
    fresh_detection_recent: bool
    screen_replay_streak: int
    screen_replay_streak_threshold: int
    screen_replay_votes_detail: str
    screen_replay_blocked_by_recovery: bool
>>>>>>> Stashed changes
    suspicion_reasons: tuple[str, ...]
    unstable_signal: bool
    no_face_cooldown_active: bool
    stable_live_hold_active: bool
    passive_window_score: float
    active_frame_score_mean: float
    active_frame_evidence_mean: float
    background_reaction_ms: float = 0.0
    temporal_aggregation_ms: float = 0.0
    @property
    def decision_confidence(self) -> float:
        """Backward-compatible alias for window-level decision confidence."""
        return self.window_confidence


class TemporalLivenessAggregator:
    """Rolling temporal aggregation for frame-level liveness output."""

    def __init__(
        self,
        *,
        window_seconds: float = 2.0,
        baseline_seconds: float = 0.75,
        max_entries: int = 45,
        ema_alpha: float = 0.25,
        no_face_consecutive_threshold: int = 6,
        face_return_grace_seconds: float = 1.0,
        face_loss_reset_seconds: float = 3.0,
        active_decay_seconds: float = 1.25,
        min_trusted_face_size_ratio: float = 0.08,
        occlusion_no_face_threshold: int = 6,
    ) -> None:
        self._window_seconds = window_seconds
        self._max_entries = max_entries
        self._buffer: deque[FrameMetrics] = deque()
        self._ema_alpha = ema_alpha
        self._no_face_consecutive_threshold = no_face_consecutive_threshold
        self._occlusion_no_face_threshold = occlusion_no_face_threshold
        self._face_return_grace_seconds = face_return_grace_seconds
        self._face_loss_reset_seconds = face_loss_reset_seconds
        self._min_trusted_face_size_ratio = min_trusted_face_size_ratio
        self._settings = get_settings()
        self._background_reaction_evaluator = BackgroundActiveReactionEvaluator(
            decay_seconds=active_decay_seconds,
            min_face_size_ratio=min_trusted_face_size_ratio,
            security_profile=PREVIEW_SECURITY_PROFILE,
            strict_sigmoid_midpoint=self._settings.get_strict_sigmoid_config()["midpoint"],
            strict_sigmoid_steepness=self._settings.get_strict_sigmoid_config()["steepness"],
            strict_sigmoid_scale=self._settings.get_strict_sigmoid_config()["scale"],
        )
        self._baseline_calibrator = LiveSessionBaselineCalibrator(baseline_seconds=baseline_seconds)
        self._puzzle_controller = (
            PreviewBiometricPuzzleController()
            if self._settings.DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED
            else None
        )
        self._ema_score: Optional[float] = None
        self._debug_decision_history: deque[str] = deque(maxlen=8)
        self._last_debug_decision_state: Optional[str] = None
        self._no_face_live_cooldown_frames = 0
        self._stable_live_hold_frames = 0
<<<<<<< Updated upstream
=======
        self._screen_replay_soft_streak = 0
        self._screen_replay_soft_history: deque[bool] = deque(maxlen=10)
        self._screen_replay_soft_off_streak = 0
        self._screen_replay_recent_weak_streak = 0
>>>>>>> Stashed changes
        self._low_quality_recovery_frames = 0

    def add(self, metrics: FrameMetrics) -> AggregatedMetrics:
        """Add a frame result and return the updated temporal summary."""
        aggregation_started = time.perf_counter()
        self._buffer.append(metrics)
        self._evict_old_entries(reference_timestamp=metrics.timestamp)
        self._reset_long_face_loss_state(metrics.timestamp)
        session_baseline = self._baseline_calibrator.update(
            BaselineCalibrationFrame(
                timestamp=metrics.timestamp,
                face_detected=metrics.face_detected,
                face_quality=metrics.face_quality,
                blur_score=metrics.blur_score,
                brightness=metrics.brightness,
                ear_current=metrics.ear_current,
                mar_current=metrics.mar_current,
                yaw_current=metrics.yaw_current,
                pitch_current=metrics.pitch_current,
                roll_current=metrics.roll_current,
                smile_score=_maybe_float(metrics.details.get("smile")),
            )
        )

        if self._ema_score is None:
            self._ema_score = metrics.raw_score
        else:
            self._ema_score = self._ema_alpha * metrics.raw_score + (1.0 - self._ema_alpha) * self._ema_score

        recent_entries = list(self._buffer)
        effective_entries = _effective_entries_for_analysis(
            recent_entries,
            face_return_grace_seconds=self._face_return_grace_seconds,
        )
        scores = [item.raw_score for item in effective_entries]
        score_mean = float(np.mean(scores))
        # Keep score smoothing tied only to score history so it remains a temporal
        # stabilization of liveness, not a proxy for decision trustworthiness.
        smoothed_score = 0.65 * float(self._ema_score) + 0.35 * score_mean
        passive_window_score = _mean([entry.passive_score for entry in effective_entries]) or 0.0
        live_count = sum(1 for item in effective_entries if item.is_live)
        face_present_ratio = sum(1 for item in effective_entries if item.face_detected) / max(len(effective_entries), 1)
        consecutive_no_face_frames = _count_consecutive_no_face(recent_entries)
        critical_occ = bool(_maybe_float(metrics.details.get("critical_occ")) or 0.0)
        critical_occ_score = float(_maybe_float(metrics.details.get("critical_occ_score")) or 0.0)
        critical_occ_regions = tuple(metrics.details.get("critical_occ_regions") or ())
        critical_region_visibility = {
            region_name: float(_maybe_float(metrics.details.get(f"critical_vis_{region_name}")) or 0.0)
            for region_name in ("left_eye", "right_eye", "nose", "mouth", "lower_face")
        }
        critical_occ_streak = int(_maybe_float(metrics.details.get("critical_occ_streak")) or 0.0)
        critical_clear_streak = int(_maybe_float(metrics.details.get("critical_clear_streak")) or 0.0)
        face_usable_raw = _maybe_float(metrics.details.get("face_usable"))
        face_usable = bool(face_usable_raw) if face_usable_raw is not None else (metrics.face_detected and not critical_occ)
        face_usability_reason = str(
            metrics.details.get("face_usability_reason")
            or ("critical_face_region_occluded" if critical_occ else ("face_usable" if face_usable else "no_face_detected"))
        )
        face_usability_state = str(
            metrics.details.get("face_usability_state")
            or ("OCCLUDED_CONFIRMED" if critical_occ else ("CLEAR" if face_usable else "NO_FACE"))
        )
        face_usability_blocked = bool(_maybe_float(metrics.details.get("face_usability_blocked")) or 0.0)
        if "face_usability_blocked" not in metrics.details:
            face_usability_blocked = critical_occ or not face_usable
        critical_occ_state = str(metrics.details.get("critical_occ_state") or face_usability_state)
        critical_occ_reason = str(metrics.details.get("critical_occ_reason") or face_usability_reason)
        liveness_skipped_due_to_face_usability = bool(
            _maybe_float(metrics.details.get("liveness_skipped_due_to_face_usability")) or 0.0
        )
        usability_reason_recent = face_usability_reason in {
            "critical_face_region_occluded",
            "recovering_face_usability",
            "no_face_detected",
        }
        consecutive_occluded_frames = critical_occ_streak
        warm_recovery = _is_warm_recovery_candidate(
            recent_entries=recent_entries,
            current_frame=metrics,
            face_return_grace_seconds=self._face_return_grace_seconds,
        )
        temporal_signal_summary = _compute_temporal_signal_summary(
            effective_entries,
            background_reaction_evaluator=self._background_reaction_evaluator,
            passive_window_score=passive_window_score,
            session_baseline=session_baseline,
        )
        puzzle_summary = self._evaluate_puzzle(metrics, temporal_signal_summary)
        temporal_signal_summary.update(
            _compute_preview_active_fusion(
                background_active_evidence=float(temporal_signal_summary.get("combined_active_evidence") or 0.0),
                background_active_score=float(temporal_signal_summary.get("combined_active_score") or 0.0),
                background_supported_score=float(temporal_signal_summary.get("supported_score") or 0.0),
                passive_window_score=passive_window_score,
                puzzle_summary=puzzle_summary,
                settings=self._settings,
            )
        )
        decision_base_score = _effective_decision_score(
            smoothed_score=smoothed_score,
            temporal_signal_summary=temporal_signal_summary,
        )
        strict_decision_layers = _preview_strict_exam_decision_layers(
            current_frame=metrics,
            temporal_signal_summary=temporal_signal_summary,
            base_decision_score=decision_base_score,
        )
        temporal_signal_summary.update(strict_decision_layers)
        low_quality = _is_low_quality(metrics)
        quality_reason = _quality_block_reason(metrics)
        if quality_reason == "face_too_small":
            self._low_quality_recovery_frames = max(self._low_quality_recovery_frames, 10)
        sufficient_evidence = _has_sufficient_evidence(
            recent_entries=effective_entries,
            temporal_signal_summary=temporal_signal_summary,
            current_frame=metrics,
            min_trusted_face_size_ratio=self._min_trusted_face_size_ratio,
            warm_recovery=warm_recovery,
        )
        score_variance = float(pvariance(scores)) if len(scores) > 1 else 0.0
        temporal_consistency = _calculate_temporal_consistency(score_variance)
        directional_agreement_mean = _mean(
            [entry.directional_agreement for entry in effective_entries if entry.directional_agreement is not None]
        ) or 0.0
        face_quality_mean = _mean(
            [entry.face_quality for entry in effective_entries if entry.face_quality is not None]
        ) or 0.0
        face_size_mean = _mean(
            [entry.face_size_ratio for entry in effective_entries if entry.face_size_ratio is not None]
        ) or 0.0
        face_size_adequacy = _mean(
            [_face_size_adequacy(entry.face_size_ratio) for entry in effective_entries if entry.face_size_ratio is not None]
        ) or 0.0
        blur_adequacy = _mean(
            [_blur_adequacy(entry.blur_score) for entry in effective_entries if entry.blur_score is not None]
        ) or 0.0
        brightness_adequacy = _mean(
            [_brightness_adequacy(entry.brightness) for entry in effective_entries]
        ) or 0.0
        frame_confidence_mean = _mean([entry.frame_confidence for entry in effective_entries]) or 0.0
        window_confidence = _calculate_window_confidence(
            recent_entries=effective_entries,
            face_present_ratio=face_present_ratio,
            temporal_signal_summary=temporal_signal_summary,
            sufficient_evidence=sufficient_evidence,
            temporal_consistency=temporal_consistency,
            frame_confidence_mean=frame_confidence_mean,
            directional_agreement_mean=directional_agreement_mean,
            face_quality_mean=face_quality_mean,
            face_size_adequacy=face_size_adequacy,
            blur_adequacy=blur_adequacy,
            brightness_adequacy=brightness_adequacy,
        )
        temporal_signal_summary.update(
            self._maybe_trigger_debug_puzzle(
                metrics=metrics,
                temporal_signal_summary=temporal_signal_summary,
                smoothed_score=smoothed_score,
                window_confidence=window_confidence,
            )
        )
        debug_decision_summary = self._apply_debug_decision_layer(
            metrics=metrics,
<<<<<<< Updated upstream
            recent_entries=recent_entries,
=======
            recent_entries=effective_entries,
>>>>>>> Stashed changes
            temporal_signal_summary=temporal_signal_summary,
            smoothed_score=smoothed_score,
            sufficient_evidence=sufficient_evidence,
            window_confidence=window_confidence,
            low_quality=low_quality,
            face_present_ratio=face_present_ratio,
            consecutive_no_face_frames=consecutive_no_face_frames,
        )
        debug_decision_summary.update(
            self._apply_critical_region_visibility_override(
                original_status=str(debug_decision_summary["debug_decision_state"]),
                face_usable=face_usable,
                face_usability_reason=face_usability_reason,
                face_usability_state=face_usability_state,
                face_usability_blocked=face_usability_blocked,
                critical_occ=critical_occ,
                critical_occ_score=critical_occ_score,
                critical_occ_regions=critical_occ_regions,
                critical_region_visibility=critical_region_visibility,
                critical_occ_streak=critical_occ_streak,
                critical_clear_streak=critical_clear_streak,
                critical_occ_state=critical_occ_state,
                critical_occ_reason=critical_occ_reason,
                liveness_skipped_due_to_face_usability=liveness_skipped_due_to_face_usability,
            )
        )
        temporal_signal_summary.update(debug_decision_summary)
        decision_state = _resolve_decision_state(
            recent_entries=effective_entries,
            current_frame=metrics,
            temporal_signal_summary=temporal_signal_summary,
            face_present_ratio=face_present_ratio,
            consecutive_no_face_frames=consecutive_no_face_frames,
            warm_recovery=warm_recovery,
            low_quality=low_quality,
            sufficient_evidence=sufficient_evidence,
            baseline_sample_count=session_baseline.sample_count if session_baseline else 0,
            smoothed_score=float(debug_decision_summary["adjusted_score"]),
            decision_confidence=window_confidence,
            no_face_consecutive_threshold=self._no_face_consecutive_threshold,
        )
        # Apply critical-region visibility gate override to the authoritative decision.
        # _resolve_decision_state only considers face_detected=False for NO_FACE; it
        # doesn't know about pixel-level occlusion. When the gate detects persistent
        # occlusion (e.g. hand over mouth/nose) it sets final_status_after_occ_gate to
        # "NO_FACE" or "INSUFFICIENT_EVIDENCE". Only promote to a more restrictive state.
        _OCC_RESTRICTIVENESS = {"NO_FACE": 0, "INSUFFICIENT_EVIDENCE": 1, "LOW_QUALITY": 2, "LIVE": 3, "SPOOF": 3}
        occ_gate_status = str(temporal_signal_summary.get("final_status_after_occ_gate") or "")
        if occ_gate_status and occ_gate_status != str(temporal_signal_summary.get("original_status_before_occ_gate") or ""):
            if _OCC_RESTRICTIVENESS.get(occ_gate_status, 1) < _OCC_RESTRICTIVENESS.get(decision_state, 1):
                decision_state = occ_gate_status
        self._update_debug_state(decision_state)
        aggregation_elapsed_ms = (time.perf_counter() - aggregation_started) * 1000.0
        aggregate_field_names = {item.name for item in fields(AggregatedMetrics)}
        aggregate = AggregatedMetrics(
            sample_count=len(effective_entries),
            window_seconds=self._window_seconds,
            decision_state=decision_state,
            ema_score=float(self._ema_score),
            score_mean=score_mean,
            supported_score=float(
                temporal_signal_summary.get("final_supported_score")
                if temporal_signal_summary.get("final_supported_score") is not None
                else temporal_signal_summary["supported_score"]
            ),
            smoothed_score=smoothed_score,
            window_confidence=window_confidence,
            score_variance=score_variance,
            min_score=min(scores),
            max_score=max(scores),
            stable_live_ratio=live_count / max(len(effective_entries), 1),
            face_present_ratio=face_present_ratio,
            consecutive_no_face_frames=consecutive_no_face_frames,
            consecutive_occluded_frames=consecutive_occluded_frames,
            warm_recovery=warm_recovery,
            low_quality=low_quality,
            sufficient_evidence=sufficient_evidence,
            temporal_consistency=temporal_consistency,
            frame_confidence_mean=frame_confidence_mean,
            directional_agreement_mean=directional_agreement_mean,
            face_quality_mean=face_quality_mean,
            face_size_mean=face_size_mean,
            face_size_adequacy=face_size_adequacy,
            blur_adequacy=blur_adequacy,
            brightness_adequacy=brightness_adequacy,
            background_reaction_ms=float(temporal_signal_summary.get("background_reaction_ms") or 0.0),
            temporal_aggregation_ms=aggregation_elapsed_ms,
            **{
                key: value
                for key, value in temporal_signal_summary.items()
                if key in aggregate_field_names and key not in {"supported_score", "background_reaction_ms"}
            },
        )
        if self._low_quality_recovery_frames > 0 and quality_reason != "face_too_small":
            self._low_quality_recovery_frames -= 1
        return aggregate

    @property
    def window_seconds(self) -> float:
        """Expose the configured rolling time window."""
        return self._window_seconds

    @property
    def max_entries(self) -> int:
        """Expose the maximum retained entries inside the time window."""
        return self._max_entries

    def get_recent_entries(self, now: Optional[float] = None) -> list[FrameMetrics]:
        """Return the current entries within the configured time window."""
        self._evict_old_entries(reference_timestamp=now or time.time())
        return list(self._buffer)

    def _evict_old_entries(self, reference_timestamp: float) -> None:
        min_timestamp = reference_timestamp - self._window_seconds
        while self._buffer and self._buffer[0].timestamp < min_timestamp:
            self._buffer.popleft()
        while len(self._buffer) > self._max_entries:
            self._buffer.popleft()

    def _reset_long_face_loss_state(self, reference_timestamp: float) -> None:
        no_face_run_duration = _tail_no_face_run_duration(list(self._buffer))
        if no_face_run_duration < self._face_loss_reset_seconds:
            return
        self._background_reaction_evaluator.reset()
        self._baseline_calibrator.reset()
        if self._puzzle_controller is not None:
            self._puzzle_controller.reset()
        self._ema_score = None
        self._debug_decision_history.clear()
        self._last_debug_decision_state = None
        self._no_face_live_cooldown_frames = 0
        self._stable_live_hold_frames = 0
        self._screen_replay_soft_streak = 0
        self._screen_replay_soft_history.clear()
        self._screen_replay_soft_off_streak = 0
        self._screen_replay_recent_weak_streak = 0
        self._low_quality_recovery_frames = 0

    def start_puzzle_session(self) -> Optional[PreviewPuzzleSummary]:
        """Start a preview puzzle session using the configured defaults."""
        if self._puzzle_controller is None:
            return None
        return self._puzzle_controller.start_session(
            difficulty=self._settings.DEV_LIVENESS_PREVIEW_PUZZLE_DIFFICULTY,
            min_steps=self._settings.DEV_LIVENESS_PREVIEW_PUZZLE_MIN_STEPS,
            max_steps=self._settings.DEV_LIVENESS_PREVIEW_PUZZLE_MAX_STEPS,
            timeout_seconds=self._settings.DEV_LIVENESS_PREVIEW_PUZZLE_TIMEOUT_SECONDS,
        )

    def reset_puzzle_session(self) -> None:
        """Reset the current preview puzzle session."""
        if self._puzzle_controller is not None:
            self._puzzle_controller.reset()

    def _evaluate_puzzle(
        self,
        metrics: FrameMetrics,
        temporal_signal_summary: dict[str, Any],
    ) -> PreviewPuzzleSummary:
        if self._puzzle_controller is None:
            return _idle_puzzle_summary()
        return self._puzzle_controller.evaluate(
            frame_timestamp=metrics.timestamp,
            current_frame_details=metrics.details,
            temporal_signal_summary=temporal_signal_summary,
        )

    def _maybe_trigger_debug_puzzle(
        self,
        *,
        metrics: FrameMetrics,
        temporal_signal_summary: dict[str, Any],
        smoothed_score: float,
        window_confidence: float,
    ) -> dict[str, Any]:
        puzzle_status = str(temporal_signal_summary.get("puzzle_status") or "idle")
        current_active_score = float(temporal_signal_summary.get("final_active_score") or 0.0)
        device_replay_risk = _device_spoof_value(metrics, "device_replay_risk") or 0.0
        sudden_face_return = bool(
            self._last_debug_decision_state == "NO_FACE"
            and metrics.face_detected
            and metrics.is_live
        )
        unstable_signal = _decision_history_unstable(list(self._debug_decision_history))
        trigger_reasons: list[str] = []
        if device_replay_risk > 0.5:
            trigger_reasons.append("HIGH_REPLAY_RISK")
        if smoothed_score > 80.0 and current_active_score < 60.0:
            trigger_reasons.append("LOW_ACTIVE_SCORE")
        if window_confidence < 0.6:
            trigger_reasons.append("LOW_CONFIDENCE")
        if sudden_face_return:
            trigger_reasons.append("SUDDEN_FACE_RETURN")
        if unstable_signal:
            trigger_reasons.append("UNSTABLE_SIGNAL")

        puzzle_required = bool(trigger_reasons)
        if (
            puzzle_required
            and puzzle_status == "idle"
            and self._puzzle_controller is not None
        ):
            started = self.start_puzzle_session()
            if started is not None:
                temporal_signal_summary.update(
                    _compute_preview_active_fusion(
                        background_active_evidence=float(temporal_signal_summary.get("combined_active_evidence") or 0.0),
                        background_active_score=float(temporal_signal_summary.get("combined_active_score") or 0.0),
                        background_supported_score=float(temporal_signal_summary.get("supported_score") or 0.0),
                        passive_window_score=float(temporal_signal_summary.get("passive_window_score") or 0.0),
                        puzzle_summary=started,
                        settings=self._settings,
                    )
                )
                temporal_signal_summary.update(
                    {
                        "puzzle_progress": started.progress,
                        "puzzle_current_step": started.current_step,
                        "puzzle_completed_steps": started.completed_steps,
                        "puzzle_total_steps": started.total_steps,
                        "puzzle_status": started.status,
                        "puzzle_confidence": started.confidence,
                        "puzzle_success": started.success,
                        "puzzle_fusion_active": started.fusion_active,
                        "puzzle_sequence_label": started.sequence_label,
                    }
                )
                puzzle_status = started.status

        puzzle_result = _current_puzzle_result_from_summary(temporal_signal_summary)
        return {
            "puzzle_required": puzzle_required,
            "puzzle_trigger_reasons": tuple(trigger_reasons),
            "puzzle_result": puzzle_result,
            "unstable_signal": unstable_signal,
        }

    def _apply_debug_decision_layer(
        self,
        *,
        metrics: FrameMetrics,
        recent_entries: list[FrameMetrics],
        temporal_signal_summary: dict[str, Any],
        smoothed_score: float,
        sufficient_evidence: bool,
        window_confidence: float,
        low_quality: bool,
        face_present_ratio: float,
        consecutive_no_face_frames: int,
    ) -> dict[str, Any]:
        device_replay_risk = _device_spoof_value(metrics, "device_replay_risk") or 0.0
        hard_replay_cues = sum(
            (
                int(_is_moire_high(metrics)),
                int(_is_flicker_high(metrics)),
                int(_is_confirmed_screen_device(metrics)),
                int(_is_flash_replay_high(metrics)),
            )
        )
        replay_veto = bool(
            device_replay_risk > 0.85
            or (device_replay_risk > 0.75 and hard_replay_cues >= 1)
        )
        base_score = float(smoothed_score)
        adjusted_score = base_score
        flash_response_score = _device_spoof_value(metrics, "flash_response_score") or 0.0
        flash_replay_risk = _device_spoof_value(metrics, "flash_replay_risk") or 0.0
        flash_samples = _maybe_float(metrics.details.get("flash_response_sample_count")) or 0.0
        positive_flash_response = bool(
            flash_samples >= 1.0
            and flash_response_score >= 0.55
            and flash_replay_risk <= 0.45
        )
        negative_flash_response = bool(
            flash_samples >= 1.0
            and flash_replay_risk >= 0.70
            and flash_response_score <= 0.30
        )
        replay_penalty = 0.0
        if not replay_veto and device_replay_risk >= 0.65:
            replay_penalty = 0.45 * device_replay_risk
            if positive_flash_response:
                replay_penalty *= 0.35
        adjusted_score = base_score * (1.0 - replay_penalty)

        puzzle_result = _current_puzzle_result_from_summary(temporal_signal_summary)
        puzzle_failed = puzzle_result is not None and puzzle_result < 0.5
        strong_replay_for_puzzle_failure = bool(
            replay_veto
            or _is_device_replay_spoof_detected(metrics)
            or device_replay_risk >= 0.70
        )
        debug_active_score = float(temporal_signal_summary.get("final_active_score") or 0.0)
        live_active_ready = bool(debug_active_score > 60.0)
        reflect_compact = _maybe_float(metrics.details.get("reflection_compact_highlight_score")) or 0.0
<<<<<<< Updated upstream
        motion_anomaly_score = float(temporal_signal_summary.get("motion_anomaly_score") or 0.0)
        signal_inconsistency_score = float(temporal_signal_summary.get("signal_inconsistency_score") or 0.0)
        quality_reason = _quality_block_reason(metrics)
        bbox_reuse = bool(
            _maybe_float(metrics.details.get("bbox_reuse"))
            or metrics.held_from_previous
            or metrics.reused_face_detection
        )
        landmark_visible_raw = _maybe_float(metrics.details.get("landmark_face_visible"))
        landmark_face_visible = bool(landmark_visible_raw) if landmark_visible_raw is not None else metrics.face_detected
        detector_unstable = bool(
            bbox_reuse
            or (metrics.error is not None and "holding last success" in metrics.error.lower())
            or (not metrics.face_detected and landmark_face_visible)
        )
        last_successful_landmark_visibility_score = 0.0
        frames_since_last_success = 9999
        prior_entries = recent_entries[:-1] if recent_entries and recent_entries[-1] is metrics else recent_entries
        for frames_ago, entry in enumerate(reversed(prior_entries), start=1):
            if entry.face_detected:
                frames_since_last_success = frames_ago
                last_successful_landmark_visibility_score = (
                    _maybe_float(entry.details.get("landmark_visibility_score"))
                    or (1.0 if bool(_maybe_float(entry.details.get("landmark_face_visible")) or 0.0) else 0.0)
                )
                break
        detector_miss_face_likely_present = bool(
            not metrics.face_detected
            and last_successful_landmark_visibility_score >= 0.70
            and frames_since_last_success <= 15
        )

        if low_quality and quality_reason == "face_too_small":
            self._low_quality_recovery_frames = 12
        elif self._low_quality_recovery_frames > 0:
            self._low_quality_recovery_frames -= 1

        recovery_after_low_quality = bool(
            self._low_quality_recovery_frames > 0 and quality_reason != "face_too_small"
        )
        spoof_streak_frozen = recovery_after_low_quality
=======
        screen_frame_risk = _device_spoof_value(metrics, "screen_frame_risk") or 0.0
        moire_risk = _device_spoof_value(metrics, "moire_risk") or 0.0
        flicker_risk = _device_spoof_value(metrics, "flicker_risk") or 0.0
        flash_planar = _maybe_float(metrics.details.get("planar_surface_risk")) or 0.0
        quality_reason = _quality_block_reason(metrics)
        landmark_face_visible_raw = _maybe_float(metrics.details.get("landmark_face_visible"))
        landmark_face_visible = bool(landmark_face_visible_raw) if landmark_face_visible_raw is not None else True
        bbox_reuse = bool(metrics.reused_face_detection or metrics.held_from_previous)
        detector_unstable = bbox_reuse or (not metrics.face_detected and landmark_face_visible)
>>>>>>> Stashed changes
        no_face_cooldown_active = self._no_face_live_cooldown_frames > 0
        unstable_non_spoof = bool(
            detector_unstable
            or bool(temporal_signal_summary.get("unstable_signal"))
<<<<<<< Updated upstream
            or not live_active_ready
            or recovery_after_low_quality
            or (low_quality and quality_reason == "face_too_small")
        )
        flash_planar_risk = _maybe_float(metrics.details.get("planar_surface_risk")) or 0.0
        # Use raw depth signals, not geometry_consistency. Cheek balance (part of geometry_consistency)
        # is always high for any symmetric face — real or flat — so it doesn't discriminate.
        # region_std and nose_cheek_delta directly measure 3D depth variation from the flash gradient.
        flash_region_std = _maybe_float(metrics.details.get("flash_region_strength_std")) or 1.0
        flash_nose_cheek = _maybe_float(metrics.details.get("flash_nose_cheek_delta")) or 1.0
        # Paper/phone: region_std ≈ 0.03-0.07, nose_cheek ≈ 0.02-0.05 (uniform flat response).
        # Real 3D face: region_std ≈ 0.12-0.18, nose_cheek ≈ 0.08-0.15 (nose protrudes, gets more flash).
        flash_planar_spoof = bool(
            flash_samples >= 1.0
            and flash_planar_risk >= 0.72
            and flash_region_std <= 0.10
            and flash_nose_cheek <= 0.07
        )
        flash_replay_strong = bool(
            (negative_flash_response and (reflect_compact >= 0.55 or flash_planar_risk >= 0.65))
            or flash_planar_spoof
        )
        motion_spoof = False
        spoof_gate = 0 if metrics.held_from_previous else int(_is_device_replay_spoof_detected(metrics))
        explicit_replay_evidence = bool(replay_veto or flash_replay_strong)
        unstable_non_spoof_case = bool(
            bbox_reuse
            or metrics.held_from_previous
            or no_face_cooldown_active
            or recovery_after_low_quality
            or detector_unstable
            or (not live_active_ready and not explicit_replay_evidence)
        )
        spoof_reason_explicit = bool(
            replay_veto
            or spoof_gate
            or flash_replay_strong
=======
            or (low_quality and quality_reason == "face_too_small")
            or not landmark_face_visible
        )
        recovery_after_low_quality = bool(
            self._low_quality_recovery_frames > 0 and quality_reason != "face_too_small"
        )
        screen_replay_blocked_by_recovery = bool(
            recovery_after_low_quality or (low_quality and quality_reason == "face_too_small")
        )
        screen_vote_names: list[str] = []
        if screen_frame_risk >= 0.30:
            screen_vote_names.append("screen_frame")
        if device_replay_risk >= 0.30:
            screen_vote_names.append("device_replay")
        if moire_risk >= 0.50:
            screen_vote_names.append("moire")
        if reflect_compact >= 0.60:
            screen_vote_names.append("reflect_compact")
        if flash_planar >= 0.70:
            screen_vote_names.append("flash_planar")
        if flicker_risk >= 0.25:
            screen_vote_names.append("flicker")
        screen_replay_soft_votes = len(screen_vote_names)
        screen_replay_soft = screen_replay_soft_votes >= 3
        screen_replay_streak_threshold = 8
        streak_increment_blocked = bool(
            screen_replay_blocked_by_recovery
            or bbox_reuse
            or detector_unstable
            or not landmark_face_visible
            or (low_quality and quality_reason == "face_too_small")
        )
        blur_adequacy = _maybe_float(temporal_signal_summary.get("blur_adequacy")) or 0.0
        brightness_value = float(metrics.brightness or 0.0)
        strong_screen_signal = bool(
            moire_risk >= 0.55
            or flicker_risk >= 0.70
            or blur_adequacy <= 0.50
            or brightness_value >= 195.0
        )
        effective_unstable = False if strong_screen_signal else unstable_non_spoof
        effective_blocked = bool(streak_increment_blocked and not strong_screen_signal)
        valid_screen_frame = bool(metrics.face_detected and landmark_face_visible)
        stable_live_ratio = (
            sum(1 for entry in recent_entries if entry.is_live) / max(len(recent_entries), 1)
            if recent_entries
            else 0.0
        )
        face_size_value = float(metrics.face_size_ratio or 0.0)
        face_size_adequacy_value = _face_size_adequacy(metrics.face_size_ratio)
        fresh_detection_recent = bool(
            any(
                entry.face_detected and not entry.reused_face_detection and not entry.held_from_previous
                for entry in recent_entries[-3:]
            )
        )
        screen_replay_streak_decay_reason = "-"
        screen_replay_history_updated_before_decision = False
        if valid_screen_frame:
            self._screen_replay_soft_history.append(screen_replay_soft)
            screen_replay_history_updated_before_decision = True
            if screen_replay_soft:
                self._screen_replay_soft_off_streak = 0
            else:
                self._screen_replay_soft_off_streak += 1
        valid_history_count = len(self._screen_replay_soft_history)
        soft_history_count = sum(1 for value in self._screen_replay_soft_history if value)
        screen_replay_recent_values = "".join("1" if value else "0" for value in self._screen_replay_soft_history) or "-"
        screen_replay_soft_recent_ratio = (
            soft_history_count / max(valid_history_count, 1)
            if valid_history_count > 0
            else 0.0
        )
        if not metrics.face_detected and (consecutive_no_face_frames >= 3 or face_present_ratio <= 0.15):
            self._screen_replay_soft_streak = 0
            self._screen_replay_soft_history.clear()
            self._screen_replay_soft_off_streak = 0
            screen_replay_streak_decay_reason = "context_reset"
        elif valid_screen_frame and not screen_replay_soft and self._screen_replay_soft_off_streak >= 8:
            self._screen_replay_soft_streak = 0
            screen_replay_streak_decay_reason = "soft_off_reset"
        elif stable_live_ratio >= 1.0 and adjusted_score > 80.0 and not screen_replay_soft:
            self._screen_replay_soft_streak = max(0, self._screen_replay_soft_streak - 3)
            screen_replay_streak_decay_reason = "stable_live_aggressive"
        elif screen_replay_soft and not effective_unstable and not effective_blocked:
            self._screen_replay_soft_streak += 1
        else:
            self._screen_replay_soft_streak = max(0, self._screen_replay_soft_streak - 1)
            if screen_replay_streak_decay_reason == "-":
                screen_replay_streak_decay_reason = "default_decay"
        screen_replay_streak = self._screen_replay_soft_streak
        screen_replay_strong = screen_replay_streak >= screen_replay_streak_threshold
        spoof_gate = int(_is_device_replay_spoof_detected(metrics))
        screen_live_suppressed = False
        screen_replay_recovery_override = False
        screen_replay_confirm_reason = "-"
        if screen_replay_blocked_by_recovery:
            screen_replay_recovery_override = bool(
                screen_replay_soft_votes >= 5
                and screen_replay_soft_recent_ratio >= 0.90
                and (
                    flicker_risk >= 0.70
                    or flash_planar >= 0.80
                    or screen_frame_risk >= 0.40
                )
            )
        screen_replay_confirm_candidate = bool(
            screen_replay_soft
            and screen_replay_streak >= screen_replay_streak_threshold
            and screen_replay_soft_recent_ratio >= 0.80
            and fresh_detection_recent
            and not replay_veto
            and spoof_gate == 0
            and not screen_replay_blocked_by_recovery
        )
        if screen_replay_confirm_candidate:
            screen_replay_confirm_reason = "recent_ratio_confirmed"
        elif screen_replay_recovery_override:
            screen_replay_confirm_candidate = True
            screen_replay_confirm_reason = "recovery_override"
        if valid_screen_frame and screen_replay_soft_recent_ratio < 0.50:
            self._screen_replay_recent_weak_streak += 1
        elif valid_screen_frame:
            self._screen_replay_recent_weak_streak = 0
        if not metrics.face_detected:
            self._screen_replay_recent_weak_streak = 0
        screen_replay_confirmed = bool(screen_replay_confirm_candidate)
        spoof_reason_explicit = bool(
            replay_veto
            or _is_device_replay_spoof_detected(metrics)
            or (negative_flash_response and reflect_compact >= 0.65)
>>>>>>> Stashed changes
            or (puzzle_failed and strong_replay_for_puzzle_failure)
        )
        strong_spoof_evidence = bool(replay_veto)
        decision_guard_reason = "-"
<<<<<<< Updated upstream
        high_replay_risk_blocked = False
=======
>>>>>>> Stashed changes

        suspicion_reasons: list[str] = []
        if replay_veto:
            suspicion_reasons.append("HIGH_REPLAY_RISK")
        elif device_replay_risk > 0.4:
            suspicion_reasons.append("HIGH_REPLAY_RISK")
        if screen_replay_soft:
            suspicion_reasons.append("SCREEN_REPLAY_SOFT")
        if negative_flash_response:
            suspicion_reasons.append("FLASH_REPLAY_RISK")
        elif positive_flash_response:
            suspicion_reasons.append("FLASH_LIVE_RESPONSE")
        if not live_active_ready:
            suspicion_reasons.append("LOW_ACTIVE_SCORE")
        if window_confidence < 0.6:
            suspicion_reasons.append("LOW_CONFIDENCE")
        if bool(temporal_signal_summary.get("unstable_signal")):
            suspicion_reasons.append("UNSTABLE_SIGNAL")
        if puzzle_failed:
            suspicion_reasons.append("PUZZLE_FAILED")
        if detector_miss_face_likely_present:
            suspicion_reasons.append("DETECTOR_MISS_FACE_LIKELY_PRESENT")
        if bool(_maybe_float(metrics.details.get("critical_occ")) or 0.0):
            suspicion_reasons.append("FACE_OCCLUDED")
        face_usability_blocked = bool(_maybe_float(metrics.details.get("face_usability_blocked")) or 0.0)
        face_usability_reason = str(metrics.details.get("face_usability_reason") or "-")
        usability_reason_recent = face_usability_reason in {
            "critical_face_region_occluded",
            "recovering_face_usability",
            "no_face_detected",
        }

        no_face_due_to_missing_current = (
            not metrics.face_detected
            and not metrics.held_from_previous
        )
        no_face_due_to_window = (
            consecutive_no_face_frames >= self._no_face_consecutive_threshold
            or face_present_ratio <= 0.15
        )
        no_face = no_face_due_to_missing_current or no_face_due_to_window
        face_likely_present_now = bool(
            metrics.face_detected
            or metrics.held_from_previous
            or bbox_reuse
            or landmark_face_visible
            or detector_miss_face_likely_present
        )
        if no_face and face_likely_present_now:
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "face_likely_present_no_face_suppressed"
        elif no_face and detector_miss_face_likely_present:
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "detector_miss_face_likely_present"
        elif no_face:
            raw_decision = "NO_FACE"
        elif low_quality:
            raw_decision = "LOW_QUALITY"
        elif replay_veto:
            raw_decision = "SPOOF"
        elif negative_flash_response and (reflect_compact >= 0.55 or flash_planar_risk >= 0.65):
            raw_decision = "SPOOF"
        elif flash_planar_spoof:
            raw_decision = "SPOOF"
        elif puzzle_failed and strong_replay_for_puzzle_failure:
            raw_decision = "SPOOF"
        elif not sufficient_evidence:
            raw_decision = "INSUFFICIENT_EVIDENCE"
        elif adjusted_score > 75.0 and live_active_ready and window_confidence > 0.6:
            raw_decision = "LIVE"
        else:
            raw_decision = "SPOOF"

<<<<<<< Updated upstream
        if raw_decision == "LIVE" and no_face_cooldown_active:
=======
        if raw_decision == "LIKELY_LIVE" and no_face_cooldown_active:
>>>>>>> Stashed changes
            raw_decision = "INSUFFICIENT_EVIDENCE"
            suspicion_reasons.append("POST_NO_FACE_COOLDOWN")
        if (
            raw_decision == "LIKELY_LIVE"
            and strong_screen_signal
            and screen_replay_soft
            and screen_replay_soft_votes >= 3
            and fresh_detection_recent
            and metrics.face_detected
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "screen_soft_live_suppressed"
            screen_live_suppressed = True
        elif (
            raw_decision == "LIKELY_LIVE"
            and strong_screen_signal
            and valid_history_count >= 5
            and screen_replay_soft_recent_ratio >= 0.50
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "screen_suspicious_live_suppressed"
            screen_live_suppressed = True
        elif (
            raw_decision == "LIKELY_LIVE"
            and screen_replay_streak >= screen_replay_streak_threshold
            and (
                not screen_replay_soft
                or valid_history_count < 5
                or screen_replay_soft_recent_ratio < 0.60
                or not fresh_detection_recent
            )
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "screen_replay_recent_weak"

        if (
            raw_decision == "SPOOF"
            and unstable_non_spoof_case
            and spoof_gate == 1
            and not replay_veto
            and not motion_spoof
            and not flash_replay_strong
            and not strong_spoof_evidence
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "unstable_non_spoof_guard"
            spoof_streak_frozen = True
            high_replay_risk_blocked = True
        elif (
            raw_decision == "SPOOF"
            and unstable_non_spoof_case
            and not replay_veto
            and not motion_spoof
            and not flash_replay_strong
            and not strong_spoof_evidence
            and not spoof_reason_explicit
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "unstable_non_spoof_guard"
            spoof_streak_frozen = True
            high_replay_risk_blocked = True
        if (
            raw_decision == "SPOOF"
            and not spoof_reason_explicit
            and recovery_after_low_quality
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "recovery_after_low_quality"
            high_replay_risk_blocked = True
        elif (
            raw_decision == "SPOOF"
            and not spoof_reason_explicit
            and unstable_non_spoof
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "unstable_non_spoof"
            high_replay_risk_blocked = True
        elif (
            raw_decision == "SPOOF"
            and not spoof_reason_explicit
            and not strong_spoof_evidence
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            decision_guard_reason = "no_explicit_spoof_reason"

        stable_live_hold_active = False
        if (
            raw_decision == "SPOOF"
            and self._stable_live_hold_frames > 0
            and metrics.face_detected
            and not replay_veto
            and (puzzle_result is None or puzzle_result >= 0.5)
            and window_confidence >= 0.52
        ):
            raw_decision = "LIVE"
            stable_live_hold_active = True
            suspicion_reasons.append("LIVE_HOLD")
        if (
            raw_decision == "LIKELY_SPOOF"
            and not spoof_reason_explicit
            and not strong_spoof_evidence
        ):
            raw_decision = "INSUFFICIENT_EVIDENCE"
            if decision_guard_reason == "-":
                decision_guard_reason = "no_explicit_spoof_reason"
        if screen_replay_confirmed:
            suspicion_reasons.append("SCREEN_REPLAY_CONFIRMED")
            raw_decision = "LIKELY_SPOOF"
            if decision_guard_reason in {"-", "no_explicit_spoof_reason"}:
                decision_guard_reason = screen_replay_confirm_reason

        return {
            "replay_veto": replay_veto,
            "adjusted_score": adjusted_score,
            "debug_active_score": debug_active_score,
            "puzzle_required": bool(temporal_signal_summary.get("puzzle_required")),
            "puzzle_result": puzzle_result,
            "preview_flash_live_response": positive_flash_response,
            "preview_flash_replay_support": negative_flash_response,
            "spoof_reason_explicit": spoof_reason_explicit,
            "strong_spoof_evidence": strong_spoof_evidence,
            "unstable_non_spoof": unstable_non_spoof,
            "decision_guard_reason": decision_guard_reason,
<<<<<<< Updated upstream
            "motion_spoof": motion_spoof,
            "flash_replay_strong": flash_replay_strong,
            "recovery_after_low_quality": recovery_after_low_quality,
            "recovery_frames_left": self._low_quality_recovery_frames,
            "spoof_streak_frozen": spoof_streak_frozen,
            "high_replay_risk_blocked": high_replay_risk_blocked,
=======
            "strong_screen_signal": strong_screen_signal,
            "effective_unstable": effective_unstable,
            "screen_replay_confirmed": screen_replay_confirmed,
            "screen_live_suppressed": screen_live_suppressed,
            "screen_replay_recovery_override": screen_replay_recovery_override,
            "screen_replay_confirm_reason": screen_replay_confirm_reason,
            "screen_replay_soft": screen_replay_soft,
            "screen_replay_soft_votes": screen_replay_soft_votes,
            "screen_replay_soft_recent_ratio": screen_replay_soft_recent_ratio,
            "screen_replay_recent_valid_count": valid_history_count,
            "screen_replay_recent_soft_count": soft_history_count,
            "screen_replay_history_updated_before_decision": screen_replay_history_updated_before_decision,
            "screen_replay_recent_values": screen_replay_recent_values,
            "screen_replay_soft_off_streak": self._screen_replay_soft_off_streak,
            "screen_replay_streak_decay_reason": screen_replay_streak_decay_reason,
            "fresh_detection_recent": fresh_detection_recent,
            "screen_replay_streak": screen_replay_streak,
            "screen_replay_streak_threshold": screen_replay_streak_threshold,
            "screen_replay_votes_detail": ",".join(screen_vote_names) if screen_vote_names else "-",
            "screen_replay_blocked_by_recovery": screen_replay_blocked_by_recovery,
>>>>>>> Stashed changes
            "suspicion_reasons": tuple(dict.fromkeys(suspicion_reasons)),
            "unstable_signal": bool(temporal_signal_summary.get("unstable_signal")),
            "no_face_cooldown_active": no_face_cooldown_active,
            "stable_live_hold_active": stable_live_hold_active,
            "debug_decision_state": raw_decision,
        }

    def _apply_critical_region_visibility_override(
        self,
        *,
        original_status: str,
        face_usable: bool,
        face_usability_reason: str,
        face_usability_state: str,
        face_usability_blocked: bool,
        critical_occ: bool,
        critical_occ_score: float,
        critical_occ_regions: tuple[str, ...],
        critical_region_visibility: dict[str, float],
        critical_occ_streak: int,
        critical_clear_streak: int,
        critical_occ_state: str,
        critical_occ_reason: str,
        liveness_skipped_due_to_face_usability: bool,
    ) -> dict[str, Any]:
        final_status = "NO_FACE" if face_usability_blocked or not face_usable else original_status
        return {
            "face_usable": face_usable,
            "face_usability_reason": face_usability_reason,
            "face_usability_state": face_usability_state,
            "face_usability_blocked": face_usability_blocked,
            "critical_occ": critical_occ,
            "critical_occ_score": critical_occ_score,
            "critical_occ_regions": critical_occ_regions,
            "critical_region_visibility": critical_region_visibility,
            "critical_occ_streak": critical_occ_streak,
            "critical_clear_streak": critical_clear_streak,
            "critical_occ_state": critical_occ_state,
            "critical_occ_reason": critical_occ_reason,
            "original_status_before_occ_gate": original_status,
            "final_status_after_occ_gate": final_status,
            "debug_decision_state": final_status,
            "liveness_skipped_due_to_face_usability": liveness_skipped_due_to_face_usability,
        }

    def _update_debug_state(self, decision_state: str) -> None:
        self._debug_decision_history.append(decision_state)
        self._last_debug_decision_state = decision_state
        if decision_state == "NO_FACE":
            self._no_face_live_cooldown_frames = 6
            self._stable_live_hold_frames = 0
            return
        if self._no_face_live_cooldown_frames > 0:
            self._no_face_live_cooldown_frames -= 1
        if decision_state == "LIVE":
            self._stable_live_hold_frames = 4
        elif self._stable_live_hold_frames > 0:
            self._stable_live_hold_frames -= 1


class LivenessPreviewFrameProcessor:
    """Thin adapter around the existing detector stack for ndarray frames."""

    def __init__(
        self,
        face_detector: IFaceDetector,
        liveness_detector: ILivenessDetector,
        settings: Settings,
        landmark_detector: Optional[ILandmarkDetector] = None,
    ) -> None:
        self._face_detector = face_detector
        self._liveness_detector = liveness_detector
        self._settings = settings
        self._landmark_detector = landmark_detector
        self._frame_index = 0
        self._cached_bounding_box: Optional[tuple[int, int, int, int]] = None
        self._cached_additional_bounding_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self._cached_face_signal_metrics: Any = None
        self._cached_liveness_result: Optional[LivenessResult] = None
        self._last_successful_metrics: Optional[FrameMetrics] = None
        self._last_successful_frame_index = 0
        self._last_successful_bbox: Optional[tuple[int, int, int, int]] = None
        self._last_successful_face_metrics: Optional[FrameMetrics] = None
        self._last_successful_bbox_cleared_reason = "-"
        self._consecutive_face_seen_frames = 0
        self._face_usability_gate = FaceUsabilityGate()
        self._device_spoof_risk_evaluator = DeviceSpoofRiskEvaluator(
            enable_flash_replay=settings.DEV_LIVENESS_PREVIEW_FLASH_REPLAY_ENABLED,
            flash_interval_seconds=settings.DEV_LIVENESS_PREVIEW_FLASH_INTERVAL_SECONDS,
            flash_history_size=settings.DEV_LIVENESS_PREVIEW_FLASH_HISTORY_SIZE,
            replay_fusion_weights={
                "moire": settings.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_MOIRE,
                "reflection": settings.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_REFLECTION,
                "flicker": settings.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLICKER,
                "flash": settings.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLASH,
                "screen_frame": settings.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_SCREEN_FRAME,
            },
        )

    def process_frame(self, frame: np.ndarray) -> FrameMetrics:
        """Synchronously process one frame via the async detector APIs."""
        self._frame_index += 1
        frame_timestamp = time.time()
        profiling: dict[str, float] = {}
        reused_face_detection = False
        reused_landmarks = False
        reused_liveness = False
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        inference_frame, inference_scale = self._resize_for_inference(frame)

        detection_started = time.perf_counter()
        bounding_box: Optional[tuple[int, int, int, int]]
        additional_bounding_boxes: tuple[tuple[int, int, int, int], ...] = ()
        should_detect = (
            self._cached_bounding_box is None
            or self._frame_index == 1
            or (self._frame_index - 1) % self._settings.DEV_LIVENESS_PREVIEW_DETECT_EVERY_N_FRAMES == 0
        )
        try:
            last_successful_bbox_available_before_detection = self._last_successful_bbox is not None
            if should_detect:
                detection = asyncio.run(self._face_detector.detect(inference_frame))
                if not detection.found or detection.bounding_box is None:
                    self._cached_bounding_box = None
                    self._cached_additional_bounding_boxes = ()
                    self._cached_face_signal_metrics = None
                    self._cached_liveness_result = None
                    profiling["face_detection_ms"] = (time.perf_counter() - detection_started) * 1000.0
                    return self._error_metrics(
                        brightness=brightness,
                        blur_score=None,
                        face_detected=False,
                        error="No face detected",
                        profiling=profiling,
                        inference_scale=inference_scale,
                        extra_details=self._face_usability_details(
                            face_usable=False,
                            face_usability_reason="no_face_detected",
                            face_usability_state="NO_FACE",
                            face_usability_blocked=True,
                            critical_occ=False,
                            critical_occ_score=0.0,
                            critical_occ_regions=(),
                            critical_region_visibility={},
                            liveness_skipped_due_to_face_usability=True,
                            preview_bbox=None,
                        ),
                    )
                else:
                    bounding_box = self._expand_bounding_box(
                        self._scale_bounding_box(detection.bounding_box, inference_scale, frame.shape),
                        frame.shape,
                    )
                    additional_bounding_boxes = tuple(
                        self._expand_bounding_box(
                            self._scale_bounding_box(candidate_bbox, inference_scale, frame.shape),
                            frame.shape,
                        )
                        for candidate_bbox in detection.additional_bounding_boxes
                    )
                    self._cached_bounding_box = bounding_box
                    self._cached_additional_bounding_boxes = additional_bounding_boxes
                    self._last_successful_bbox = bounding_box
                    self._last_successful_bbox_cleared_reason = "-"
            else:
                reused_face_detection = True
                bounding_box = self._cached_bounding_box
                additional_bounding_boxes = self._cached_additional_bounding_boxes
        except Exception as exc:
            self._cached_bounding_box = None
            self._cached_additional_bounding_boxes = ()
            self._cached_face_signal_metrics = None
            self._cached_liveness_result = None
            profiling["face_detection_ms"] = (time.perf_counter() - detection_started) * 1000.0
            return self._error_metrics(
                brightness=brightness,
                blur_score=None,
                face_detected=False,
                error=f"Face detection error: {exc}",
                profiling=profiling,
                inference_scale=inference_scale,
            )
        profiling["face_detection_ms"] = (time.perf_counter() - detection_started) * 1000.0

        if bounding_box is None:
            self._cached_face_signal_metrics = None
            self._cached_liveness_result = None
            return self._error_metrics(
                brightness=brightness,
                blur_score=None,
                face_detected=False,
                error="No face detected",
                profiling=profiling,
                inference_scale=inference_scale,
                extra_details=self._face_usability_details(
                    face_usable=False,
                    face_usability_reason="no_face_detected",
                    face_usability_state="NO_FACE",
                    face_usability_blocked=True,
                    critical_occ=False,
                    critical_occ_score=0.0,
                    critical_occ_regions=(),
                    critical_region_visibility={},
                    liveness_skipped_due_to_face_usability=True,
                    preview_bbox=None,
                ),
            )

        face_region = self._crop_face_region(frame, bounding_box)
        if face_region.size == 0:
            self._cached_bounding_box = None
            self._cached_additional_bounding_boxes = ()
            self._cached_face_signal_metrics = None
            self._cached_liveness_result = None
            return self._error_metrics(
                brightness=brightness,
                blur_score=None,
                face_detected=False,
                error="No face detected",
                profiling=profiling,
                inference_scale=inference_scale,
                extra_details=self._face_usability_details(
                    face_usable=False,
                    face_usability_reason="no_face_detected",
                    face_usability_state="NO_FACE",
                    face_usability_blocked=True,
                    critical_occ=False,
                    critical_occ_score=0.0,
                    critical_occ_regions=(),
                    critical_region_visibility={},
                    liveness_skipped_due_to_face_usability=True,
                    preview_bbox=None,
                ),
            )

        blur_started = time.perf_counter()
        blur_score = self._compute_blur(face_region)
        profiling["blur_ms"] = (time.perf_counter() - blur_started) * 1000.0
        face_region_brightness = float(np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)))
        _, _, face_width, face_height = bounding_box
        frame_height, frame_width = frame.shape[:2]
        face_size_ratio = (face_width * face_height) / max(frame_width * frame_height, 1)
        preview_lower_face_texture = self._compute_lower_face_texture(face_region)
        usability_started = time.perf_counter()
        face_usability = self._face_usability_gate.evaluate(
            frame=frame,
            face_bbox=bounding_box,
            preview_details={
                "preview_bbox_x": float(bounding_box[0]),
                "preview_bbox_y": float(bounding_box[1]),
                "preview_bbox_w": float(bounding_box[2]),
                "preview_bbox_h": float(bounding_box[3]),
                "preview_lower_face_texture": preview_lower_face_texture,
            },
            blur_score=blur_score,
        )
        profiling["face_usability_ms"] = (time.perf_counter() - usability_started) * 1000.0
        if not face_usability.usable:
            self._cached_face_signal_metrics = None
            self._cached_liveness_result = None
            self._consecutive_face_seen_frames = 0
            return self._error_metrics(
                brightness=face_region_brightness,
                blur_score=blur_score,
                face_detected=False,
                error="Please keep your full face visible.",
                profiling=profiling,
                inference_scale=inference_scale,
                extra_details=self._face_usability_details(
                    face_usable=face_usability.usable,
                    face_usability_reason=face_usability.reason,
                    face_usability_state=face_usability.state,
                    face_usability_blocked=face_usability.blocked,
                    critical_occ=face_usability.occluded,
                    critical_occ_score=face_usability.occlusion_score,
                    critical_occ_regions=face_usability.occluded_regions,
                    critical_region_visibility=face_usability.visibility_scores,
                    liveness_skipped_due_to_face_usability=True,
                    preview_bbox=bounding_box,
                ),
            )
        landmark_started = time.perf_counter()
        should_extract_landmarks = (
            self._cached_face_signal_metrics is None
            or self._frame_index == 1
            or (self._frame_index - 1) % self._settings.DEV_LIVENESS_PREVIEW_LANDMARK_EVERY_N_FRAMES == 0
        )
        if should_extract_landmarks:
            face_signal_metrics = extract_face_signal_metrics(
                face_region_bgr=face_region,
                landmark_detector=self._landmark_detector,
                face_quality=None,
                blur_score=blur_score,
                brightness=face_region_brightness,
            )
            self._cached_face_signal_metrics = face_signal_metrics
        else:
            reused_landmarks = True
            face_signal_metrics = self._cached_face_signal_metrics
        profiling["landmark_ms"] = (time.perf_counter() - landmark_started) * 1000.0

        liveness_started = time.perf_counter()
        should_run_liveness = (
            self._cached_liveness_result is None
            or self._frame_index == 1
            or (self._frame_index - 1) % self._settings.DEV_LIVENESS_PREVIEW_LIVENESS_EVERY_N_FRAMES == 0
        )
        try:
            if should_run_liveness:
                liveness_result = asyncio.run(self._liveness_detector.check_liveness(face_region))
                self._cached_liveness_result = liveness_result
            else:
                reused_liveness = True
                liveness_result = self._cached_liveness_result
        except FaceNotDetectedError:
            self._cached_bounding_box = None
            self._cached_additional_bounding_boxes = ()
            self._cached_face_signal_metrics = None
            self._cached_liveness_result = None
            profiling["liveness_ms"] = (time.perf_counter() - liveness_started) * 1000.0
            return self._error_metrics(
                brightness=face_region_brightness,
                blur_score=blur_score,
                face_detected=False,
                error="No face detected in cropped region",
                profiling=profiling,
                inference_scale=inference_scale,
                extra_details=self._face_usability_details(
                    face_usable=False,
                    face_usability_reason="no_face_detected",
                    face_usability_state="NO_FACE",
                    face_usability_blocked=True,
                    critical_occ=False,
                    critical_occ_score=0.0,
                    critical_occ_regions=(),
                    critical_region_visibility={},
                    liveness_skipped_due_to_face_usability=True,
                    preview_bbox=bounding_box,
                ),
            )
        except LivenessCheckError as exc:
            self._cached_liveness_result = None
            profiling["liveness_ms"] = (time.perf_counter() - liveness_started) * 1000.0
            return self._error_metrics(
                brightness=face_region_brightness,
                blur_score=blur_score,
                face_detected=True,
                error=str(exc),
                profiling=profiling,
                inference_scale=inference_scale,
            )
        except Exception as exc:
            self._cached_liveness_result = None
            profiling["liveness_ms"] = (time.perf_counter() - liveness_started) * 1000.0
            return self._error_metrics(
                brightness=face_region_brightness,
                blur_score=blur_score,
                face_detected=True,
                error=f"Liveness error: {exc}",
                profiling=profiling,
                inference_scale=inference_scale,
            )
        profiling["liveness_ms"] = (time.perf_counter() - liveness_started) * 1000.0

        metrics = self._build_metrics(
            frame=frame,
            face_region=face_region,
            bounding_box=bounding_box,
            additional_bounding_boxes=additional_bounding_boxes,
            face_usability=face_usability,
            result=liveness_result,
            brightness=face_region_brightness,
            blur_score=blur_score,
            face_signal_metrics=face_signal_metrics,
            face_size_ratio=face_size_ratio,
            profiling=profiling,
            reused_face_detection=reused_face_detection,
            reused_landmarks=reused_landmarks,
            reused_liveness=reused_liveness,
            inference_scale=inference_scale,
            frame_timestamp=frame_timestamp,
            last_successful_bbox_available_before_detection=last_successful_bbox_available_before_detection,
            last_successful_bbox_available_after_detection=self._last_successful_bbox is not None,
            last_successful_bbox_cleared_reason=self._last_successful_bbox_cleared_reason,
        )
        self._last_successful_metrics = metrics
        self._last_successful_face_metrics = metrics
        self._last_successful_frame_index = self._frame_index
        self._consecutive_face_seen_frames += 1
        return metrics

    def get_flash_visual_state(self) -> dict[str, float | str | bool | None]:
        """Expose the current flash debug state to the preview renderer."""
        return self._device_spoof_risk_evaluator.get_flash_visual_state()

    def _should_reuse_cached_box_on_miss(self) -> bool:
        if self._cached_bounding_box is None or self._last_successful_metrics is None:
            return False
        hold_frames = self._settings.DEV_LIVENESS_PREVIEW_HOLD_LAST_SUCCESS_FRAMES
        if hold_frames <= 0:
            return False
        frames_since_success = self._frame_index - self._last_successful_frame_index
        return frames_since_success <= max(2, hold_frames * 2)

    def enrich_device_spoof_with_history(
        self,
        metrics: FrameMetrics,
        recent_entries: list[FrameMetrics],
    ) -> FrameMetrics:
        """Update temporal spoof metrics using the existing preview frame window."""
        if metrics.device_spoof is None:
            return metrics

        temporal_signal_history: list[dict[str, float]] = []
        for entry in [*recent_entries, metrics]:
            sample = self._device_spoof_risk_evaluator.temporal_signal_sample_from_details(entry.details)
            if sample is not None:
                temporal_signal_history.append(sample)

        updated_device_spoof = self._device_spoof_risk_evaluator.update_with_temporal_history(
            metrics.device_spoof,
            temporal_signal_history,
        )
        updated_details = dict(metrics.details)
        updated_details.update(updated_device_spoof.details)
        updated_details.update(updated_device_spoof.to_dict())
        depth_temporal_flat_risk = _compute_depth_temporal_flat_risk([*recent_entries, metrics])
        updated_details["preview_depth_temporal_flat_risk"] = depth_temporal_flat_risk
        updated_details["preview_depth_flat_combined_risk"] = max(
            _maybe_float(updated_details.get("depth_flat_risk")) or 0.0,
            depth_temporal_flat_risk,
        )
        updated_details["preview_spoof_support_count"] = _spoof_support_count_from_details(updated_details)
        updated_details["preview_moire_raw_metric"] = _maybe_float(updated_details.get("moire_score")) or 0.0
        updated_details["preview_moire_normalized_risk"] = _maybe_float(updated_details.get("moire_risk")) or 0.0
        updated_details["preview_strict_moire_support_contribution"] = 0.0
        updated_details["preview_strict_cutout_support_contribution"] = 0.0
        updated_details["preview_weighted_spoof_support_score"] = _weighted_spoof_support_score_from_details(updated_details)
        updated_details["preview_spoof_support_streak"] = _spoof_support_streak(
            [*recent_entries, replace(metrics, details=updated_details, device_spoof=updated_device_spoof)]
        )
        updated_details["preview_support_based_spoof_candidate"] = _is_support_based_spoof_candidate_from_details(
            updated_details,
            updated_device_spoof,
        )
        return replace(
            metrics,
            details=updated_details,
            device_spoof=updated_device_spoof,
        )

    def _resize_for_inference(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        max_side = self._settings.DEV_LIVENESS_PREVIEW_INFERENCE_MAX_SIDE
        if max_side <= 0:
            return frame, 1.0
        height, width = frame.shape[:2]
        current_max_side = max(height, width)
        if current_max_side <= max_side:
            return frame, 1.0
        scale = max_side / float(current_max_side)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _scale_bounding_box(
        bounding_box: tuple[int, int, int, int],
        inference_scale: float,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        if abs(inference_scale - 1.0) <= 1e-6:
            return LivenessPreviewFrameProcessor._clamp_bounding_box(bounding_box, frame_shape)
        x, y, width, height = bounding_box
        scaled = (
            int(round(x / inference_scale)),
            int(round(y / inference_scale)),
            int(round(width / inference_scale)),
            int(round(height / inference_scale)),
        )
        return LivenessPreviewFrameProcessor._clamp_bounding_box(scaled, frame_shape)

    @staticmethod
    def _clamp_bounding_box(
        bounding_box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = bounding_box
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = max(1, min(width, frame_width - x))
        height = max(1, min(height, frame_height - y))
        return (x, y, width, height)

    def _expand_bounding_box(
        self,
        bounding_box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = bounding_box
        side_padding = int(round(width * self._settings.DEV_LIVENESS_PREVIEW_FACE_BOX_SIDE_PADDING_RATIO))
        top_padding = int(round(height * self._settings.DEV_LIVENESS_PREVIEW_FACE_BOX_TOP_PADDING_RATIO))
        bottom_padding = int(round(height * self._settings.DEV_LIVENESS_PREVIEW_FACE_BOX_BOTTOM_PADDING_RATIO))
        expanded = (
            x - side_padding,
            y - top_padding,
            width + (2 * side_padding),
            height + top_padding + bottom_padding,
        )
        return self._clamp_bounding_box(expanded, frame_shape)

    @staticmethod
    def _crop_face_region(frame: np.ndarray, bounding_box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bounding_box
        return frame[y : y + height, x : x + width]

    @staticmethod
    def _compute_blur(face_region: np.ndarray) -> float:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_lower_face_texture(face_region: np.ndarray) -> float:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        if height < 10:
            return 0.0
        return float(cv2.Laplacian(gray[height // 2 :], cv2.CV_64F).var())

    def _error_metrics(
        self,
        *,
        brightness: float,
        blur_score: Optional[float],
        face_detected: bool,
        error: str,
        profiling: Optional[dict[str, float]] = None,
        inference_scale: float = 1.0,
        extra_details: Optional[dict[str, Any]] = None,
    ) -> FrameMetrics:
        last_face_seen_age = (
            float(self._frame_index - self._last_successful_frame_index)
            if self._last_successful_metrics is not None
            else -1.0
        )
        last_successful_bbox_available_before_detection = self._last_successful_bbox is not None
        lowered_error = error.lower()
        previous_face_or_reuse = bool(
            self._last_successful_face_metrics is not None
            and (
                self._last_successful_face_metrics.face_detected
                or self._last_successful_face_metrics.held_from_previous
                or self._last_successful_face_metrics.reused_face_detection
            )
        )
        should_use_no_face_grace = (
            not face_detected
            and self._last_successful_bbox is not None
            and self._last_successful_face_metrics is not None
            and last_face_seen_age >= 0.0
            and last_face_seen_age <= 5.0
            and previous_face_or_reuse
            and ("circuit breaker" in lowered_error)
        )
        if should_use_no_face_grace or self._should_hold_last_success(error):
            return self._hold_last_success(
                error=error,
                profiling=profiling,
                inference_scale=inference_scale,
            )
        if self._last_successful_bbox is not None and last_face_seen_age > 5.0:
            self._last_successful_bbox = None
            self._last_successful_bbox_cleared_reason = "grace_expired"
        self._consecutive_face_seen_frames = 0
        details = {
            "preview_detector_unstable": 1.0 if not face_detected else 0.0,
            "preview_last_face_seen_age": last_face_seen_age,
            "preview_no_face_grace_active": 0.0,
            "preview_no_face_grace_expired": 1.0 if (last_face_seen_age > 5.0 and self._last_successful_metrics is not None) else 0.0,
            "preview_no_face_grace_reason": "-",
            "preview_last_successful_bbox_available": 1.0 if self._last_successful_bbox is not None else 0.0,
            "preview_last_successful_bbox_available_before_detection": 1.0 if last_successful_bbox_available_before_detection else 0.0,
            "preview_last_successful_bbox_available_after_detection": 1.0 if self._last_successful_bbox is not None else 0.0,
            "preview_last_successful_bbox_cleared_reason": self._last_successful_bbox_cleared_reason,
        }
        if extra_details:
            details.update(extra_details)
        return FrameMetrics(
            timestamp=time.time(),
            face_detected=face_detected,
            is_live=False,
            raw_score=0.0,
            confidence=0.0,
            passive_score=0.0,
            active_score=0.0,
            active_evidence=None,
            directional_agreement=None,
            face_quality=None,
            face_size_ratio=None,
            blur_score=blur_score,
            brightness=brightness,
            ear_current=None,
            mar_current=None,
            yaw_current=None,
            pitch_current=None,
            roll_current=None,
            landmark_model=None,
            background_active_mode="error" if face_detected else "no_face",
            background_active_detected=False,
            device_spoof=None,
            details=details,
            error=error,
            profiling=profiling or {},
            inference_scale=inference_scale,
        )

    def _should_hold_last_success(self, error: str) -> bool:
        if self._last_successful_face_metrics is None or self._last_successful_bbox is None:
            return False
        max_hold_frames = 5
        if max_hold_frames <= 0:
            return False
        frames_since_success = self._frame_index - self._last_successful_frame_index
        if frames_since_success > max_hold_frames:
            return False
        previous_face_or_reuse = bool(
            self._last_successful_face_metrics.face_detected
            or self._last_successful_face_metrics.held_from_previous
            or self._last_successful_face_metrics.reused_face_detection
        )
        if not previous_face_or_reuse:
            return False
        lowered_error = error.lower()
        return "circuit breaker" in lowered_error

    def _hold_last_success(
        self,
        *,
        error: str,
        profiling: Optional[dict[str, float]],
        inference_scale: float,
    ) -> FrameMetrics:
        assert self._last_successful_face_metrics is not None
        previous = self._last_successful_face_metrics
        details = dict(previous.details)
        details["preview_hold_last_success"] = True
        details["preview_hold_error"] = error
        details["preview_detector_unstable"] = 1.0
        details["preview_last_face_seen_age"] = float(self._frame_index - self._last_successful_frame_index)
        details["preview_no_face_grace_active"] = 1.0
        details["preview_no_face_grace_expired"] = 0.0
        details["preview_no_face_grace_reason"] = "last_bbox_recent"
        details["preview_last_successful_bbox_available"] = 1.0
        details["preview_last_successful_bbox_available_before_detection"] = 1.0
        details["preview_last_successful_bbox_available_after_detection"] = 1.0
        details["preview_last_successful_bbox_cleared_reason"] = self._last_successful_bbox_cleared_reason
        return FrameMetrics(
            timestamp=time.time(),
            face_detected=True,
            is_live=previous.is_live,
            raw_score=previous.raw_score,
            confidence=previous.confidence,
            passive_score=previous.passive_score,
            active_score=previous.active_score,
            active_evidence=previous.active_evidence,
            directional_agreement=previous.directional_agreement,
            face_quality=previous.face_quality,
            face_size_ratio=previous.face_size_ratio,
            blur_score=previous.blur_score,
            brightness=previous.brightness,
            ear_current=previous.ear_current,
            mar_current=previous.mar_current,
            yaw_current=previous.yaw_current,
            pitch_current=previous.pitch_current,
            roll_current=previous.roll_current,
            landmark_model=previous.landmark_model,
            background_active_mode=previous.background_active_mode,
            background_active_detected=previous.background_active_detected,
            device_spoof=previous.device_spoof,
            details=details,
            error=f"{error} (holding last success)",
            profiling=profiling or {},
            reused_face_detection=True,
            reused_landmarks=True,
            reused_liveness=True,
            held_from_previous=True,
            inference_scale=inference_scale,
        )

    def _build_metrics(
        self,
        *,
        frame: np.ndarray,
        face_region: np.ndarray,
        bounding_box: tuple[int, int, int, int],
        additional_bounding_boxes: tuple[tuple[int, int, int, int], ...],
        face_usability,
        result: LivenessResult,
        brightness: float,
        blur_score: float,
        face_signal_metrics,
        face_size_ratio: Optional[float],
        profiling: Optional[dict[str, float]] = None,
        reused_face_detection: bool = False,
        reused_landmarks: bool = False,
        reused_liveness: bool = False,
        inference_scale: float = 1.0,
        frame_timestamp: Optional[float] = None,
        last_successful_bbox_available_before_detection: bool = False,
        last_successful_bbox_available_after_detection: bool = False,
        last_successful_bbox_cleared_reason: str = "-",
    ) -> FrameMetrics:
        details = dict(result.details)
        details.update(face_signal_metrics.to_dict())
        # Laplacian variance of the bottom half of the face crop Ã¢â‚¬â€ used to detect a hand or
        # object covering the mouth/nose while the eyes remain open (EAR is still normal but
        # the lower face region becomes much more uniform than real facial skin texture).
        details["preview_lower_face_texture"] = self._compute_lower_face_texture(face_region)
        face_quality = _coalesce_float(_maybe_float(details.get("face_quality")), face_signal_metrics.face_quality)
        ear_current = _coalesce_float(_maybe_float(details.get("ear_current")), face_signal_metrics.ear_current)
        mar_current = _coalesce_float(_maybe_float(details.get("mar_current")), face_signal_metrics.mar_current)
        yaw_current = _coalesce_float(_maybe_float(details.get("yaw_current")), face_signal_metrics.yaw_current)
        pitch_current = _coalesce_float(_maybe_float(details.get("pitch_current")), face_signal_metrics.pitch_current)
        roll_current = _coalesce_float(_maybe_float(details.get("roll_current")), face_signal_metrics.roll_current)
        detector_active_score = float(details.get("active_score") or 0.0)
        detector_active_evidence = _maybe_float(details.get("active_evidence"))
        frame_active_evidence = self._compute_preview_frame_active_evidence(
            details=details,
            face_quality=face_quality,
            face_size_ratio=face_size_ratio,
            ear_current=ear_current,
            mar_current=mar_current,
            yaw_current=yaw_current,
        )
        standard_frame_active_score = _standard_active_support_score(frame_active_evidence)
        frame_active_score = standard_frame_active_score
        details["detector_active_score"] = detector_active_score
        details["detector_active_evidence"] = detector_active_evidence
        details["preview_frame_active_score"] = frame_active_score
        details["preview_frame_active_evidence"] = frame_active_evidence
        details["preview_frame_active_score_standard"] = standard_frame_active_score
        details["preview_frame_active_score_mapping"] = "standard_sqrt"
        details["security_profile"] = PREVIEW_SECURITY_PROFILE
        details["preview_bbox_x"] = float(bounding_box[0])
        details["preview_bbox_y"] = float(bounding_box[1])
        details["preview_bbox_w"] = float(bounding_box[2])
        details["preview_bbox_h"] = float(bounding_box[3])
        details.update(
            self._face_usability_details(
                face_usable=face_usability.usable,
                face_usability_reason=face_usability.reason,
                face_usability_state=face_usability.state,
                face_usability_blocked=face_usability.blocked,
                critical_occ=face_usability.occluded,
                critical_occ_score=face_usability.occlusion_score,
                critical_occ_regions=face_usability.occluded_regions,
                critical_region_visibility=face_usability.visibility_scores,
                liveness_skipped_due_to_face_usability=False,
                preview_bbox=bounding_box,
                critical_occ_streak=face_usability.occlusion_streak,
                critical_clear_streak=face_usability.clear_streak,
            )
        )
        details["preview_detector_unstable"] = 0.0
        details["preview_last_face_seen_age"] = 0.0
        details["preview_no_face_grace_active"] = 0.0
        details["preview_no_face_grace_expired"] = 0.0
        details["preview_no_face_grace_reason"] = "-"
        details["preview_last_successful_bbox_available"] = 1.0 if self._last_successful_bbox is not None else 0.0
        details["preview_last_successful_bbox_available_before_detection"] = 1.0 if last_successful_bbox_available_before_detection else 0.0
        details["preview_last_successful_bbox_available_after_detection"] = 1.0 if last_successful_bbox_available_after_detection else 0.0
        details["preview_last_successful_bbox_cleared_reason"] = last_successful_bbox_cleared_reason
        details["preview_additional_bboxes"] = [
            [float(x), float(y), float(width), float(height)]
            for x, y, width, height in additional_bounding_boxes
        ]
        device_spoof = self._device_spoof_risk_evaluator.evaluate(
            frame_bgr=frame,
            face_region_bgr=face_region,
            face_bounding_box=bounding_box,
            frame_timestamp=frame_timestamp,
            flash_skin_regions=face_signal_metrics.flash_skin_regions,
            bbox_reused=reused_face_detection,
        )
        details.update(device_spoof.to_dict())
        return FrameMetrics(
            timestamp=float(frame_timestamp or time.time()),
            face_detected=True,
            is_live=result.is_live,
            raw_score=result.score,
            confidence=result.confidence,
            passive_score=float(details.get("passive_score") or 0.0),
            active_score=frame_active_score,
            active_evidence=frame_active_evidence,
            directional_agreement=_maybe_float(details.get("directional_agreement")),
            face_quality=face_quality,
            face_size_ratio=face_size_ratio,
            blur_score=blur_score,
            brightness=brightness,
            ear_current=ear_current,
            mar_current=mar_current,
            yaw_current=yaw_current,
            pitch_current=pitch_current,
            roll_current=roll_current,
            landmark_model=details.get("landmark_model") or face_signal_metrics.landmark_model,
            background_active_mode=details.get("background_active_mode") or result.challenge,
            background_active_detected=bool(
                details.get("background_active_reaction_detected")
                if details.get("background_active_reaction_detected") is not None
                else result.challenge_completed
            ),
            device_spoof=device_spoof,
            details=details,
            error=None,
            profiling=profiling or {},
            reused_face_detection=reused_face_detection,
            reused_landmarks=reused_landmarks,
            reused_liveness=reused_liveness,
            held_from_previous=False,
            inference_scale=inference_scale,
        )

    @staticmethod
    def _face_usability_details(
        *,
        face_usable: bool,
        face_usability_reason: str,
        face_usability_state: str,
        face_usability_blocked: bool,
        critical_occ: bool,
        critical_occ_score: float,
        critical_occ_regions: tuple[str, ...],
        critical_region_visibility: dict[str, float],
        liveness_skipped_due_to_face_usability: bool,
        preview_bbox: Optional[tuple[int, int, int, int]],
        critical_occ_streak: int = 0,
        critical_clear_streak: int = 0,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "face_usable": 1.0 if face_usable else 0.0,
            "face_usability_reason": face_usability_reason,
            "face_usability_state": face_usability_state,
            "face_usability_blocked": 1.0 if face_usability_blocked else 0.0,
            "critical_occ": 1.0 if critical_occ else 0.0,
            "critical_occ_score": float(critical_occ_score),
            "critical_occ_regions": list(critical_occ_regions),
            "critical_occ_streak": float(critical_occ_streak),
            "critical_clear_streak": float(critical_clear_streak),
            "critical_occ_state": face_usability_state,
            "critical_occ_reason": face_usability_reason,
            "critical_region_visibility": dict(critical_region_visibility),
            "liveness_skipped_due_to_face_usability": 1.0 if liveness_skipped_due_to_face_usability else 0.0,
        }
        for region_name in ("left_eye", "right_eye", "nose", "mouth", "lower_face"):
            details[f"critical_vis_{region_name}"] = float(critical_region_visibility.get(region_name, 0.0))
        if preview_bbox is not None:
            details["preview_bbox_x"] = float(preview_bbox[0])
            details["preview_bbox_y"] = float(preview_bbox[1])
            details["preview_bbox_w"] = float(preview_bbox[2])
            details["preview_bbox_h"] = float(preview_bbox[3])
        return details

    def _compute_preview_frame_active_evidence(
        self,
        *,
        details: dict[str, Any],
        face_quality: Optional[float],
        face_size_ratio: Optional[float],
        ear_current: Optional[float],
        mar_current: Optional[float],
        yaw_current: Optional[float],
    ) -> float:
        blink_score = _maybe_float(details.get("blink"))
        smile_score = _maybe_float(details.get("smile"))
        mar_baseline = _maybe_float(details.get("mar_baseline"))
        yaw_baseline = _maybe_float(details.get("yaw_baseline"))

        blink_support = _clamp01((blink_score - 70.0) / 22.0) if blink_score is not None else 0.0
        smile_support = _clamp01((smile_score - 58.0) / 26.0) if smile_score is not None else 0.0

        mouth_support = 0.0
        if mar_current is not None:
            if mar_baseline is not None and mar_baseline > 1e-6:
                mouth_support = _clamp01((mar_current / mar_baseline - 1.18) / 0.50)
            else:
                mouth_support = _clamp01((mar_current - 0.36) / 0.22)

        head_support = 0.0
        if yaw_current is not None:
            yaw_delta = abs(yaw_current - (yaw_baseline or 0.0))
            head_support = _clamp01((yaw_delta - 7.0) / 11.0)

        ranked = sorted(
            [blink_support, smile_support, mouth_support, head_support],
            reverse=True,
        )
        primary = ranked[0] if ranked else 0.0
        secondary = ranked[1] if len(ranked) > 1 else 0.0
        raw_frame_evidence = _clamp01(0.70 * primary + 0.30 * secondary)

        size_trust = _clamp01((face_size_ratio or 0.0) / max(self._settings.DEV_LIVENESS_PREVIEW_MIN_TRUSTED_FACE_SIZE_RATIO, 1e-6))
        quality_trust = _clamp01(face_quality or 0.0)
        effective_trust = _clamp01(0.70 + 0.20 * size_trust + 0.10 * quality_trust)
        return _clamp01(raw_frame_evidence * effective_trust)


class LiveLivenessPreview:
    """Webcam preview window for observing frame and temporal liveness metrics."""

    WINDOW_NAME = "Dev Liveness Preview"
    OVERLAY_FONT_SCALE = 0.43
    OVERLAY_LINE_HEIGHT = 18
    OVERLAY_TEXT_THICKNESS = 1
    def __init__(
        self,
        *,
        settings: Settings,
        frame_processor: LivenessPreviewFrameProcessor,
        temporal_aggregator: TemporalLivenessAggregator,
    ) -> None:
        self._settings = settings
        self._frame_processor = frame_processor
        self._temporal_aggregator = temporal_aggregator
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the preview loop in a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.debug("Dev liveness preview already running")
            return

        self._thread = threading.Thread(
            target=self.run,
            name="dev-liveness-preview",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the preview thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run(self) -> None:
        """Open the webcam and continuously render preview overlays."""
        logger.info(
            "Starting dev liveness preview: profile=%s camera_index=%s window_seconds=%.2f max_entries=%s ema_alpha=%.2f infer_max_side=%s detect_every=%s landmark_every=%s liveness_every=%s",
            PREVIEW_SECURITY_PROFILE,
            self._settings.DEV_LIVENESS_PREVIEW_CAMERA_INDEX,
            self._temporal_aggregator.window_seconds,
            self._temporal_aggregator.max_entries,
            self._settings.DEV_LIVENESS_PREVIEW_EMA_ALPHA,
            self._settings.DEV_LIVENESS_PREVIEW_INFERENCE_MAX_SIDE,
            self._settings.DEV_LIVENESS_PREVIEW_DETECT_EVERY_N_FRAMES,
            self._settings.DEV_LIVENESS_PREVIEW_LANDMARK_EVERY_N_FRAMES,
            self._settings.DEV_LIVENESS_PREVIEW_LIVENESS_EVERY_N_FRAMES,
        )
        capture = cv2.VideoCapture(self._settings.DEV_LIVENESS_PREVIEW_CAMERA_INDEX)
        if not capture.isOpened():
            logger.warning("Dev liveness preview could not open webcam %s", self._settings.DEV_LIVENESS_PREVIEW_CAMERA_INDEX)
            return

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_frame_started = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                capture_started = time.perf_counter()
                ok, frame = capture.read()
                capture_ms = (time.perf_counter() - capture_started) * 1000.0
                if not ok or frame is None:
                    logger.warning("Dev liveness preview failed to read webcam frame")
                    break

                frame_started = time.perf_counter()
                display_fps = 1.0 / max(frame_started - last_frame_started, 1e-6)
                last_frame_started = frame_started

                inference_started = time.perf_counter()
                frame_metrics = self._frame_processor.process_frame(frame)
                recent_entries = self._temporal_aggregator.get_recent_entries(now=frame_metrics.timestamp)
                frame_metrics = self._frame_processor.enrich_device_spoof_with_history(frame_metrics, recent_entries)
                aggregate = self._temporal_aggregator.add(frame_metrics)
                inference_elapsed = time.perf_counter() - inference_started
                inference_fps = 1.0 / max(inference_elapsed, 1e-6)
                overlay_started = time.perf_counter()
                overlay = self._render_overlay(frame, frame_metrics, aggregate, display_fps, inference_fps)
                overlay_ms = (time.perf_counter() - overlay_started) * 1000.0
                cv2.imshow(self.WINDOW_NAME, overlay)

                if self._window_closed():
                    break

                frame_count += 1
                self._maybe_log_frame(
                    frame_count,
                    frame_metrics,
                    aggregate,
                    display_fps=display_fps,
                    inference_fps=inference_fps,
                    capture_ms=capture_ms,
                    overlay_ms=overlay_ms,
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    puzzle_summary = self._temporal_aggregator.start_puzzle_session()
                    if puzzle_summary is not None:
                        logger.info(
                            "dev_liveness_preview puzzle_start status=%s steps=%s sequence=%s",
                            puzzle_summary.status,
                            puzzle_summary.total_steps,
                            puzzle_summary.sequence_label,
                        )
                    continue
                if key == ord("r"):
                    self._temporal_aggregator.reset_puzzle_session()
                    logger.info("dev_liveness_preview puzzle_reset")
                    continue
                if key in (ord("q"), 27) or self._window_closed():
                    break
        finally:
            capture.release()
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            logger.info("Dev liveness preview stopped")

    def _maybe_log_frame(
        self,
        frame_count: int,
        frame_metrics: FrameMetrics,
        aggregate: AggregatedMetrics,
        *,
        display_fps: float,
        inference_fps: float,
        capture_ms: float,
        overlay_ms: float,
    ) -> None:
        cadence = max(1, int(self._settings.DEV_LIVENESS_PREVIEW_LOG_EVERY_N_FRAMES))
        if frame_count > 3 and frame_count % cadence != 0:
            return

        logger.info(
            "dev_liveness_preview frame=%s raw=%.1f smooth=%.1f final_active=%.2f final_active_score=%.1f conf=%.2f stable_live=%.2f display_fps=%.1f inference_fps=%.1f capture_ms=%.1f detect_ms=%.1f landmark_ms=%.1f liveness_ms=%.1f bg_ms=%.1f agg_ms=%.1f overlay_ms=%.1f reuse=det:%s lm:%s liv:%s scale=%.2f face=%s error=%s",
            frame_count,
            frame_metrics.raw_score,
            aggregate.smoothed_score,
            aggregate.final_active_evidence,
            aggregate.final_active_score,
            aggregate.window_confidence,
            aggregate.stable_live_ratio,
            display_fps,
            inference_fps,
            capture_ms,
            frame_metrics.profiling.get("face_detection_ms", 0.0),
            frame_metrics.profiling.get("landmark_ms", 0.0),
            frame_metrics.profiling.get("liveness_ms", 0.0),
            aggregate.background_reaction_ms,
            aggregate.temporal_aggregation_ms,
            overlay_ms,
            frame_metrics.reused_face_detection,
            frame_metrics.reused_landmarks,
            frame_metrics.reused_liveness,
            frame_metrics.inference_scale,
            frame_metrics.face_detected,
            frame_metrics.error,
        )
        logger.info(
            "dev_liveness_preview metrics %s",
            " | ".join(
                [
                    f"profile={PREVIEW_SECURITY_PROFILE}",
                    f"status={aggregate.decision_state}",
                    f"original_status_before_occ_gate={aggregate.original_status_before_occ_gate}",
                    f"final_status_after_occ_gate={aggregate.final_status_after_occ_gate}",
                    f"frame_score={frame_metrics.raw_score:.1f}",
                    f"smoothed={aggregate.smoothed_score:.1f}",
                    f"frame_conf={frame_metrics.frame_confidence:.2f}",
                    f"window_conf={aggregate.window_confidence:.2f}",
                    f"passive={frame_metrics.passive_score:.1f}",
                    f"passive_win={aggregate.passive_window_score:.1f}",
                    f"frame_active={frame_metrics.active_score:.1f}",
                    f"frame_active_ev={_format_optional(frame_metrics.active_evidence)}",
                    f"frame_active_std={_format_optional(_maybe_float(frame_metrics.details.get('preview_frame_active_score_standard')))}",
                    f"detector_active={_format_optional(_maybe_float(frame_metrics.details.get('detector_active_score')))}",
                    f"detector_active_ev={_format_optional(_maybe_float(frame_metrics.details.get('detector_active_evidence')))}",
                    f"bg_active_raw={_format_optional(aggregate.raw_active_evidence)}",
                    f"bg_active_temp={_format_optional(aggregate.background_active_evidence)}",
                    f"bg_active_score={aggregate.active_score_standard_mapping:0.1f}",
                    f"puzzle_status={aggregate.puzzle_status}",
                    f"puzzle_step={aggregate.puzzle_current_step}",
                    f"puzzle_progress={aggregate.puzzle_progress:0.2f}",
                    f"puzzle_ev={aggregate.puzzle_active_evidence:0.2f}",
                    f"puzzle_conf={aggregate.puzzle_confidence:0.2f}",
                    f"fusion_active={int(aggregate.puzzle_fusion_active)}",
                    f"final_active={aggregate.final_active_evidence:0.2f}",
                    f"final_active_score={aggregate.final_active_score:0.1f}",
                    f"final_supported={aggregate.final_supported_score:0.1f}",
                    f"replay_veto={int(aggregate.replay_veto)}",
                    f"adjusted_score={aggregate.adjusted_score:0.1f}",
                    f"debug_active={aggregate.debug_active_score:0.1f}",
                    f"puzzle_required={int(aggregate.puzzle_required)}",
                    f"puzzle_result={_format_optional(aggregate.puzzle_result)}",
                    f"flash_live_resp={int(aggregate.preview_flash_live_response)}",
                    f"flash_replay_sup={int(aggregate.preview_flash_replay_support)}",
                    f"spoof_reason_explicit={int(aggregate.spoof_reason_explicit)}",
                    f"strong_spoof_evidence={int(aggregate.strong_spoof_evidence)}",
                    f"unstable_non_spoof={int(aggregate.unstable_non_spoof)}",
                    f"decision_guard_reason={aggregate.decision_guard_reason}",
<<<<<<< Updated upstream
                    f"motion_spoof={int(aggregate.motion_spoof)}",
                    f"flash_replay_strong={int(aggregate.flash_replay_strong)}",
                    f"recovery_after_low_quality={int(aggregate.recovery_after_low_quality)}",
                    f"recovery_frames_left={aggregate.recovery_frames_left}",
                    f"spoof_streak_frozen={int(aggregate.spoof_streak_frozen)}",
                    f"high_replay_risk_blocked={int(aggregate.high_replay_risk_blocked)}",
=======
                    f"strong_screen_signal={int(aggregate.strong_screen_signal)}",
                    f"effective_unstable={int(aggregate.effective_unstable)}",
                    f"screen_replay_confirmed={int(aggregate.screen_replay_confirmed)}",
                    f"screen_live_suppressed={int(aggregate.screen_live_suppressed)}",
                    f"screen_replay_recovery_override={int(aggregate.screen_replay_recovery_override)}",
                    f"screen_replay_confirm_reason={aggregate.screen_replay_confirm_reason}",
                    f"screen_replay_soft={int(aggregate.screen_replay_soft)}",
                    f"screen_replay_soft_votes={aggregate.screen_replay_soft_votes}",
                    f"screen_replay_soft_recent_ratio={aggregate.screen_replay_soft_recent_ratio:.2f}",
                    f"screen_replay_recent_valid_count={aggregate.screen_replay_recent_valid_count}",
                    f"screen_replay_recent_soft_count={aggregate.screen_replay_recent_soft_count}",
                    f"screen_replay_history_updated_before_decision={int(aggregate.screen_replay_history_updated_before_decision)}",
                    f"screen_replay_recent_values={aggregate.screen_replay_recent_values}",
                    f"screen_replay_soft_off_streak={aggregate.screen_replay_soft_off_streak}",
                    f"screen_replay_streak_decay_reason={aggregate.screen_replay_streak_decay_reason}",
                    f"fresh_detection_recent={int(aggregate.fresh_detection_recent)}",
                    f"screen_replay_streak={aggregate.screen_replay_streak}",
                    f"screen_replay_streak_threshold={aggregate.screen_replay_streak_threshold}",
                    f"screen_replay_votes_detail={aggregate.screen_replay_votes_detail}",
                    f"screen_replay_blocked_by_recovery={int(aggregate.screen_replay_blocked_by_recovery)}",
>>>>>>> Stashed changes
                    f"flash_match={_format_optional(_maybe_float(frame_metrics.details.get('flash_color_match_score')))}",
                    f"flash_skin_mask_coverage={_format_optional(_maybe_float(frame_metrics.details.get('flash_skin_mask_coverage')))}",
                    f"flash_channel_response={_format_optional(_maybe_float(frame_metrics.details.get('flash_channel_response')))}",
                    f"flash_baseline_frames={_format_optional(_maybe_float(frame_metrics.details.get('flash_baseline_frames')))}",
                    f"flash_response_samples={_format_optional(_maybe_float(frame_metrics.details.get('flash_response_sample_count')))}",
                    f"flash_specular={_format_optional(_maybe_float(frame_metrics.details.get('specular_hotspot_risk')))}",
                    f"flash_diffuse={_format_optional(_maybe_float(frame_metrics.details.get('diffuse_response_score')))}",
                    f"flash_geom={_format_optional(_maybe_float(frame_metrics.details.get('geometry_response_consistency')))}",
                    f"flash_planar={_format_optional(_maybe_float(frame_metrics.details.get('planar_surface_risk')))}",
<<<<<<< Updated upstream
                    f"held_from_previous={int(frame_metrics.held_from_previous)}",
                    f"detector_unstable={int(bool(_maybe_float(frame_metrics.details.get('preview_detector_unstable')) or 0.0))}",
                    f"last_face_seen_age={_format_optional(_maybe_float(frame_metrics.details.get('preview_last_face_seen_age')))}",
                    f"no_face_grace_active={int(bool(_maybe_float(frame_metrics.details.get('preview_no_face_grace_active')) or 0.0))}",
                    f"no_face_grace_expired={int(bool(_maybe_float(frame_metrics.details.get('preview_no_face_grace_expired')) or 0.0))}",
                    f"no_face_grace_reason={frame_metrics.details.get('preview_no_face_grace_reason') or '-'}",
                    f"last_successful_bbox_available={int(bool(_maybe_float(frame_metrics.details.get('preview_last_successful_bbox_available')) or 0.0))}",
                    f"last_successful_bbox_available_before_detection={int(bool(_maybe_float(frame_metrics.details.get('preview_last_successful_bbox_available_before_detection')) or 0.0))}",
                    f"last_successful_bbox_available_after_detection={int(bool(_maybe_float(frame_metrics.details.get('preview_last_successful_bbox_available_after_detection')) or 0.0))}",
                    f"last_successful_bbox_cleared_reason={frame_metrics.details.get('preview_last_successful_bbox_cleared_reason') or '-'}",
=======
                    f"flash_skin_mask_coverage={_format_optional(_maybe_float(frame_metrics.details.get('flash_skin_mask_coverage')))}",
                    f"flash_channel_response={_format_optional(_maybe_float(frame_metrics.details.get('flash_channel_response')))}",
                    f"flash_baseline_frames={_format_optional(_maybe_float(frame_metrics.details.get('flash_baseline_frames')))}",
                    f"flash_response_samples={_format_optional(_maybe_float(frame_metrics.details.get('flash_response_samples')))}",
>>>>>>> Stashed changes
                    f"unstable_signal={int(aggregate.unstable_signal)}",
                    f"no_face_cooldown={int(aggregate.no_face_cooldown_active)}",
                    f"stable_live_hold={int(aggregate.stable_live_hold_active)}",
                    f"suspicion={','.join(aggregate.suspicion_reasons) if aggregate.suspicion_reasons else '-'}",
                    f"moire={_format_optional(_device_spoof_value(frame_metrics, 'moire_risk'))}",
                    f"moire_raw={_format_optional(_maybe_float(frame_metrics.details.get('preview_moire_raw_metric')))}",
                    f"moire_sel={_format_optional(_maybe_float(frame_metrics.details.get('moire_orientation_selectivity')))}",
                    f"moire_fft={_format_optional(_maybe_float(frame_metrics.details.get('moire_fft_risk')))}",
                    f"spoof_support_w={_format_optional(_maybe_float(frame_metrics.details.get('preview_weighted_spoof_support_score')))}",
                    f"depth_flat={_format_optional(_maybe_float(frame_metrics.details.get('preview_depth_flat_combined_risk')))}",
                    f"reflection={_format_optional(_device_spoof_value(frame_metrics, 'reflection_risk'))}",
                    f"cutout={_format_optional(_device_spoof_value(frame_metrics, 'cutout_spoof_support'))}",
                    f"hole_cutout={_format_optional(_device_spoof_value(frame_metrics, 'hole_cutout_risk'))}",
                    f"focal_blur={_format_optional(_device_spoof_value(frame_metrics, 'focal_blur_anomaly_risk'))}",
                    f"screen_frame={_format_optional(_device_spoof_value(frame_metrics, 'screen_frame_risk'))}",
                    f"screen_conf={int(_is_confirmed_screen_device(frame_metrics))}",
                    f"reflect_clip={_format_optional(_maybe_float(frame_metrics.details.get('reflection_clipped_ratio')))}",
                    f"reflect_compact={_format_optional(_maybe_float(frame_metrics.details.get('reflection_compact_highlight_score')))}",
                    f"reflect_glossy={_format_optional(_maybe_float(frame_metrics.details.get('reflection_glossy_patch_ratio')))}",
                    f"flicker={_format_optional(_device_spoof_value(frame_metrics, 'flicker_risk'))}",
                    f"device_replay={_format_optional(_device_spoof_value(frame_metrics, 'device_replay_risk'))}",
                    f"sf_hi={int(_is_screen_frame_high(frame_metrics))}",
                    f"moire_hi={int(_is_moire_high(frame_metrics))}",
                    f"depth_hi={int(_is_depth_flat(frame_metrics))}",
                    f"reflect_hi={int(_is_reflection_high(frame_metrics))}",
                    f"flicker_hi={int(_is_flicker_high(frame_metrics))}",
                    f"cutout_hi={int(_is_cutout_high(frame_metrics))}",
                    f"uniface_neg={int(_is_uniface_negative(frame_metrics))}",
                    f"spoof_support={_spoof_support_count(frame_metrics)}",
                    f"spoof_streak={int(_maybe_float(frame_metrics.details.get('preview_spoof_support_streak')) or 0.0)}",
                    f"spoof_gate={int(_is_device_replay_spoof_detected(frame_metrics))}",
                    f"bbox={_format_bbox(frame_metrics)}",
                    f"bbox_reuse={int(frame_metrics.reused_face_detection)}",
                    f"face={int(frame_metrics.face_detected)}",
                    f"face_usable={int(aggregate.face_usable)}",
                    f"face_usability_reason={aggregate.face_usability_reason}",
                    f"face_usability_state={aggregate.face_usability_state}",
                    f"face_usability_blocked={int(aggregate.face_usability_blocked)}",
                    f"liveness_skipped_due_to_face_usability={int(aggregate.liveness_skipped_due_to_face_usability)}",
                    f"face_quality={_format_optional(frame_metrics.face_quality)}",
                    f"face_size={_format_optional(frame_metrics.face_size_ratio)}",
                    f"brightness={frame_metrics.brightness:.1f}",
                    f"blur={_format_optional(frame_metrics.blur_score)}",
                    f"ear={_format_optional(frame_metrics.ear_current)}",
                    f"mar={_format_optional(frame_metrics.mar_current)}",
                    f"yaw={_format_optional(frame_metrics.yaw_current)}",
                    f"pitch={_format_optional(frame_metrics.pitch_current)}",
                    f"roll={_format_optional(frame_metrics.roll_current)}",
                    f"baseline={'ready' if aggregate.baseline_ready else 'calibrating'}",
                    f"baseline_n={aggregate.baseline_sample_count}",
                    f"ear_base={_format_optional(aggregate.ear_baseline)}",
                    f"mar_base={_format_optional(aggregate.mar_baseline)}",
                    f"smile_base={_format_optional(aggregate.smile_baseline)}",
                    f"yaw_base={_format_optional(aggregate.yaw_baseline)}",
                    f"ear_drop={_format_optional(aggregate.ear_drop_ratio)}",
                    f"mar_rise={_format_optional(aggregate.mar_rise_ratio)}",
                    f"blink_ev={_format_optional(aggregate.blink_evidence)}",
                    f"smile_ev={_format_optional(aggregate.smile_evidence)}",
                    f"mouth_ev={_format_optional(aggregate.mouth_open_evidence)}",
                    f"turn_left_ev={_format_optional(aggregate.head_turn_left_evidence)}",
                    f"turn_right_ev={_format_optional(aggregate.head_turn_right_evidence)}",
                    f"primary={aggregate.primary_event:.2f}",
                    f"secondary={aggregate.secondary_event:.2f}",
                    f"raw_react={aggregate.raw_reaction_evidence:.2f}",
                    f"base_trust={aggregate.base_active_trust:.2f}",
                    f"trust_pen={aggregate.trust_penalty:.2f}",
                    f"trust={aggregate.effective_trust:.2f}",
                    f"trusted_react={aggregate.trusted_reaction_evidence:.2f}",
                    f"blink_anom={aggregate.blink_anomaly_score:.2f}",
                    f"motion_anom={aggregate.motion_anomaly_score:.2f}",
                    f"signal_incons={aggregate.signal_inconsistency_score:.2f}",
                    f"active_spoof_sup={aggregate.spoof_support_score:.2f}",
                    f"persist1={aggregate.persisted_primary:.2f}",
                    f"persist2={aggregate.persisted_secondary:.2f}",
                    f"persist_react={aggregate.persisted_reaction_evidence:.2f}",
                    f"variance={aggregate.score_variance:.2f}",
                    f"score_mean={aggregate.score_mean:.1f}",
                    f"ema={aggregate.ema_score:.1f}",
                    f"temp_consistency={aggregate.temporal_consistency:.2f}",
                    f"dir_agreement={_format_optional(frame_metrics.directional_agreement)}",
                    f"dir_agreement_mean={aggregate.directional_agreement_mean:.2f}",
                    f"face_quality_mean={aggregate.face_quality_mean:.2f}",
                    f"face_size_mean={aggregate.face_size_mean:.3f}",
                    f"face_size_adequacy={aggregate.face_size_adequacy:.2f}",
                    f"blur_adequacy={aggregate.blur_adequacy:.2f}",
                    f"brightness_adequacy={aggregate.brightness_adequacy:.2f}",
                    f"stable_live={aggregate.stable_live_ratio:.2f}",
                    f"face_ratio={aggregate.face_present_ratio:.2f}",
                    f"no_face={aggregate.consecutive_no_face_frames}",
                    f"occluded={aggregate.consecutive_occluded_frames}",
                    f"critical_occ={int(aggregate.critical_occ)}",
                    f"critical_occ_score={aggregate.critical_occ_score:.2f}",
                    f"critical_occ_regions={','.join(aggregate.critical_occ_regions) if aggregate.critical_occ_regions else '-'}",
                    f"critical_occ_streak={aggregate.critical_occ_streak}",
                    f"critical_clear_streak={aggregate.critical_clear_streak}",
                    f"critical_occ_state={aggregate.critical_occ_state}",
                    f"critical_occ_reason={aggregate.critical_occ_reason}",
                    f"lm_vis={_format_optional(frame_metrics.details.get('landmark_visibility_score'))}",
                    f"recovery={'warm' if aggregate.warm_recovery else 'normal'}",
                    f"evidence={'sufficient' if aggregate.sufficient_evidence else 'pending'}",
                    f"quality_state={'low' if aggregate.low_quality else 'ok'}",
                    f"quality_blocked={int(_is_quality_blocked(frame_metrics, aggregate))}",
                    f"quality_reason={_quality_block_reason(frame_metrics)}",
                    f"bg_mode={frame_metrics.background_active_mode}",
                    f"bg_detect={int(frame_metrics.background_active_detected)}",
                    f"error={frame_metrics.error or '-'}",
                ]
            ),
        )

    def _window_closed(self) -> bool:
        """Return True when the preview window has been closed by the user."""
        try:
            visible = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            autosize = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)
            aspect_ratio = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
            return visible < 1 or autosize < 0 or aspect_ratio < 0
        except cv2.error:
            return True

    def _render_overlay(
        self,
        frame: np.ndarray,
        frame_metrics: FrameMetrics,
        aggregate: AggregatedMetrics,
        display_fps: float,
        inference_fps: float,
    ) -> np.ndarray:
        overlay = frame.copy()
        self._apply_flash_debug_stimulus(overlay)
        flash_state = self._frame_processor.get_flash_visual_state()
        flash_visible = bool(flash_state.get("enabled") and flash_state.get("visible"))
        status_color = _status_color(frame_metrics, aggregate)
        status_text = _status_text(frame_metrics, aggregate)
        _show_scores = aggregate.decision_state != "NO_FACE"
        lines: list[tuple[str, str]] = [
            ("STATUS", status_text),
            ("Profile", PREVIEW_SECURITY_PROFILE),
            *([
                ("Frame score", f"{frame_metrics.raw_score:5.1f}"),
                ("Smoothed", f"{aggregate.smoothed_score:5.1f}"),
                ("Frame conf.", f"{frame_metrics.frame_confidence:0.2f}"),
                ("Window conf.", f"{aggregate.window_confidence:0.2f}"),
            ] if _show_scores else []),
            ("Decision", aggregate.decision_state),
            ("Orig/Fnl", f"{aggregate.original_status_before_occ_gate} / {aggregate.final_status_after_occ_gate}"),
            *([
                ("Passive", f"{frame_metrics.passive_score:5.1f} / win {aggregate.passive_window_score:5.1f}"),
                ("Frame active", f"{frame_metrics.active_score:5.1f}"),
                ("BG active", f"{aggregate.background_active_score:5.1f}"),
                ("Puzzle", f"{aggregate.puzzle_status} / {aggregate.puzzle_current_step}"),
                ("Final active", f"{aggregate.final_active_score:5.1f}"),
                ("Device replay", _format_optional(_device_spoof_value(frame_metrics, "device_replay_risk"))),
                ("Replay veto", str(int(aggregate.replay_veto))),
                ("Adjusted score", f"{aggregate.adjusted_score:5.1f}"),
                ("Puzzle req.", str(int(aggregate.puzzle_required))),
                ("Puzzle result", _format_optional(aggregate.puzzle_result)),
                ("Flash live", str(int(aggregate.preview_flash_live_response))),
                ("Flash replay", str(int(aggregate.preview_flash_replay_support))),
            ] if _show_scores else []),
            ("Face", "YES" if frame_metrics.face_detected else "NO"),
            ("Face usable", "YES" if aggregate.face_usable else "NO"),
            ("Usability reason", aggregate.face_usability_reason),
            ("Usability state", aggregate.face_usability_state),
            ("Window", f"{aggregate.window_seconds:0.1f}s / {aggregate.sample_count} samples"),
            ("FPS", f"{display_fps:0.1f} / inf {inference_fps:0.1f}"),
        ]

        if self._settings.DEV_LIVENESS_PREVIEW_SHOW_DEBUG:
            lines.extend(
                [
                    ("Moire risk", _format_optional(_device_spoof_value(frame_metrics, "moire_risk"))),
                    ("Moire raw", _format_optional(_maybe_float(frame_metrics.details.get("preview_moire_raw_metric")))),
                    ("Moire select.", _format_optional(_maybe_float(frame_metrics.details.get("moire_orientation_selectivity")))),
                    ("Moire FFT", _format_optional(_maybe_float(frame_metrics.details.get("moire_fft_risk")))),
                    ("Moire std mean", _format_optional(_maybe_float(frame_metrics.details.get("moire_response_std_mean")))),
                    ("Depth flat", _format_optional(_maybe_float(frame_metrics.details.get("preview_depth_flat_combined_risk")))),
                    ("Depth range", _format_optional(_maybe_float(frame_metrics.details.get("depth_range")))),
                    ("Nose-cheek dz", _format_optional(_maybe_float(frame_metrics.details.get("nose_cheek_depth_delta")))),
                    ("Reflection risk", _format_optional(_device_spoof_value(frame_metrics, "reflection_risk"))),
                    ("Cutout risk", _format_optional(_device_spoof_value(frame_metrics, "cutout_spoof_support"))),
                    ("Hole cutout", _format_optional(_device_spoof_value(frame_metrics, "hole_cutout_risk"))),
                    ("Focal blur", _format_optional(_device_spoof_value(frame_metrics, "focal_blur_anomaly_risk"))),
                    ("Screen frame", _format_optional(_device_spoof_value(frame_metrics, "screen_frame_risk"))),
                    ("Screen confirmed", str(int(_is_confirmed_screen_device(frame_metrics)))),
                    ("Flicker risk", _format_optional(_device_spoof_value(frame_metrics, "flicker_risk"))),
                    ("Flash score", _format_optional(_device_spoof_value(frame_metrics, "flash_response_score"))),
                    ("Flash strength", _format_optional(_device_spoof_value(frame_metrics, "flash_response_strength"))),
                    ("Flash consist.", _format_optional(_device_spoof_value(frame_metrics, "flash_response_consistency"))),
                    ("Flash risk", _format_optional(_device_spoof_value(frame_metrics, "flash_replay_risk"))),
                    ("Flash phase", str(frame_metrics.details.get("flash_challenge_phase") or "-")),
                    ("Flash color", str(frame_metrics.details.get("flash_challenge_color") or "-")),
                    ("Flash visible", str(int(_maybe_float(frame_metrics.details.get("flash_challenge_visible")) or 0.0))),
                    ("Flash samples", str(int(_maybe_float(frame_metrics.details.get("flash_response_sample_count")) or 0.0))),
                    ("Flash match", _format_optional(_maybe_float(frame_metrics.details.get("flash_color_match_score")))),
                    ("Flash specular", _format_optional(_maybe_float(frame_metrics.details.get("specular_hotspot_risk")))),
                    ("Flash diffuse", _format_optional(_maybe_float(frame_metrics.details.get("diffuse_response_score")))),
                    ("Flash geom.", _format_optional(_maybe_float(frame_metrics.details.get("geometry_response_consistency")))),
                    ("Flash planar", _format_optional(_maybe_float(frame_metrics.details.get("planar_surface_risk")))),
                    ("Pre-flash cap.", str(int(_maybe_float(frame_metrics.details.get("pre_flash_captured")) or 0.0))),
                    ("Flash frame cap.", str(int(_maybe_float(frame_metrics.details.get("flash_frame_captured")) or 0.0))),
                    ("Device replay", _format_optional(_device_spoof_value(frame_metrics, "device_replay_risk"))),
                    ("Screen hi", str(int(_is_screen_frame_high(frame_metrics)))),
                    ("Moire hi", str(int(_is_moire_high(frame_metrics)))),
                    ("Depth hi", str(int(_is_depth_flat(frame_metrics)))),
                    ("Reflect hi", str(int(_is_reflection_high(frame_metrics)))),
                    ("Flicker hi", str(int(_is_flicker_high(frame_metrics)))),
                    ("Flash hi", str(int(_is_flash_replay_high(frame_metrics)))),
                    ("Cutout hi", str(int(_is_cutout_high(frame_metrics)))),
                    ("UniFace neg", str(int(_is_uniface_negative(frame_metrics)))),
                    ("Spoof support", str(_spoof_support_count(frame_metrics))),
                    ("Spoof support w.", _format_optional(_maybe_float(frame_metrics.details.get("preview_weighted_spoof_support_score")))),
                    ("Spoof streak", str(int(_maybe_float(frame_metrics.details.get("preview_spoof_support_streak")) or 0.0))),
                    ("Spoof gate", str(int(_is_device_replay_spoof_detected(frame_metrics)))),
                    ("BBox", _format_bbox(frame_metrics)),
                    ("BBox reuse", str(int(frame_metrics.reused_face_detection))),
                    ("--- Liveness Debug ---", ""),
                    ("Frame active ev.", _format_optional(frame_metrics.active_evidence)),
                    ("Frame act score", _format_optional(_maybe_float(frame_metrics.details.get("preview_frame_active_score_standard")))),
                    ("Detector active", _format_optional(_maybe_float(frame_metrics.details.get("detector_active_score")))),
                    ("Detector act ev.", _format_optional(_maybe_float(frame_metrics.details.get("detector_active_evidence")))),
                    ("BG active raw", _format_optional(aggregate.raw_active_evidence)),
                    ("BG active temp", _format_optional(aggregate.background_active_evidence)),
                    ("BG act score", f"{aggregate.active_score_standard_mapping:0.1f}"),
                    ("Puzzle status", aggregate.puzzle_status),
                    ("Puzzle step", aggregate.puzzle_current_step),
                    ("Puzzle progress", f"{aggregate.puzzle_completed_steps}/{aggregate.puzzle_total_steps} ({aggregate.puzzle_progress:0.2f})"),
                    ("Puzzle seq.", aggregate.puzzle_sequence_label),
                    ("Puzzle ev.", f"{aggregate.puzzle_active_evidence:0.2f}"),
                    ("Puzzle conf.", f"{aggregate.puzzle_confidence:0.2f}"),
                    ("Puzzle success", str(int(aggregate.puzzle_success))),
                    ("Fusion active", str(int(aggregate.puzzle_fusion_active))),
                    ("Final active ev.", f"{aggregate.final_active_evidence:0.2f}"),
                    ("Final act score", f"{aggregate.final_active_score:0.1f}"),
                    ("Final support", f"{aggregate.final_supported_score:0.1f}"),
                    ("Replay veto", str(int(aggregate.replay_veto))),
                    ("Adjusted score", f"{aggregate.adjusted_score:0.1f}"),
                    ("Debug active", f"{aggregate.debug_active_score:0.1f}"),
                    ("Puzzle required", str(int(aggregate.puzzle_required))),
                    ("Puzzle result", _format_optional(aggregate.puzzle_result)),
                    ("Flash live resp.", str(int(aggregate.preview_flash_live_response))),
                    ("Flash replay sup.", str(int(aggregate.preview_flash_replay_support))),
                    ("Unstable signal", str(int(aggregate.unstable_signal))),
                    ("No-face cooldown", str(int(aggregate.no_face_cooldown_active))),
                    ("Stable live hold", str(int(aggregate.stable_live_hold_active))),
                    ("Critical occ.", str(int(aggregate.critical_occ))),
                    ("Critical occ score", f"{aggregate.critical_occ_score:0.2f}"),
                    ("Critical occ regions", ", ".join(aggregate.critical_occ_regions) if aggregate.critical_occ_regions else "-"),
                    ("Critical occ streak", str(aggregate.critical_occ_streak)),
                    ("Critical clear streak", str(aggregate.critical_clear_streak)),
                    ("Critical occ state", aggregate.critical_occ_state),
                    ("Critical occ reason", aggregate.critical_occ_reason),
                    ("Suspicion", ", ".join(aggregate.suspicion_reasons) if aggregate.suspicion_reasons else "-"),
                    ("Frame conf. mean", f"{aggregate.frame_confidence_mean:0.2f}"),
                    ("Directional agr.", _format_optional(frame_metrics.directional_agreement)),
                    ("Face quality", _format_optional(frame_metrics.face_quality)),
                    ("Face size", _format_optional(frame_metrics.face_size_ratio)),
                    ("Blur", _format_optional(frame_metrics.blur_score)),
                    ("Brightness", f"{frame_metrics.brightness:0.1f}"),
                    ("EAR", _format_optional(frame_metrics.ear_current)),
                    ("MAR", _format_optional(frame_metrics.mar_current)),
                    ("Yaw", _format_optional(frame_metrics.yaw_current)),
                    ("Pitch", _format_optional(frame_metrics.pitch_current)),
                    ("Roll", _format_optional(frame_metrics.roll_current)),
                    ("Baseline", f"{'READY' if aggregate.baseline_ready else 'CALIBRATING'} ({aggregate.baseline_sample_count})"),
                    ("EAR base", _format_optional(aggregate.ear_baseline)),
                    ("MAR base", _format_optional(aggregate.mar_baseline)),
                    ("Smile base", _format_optional(aggregate.smile_baseline)),
                    ("Yaw base", _format_optional(aggregate.yaw_baseline)),
                    ("EAR drop", _format_optional(aggregate.ear_drop_ratio)),
                    ("MAR rise", _format_optional(aggregate.mar_rise_ratio)),
                    ("EMA raw", f"{aggregate.ema_score:0.1f}"),
                    ("Blink ev.", _format_optional(aggregate.blink_evidence)),
                    ("Smile ev.", _format_optional(aggregate.smile_evidence)),
                    ("Mouth ev.", _format_optional(aggregate.mouth_open_evidence)),
                    ("Turn L/R ev.", f"{_format_optional(aggregate.head_turn_left_evidence)} / {_format_optional(aggregate.head_turn_right_evidence)}"),
                    ("Primary/2nd", f"{aggregate.primary_event:0.2f} / {aggregate.secondary_event:0.2f}"),
                    ("Raw react", f"{aggregate.raw_reaction_evidence:0.2f}"),
                    ("Base trust", f"{aggregate.base_active_trust:0.2f}"),
                    ("Trust penalty", f"{aggregate.trust_penalty:0.2f}"),
                    ("Eff trust", f"{aggregate.effective_trust:0.2f}"),
                    ("Trusted react", f"{aggregate.trusted_reaction_evidence:0.2f}"),
                    ("Blink anom.", f"{aggregate.blink_anomaly_score:0.2f}"),
                    ("Motion anom.", f"{aggregate.motion_anomaly_score:0.2f}"),
                    ("Signal incons.", f"{aggregate.signal_inconsistency_score:0.2f}"),
                    ("Active spoof sup.", f"{aggregate.spoof_support_score:0.2f}"),
                    ("Persist 1/2", f"{aggregate.persisted_primary:0.2f} / {aggregate.persisted_secondary:0.2f}"),
                    ("Persist react", f"{aggregate.persisted_reaction_evidence:0.2f}"),
                    ("Variance", f"{aggregate.score_variance:0.2f}"),
                    ("Score mean", f"{aggregate.score_mean:0.1f}"),
                    ("Score EMA", f"{aggregate.ema_score:0.1f}"),
                    ("Temp consist.", f"{aggregate.temporal_consistency:0.2f}"),
                    ("Dir. agr mean", f"{aggregate.directional_agreement_mean:0.2f}"),
                    ("Face qual mean", f"{aggregate.face_quality_mean:0.2f}"),
                    ("Face size mean", f"{aggregate.face_size_mean:0.3f}"),
                    ("Face size adq.", f"{aggregate.face_size_adequacy:0.2f}"),
                    ("Blur adequacy", f"{aggregate.blur_adequacy:0.2f}"),
                    ("Bright adequacy", f"{aggregate.brightness_adequacy:0.2f}"),
                    ("Detect ms", f"{frame_metrics.profiling.get('face_detection_ms', 0.0):0.1f}"),
                    ("Landmark ms", f"{frame_metrics.profiling.get('landmark_ms', 0.0):0.1f}"),
                    ("Liveness ms", f"{frame_metrics.profiling.get('liveness_ms', 0.0):0.1f}"),
                    ("BG react ms", f"{aggregate.background_reaction_ms:0.1f}"),
                    ("Agg ms", f"{aggregate.temporal_aggregation_ms:0.1f}"),
                    ("Infer scale", f"{frame_metrics.inference_scale:0.2f}"),
                    ("Reuse", f"D{int(frame_metrics.reused_face_detection)} L{int(frame_metrics.reused_landmarks)} V{int(frame_metrics.reused_liveness)} H{int(frame_metrics.held_from_previous)}"),
                    ("Stable live", f"{aggregate.stable_live_ratio:0.2f}"),
                    ("Face ratio", f"{aggregate.face_present_ratio:0.2f} / no-face {aggregate.consecutive_no_face_frames}"),
                    ("Recovery", "WARM" if aggregate.warm_recovery else "NORMAL"),
                    ("Evidence", "SUFFICIENT" if aggregate.sufficient_evidence else "PENDING"),
                    ("Quality", "LOW" if aggregate.low_quality else "OK"),
                    ("BG mode", frame_metrics.background_active_mode),
                    ("BG detect", "YES" if frame_metrics.background_active_detected else "NO"),
                    ("Controls", "P=start puzzle R=reset Q=quit"),
                ]
            )

        rendered_lines = lines if not flash_visible else lines[:8]
        panel_width = 540 if not flash_visible else 280
        panel_bottom = min(
            overlay.shape[0] - 6,
            10 + len(rendered_lines) * self.OVERLAY_LINE_HEIGHT + (30 if frame_metrics.error else 0),
        )
        if not flash_visible:
            cv2.rectangle(overlay, (6, 4), (panel_width, panel_bottom), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, overlay)

        y = 18
        for index, (label, value) in enumerate(rendered_lines):
            color = status_color if index == 0 else (0, 0, 0)
            if label.startswith("--- "):
                color = (40, 40, 40)
            cv2.putText(
                overlay,
                label if label.startswith("--- ") else f"{label}: {value}",
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.OVERLAY_FONT_SCALE,
                color,
                self.OVERLAY_TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            y += self.OVERLAY_LINE_HEIGHT

        if flash_visible:
            cv2.putText(
                overlay,
                f"FLASH {str(flash_state.get('color') or '-').upper()}",
                (max(20, overlay.shape[1] // 2 - 110), max(70, overlay.shape[0] - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if frame_metrics.error:
            cv2.putText(
                overlay,
                frame_metrics.error.upper(),
                (14, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        self._draw_face_bbox(overlay, frame_metrics, status_color)
        border_color = status_color
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), border_color, 4)
        return overlay

    def _apply_flash_debug_stimulus(self, overlay: np.ndarray) -> None:
        flash_state = self._frame_processor.get_flash_visual_state()
        if not flash_state.get("enabled") or not flash_state.get("visible"):
            return
        color_name = str(flash_state.get("color") or "white")
        bgr = {
            "red": (40, 40, 255),
            "green": (60, 255, 60),
            "blue": (255, 90, 40),
            "yellow": (40, 255, 255),
            "white": (255, 255, 255),
        }.get(color_name, (255, 255, 255))
        stimulus = overlay.copy()
        cv2.rectangle(stimulus, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), bgr, -1)
        cv2.addWeighted(stimulus, 0.84, overlay, 0.16, 0, overlay)
        cv2.putText(
            overlay,
            f"FLASH {color_name.upper()}",
            (max(20, overlay.shape[1] // 2 - 90), 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0) if color_name == "white" else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_face_bbox(
        self,
        overlay: np.ndarray,
        frame_metrics: FrameMetrics,
        color: tuple[int, int, int],
    ) -> None:
        bbox = _extract_bbox(frame_metrics)
        if bbox is None:
            return

        for index, extra_bbox in enumerate(_extract_additional_bboxes(frame_metrics), start=2):
            extra_x, extra_y, extra_width, extra_height = extra_bbox
            cv2.rectangle(
                overlay,
                (extra_x, extra_y),
                (extra_x + extra_width, extra_y + extra_height),
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                overlay,
                (extra_x, extra_y),
                (extra_x + extra_width, extra_y + extra_height),
                (0, 215, 255),
                1,
            )
            cv2.putText(
                overlay,
                f"face {index}",
                (extra_x, max(24, extra_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

        x, y, width, height = bbox
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 255), 3)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        label = f"face 1 {x},{y} {width}x{height}"
        if frame_metrics.reused_face_detection:
            label += " reuse"
        label_y = max(24, y - 10)
        cv2.putText(
            overlay,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def create_dev_liveness_preview(
    *,
    settings: Settings,
    face_detector: IFaceDetector,
    liveness_detector: ILivenessDetector,
    landmark_detector: Optional[ILandmarkDetector] = None,
) -> LiveLivenessPreview:
    """Construct a preview instance from existing detector components."""
    return LiveLivenessPreview(
        settings=settings,
        frame_processor=LivenessPreviewFrameProcessor(
            face_detector=face_detector,
            liveness_detector=liveness_detector,
            settings=settings,
            landmark_detector=landmark_detector,
        ),
        temporal_aggregator=TemporalLivenessAggregator(
            window_seconds=settings.DEV_LIVENESS_PREVIEW_WINDOW_SECONDS,
            baseline_seconds=settings.DEV_LIVENESS_PREVIEW_BASELINE_SECONDS,
            max_entries=settings.DEV_LIVENESS_PREVIEW_BUFFER_SIZE,
            ema_alpha=settings.DEV_LIVENESS_PREVIEW_EMA_ALPHA,
            no_face_consecutive_threshold=settings.DEV_LIVENESS_PREVIEW_NO_FACE_CONSECUTIVE_THRESHOLD,
            face_return_grace_seconds=settings.DEV_LIVENESS_PREVIEW_FACE_RETURN_GRACE_SECONDS,
            face_loss_reset_seconds=settings.DEV_LIVENESS_PREVIEW_FACE_LOSS_RESET_SECONDS,
            active_decay_seconds=settings.DEV_LIVENESS_PREVIEW_ACTIVE_DECAY_SECONDS,
            min_trusted_face_size_ratio=settings.DEV_LIVENESS_PREVIEW_MIN_TRUSTED_FACE_SIZE_RATIO,
            occlusion_no_face_threshold=settings.DEV_LIVENESS_PREVIEW_OCCLUSION_NO_FACE_THRESHOLD,
        ),
    )


def _status_color(frame_metrics: FrameMetrics, aggregate: AggregatedMetrics) -> tuple[int, int, int]:
    state = aggregate.decision_state
    if state == "NO_FACE":
        return (0, 165, 255)
    if state == "LOW_QUALITY":
        return (0, 215, 255)
    if state == "INSUFFICIENT_EVIDENCE":
        return (0, 215, 255)
    if state == "LIVE":
        return (0, 200, 0)
    return (0, 0, 220)


def _status_text(frame_metrics: FrameMetrics, aggregate: AggregatedMetrics) -> str:
    return aggregate.decision_state


def _extract_bbox(frame_metrics: FrameMetrics) -> Optional[tuple[int, int, int, int]]:
    x = _maybe_float(frame_metrics.details.get("preview_bbox_x"))
    y = _maybe_float(frame_metrics.details.get("preview_bbox_y"))
    width = _maybe_float(frame_metrics.details.get("preview_bbox_w"))
    height = _maybe_float(frame_metrics.details.get("preview_bbox_h"))
    if None in (x, y, width, height):
        return None
    return (int(x), int(y), int(width), int(height))


def _extract_additional_bboxes(frame_metrics: FrameMetrics) -> list[tuple[int, int, int, int]]:
    raw_bboxes = frame_metrics.details.get("preview_additional_bboxes")
    if not isinstance(raw_bboxes, list):
        return []

    parsed: list[tuple[int, int, int, int]] = []
    for item in raw_bboxes:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        x, y, width, height = (_maybe_float(value) for value in item)
        if None in (x, y, width, height):
            continue
        parsed.append((int(x), int(y), int(width), int(height)))
    return parsed


def _format_bbox(frame_metrics: FrameMetrics) -> str:
    bbox = _extract_bbox(frame_metrics)
    if bbox is None:
        return "-"
    x, y, width, height = bbox
    extra_count = len(_extract_additional_bboxes(frame_metrics))
    return f"{x},{y},{width},{height}" + (f" (+{extra_count})" if extra_count else "")


def _format_optional(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:0.2f}"


def _device_spoof_value(frame_metrics: FrameMetrics, field_name: str) -> Optional[float]:
    if frame_metrics.device_spoof is None:
        return None
    value = getattr(frame_metrics.device_spoof, field_name, None)
    return float(value) if value is not None else None


def _is_device_replay_spoof_detected(frame_metrics: FrameMetrics) -> bool:
    if frame_metrics.device_spoof is None:
        return False

    device_replay_risk = getattr(frame_metrics.device_spoof, "device_replay_risk", None) or 0.0
    reflection_risk = getattr(frame_metrics.device_spoof, "reflection_risk", None) or 0.0
    flash_replay_risk = getattr(frame_metrics.device_spoof, "flash_replay_risk", None) or 0.0
    screen_frame_risk = getattr(frame_metrics.device_spoof, "screen_frame_risk", None) or 0.0
    reflect_compact = float(
        frame_metrics.details.get("reflection_compact_highlight_score") or 0.0
    )
    face_center_inside = float(frame_metrics.details.get("screen_frame_face_center_inside") or 0.0)
    flash_response_score = float(frame_metrics.details.get("flash_response_score") or 0.0)
    flash_samples = float(frame_metrics.details.get("flash_response_sample_count") or 0.0)
    weighted_support_score = _weighted_spoof_support_score_from_details(frame_metrics.details)
    support_streak = int(_maybe_float(frame_metrics.details.get("preview_spoof_support_streak")) or 0.0)
    confirmed_screen = _is_confirmed_screen_device(frame_metrics)
    hard_replay_cues = sum(
        (
            int(_is_moire_high(frame_metrics)),
            int(_is_flicker_high(frame_metrics)),
            int(confirmed_screen),
            int(_is_flash_replay_high(frame_metrics)),
        )
    )

    hard_reflection_rule = (
        reflect_compact >= 0.50
        and reflection_risk >= 0.60
        and device_replay_risk >= 0.75
        and (
            _is_screen_frame_supportive(frame_metrics)
            or _is_moire_high(frame_metrics)
            or _is_flash_replay_high(frame_metrics)
            or _is_flicker_high(frame_metrics)
        )
    )
    hard_confirmed_screen_rule = confirmed_screen and face_center_inside >= 0.55
    hard_screen_rule = (
        screen_frame_risk >= 0.72
        and device_replay_risk >= 0.68
        and face_center_inside >= 0.5
    )
    hard_flash_rule = (
        flash_samples >= 1.0
        and flash_replay_risk >= 0.75
        and flash_response_score <= 0.30
        and device_replay_risk >= 0.50
    )
    live_motion_override = _has_strong_live_override(frame_metrics)
    strong_spoof_corroboration = _has_strong_spoof_corroboration(frame_metrics)
    support_based_rule = (
        device_replay_risk >= 0.48
        and weighted_support_score >= 2.0
        and support_streak >= 4
        and hard_replay_cues >= 1
        and strong_spoof_corroboration
        and not (live_motion_override and not hard_reflection_rule and not hard_screen_rule)
    )

    return (
        hard_reflection_rule
        or hard_confirmed_screen_rule
        or hard_screen_rule
        or hard_flash_rule
        or support_based_rule
    )


def _is_screen_frame_high(frame_metrics: FrameMetrics) -> bool:
    return (_device_spoof_value(frame_metrics, "screen_frame_risk") or 0.0) >= 0.72


def _is_confirmed_screen_device(frame_metrics: FrameMetrics) -> bool:
    screen_source = _maybe_float(frame_metrics.details.get("screen_frame_source")) or 0.0
    screen_frame_risk = _device_spoof_value(frame_metrics, "screen_frame_risk") or 0.0
    candidate_found = _maybe_float(frame_metrics.details.get("boundary_candidate_found")) or 0.0
    partial_candidate_found = _maybe_float(frame_metrics.details.get("boundary_partial_candidate_found")) or 0.0
    candidate_score = _maybe_float(frame_metrics.details.get("screen_frame_candidate_score")) or 0.0
    contour_score = _maybe_float(frame_metrics.details.get("boundary_contour_score")) or 0.0
    partial_score = _maybe_float(frame_metrics.details.get("boundary_partial_score")) or 0.0
    aspect_score = _maybe_float(frame_metrics.details.get("screen_frame_aspect_score")) or 0.0
    face_center_inside = _maybe_float(frame_metrics.details.get("screen_frame_face_center_inside")) or 0.0
    contour_area_ratio = _maybe_float(frame_metrics.details.get("screen_frame_area_ratio")) or 0.0
    partial_area_ratio = _maybe_float(frame_metrics.details.get("boundary_partial_candidate_area_ratio")) or 0.0
    candidate_area_ratio = max(contour_area_ratio, partial_area_ratio)
    any_candidate_found = candidate_found >= 0.5 or partial_candidate_found >= 0.5
    strong_geometry = contour_score >= 0.58 or partial_score >= 0.68
    centered_candidate = face_center_inside >= 0.48
    large_enough_candidate = candidate_area_ratio >= 0.14
    plausible_device_shape = aspect_score >= 0.22
    detector_confirmed = (
        screen_source >= 0.5
        and any_candidate_found
        and centered_candidate
        and large_enough_candidate
    )
    geometric_confirmed = (
        any_candidate_found
        and centered_candidate
        and large_enough_candidate
        and plausible_device_shape
        and screen_frame_risk >= 0.62
        and candidate_score >= 0.58
        and strong_geometry
    )

    return bool(detector_confirmed or geometric_confirmed)


def _is_screen_frame_supportive(frame_metrics: FrameMetrics) -> bool:
    screen_frame_risk = _device_spoof_value(frame_metrics, "screen_frame_risk") or 0.0
    face_center_inside = _maybe_float(frame_metrics.details.get("screen_frame_face_center_inside")) or 0.0
    return screen_frame_risk >= 0.25 or face_center_inside >= 0.45


def _is_moire_high(frame_metrics: FrameMetrics) -> bool:
    moire_risk = _device_spoof_value(frame_metrics, "moire_risk") or 0.0
    moire_fft = _maybe_float(frame_metrics.details.get("moire_fft_risk")) or 0.0
    moire_selectivity = _maybe_float(frame_metrics.details.get("moire_orientation_selectivity")) or 0.0
    return moire_risk >= 0.70 and (
        moire_fft >= 0.55
        or moire_selectivity >= 0.35
    )


def _is_reflection_high(frame_metrics: FrameMetrics) -> bool:
    reflection_risk = _device_spoof_value(frame_metrics, "reflection_risk") or 0.0
    reflect_compact = _maybe_float(frame_metrics.details.get("reflection_compact_highlight_score")) or 0.0
    return reflection_risk >= 0.60 or reflect_compact >= 0.50


def _has_strong_live_override(frame_metrics: FrameMetrics) -> bool:
    active_evidence = frame_metrics.active_evidence or 0.0
    active_score = frame_metrics.active_score or 0.0
    return bool(
        frame_metrics.face_detected
        and frame_metrics.is_live
        and frame_metrics.raw_score >= 80.0
        and frame_metrics.frame_confidence >= 0.58
        and (active_evidence >= 0.35 or active_score >= 65.0)
    )


def _has_strong_spoof_corroboration(frame_metrics: FrameMetrics) -> bool:
    return _has_strong_spoof_corroboration_from_details(frame_metrics.details)


def _is_flicker_high(frame_metrics: FrameMetrics) -> bool:
    return (_device_spoof_value(frame_metrics, "flicker_risk") or 0.0) >= 0.70


def _is_flash_replay_high(frame_metrics: FrameMetrics) -> bool:
    return (_device_spoof_value(frame_metrics, "flash_replay_risk") or 0.0) >= 0.70


def _is_cutout_high(frame_metrics: FrameMetrics) -> bool:
    hole_cutout_risk = _device_spoof_value(frame_metrics, "hole_cutout_risk") or 0.0
    focal_blur_risk = _device_spoof_value(frame_metrics, "focal_blur_anomaly_risk") or 0.0
    cutout_support = _device_spoof_value(frame_metrics, "cutout_spoof_support") or 0.0
    return cutout_support >= 0.62 or (
        hole_cutout_risk >= 0.64 and focal_blur_risk >= 0.50
    )


def _is_depth_flat(frame_metrics: FrameMetrics) -> bool:
    return (_maybe_float(frame_metrics.details.get("preview_depth_flat_combined_risk")) or 0.0) >= 0.72


def _is_uniface_negative(frame_metrics: FrameMetrics) -> bool:
    uniface_score = _maybe_float(frame_metrics.details.get("uniface_score"))
    uniface_is_live = frame_metrics.details.get("uniface_is_live")
    if uniface_score is None and uniface_is_live is None:
        return False
    return bool(
        (uniface_is_live is False and (uniface_score is None or uniface_score < 65.0))
        or (uniface_score is not None and uniface_score < 55.0)
    )


def _spoof_support_count(frame_metrics: FrameMetrics) -> int:
    return _spoof_support_count_from_details(frame_metrics.details)


def _spoof_support_count_from_details(details: dict[str, Any]) -> int:
    strict_exam = _is_strict_exam_profile(details)
    support_flags = (
        _is_detail_screen_frame_supportive(details),
        _is_detail_moire_high(details),
        _is_detail_reflection_high(details),
        _is_detail_flicker_high(details),
        _is_detail_flash_replay_high(details),
        strict_exam and _is_detail_cutout_high(details),
        _is_detail_uniface_negative(details),
        _is_detail_depth_flat(details),
    )
    return int(sum(1 for flag in support_flags if flag))


def _strict_moire_support_contribution_from_details(details: dict[str, Any]) -> float:
    if not _is_strict_exam_profile(details):
        return 0.0
    moire_risk = _maybe_float(details.get("moire_risk")) or 0.0
    moire_fft_risk = _maybe_float(details.get("moire_fft_risk")) or 0.0
    moire_selectivity = _maybe_float(details.get("moire_orientation_selectivity")) or 0.0
    normalized_selectivity = _normalize(moire_selectivity, 0.20, 0.45)
    moire_support_strength = _clamp01(
        0.50 * moire_risk
        + 0.30 * moire_fft_risk
        + 0.20 * normalized_selectivity
    ) or 0.0
    config = get_settings().get_strict_micro_texture_config()
    return float(config["moire_support_weight"] * moire_support_strength)


def _strict_cutout_support_contribution_from_details(details: dict[str, Any]) -> float:
    if not _is_strict_exam_profile(details):
        return 0.0
    hole_cutout_risk = _maybe_float(details.get("hole_cutout_risk")) or 0.0
    focal_blur_risk = _maybe_float(details.get("focal_blur_anomaly_risk")) or 0.0
    cutout_support = _maybe_float(details.get("cutout_spoof_support")) or 0.0
    cutout_strength = _clamp01(
        0.30 * hole_cutout_risk
        + 0.25 * focal_blur_risk
        + 0.45 * cutout_support
    ) or 0.0
    config = get_settings().get_strict_micro_texture_config()
    return float(config["cutout_support_weight"] * cutout_strength)


def _weighted_spoof_support_score_from_details(details: dict[str, Any]) -> float:
    base_support_count = float(_spoof_support_count_from_details(details))
    strict_moire_contribution = _strict_moire_support_contribution_from_details(details)
    strict_cutout_contribution = _strict_cutout_support_contribution_from_details(details)
    weighted_score = base_support_count
    if strict_moire_contribution > 0.0:
        weighted_score += (
            max(0.0, strict_moire_contribution - 1.0)
            if _is_detail_moire_high(details)
            else strict_moire_contribution
        )
    if strict_cutout_contribution > 0.0:
        weighted_score += (
            max(0.0, strict_cutout_contribution - 1.0)
            if _is_detail_cutout_high(details)
            else strict_cutout_contribution
        )
    return weighted_score


def _spoof_support_streak(entries: list[FrameMetrics]) -> int:
    streak = 0
    for entry in reversed(entries):
        if not _is_support_based_spoof_candidate(entry):
            break
        streak += 1
    return streak


def _is_support_based_spoof_candidate(frame_metrics: FrameMetrics) -> bool:
    if frame_metrics.device_spoof is None:
        return False
    return _is_support_based_spoof_candidate_from_details(frame_metrics.details, frame_metrics.device_spoof)


def _is_support_based_spoof_candidate_from_details(
    details: dict[str, Any],
    device_spoof: DeviceSpoofRiskAssessment,
) -> bool:
    support_count = _spoof_support_count_from_details(details)
    reflect_compact = _maybe_float(details.get("reflection_compact_highlight_score")) or 0.0
    return support_count >= 2 and reflect_compact >= 0.65


def _compute_depth_temporal_flat_risk(entries: list[FrameMetrics]) -> float:
    yaw_values = [
        float(entry.yaw_current)
        for entry in entries
        if entry.face_detected and entry.yaw_current is not None
    ]
    asymmetry_values = [
        _maybe_float(entry.details.get("cheek_depth_asymmetry"))
        for entry in entries
        if entry.face_detected
    ]
    asymmetry_values = [value for value in asymmetry_values if value is not None]
    nose_cheek_values = [
        _maybe_float(entry.details.get("nose_cheek_depth_delta"))
        for entry in entries
        if entry.face_detected
    ]
    nose_cheek_values = [value for value in nose_cheek_values if value is not None]

    if len(yaw_values) < 3 or not asymmetry_values or not nose_cheek_values:
        return 0.0

    yaw_span = max(yaw_values) - min(yaw_values)
    if yaw_span < 8.0:
        return 0.0

    asymmetry_mean = float(np.mean(asymmetry_values))
    asymmetry_span = max(asymmetry_values) - min(asymmetry_values)
    nose_cheek_mean = float(np.mean(nose_cheek_values))
    nose_cheek_span = max(nose_cheek_values) - min(nose_cheek_values)

    yaw_factor = _normalize(yaw_span, 8.0, 24.0)
    flatness = (
        0.35 * _inverse_normalize(asymmetry_mean, 0.010, 0.050)
        + 0.25 * _inverse_normalize(asymmetry_span, 0.008, 0.035)
        + 0.25 * _inverse_normalize(nose_cheek_mean, 0.015, 0.075)
        + 0.15 * _inverse_normalize(nose_cheek_span, 0.006, 0.030)
    )
    return max(0.0, min(1.0, yaw_factor * flatness))


def _is_detail_moire_high(details: dict[str, Any]) -> bool:
    strict_exam = _is_strict_exam_profile(details)
    moire_risk = _maybe_float(details.get("moire_risk")) or 0.0
    moire_fft = _maybe_float(details.get("moire_fft_risk")) or 0.0
    moire_selectivity = _maybe_float(details.get("moire_orientation_selectivity")) or 0.0
    return moire_risk >= (0.62 if strict_exam else 0.70) and (
        moire_fft >= (0.48 if strict_exam else 0.55)
        or moire_selectivity >= (0.30 if strict_exam else 0.35)
    )


def _is_detail_reflection_high(details: dict[str, Any]) -> bool:
    reflection_risk = _maybe_float(details.get("reflection_risk")) or 0.0
    reflect_compact = _maybe_float(details.get("reflection_compact_highlight_score")) or 0.0
    return reflection_risk >= 0.60 or reflect_compact >= 0.50


def _is_detail_flicker_high(details: dict[str, Any]) -> bool:
    return (_maybe_float(details.get("flicker_risk")) or 0.0) >= 0.70


def _is_detail_flash_replay_high(details: dict[str, Any]) -> bool:
    return (_maybe_float(details.get("flash_replay_risk")) or 0.0) >= 0.70


def _is_detail_cutout_high(details: dict[str, Any]) -> bool:
    hole_cutout_risk = _maybe_float(details.get("hole_cutout_risk")) or 0.0
    focal_blur_risk = _maybe_float(details.get("focal_blur_anomaly_risk")) or 0.0
    cutout_support = _maybe_float(details.get("cutout_spoof_support")) or 0.0
    return cutout_support >= 0.62 or (hole_cutout_risk >= 0.64 and focal_blur_risk >= 0.50)


def _has_strong_spoof_corroboration_from_details(details: dict[str, Any]) -> bool:
    strict_exam = _is_strict_exam_profile(details)
    return bool(
        _is_detail_moire_high(details)
        or _is_detail_depth_flat(details)
        or _is_detail_uniface_negative(details)
        or _is_detail_screen_frame_high(details)
        or _is_detail_flash_replay_high(details)
        or (strict_exam and _is_detail_cutout_high(details))
        or (_is_detail_screen_frame_supportive(details) and _is_detail_flicker_high(details))
    )


def _is_detail_screen_frame_supportive(details: dict[str, Any]) -> bool:
    screen_frame_risk = _maybe_float(details.get("screen_frame_risk")) or 0.0
    face_center_inside = _maybe_float(details.get("screen_frame_face_center_inside")) or 0.0
    return screen_frame_risk >= 0.25 or face_center_inside >= 0.45


def _is_detail_screen_frame_high(details: dict[str, Any]) -> bool:
    return (_maybe_float(details.get("screen_frame_risk")) or 0.0) >= 0.72


def _is_detail_uniface_negative(details: dict[str, Any]) -> bool:
    uniface_score = _maybe_float(details.get("uniface_score"))
    uniface_is_live = details.get("uniface_is_live")
    if uniface_score is None and uniface_is_live is None:
        return False
    return bool(
        (uniface_is_live is False and (uniface_score is None or uniface_score < 65.0))
        or (uniface_score is not None and uniface_score < 55.0)
    )


def _is_detail_depth_flat(details: dict[str, Any]) -> bool:
    return (_maybe_float(details.get("preview_depth_flat_combined_risk")) or 0.0) >= 0.72


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce_float(primary: Optional[float], fallback: Optional[float]) -> Optional[float]:
    return primary if primary is not None else fallback


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _normalized_sigmoid_score(
    evidence: float,
    *,
    midpoint: float,
    steepness: float,
    scale: float,
) -> float:
    clipped_evidence = max(0.0, min(1.0, float(evidence)))

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-steepness * (x - midpoint))))

    low = _sigmoid(0.0)
    high = _sigmoid(1.0)
    if high - low <= 1e-6:
        return 0.0
    normalized = (_sigmoid(clipped_evidence) - low) / (high - low)
    return max(0.0, min(float(scale), float(scale) * normalized))


def _idle_puzzle_summary() -> PreviewPuzzleSummary:
    return PreviewPuzzleSummary(
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
    )


def _decision_history_unstable(history: list[str]) -> bool:
    recent = [state for state in history[-6:] if state]
    if len(recent) < 4:
        return False
    transitions = sum(1 for previous, current in zip(recent, recent[1:]) if previous != current)
    return transitions >= 3 or len(set(recent)) >= 3


def _current_puzzle_result_from_summary(temporal_signal_summary: dict[str, Any]) -> Optional[float]:
    puzzle_status = str(temporal_signal_summary.get("puzzle_status") or "idle")
    if puzzle_status == "idle":
        return None
    return float(temporal_signal_summary.get("puzzle_active_evidence") or 0.0)


def _compute_preview_active_fusion(
    *,
    background_active_evidence: float,
    background_active_score: float,
    background_supported_score: float,
    passive_window_score: float,
    puzzle_summary: PreviewPuzzleSummary,
    settings: Settings,
) -> dict[str, Any]:
    background_evidence = _clamp01(background_active_evidence) or 0.0
    background_score = max(0.0, min(100.0, float(background_active_score)))
    puzzle_evidence = _clamp01(puzzle_summary.active_evidence) or 0.0
    final_active_evidence = background_evidence
    fusion_active = bool(puzzle_summary.fusion_active)

    final_active_score_standard = _standard_active_support_score(final_active_evidence)
    final_active_score_mapping_mode = "standard_sqrt"
    final_active_score = final_active_score_standard

    final_supported_score = float(background_supported_score)

    return {
        "background_active_evidence": background_evidence,
        "background_active_score": background_score,
        "puzzle_active_evidence": puzzle_evidence,
        "puzzle_progress": float(puzzle_summary.progress),
        "puzzle_current_step": puzzle_summary.current_step,
        "puzzle_completed_steps": int(puzzle_summary.completed_steps),
        "puzzle_total_steps": int(puzzle_summary.total_steps),
        "puzzle_status": puzzle_summary.status,
        "puzzle_confidence": float(puzzle_summary.confidence),
        "puzzle_success": bool(puzzle_summary.success),
        "puzzle_fusion_active": fusion_active,
        "puzzle_sequence_label": puzzle_summary.sequence_label,
        "final_active_evidence": final_active_evidence,
        "final_active_score": final_active_score,
        "final_active_score_mapping_mode": final_active_score_mapping_mode,
        "final_active_score_standard_mapping": final_active_score_standard,
        "final_active_score_strict_mapping": final_active_score_standard,
        "final_supported_score": final_supported_score,
    }


def _effective_decision_score(
    *,
    smoothed_score: float,
    temporal_signal_summary: dict[str, Any],
) -> float:
    return smoothed_score


def _preview_strict_exam_decision_layers(
    *,
    current_frame: FrameMetrics,
    temporal_signal_summary: dict[str, Any],
    base_decision_score: float,
) -> dict[str, Any]:
    return {
        "strict_decision_score": float(base_decision_score),
        "strict_replay_penalty": 0.0,
        "strict_spoof_support_penalty": 0.0,
        "strict_challenge_penalty": 0.0,
        "strict_hard_block": False,
        "strict_hard_replay_cues": 0.0,
    }


def _inverse_normalize(value: float, low: float, high: float) -> float:
    return 1.0 - _normalize(value, low, high)


def _compute_temporal_signal_summary(
    entries: list[FrameMetrics],
    *,
    background_reaction_evaluator: Optional[BackgroundActiveReactionEvaluator] = None,
    passive_window_score: Optional[float] = None,
    session_baseline: Optional[SessionBaseline] = None,
) -> dict[str, Any]:
    ear_values = [entry.ear_current for entry in entries if entry.ear_current is not None]
    mar_values = [entry.mar_current for entry in entries if entry.mar_current is not None]
    yaw_values = [entry.yaw_current for entry in entries if entry.yaw_current is not None]
    pitch_values = [entry.pitch_current for entry in entries if entry.pitch_current is not None]
    roll_values = [entry.roll_current for entry in entries if entry.roll_current is not None]

    ear_baseline = session_baseline.ear_baseline if session_baseline else _first_available(entries, "ear_baseline")
    mar_baseline = session_baseline.mar_baseline if session_baseline else _first_available(entries, "mar_baseline")
    smile_baseline = session_baseline.smile_baseline if session_baseline else _first_available(entries, "smile_baseline")
    yaw_baseline = session_baseline.yaw_baseline if session_baseline else _first_available(entries, "yaw_baseline")
    pitch_baseline = session_baseline.pitch_baseline if session_baseline else _first_available(entries, "pitch_baseline")
    roll_baseline = session_baseline.roll_baseline if session_baseline else _first_available(entries, "roll_baseline")

    ear_mean = _mean(ear_values)
    ear_min = min(ear_values) if ear_values else None
    ear_max = max(ear_values) if ear_values else None
    ear_drop = (ear_max - ear_min) if ear_min is not None and ear_max is not None else None
    ear_drop_ratio = _ratio(ear_drop, ear_baseline or ear_max)

    mar_mean = _mean(mar_values)
    mar_min = min(mar_values) if mar_values else None
    mar_max = max(mar_values) if mar_values else None
    mar_rise = (mar_max - mar_min) if mar_min is not None and mar_max is not None else None
    mar_rise_ratio = _ratio(mar_rise, mar_baseline or mar_mean)

    yaw_mean = _mean(yaw_values)
    yaw_left_peak = max([-value for value in yaw_values if value < 0], default=None)
    yaw_right_peak = max([value for value in yaw_values if value > 0], default=None)
    pitch_mean = _mean(pitch_values)
    roll_mean = _mean(roll_values)

    reaction_summary = None
    background_reaction_ms = 0.0
    if background_reaction_evaluator is not None:
        reaction_started = time.perf_counter()
        reaction_summary = background_reaction_evaluator.evaluate(
            [
                ReactionSignalFrame(
                    timestamp=entry.timestamp,
                    face_detected=entry.face_detected,
                    active_score=entry.active_score,
                    active_evidence=entry.active_evidence,
                    ear_current=entry.ear_current,
                    mar_current=entry.mar_current,
                    yaw_current=entry.yaw_current,
                    pitch_current=entry.pitch_current,
                    roll_current=entry.roll_current,
                    face_quality=entry.face_quality,
                    face_size_ratio=entry.face_size_ratio,
                    smile_score=_maybe_float(entry.details.get("smile")),
                    blink_score=_maybe_float(entry.details.get("blink")),
                    ear_baseline=ear_baseline if ear_baseline is not None else _maybe_float(entry.details.get("ear_baseline")),
                    mar_baseline=mar_baseline if mar_baseline is not None else _maybe_float(entry.details.get("mar_baseline")),
                    smile_baseline=smile_baseline if smile_baseline is not None else _maybe_float(entry.details.get("smile_baseline")),
                    yaw_baseline=yaw_baseline if yaw_baseline is not None else _maybe_float(entry.details.get("yaw_baseline")),
                    pitch_baseline=pitch_baseline if pitch_baseline is not None else _maybe_float(entry.details.get("pitch_baseline")),
                    roll_baseline=roll_baseline if roll_baseline is not None else _maybe_float(entry.details.get("roll_baseline")),
                )
                for entry in entries
            ],
            passive_window_score=passive_window_score or 0.0,
        )
        background_reaction_ms = (time.perf_counter() - reaction_started) * 1000.0

    blink_evidence = reaction_summary.blink_evidence if reaction_summary else _clamp01(_ratio(ear_drop, (ear_baseline or ear_max or 0.0) * 0.35 if (ear_baseline or ear_max) else None))
    smile_evidence = reaction_summary.smile_evidence if reaction_summary else _clamp01(_ratio(mar_rise, (mar_baseline or mar_max or 0.0) * 0.45 if (mar_baseline or mar_max) else None))
    mouth_open_evidence = reaction_summary.mouth_open_evidence if reaction_summary else _clamp01(_ratio(mar_max, (mar_baseline or mar_mean or 0.0) * 1.7 if (mar_baseline or mar_mean) else None))
    head_turn_left_evidence = reaction_summary.head_turn_left_evidence if reaction_summary else _clamp01(_ratio(yaw_left_peak, 15.0))
    head_turn_right_evidence = reaction_summary.head_turn_right_evidence if reaction_summary else _clamp01(_ratio(yaw_right_peak, 15.0))
    primary_event = reaction_summary.primary_event if reaction_summary else 0.0
    secondary_event = reaction_summary.secondary_event if reaction_summary else 0.0
    raw_reaction_evidence = reaction_summary.raw_reaction_evidence if reaction_summary else 0.0
    effective_trust = reaction_summary.effective_trust if reaction_summary else 0.65
    trusted_reaction_evidence = reaction_summary.trusted_reaction_evidence if reaction_summary else 0.0
    persisted_primary = reaction_summary.persisted_primary if reaction_summary else 0.0
    persisted_secondary = reaction_summary.persisted_secondary if reaction_summary else 0.0
    persisted_reaction_evidence = reaction_summary.persisted_reaction_evidence if reaction_summary else 0.0
    combined_active_evidence = reaction_summary.combined_active_evidence if reaction_summary else 0.0
    combined_active_score = reaction_summary.combined_active_score if reaction_summary else 0.0
    raw_active_evidence = reaction_summary.raw_active_evidence if reaction_summary else 0.0
    passive_weight = reaction_summary.passive_weight if reaction_summary else 0.88
    active_weight = reaction_summary.active_weight if reaction_summary else 0.12
    passive_window_score = reaction_summary.passive_window_score if reaction_summary else (passive_window_score or 0.0)
    active_frame_score_mean = reaction_summary.active_frame_score_mean if reaction_summary else 0.0
    active_frame_evidence_mean = reaction_summary.active_frame_evidence_mean if reaction_summary else 0.0
    supported_score = (
        reaction_summary.supported_score
        if reaction_summary is not None
        else min(100.0, max(0.0, passive_window_score * passive_weight + combined_active_score * active_weight))
    )

    return {
        "supported_score": supported_score,
        "ear_mean": ear_mean,
        "ear_min": ear_min,
        "ear_max": ear_max,
        "ear_drop": ear_drop,
        "ear_drop_ratio": ear_drop_ratio,
        "mar_mean": mar_mean,
        "mar_max": mar_max,
        "mar_rise": mar_rise,
        "mar_rise_ratio": mar_rise_ratio,
        "yaw_mean": yaw_mean,
        "yaw_left_peak": yaw_left_peak,
        "yaw_right_peak": yaw_right_peak,
        "pitch_mean": pitch_mean,
        "roll_mean": roll_mean,
        "baseline_ready": session_baseline.calibrated if session_baseline else False,
        "baseline_sample_count": session_baseline.sample_count if session_baseline else 0,
        "baseline_duration_seconds": session_baseline.duration_seconds if session_baseline else 0.0,
        "ear_baseline": ear_baseline,
        "mar_baseline": mar_baseline,
        "smile_baseline": smile_baseline,
        "yaw_baseline": yaw_baseline,
        "pitch_baseline": pitch_baseline,
        "roll_baseline": roll_baseline,
        "blink_evidence": blink_evidence,
        "smile_evidence": smile_evidence,
        "mouth_open_evidence": mouth_open_evidence,
        "head_turn_left_evidence": head_turn_left_evidence,
        "head_turn_right_evidence": head_turn_right_evidence,
        "primary_event": primary_event,
        "secondary_event": secondary_event,
        "raw_reaction_evidence": raw_reaction_evidence,
        "effective_trust": effective_trust,
        "trusted_reaction_evidence": trusted_reaction_evidence,
        "base_active_trust": reaction_summary.base_active_trust if reaction_summary else effective_trust,
        "trust_penalty": reaction_summary.trust_penalty if reaction_summary else 0.0,
        "blink_anomaly_score": reaction_summary.blink_anomaly_score if reaction_summary else 0.0,
        "motion_anomaly_score": reaction_summary.motion_anomaly_score if reaction_summary else 0.0,
        "signal_inconsistency_score": reaction_summary.signal_inconsistency_score if reaction_summary else 0.0,
        "spoof_support_score": reaction_summary.spoof_support_score if reaction_summary else 0.0,
        "persisted_primary": persisted_primary,
        "persisted_secondary": persisted_secondary,
        "persisted_reaction_evidence": persisted_reaction_evidence,
        "raw_active_evidence": raw_active_evidence,
        "combined_active_evidence": combined_active_evidence,
        "combined_active_score": combined_active_score,
        "active_score_mapping_mode": reaction_summary.active_score_mapping_mode if reaction_summary else "standard_sqrt",
        "active_score_standard_mapping": reaction_summary.active_score_standard_mapping if reaction_summary else _standard_active_support_score(combined_active_evidence),
        "active_score_strict_mapping": reaction_summary.active_score_strict_mapping if reaction_summary else _standard_active_support_score(combined_active_evidence),
        "passive_window_score": passive_window_score,
        "active_frame_score_mean": active_frame_score_mean,
        "active_frame_evidence_mean": active_frame_evidence_mean,
        "background_reaction_ms": background_reaction_ms,
    }


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _standard_active_support_score(active_evidence: float) -> float:
    return min(100.0, max(0.0, 100.0 * float(np.sqrt(max(active_evidence, 0.0)))))


def _strict_sigmoid_active_support_score(active_evidence: float, settings: Settings) -> float:
    sigmoid = settings.get_strict_sigmoid_config()
    return _normalized_sigmoid_score(
        active_evidence,
        midpoint=sigmoid["midpoint"],
        steepness=sigmoid["steepness"],
        scale=sigmoid["scale"],
    )


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) <= 1e-6:
        return None
    return float(numerator / denominator)


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _count_consecutive_no_face(entries: list[FrameMetrics]) -> int:
    count = 0
    for entry in reversed(entries):
        if entry.face_detected or entry.held_from_previous:
            break
        count += 1
    return count


def _is_face_occluded(
    frame_metrics: FrameMetrics, *, baseline_mar: Optional[float] = None
) -> bool:
    """True when the face bounding box is detected but key landmark regions are blocked."""
    if not frame_metrics.face_detected:
        return False

    # EAR and MAR geometry confirms whether the eyes and mouth are readable.
    # MediaPipe visibility scores read near-zero when head boundaries are covered
    # (e.g. headscarf) even when the face is fully visible, so we trust geometry
    # over mesh-fit visibility. However, MediaPipe hallucinates mouth landmarks even
    # when the mouth is physically blocked, so MAR alone cannot detect coverage.
    # We therefore also check lower-face pixel texture: a hand or object over the
    # mouth/nose creates a uniform patch whose Laplacian variance is much lower than
    # real facial skin texture). This fires even when EAR/MAR look normal.
    ear = frame_metrics.ear_current
    mar = frame_metrics.mar_current
    if ear is not None and ear >= 0.15 and mar is not None:
        mar_threshold = baseline_mar * 0.6 if baseline_mar is not None else 0.15
        if mar >= mar_threshold:
            # Eyes and mouth geometry look normal. Do a final pixel-texture check on
            # the lower face to catch hand/object coverage that landmarks miss.
            lower_texture = _maybe_float(frame_metrics.details.get("preview_lower_face_texture"))
            blur = frame_metrics.blur_score
            if (
                lower_texture is not None
                and blur is not None
                and blur > 10.0
                and lower_texture < blur * _OCCLUSION_LOWER_TEXTURE_RATIO
            ):
                return True
            return False

    landmark_face_visible = _maybe_float(frame_metrics.details.get("landmark_face_visible"))
    if landmark_face_visible is not None and landmark_face_visible < 0.5:
        return True
    visibility_score = _maybe_float(frame_metrics.details.get("landmark_visibility_score"))
    if visibility_score is not None and visibility_score < _OCCLUSION_VISIBILITY_THRESHOLD:
        return True
    occlusion_quality = _maybe_float(frame_metrics.details.get("quality_occlusion"))
    if occlusion_quality is not None and occlusion_quality < _OCCLUSION_QUALITY_THRESHOLD:
        return True
    return False

def _count_consecutive_occluded(
    entries: list[FrameMetrics], *, baseline_mar: Optional[float] = None
) -> int:
    """Count occluded frames in the trailing streak, tolerating one isolated clear frame.

    A single non-occluded frame mid-streak (e.g. texture fluctuation) does not reset
    the count. Two or more consecutive clear frames end the streak.
    """
    count = 0
    clear_run = 0
    for entry in reversed(entries):
        if not entry.face_detected:
            break
        if _is_face_occluded(entry, baseline_mar=baseline_mar):
            count += 1
            clear_run = 0
        else:
            clear_run += 1
            if clear_run >= 2:
                break
    return count


def _tail_no_face_run_duration(entries: list[FrameMetrics]) -> float:
    no_face_tail: list[FrameMetrics] = []
    for entry in reversed(entries):
        if entry.face_detected or entry.held_from_previous:
            break
        no_face_tail.append(entry)
    if len(no_face_tail) < 2:
        return 0.0
    return max(0.0, no_face_tail[0].timestamp - no_face_tail[-1].timestamp)


def _effective_entries_for_analysis(
    entries: list[FrameMetrics],
    *,
    face_return_grace_seconds: float,
) -> list[FrameMetrics]:
    if not entries or not entries[-1].face_detected:
        return entries

    tail_count = 0
    for entry in reversed(entries[:-1]):
        if entry.face_detected or entry.held_from_previous:
            break
        tail_count += 1
    if tail_count <= 0:
        return entries

    no_face_start = len(entries) - 1 - tail_count
    no_face_run = entries[no_face_start:-1]
    if len(no_face_run) < 1:
        return entries

    no_face_duration = max(0.0, entries[-1].timestamp - no_face_run[0].timestamp)
    if no_face_duration > face_return_grace_seconds:
        return entries

    return entries[:no_face_start] + [entries[-1]]


def _is_low_quality(frame_metrics: FrameMetrics) -> bool:
    if not frame_metrics.face_detected:
        return False
    if frame_metrics.face_size_ratio is not None and frame_metrics.face_size_ratio < 0.08:
        return True
    if frame_metrics.face_quality is not None and frame_metrics.face_quality < 0.45:
        return True
    if frame_metrics.blur_score is not None and frame_metrics.blur_score < 25.0:
        return True
    if not 45.0 <= frame_metrics.brightness <= 215.0:
        return True
    return False


def _quality_block_reason(frame_metrics: FrameMetrics) -> str:
    if not frame_metrics.face_detected:
        return "-"
    if frame_metrics.face_size_ratio is not None and frame_metrics.face_size_ratio < 0.08:
        return "face_too_small"
    if frame_metrics.face_quality is not None and frame_metrics.face_quality < 0.45:
        return "face_quality_low"
    if frame_metrics.blur_score is not None and frame_metrics.blur_score < 25.0:
        return "blur_low"
    if frame_metrics.brightness < 45.0:
        return "brightness_low"
    if frame_metrics.brightness > 215.0:
        return "brightness_high"
    return "-"


def _is_quality_blocked(frame_metrics: FrameMetrics, aggregate: AggregatedMetrics) -> bool:
    return aggregate.decision_state == "INSUFFICIENT_EVIDENCE" and _quality_block_reason(frame_metrics) != "-"


def _is_warm_recovery_candidate(
    *,
    recent_entries: list[FrameMetrics],
    current_frame: FrameMetrics,
    face_return_grace_seconds: float,
) -> bool:
    if not current_frame.face_detected or len(recent_entries) < 2:
        return False

    tail_no_face: list[FrameMetrics] = []
    for entry in reversed(recent_entries[:-1]):
        if entry.face_detected or entry.held_from_previous:
            break
        tail_no_face.append(entry)

    if not tail_no_face:
        return False

    no_face_duration = max(0.0, current_frame.timestamp - tail_no_face[-1].timestamp)
    if no_face_duration > face_return_grace_seconds:
        return False

    prior_valid_faces = sum(
        1 for entry in recent_entries[: len(recent_entries) - 1 - len(tail_no_face)]
        if entry.face_detected
    )
    return prior_valid_faces >= 2


def _has_sufficient_evidence(
    *,
    recent_entries: list[FrameMetrics],
    temporal_signal_summary: dict[str, Any],
    current_frame: FrameMetrics,
    min_trusted_face_size_ratio: float,
    warm_recovery: bool,
) -> bool:
    strict_exam = _is_strict_exam_profile(current_frame.details)
    minimum_samples = 3 if (warm_recovery and strict_exam) else 2 if warm_recovery else 8 if strict_exam else 6
    if len(recent_entries) < minimum_samples:
        return False
    if not current_frame.face_detected:
        return False
    if current_frame.face_size_ratio is not None and current_frame.face_size_ratio < min_trusted_face_size_ratio:
        return False
    baseline_ready = bool(temporal_signal_summary.get("baseline_ready"))
    baseline_sample_count = int(temporal_signal_summary.get("baseline_sample_count") or 0)
    baseline_min = 2 if strict_exam else 1
    if not baseline_ready and not (warm_recovery and baseline_sample_count >= baseline_min):
        return False

    combined_active_evidence = (
        temporal_signal_summary.get("final_active_evidence")
        if temporal_signal_summary.get("final_active_evidence") is not None
        else temporal_signal_summary.get("combined_active_evidence")
    ) or 0.0
    face_present_ratio = sum(1 for entry in recent_entries if entry.face_detected) / max(len(recent_entries), 1)
    min_face_ratio = 0.45 if (warm_recovery and strict_exam) else 0.35 if warm_recovery else 0.68 if strict_exam else 0.6
    min_confidence = 0.52 if (warm_recovery and strict_exam) else 0.45 if warm_recovery else 0.66 if strict_exam else 0.60
    return face_present_ratio >= min_face_ratio and (
        combined_active_evidence >= (0.14 if strict_exam else 0.10)
        or current_frame.confidence >= min_confidence
    )


def _resolve_decision_state(
    *,
    recent_entries: list[FrameMetrics],
    current_frame: FrameMetrics,
    temporal_signal_summary: dict[str, Any],
    face_present_ratio: float,
    consecutive_no_face_frames: int,
    warm_recovery: bool,
    low_quality: bool,
    sufficient_evidence: bool,
    baseline_sample_count: int,
    smoothed_score: float,
    decision_confidence: float,
    no_face_consecutive_threshold: int,
) -> str:
    debug_decision_state = temporal_signal_summary.get("debug_decision_state")
    if isinstance(debug_decision_state, str) and debug_decision_state:
        return debug_decision_state
    if consecutive_no_face_frames >= no_face_consecutive_threshold or face_present_ratio <= 0.15:
        if _should_treat_no_face_as_insufficient(
            recent_entries=recent_entries,
            current_frame=current_frame,
            face_present_ratio=face_present_ratio,
            consecutive_no_face_frames=consecutive_no_face_frames,
            baseline_sample_count=baseline_sample_count,
            no_face_consecutive_threshold=no_face_consecutive_threshold,
        ):
            return "INSUFFICIENT_EVIDENCE"
        return "NO_FACE"
    quality_reason = _quality_block_reason(current_frame)
    quality_blocked = quality_reason != "-"
    if low_quality and quality_reason == "face_too_small":
        return "LOW_QUALITY"
    spoof_gate_active = _is_device_replay_spoof_detected(current_frame)
    strict_hard_block = bool(temporal_signal_summary.get("strict_hard_block"))
    puzzle_fusion_active = bool(temporal_signal_summary.get("puzzle_fusion_active"))
    puzzle_success = bool(temporal_signal_summary.get("puzzle_success"))
    puzzle_status = str(temporal_signal_summary.get("puzzle_status") or "idle")
    support_streak = int(_maybe_float(current_frame.details.get("preview_spoof_support_streak")) or 0.0)
    spoof_ready_despite_evidence = spoof_gate_active and support_streak >= 3
    early_live_ready = (
        current_frame.face_detected
        and current_frame.is_live
        and not low_quality
        and baseline_sample_count >= 3
        and (current_frame.face_size_ratio or 0.0) >= 0.09
        and smoothed_score >= (72.0 if warm_recovery else 76.0)
        and decision_confidence >= (0.58 if warm_recovery else 0.72)
        and not _is_device_replay_spoof_detected(current_frame)
    )
    challenge_live_block = bool(
        puzzle_fusion_active
        and not puzzle_success
        and puzzle_status in {"running", "failed", "timed_out"}
    )
    if not sufficient_evidence and early_live_ready:
        return "LIVE"
    if strict_hard_block:
        return "SPOOF"
    if spoof_gate_active:
        return "SPOOF"
    if quality_blocked and not spoof_gate_active:
        return "LOW_QUALITY"
    if not sufficient_evidence and not spoof_ready_despite_evidence:
        return "INSUFFICIENT_EVIDENCE"
    if low_quality:
        return "LOW_QUALITY"

    live_score_threshold = 68.0 if warm_recovery else 80.0
    live_conf_threshold = 0.50 if warm_recovery else 0.65
    if not challenge_live_block and smoothed_score >= live_score_threshold and decision_confidence >= live_conf_threshold:
        return "LIVE"
    if (
        not challenge_live_block
        and sufficient_evidence
        and current_frame.is_live
        and smoothed_score >= (66.0 if warm_recovery else 70.0)
        and decision_confidence >= (0.46 if warm_recovery else 0.72)
    ):
        return "LIVE"
    if smoothed_score < 55.0 and decision_confidence >= 0.55:
        return "SPOOF"
    return "INSUFFICIENT_EVIDENCE"


def _is_strict_exam_profile(details: dict[str, Any]) -> bool:
    return False


def _should_treat_no_face_as_insufficient(
    *,
    recent_entries: list[FrameMetrics],
    current_frame: FrameMetrics,
    face_present_ratio: float,
    consecutive_no_face_frames: int,
    baseline_sample_count: int,
    no_face_consecutive_threshold: int,
) -> bool:
    if current_frame.face_detected or current_frame.held_from_previous:
        return False
    if baseline_sample_count < 3:
        return False
    if not recent_entries:
        return False
    if consecutive_no_face_frames > max(2, no_face_consecutive_threshold - 2):
        return False
    if face_present_ratio < 0.18:
        return False

    prior_faces = [entry for entry in recent_entries[:-1] if entry.face_detected or entry.held_from_previous]
    if len(prior_faces) < 3:
        return False

    last_face = prior_faces[-1]
    if current_frame.timestamp - last_face.timestamp > 0.9:
        return False

    recent_face_run = 0
    for entry in reversed(recent_entries[:-1]):
        if entry.face_detected or entry.held_from_previous:
            recent_face_run += 1
            continue
        break
    if recent_face_run < 2:
        return False

    return True


def _calculate_temporal_consistency(score_variance: float) -> float:
    return max(0.0, min(1.0, 1.0 - (score_variance / 400.0)))


def _face_size_adequacy(face_size_ratio: Optional[float]) -> float:
    if face_size_ratio is None:
        return 0.0
    return max(0.0, min(1.0, face_size_ratio / 0.18))


def _blur_adequacy(blur_score: Optional[float]) -> float:
    if blur_score is None:
        return 0.0
    return max(0.0, min(1.0, blur_score / 120.0))


def _brightness_adequacy(brightness: float) -> float:
    return max(0.0, min(1.0, 1.0 - abs(brightness - 130.0) / 110.0))


def _calculate_window_confidence(
    *,
    recent_entries: list[FrameMetrics],
    face_present_ratio: float,
    temporal_signal_summary: dict[str, Any],
    sufficient_evidence: bool,
    temporal_consistency: float,
    frame_confidence_mean: float,
    directional_agreement_mean: float,
    face_quality_mean: float,
    face_size_adequacy: float,
    blur_adequacy: float,
    brightness_adequacy: float,
) -> float:
    if not recent_entries:
        return 0.0

    # Confidence is a reliability metric. It intentionally excludes the smoothed
    # liveness score and is driven only by input quality, agreement, presence,
    # temporal stability, and evidence sufficiency.
    valid_face_ratio = sum(1 for entry in recent_entries if entry.face_detected) / max(len(recent_entries), 1)
    evidence_strength = max(
        0.0,
        min(
            1.0,
            0.55
            * float(
                (
                    temporal_signal_summary.get("final_active_evidence")
                    if temporal_signal_summary.get("final_active_evidence") is not None
                    else temporal_signal_summary.get("combined_active_evidence")
                )
                or 0.0
            )
            + 0.45 * (1.0 if sufficient_evidence else 0.0),
        ),
    )
    confidence = (
        0.18 * max(0.0, min(1.0, face_quality_mean))
        + 0.16 * max(0.0, min(1.0, valid_face_ratio))
        + 0.14 * face_size_adequacy
        + 0.12 * blur_adequacy
        + 0.10 * brightness_adequacy
        + 0.10 * max(0.0, min(1.0, face_present_ratio))
        + 0.08 * max(0.0, min(1.0, frame_confidence_mean))
        + 0.12 * max(0.0, min(1.0, directional_agreement_mean))
        + 0.10 * temporal_consistency
        + 0.10 * evidence_strength
    )
    if len(recent_entries) < 4:
        confidence *= 0.75
    return max(0.0, min(1.0, confidence))


def _first_available(entries: list[FrameMetrics], field_name: str) -> Optional[float]:
    for entry in reversed(entries):
        value = _maybe_float(entry.details.get(field_name))
        if value is not None:
            return value
    return None
