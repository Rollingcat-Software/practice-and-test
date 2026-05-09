from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ReactionSignalFrame:
    timestamp: float
    face_detected: bool
    active_score: float
    active_evidence: Optional[float]
    ear_current: Optional[float]
    mar_current: Optional[float]
    yaw_current: Optional[float]
    pitch_current: Optional[float] = None
    roll_current: Optional[float] = None
    face_quality: Optional[float] = None
    face_size_ratio: Optional[float] = None
    smile_score: Optional[float] = None
    blink_score: Optional[float] = None
    ear_baseline: Optional[float] = None
    mar_baseline: Optional[float] = None
    smile_baseline: Optional[float] = None
    yaw_baseline: Optional[float] = None
    pitch_baseline: Optional[float] = None
    roll_baseline: Optional[float] = None


@dataclass(frozen=True)
class BackgroundReactionSummary:
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
    persisted_primary: float
    persisted_secondary: float
    persisted_reaction_evidence: float
    raw_active_evidence: float
    combined_active_evidence: float
    combined_active_score: float
    supported_score: float
    passive_weight: float
    active_weight: float
    passive_window_score: float
    active_frame_score_mean: float
    active_frame_evidence_mean: float
    active_score_mapping_mode: str = "standard_sqrt"
    active_score_standard_mapping: float = 0.0
    active_score_strict_mapping: float = 0.0
    base_active_trust: float = 0.0
    trust_penalty: float = 0.0
    blink_anomaly_score: float = 0.0
    motion_anomaly_score: float = 0.0
    signal_inconsistency_score: float = 0.0
    spoof_support_score: float = 0.0


@dataclass(frozen=True)
class _SignalHistorySample:
    timestamp: float
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    face_size_ratio: Optional[float]


@dataclass(frozen=True)
class _AnomalySummary:
    blink_anomaly_score: float
    motion_anomaly_score: float
    signal_inconsistency_score: float
    trust_penalty: float
    spoof_support_score: float


class BackgroundActiveReactionEvaluator:
    BLINK_DROP_RATIO = 0.16
    BLINK_RECOVERY_RATIO = 0.08
    BLINK_MIN_DROP_HOLD_SECONDS = 0.05
    SMILE_DELTA_THRESHOLD = 6.5
    SMILE_MIN_HOLD_SECONDS = 0.18
    MOUTH_OPEN_RATIO_THRESHOLD = 1.32
    MOUTH_OPEN_MIN_HOLD_SECONDS = 0.14
    HEAD_TURN_THRESHOLD_DEGREES = 9.0
    HEAD_TURN_MIN_HOLD_SECONDS = 0.12
    HEAD_TURN_RETURN_RATIO = 0.45
    MOUTH_OPEN_RETURN_RATIO = 0.60
    SUPPORTIVE_ACTIVE_BASE_BONUS = 4.0
    SUPPORTIVE_ACTIVE_MAX_BONUS = 12.0
    PRIMARY_EVENT_WEIGHT = 0.75
    SECONDARY_EVENT_WEIGHT = 0.25
    BLINK_EVENT_WEIGHT = 1.20
    MOUTH_OPEN_EVENT_WEIGHT = 0.68
    SMILE_EVENT_WEIGHT = 0.95
    HEAD_TURN_EVENT_WEIGHT = 1.00
    CURRENT_REACTION_BLEND_WEIGHT = 0.35
    PERSISTED_REACTION_BLEND_WEIGHT = 0.65
    TRUST_FLOOR = 0.65
    TRUST_RANGE = 0.35
    MAX_ANOMALY_TRUST_PENALTY = 0.45
    HISTORY_WINDOW_SECONDS = 3.0
    MAX_HISTORY_SAMPLES = 72
    BLINK_EVENT_WINDOW_SECONDS = 1.8
    BLINK_EVENT_HISTORY_LIMIT = 16
    BLINK_MIN_INTERVAL_SECONDS = 0.12
    BLINK_RATE_WINDOW_SECONDS = 1.35
    MOTION_WINDOW_SECONDS = 1.25

    def __init__(
        self,
        *,
        decay_seconds: float = 2.4,
        min_face_size_ratio: float = 0.08,
        security_profile: str = "standard",
        strict_sigmoid_midpoint: float = 0.62,
        strict_sigmoid_steepness: float = 12.0,
        strict_sigmoid_scale: float = 100.0,
    ) -> None:
        self._decay_seconds = decay_seconds
        self._min_face_size_ratio = min_face_size_ratio
        self._security_profile = str(security_profile or "standard").lower()
        self._strict_sigmoid_midpoint = float(strict_sigmoid_midpoint)
        self._strict_sigmoid_steepness = float(strict_sigmoid_steepness)
        self._strict_sigmoid_scale = float(strict_sigmoid_scale)
        self._persisted_reaction_evidence = {
            "blink": 0.0,
            "head_turn_left": 0.0,
            "head_turn_right": 0.0,
            "mouth_open": 0.0,
            "smile": 0.0,
        }
        self._recent_signal_history: deque[_SignalHistorySample] = deque(maxlen=self.MAX_HISTORY_SAMPLES)
        self._recent_blink_events: deque[float] = deque(maxlen=self.BLINK_EVENT_HISTORY_LIMIT)
        self._last_timestamp: Optional[float] = None
        self._last_history_timestamp: Optional[float] = None
        self._last_blink_event_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._persisted_reaction_evidence = {
            "blink": 0.0,
            "head_turn_left": 0.0,
            "head_turn_right": 0.0,
            "mouth_open": 0.0,
            "smile": 0.0,
        }
        self._recent_signal_history.clear()
        self._recent_blink_events.clear()
        self._last_timestamp = None
        self._last_history_timestamp = None
        self._last_blink_event_timestamp = None

    def evaluate(self, frames: Sequence[ReactionSignalFrame], *, passive_window_score: float) -> BackgroundReactionSummary:
        latest_timestamp = frames[-1].timestamp if frames else self._last_timestamp
        self._decay_persisted_evidence(latest_timestamp)
        self._prune_history(latest_timestamp)
        usable_frames = [frame for frame in frames if frame.face_detected]
        if not usable_frames:
            p1, p2, pr = self._strongest_two_evidence(self._persisted_reaction_evidence)
            return BackgroundReactionSummary(
                blink_evidence=None, smile_evidence=None, mouth_open_evidence=None,
                head_turn_left_evidence=None, head_turn_right_evidence=None,
                primary_event=0.0, secondary_event=0.0, raw_reaction_evidence=0.0,
                effective_trust=self._effective_trust(0.0, 0.0), trusted_reaction_evidence=0.0,
                persisted_primary=p1, persisted_secondary=p2, persisted_reaction_evidence=pr,
                raw_active_evidence=0.0, combined_active_evidence=pr,
                combined_active_score=self._active_score_from_evidence(pr),
                supported_score=passive_window_score, passive_weight=0.88, active_weight=0.12,
                passive_window_score=passive_window_score, active_frame_score_mean=0.0, active_frame_evidence_mean=0.0,
                active_score_mapping_mode=self._active_score_mapping_mode(),
                active_score_standard_mapping=self._standard_active_score_from_evidence(pr),
                active_score_strict_mapping=self._standard_active_score_from_evidence(pr),
            )

        ear_values = [f.ear_current for f in usable_frames if f.ear_current is not None]
        mar_values = [f.mar_current for f in usable_frames if f.mar_current is not None]
        yaw_values = [f.yaw_current for f in usable_frames if f.yaw_current is not None]
        smile_scores = [f.smile_score for f in usable_frames if f.smile_score is not None]
        frame_active_scores = [f.active_score for f in usable_frames]
        frame_active_evidence = [f.active_evidence for f in usable_frames if f.active_evidence is not None]
        base_active_trust = self._compute_base_active_trust(usable_frames)

        ear_baseline = _first_available([f.ear_baseline for f in usable_frames]) or (float(np.percentile(ear_values, 80)) if ear_values else None)
        mar_baseline = _first_available([f.mar_baseline for f in usable_frames]) or (float(np.percentile(mar_values, 25)) if mar_values else None)
        smile_baseline = _first_available([f.smile_baseline for f in usable_frames]) or (float(np.percentile(smile_scores, 25)) if smile_scores else None)
        yaw_baseline = _first_available([f.yaw_baseline for f in usable_frames]) or (float(np.median(yaw_values)) if yaw_values else None)

        blink_evidence = self.compute_blink_evidence(usable_frames, ear_baseline)
        smile_evidence = self.compute_smile_evidence(usable_frames, mar_baseline, smile_baseline)
        mouth_open_evidence = self.compute_mouth_open_evidence(usable_frames, mar_baseline)
        left_evidence = self.compute_head_turn_left_evidence(usable_frames, yaw_baseline)
        right_evidence = self.compute_head_turn_right_evidence(usable_frames, yaw_baseline)

        self._register_blink_events(self._extract_blink_event_timestamps(usable_frames, ear_baseline))
        self._register_signal_history(usable_frames)

        raw_events = {
            "blink": blink_evidence or 0.0,
            "head_turn_left": left_evidence or 0.0,
            "head_turn_right": right_evidence or 0.0,
            "mouth_open": mouth_open_evidence or 0.0,
            "smile": smile_evidence or 0.0,
        }
        weighted_events = self._apply_event_weights(raw_events)
        primary_event, secondary_event, raw_reaction_evidence = self._strongest_two_evidence(weighted_events)

        anomaly = self._compute_anomaly_summary(latest_timestamp)
        effective_trust = self._effective_trust(base_active_trust, anomaly.trust_penalty)
        trusted_reaction_evidence = _clamp01(raw_reaction_evidence * effective_trust)

        self._persist_reaction_evidence({k: v * effective_trust for k, v in weighted_events.items()})
        persisted_primary, persisted_secondary, persisted_reaction_evidence = self._strongest_two_evidence(self._persisted_reaction_evidence)
        combined_active_evidence = _clamp01(
            self.CURRENT_REACTION_BLEND_WEIGHT * trusted_reaction_evidence
            + self.PERSISTED_REACTION_BLEND_WEIGHT * persisted_reaction_evidence
        )
        combined_active_score = self._active_score_from_evidence(combined_active_evidence)

        passive_foundation = max(0.0, min(100.0, passive_window_score))
        passive_strength = _clamp01((passive_foundation - 45.0) / 35.0)
        active_bonus_budget = self.SUPPORTIVE_ACTIVE_BASE_BONUS + (self.SUPPORTIVE_ACTIVE_MAX_BONUS - self.SUPPORTIVE_ACTIVE_BASE_BONUS) * passive_strength
        supportive_active_bonus = active_bonus_budget * combined_active_evidence * combined_active_evidence * effective_trust
        supported_score = min(100.0, passive_foundation + supportive_active_bonus)
        active_contribution = max(0.0, supported_score - passive_foundation)
        active_weight = min(0.22, active_contribution / max(supported_score, 1e-6))

        return BackgroundReactionSummary(
            blink_evidence=blink_evidence, smile_evidence=smile_evidence, mouth_open_evidence=mouth_open_evidence,
            head_turn_left_evidence=left_evidence, head_turn_right_evidence=right_evidence,
            primary_event=primary_event, secondary_event=secondary_event, raw_reaction_evidence=raw_reaction_evidence,
            effective_trust=effective_trust, trusted_reaction_evidence=trusted_reaction_evidence,
            persisted_primary=persisted_primary, persisted_secondary=persisted_secondary,
            persisted_reaction_evidence=persisted_reaction_evidence, raw_active_evidence=trusted_reaction_evidence,
            combined_active_evidence=combined_active_evidence, combined_active_score=combined_active_score,
            supported_score=supported_score, passive_weight=1.0 - active_weight, active_weight=active_weight,
            passive_window_score=passive_window_score,
            active_frame_score_mean=float(np.mean(frame_active_scores)) if frame_active_scores else 0.0,
            active_frame_evidence_mean=float(np.mean(frame_active_evidence)) if frame_active_evidence else 0.0,
            active_score_mapping_mode=self._active_score_mapping_mode(),
            active_score_standard_mapping=self._standard_active_score_from_evidence(combined_active_evidence),
            active_score_strict_mapping=self._standard_active_score_from_evidence(combined_active_evidence),
            base_active_trust=base_active_trust, trust_penalty=anomaly.trust_penalty,
            blink_anomaly_score=anomaly.blink_anomaly_score, motion_anomaly_score=anomaly.motion_anomaly_score,
            signal_inconsistency_score=anomaly.signal_inconsistency_score, spoof_support_score=anomaly.spoof_support_score,
        )

    def _compute_base_active_trust(self, frames: Sequence[ReactionSignalFrame]) -> float:
        face_quality_mean = _mean([f.face_quality for f in frames if f.face_quality is not None]) or 0.0
        sizes = [f.face_size_ratio for f in frames if f.face_size_ratio is not None]
        face_size_trust = _clamp01(float(np.mean(sizes)) / max(self._min_face_size_ratio, 1e-6)) if sizes else 0.0
        return _clamp01(0.60 * face_size_trust + 0.40 * face_quality_mean)

    def _apply_event_weights(self, raw_events: dict[str, float]) -> dict[str, float]:
        return {
            "blink": _clamp01(raw_events.get("blink", 0.0) * self.BLINK_EVENT_WEIGHT),
            "head_turn_left": _clamp01(raw_events.get("head_turn_left", 0.0) * self.HEAD_TURN_EVENT_WEIGHT),
            "head_turn_right": _clamp01(raw_events.get("head_turn_right", 0.0) * self.HEAD_TURN_EVENT_WEIGHT),
            "mouth_open": _clamp01(raw_events.get("mouth_open", 0.0) * self.MOUTH_OPEN_EVENT_WEIGHT),
            "smile": _clamp01(raw_events.get("smile", 0.0) * self.SMILE_EVENT_WEIGHT),
        }

    def _active_score_from_evidence(self, active_evidence: float) -> float:
        return self._standard_active_score_from_evidence(active_evidence)

    def _active_score_mapping_mode(self) -> str:
        return "standard_sqrt"

    def _standard_active_score_from_evidence(self, active_evidence: float) -> float:
        return min(100.0, max(0.0, 100.0 * float(np.sqrt(max(active_evidence, 0.0)))))

    def _strict_active_score_from_evidence(self, active_evidence: float) -> float:
        return _normalized_sigmoid_score(
            active_evidence,
            midpoint=self._strict_sigmoid_midpoint,
            steepness=self._strict_sigmoid_steepness,
            scale=self._strict_sigmoid_scale,
        )

    def _decay_persisted_evidence(self, latest_timestamp: Optional[float]) -> None:
        if latest_timestamp is None:
            return
        if self._last_timestamp is None:
            self._last_timestamp = latest_timestamp
            return
        delta_seconds = max(0.0, latest_timestamp - self._last_timestamp)
        if delta_seconds <= 0.0:
            return
        decay_multiplier = float(np.exp(-delta_seconds / max(self._decay_seconds, 1e-6)))
        for key, value in self._persisted_reaction_evidence.items():
            self._persisted_reaction_evidence[key] = float(value) * decay_multiplier
        self._last_timestamp = latest_timestamp

    def _persist_reaction_evidence(self, current_values: dict[str, float]) -> None:
        for key, value in current_values.items():
            self._persisted_reaction_evidence[key] = max(self._persisted_reaction_evidence.get(key, 0.0), _clamp01(value))

    def _effective_trust(self, active_trust: float, anomaly_penalty: float) -> float:
        nominal = _clamp01(self.TRUST_FLOOR + self.TRUST_RANGE * _clamp01(active_trust))
        return _clamp01(nominal * (1.0 - self.MAX_ANOMALY_TRUST_PENALTY * _clamp01(anomaly_penalty)))

    def _strongest_two_evidence(self, values: dict[str, float]) -> tuple[float, float, float]:
        ranked = sorted((_clamp01(v) for v in values.values()), reverse=True)
        primary = ranked[0] if ranked else 0.0
        secondary = ranked[1] if len(ranked) > 1 else 0.0
        return primary, secondary, _clamp01(self.PRIMARY_EVENT_WEIGHT * primary + self.SECONDARY_EVENT_WEIGHT * secondary)

    def _extract_blink_event_timestamps(self, frames: Sequence[ReactionSignalFrame], ear_baseline: Optional[float]) -> list[float]:
        ear_points = [(f.timestamp, f.ear_current) for f in frames if f.ear_current is not None]
        if len(ear_points) < 3 or ear_baseline is None or ear_baseline <= 1e-6:
            return []
        low_threshold = ear_baseline * (1.0 - self.BLINK_DROP_RATIO)
        recover_threshold = ear_baseline * (1.0 - self.BLINK_RECOVERY_RATIO)
        events: list[float] = []
        in_low = False
        min_time: Optional[float] = None
        min_value: Optional[float] = None
        neutral_ready = False
        for timestamp, ear in ear_points:
            if ear is None:
                continue
            if ear >= recover_threshold:
                if in_low and min_time is not None and neutral_ready:
                    events.append(float(min_time))
                    in_low = False
                    min_time = None
                    min_value = None
                neutral_ready = True
                continue
            if ear <= low_threshold and neutral_ready:
                if not in_low:
                    in_low = True
                    min_time = float(timestamp)
                    min_value = float(ear)
                elif min_value is None or ear < min_value:
                    min_time = float(timestamp)
                    min_value = float(ear)
        return events

    def _register_blink_events(self, timestamps: Sequence[float]) -> None:
        for timestamp in timestamps:
            if self._last_blink_event_timestamp is not None and timestamp <= self._last_blink_event_timestamp + 0.04:
                continue
            self._recent_blink_events.append(float(timestamp))
            self._last_blink_event_timestamp = float(timestamp)

    def _register_signal_history(self, frames: Sequence[ReactionSignalFrame]) -> None:
        for frame in frames:
            if not frame.face_detected:
                continue
            if self._last_history_timestamp is not None and frame.timestamp <= self._last_history_timestamp:
                continue
            self._recent_signal_history.append(
                _SignalHistorySample(
                    timestamp=float(frame.timestamp),
                    yaw=_maybe_float(frame.yaw_current),
                    pitch=_maybe_float(frame.pitch_current),
                    roll=_maybe_float(frame.roll_current),
                    face_size_ratio=_maybe_float(frame.face_size_ratio),
                )
            )
            self._last_history_timestamp = float(frame.timestamp)

    def _prune_history(self, reference_timestamp: Optional[float]) -> None:
        if reference_timestamp is None:
            return
        while self._recent_signal_history and self._recent_signal_history[0].timestamp < reference_timestamp - self.HISTORY_WINDOW_SECONDS:
            self._recent_signal_history.popleft()
        while self._recent_blink_events and self._recent_blink_events[0] < reference_timestamp - self.BLINK_EVENT_WINDOW_SECONDS:
            self._recent_blink_events.popleft()

    def _compute_anomaly_summary(self, reference_timestamp: Optional[float]) -> _AnomalySummary:
        blink_score = self._compute_blink_anomaly(reference_timestamp)
        motion_score = self._compute_motion_anomaly(reference_timestamp)
        inconsistency_score = self._compute_signal_inconsistency(reference_timestamp)
        trust_penalty = _clamp01(
            0.45 * (_normalize(blink_score, lower=0.45, upper=1.0) or 0.0)
            + 0.35 * (_normalize(motion_score, lower=0.35, upper=1.0) or 0.0)
            + 0.20 * (_normalize(inconsistency_score, lower=0.30, upper=1.0) or 0.0)
        )
        spoof_support = _clamp01(0.50 * blink_score + 0.30 * motion_score + 0.20 * inconsistency_score)
        return _AnomalySummary(blink_score, motion_score, inconsistency_score, trust_penalty, spoof_support)

    def _compute_blink_anomaly(self, reference_timestamp: Optional[float]) -> float:
        if reference_timestamp is None or len(self._recent_blink_events) < 2:
            return 0.0
        recent = [t for t in self._recent_blink_events if t >= reference_timestamp - self.BLINK_RATE_WINDOW_SECONDS]
        if len(recent) < 2:
            return 0.0
        intervals = np.diff(np.asarray(recent, dtype=np.float32))
        min_interval = float(np.min(intervals)) if len(intervals) else self.BLINK_RATE_WINDOW_SECONDS
        blink_rate = len(recent) / max(self.BLINK_RATE_WINDOW_SECONDS, 1e-6)
        interval_penalty = _inverse_normalize(min_interval, lower=self.BLINK_MIN_INTERVAL_SECONDS, upper=0.26) or 0.0
        rate_penalty = _normalize(blink_rate, lower=1.8, upper=4.2) or 0.0
        return _clamp01(0.70 * interval_penalty + 0.30 * rate_penalty)

    def _compute_motion_anomaly(self, reference_timestamp: Optional[float]) -> float:
        samples = self._recent_motion_samples(reference_timestamp)
        if len(samples) < 4:
            return 0.0
        yaw = [s.yaw for s in samples]
        pitch = [s.pitch for s in samples]
        roll = [s.roll for s in samples]
        face_sizes = [s.face_size_ratio for s in samples]
        return _clamp01(
            0.35 * self._axis_jitter_score(yaw, amplitude_low=1.3, amplitude_high=6.5)
            + 0.20 * self._axis_jitter_score(pitch, amplitude_low=1.0, amplitude_high=5.0)
            + 0.20 * self._axis_jitter_score(roll, amplitude_low=1.2, amplitude_high=6.0)
            + 0.25 * self._face_size_jitter_score(face_sizes)
        )

    def _compute_signal_inconsistency(self, reference_timestamp: Optional[float]) -> float:
        samples = self._recent_motion_samples(reference_timestamp)
        if len(samples) < 4:
            return 0.0
        yaw = [s.yaw for s in samples if s.yaw is not None]
        pitch = [s.pitch for s in samples if s.pitch is not None]
        roll = [s.roll for s in samples if s.roll is not None]
        face_sizes = [s.face_size_ratio for s in samples if s.face_size_ratio is not None]
        if len(face_sizes) < 3:
            return 0.0
        pose_span = (max(yaw) - min(yaw) if yaw else 0.0) + 0.8 * (max(pitch) - min(pitch) if pitch else 0.0) + 0.6 * (max(roll) - min(roll) if roll else 0.0)
        deltas = np.abs(np.diff(np.asarray(face_sizes, dtype=np.float32)))
        mean_size = max(float(np.mean(face_sizes)), 1e-6)
        max_relative_jump = float(np.max(deltas) / mean_size) if len(deltas) else 0.0
        scale_penalty = _normalize(max_relative_jump, lower=0.10, upper=0.32) or 0.0
        low_motion_penalty = _inverse_normalize(pose_span, lower=4.0, upper=16.0) or 0.0
        return _clamp01(scale_penalty * (0.55 + 0.45 * low_motion_penalty))

    def _recent_motion_samples(self, reference_timestamp: Optional[float]) -> list[_SignalHistorySample]:
        if reference_timestamp is None:
            return list(self._recent_signal_history)
        return [s for s in self._recent_signal_history if s.timestamp >= reference_timestamp - self.MOTION_WINDOW_SECONDS]

    def _axis_jitter_score(self, values: Sequence[Optional[float]], *, amplitude_low: float, amplitude_high: float) -> float:
        filtered = np.asarray([float(v) for v in values if v is not None], dtype=np.float32)
        if len(filtered) < 4:
            return 0.0
        deltas = np.diff(filtered)
        if len(deltas) < 3:
            return 0.0
        significant = deltas[np.abs(deltas) >= amplitude_low]
        if len(significant) < 2:
            return 0.0
        sign_flips = np.sum(np.signbit(significant[:-1]) != np.signbit(significant[1:]))
        flip_ratio = float(sign_flips / max(len(significant) - 1, 1))
        amp_score = _normalize(float(np.std(deltas)), lower=amplitude_low, upper=amplitude_high) or 0.0
        flip_score = _normalize(flip_ratio, lower=0.45, upper=0.95) or 0.0
        return _clamp01(0.60 * amp_score + 0.40 * flip_score)

    def _face_size_jitter_score(self, values: Sequence[Optional[float]]) -> float:
        filtered = np.asarray([float(v) for v in values if v is not None], dtype=np.float32)
        if len(filtered) < 4:
            return 0.0
        deltas = np.abs(np.diff(filtered))
        mean_size = max(float(np.mean(filtered)), 1e-6)
        std_score = _normalize(float(np.std(deltas)), lower=0.008, upper=0.045) or 0.0
        jump_score = _normalize(float(np.max(deltas) / mean_size) if len(deltas) else 0.0, lower=0.10, upper=0.32) or 0.0
        return _clamp01(0.45 * std_score + 0.55 * jump_score)

    def compute_blink_evidence(self, frames: Sequence[ReactionSignalFrame], ear_baseline: Optional[float]) -> Optional[float]:
        ear_points = [(f.timestamp, f.ear_current) for f in frames if f.ear_current is not None]
        if len(ear_points) < 3 or ear_baseline is None or ear_baseline <= 1e-6:
            return None
        ear_values = [v for _, v in ear_points]
        ear_min = min(ear_values)
        low_threshold = ear_baseline * (1.0 - self.BLINK_DROP_RATIO)
        recover_threshold = ear_baseline * (1.0 - self.BLINK_RECOVERY_RATIO)
        low_hold = _longest_run_duration(ear_points, lambda value: value <= low_threshold)
        recovery_seen = any(value >= recover_threshold for _, value in ear_points[-max(2, len(ear_points) // 3):])
        neutral_before_drop = any(value >= recover_threshold for _, value in ear_points[: max(2, len(ear_points) // 3)])
        drop_ratio = max(0.0, (ear_baseline - ear_min) / ear_baseline)
        drop_evidence = _normalize(drop_ratio, lower=0.12, upper=0.30) or 0.0
        hold_evidence = _normalize(low_hold, lower=self.BLINK_MIN_DROP_HOLD_SECONDS, upper=0.22) or 0.0
        recovery_evidence = 1.0 if recovery_seen and neutral_before_drop else (0.35 if neutral_before_drop else 0.0)
        return _clamp01(0.50 * drop_evidence + 0.18 * hold_evidence + 0.32 * recovery_evidence)

    def compute_smile_evidence(self, frames: Sequence[ReactionSignalFrame], mar_baseline: Optional[float], smile_baseline: Optional[float]) -> Optional[float]:
        mar_values = [f.mar_current for f in frames if f.mar_current is not None]
        smile_scores = [f.smile_score for f in frames if f.smile_score is not None]
        if not mar_values:
            return None
        mar_peak = max(mar_values)
        mar_floor = min(mar_values)
        rise = mar_peak - mar_floor
        rise_ratio = rise / mar_baseline if mar_baseline is not None and mar_baseline > 1e-6 else None
        mar_evidence = _normalize(rise_ratio, lower=0.18, upper=0.52)
        smile_points = [(f.timestamp, f.smile_score) for f in frames if f.smile_score is not None]
        smile_peak = max(smile_scores) if smile_scores else None
        smile_delta = (smile_peak - smile_baseline) if smile_peak is not None and smile_baseline is not None else None
        smile_evidence = _normalize(smile_delta, lower=8.0, upper=28.0) if smile_delta is not None else (_normalize(smile_peak, lower=55.0, upper=90.0) if smile_peak is not None else None)
        hold_threshold = smile_baseline + self.SMILE_DELTA_THRESHOLD if smile_baseline is not None else None
        smile_hold = _longest_run_duration(smile_points, lambda value: value >= hold_threshold) if hold_threshold is not None else 0.0
        hold_evidence = _normalize(smile_hold, lower=self.SMILE_MIN_HOLD_SECONDS, upper=0.45)
        return _weighted_mean([(mar_evidence, 0.35), (smile_evidence, 0.40), (hold_evidence, 0.25)])

    def compute_head_turn_left_evidence(self, frames: Sequence[ReactionSignalFrame], yaw_baseline: Optional[float]) -> Optional[float]:
        yaw_points = [(f.timestamp, f.yaw_current) for f in frames if f.yaw_current is not None]
        if not yaw_points:
            return None
        baseline = yaw_baseline or 0.0
        deviations = [(ts, baseline - yaw) for ts, yaw in yaw_points if yaw < baseline]
        if not deviations:
            return None
        peak = max(dev for _, dev in deviations)
        hold = _longest_run_duration(yaw_points, lambda value: (baseline - value) >= self.HEAD_TURN_THRESHOLD_DEGREES)
        neutral_band = max(3.5, self.HEAD_TURN_THRESHOLD_DEGREES * self.HEAD_TURN_RETURN_RATIO)
        neutral_before = any((baseline - value) <= neutral_band for _, value in yaw_points[: max(2, len(yaw_points) // 3)])
        returned = any((baseline - value) <= neutral_band for _, value in yaw_points[-max(2, len(yaw_points) // 3):])
        transition = 1.0 if neutral_before and returned else (0.45 if neutral_before else 0.0)
        return _weighted_mean([(_normalize(peak, lower=8.0, upper=22.0), 0.55), (_normalize(hold, lower=self.HEAD_TURN_MIN_HOLD_SECONDS, upper=0.35), 0.20), (transition, 0.25)])

    def compute_head_turn_right_evidence(self, frames: Sequence[ReactionSignalFrame], yaw_baseline: Optional[float]) -> Optional[float]:
        yaw_points = [(f.timestamp, f.yaw_current) for f in frames if f.yaw_current is not None]
        if not yaw_points:
            return None
        baseline = yaw_baseline or 0.0
        deviations = [(ts, yaw - baseline) for ts, yaw in yaw_points if yaw > baseline]
        if not deviations:
            return None
        peak = max(dev for _, dev in deviations)
        hold = _longest_run_duration(yaw_points, lambda value: (value - baseline) >= self.HEAD_TURN_THRESHOLD_DEGREES)
        neutral_band = max(3.5, self.HEAD_TURN_THRESHOLD_DEGREES * self.HEAD_TURN_RETURN_RATIO)
        neutral_before = any((value - baseline) <= neutral_band for _, value in yaw_points[: max(2, len(yaw_points) // 3)])
        returned = any((value - baseline) <= neutral_band for _, value in yaw_points[-max(2, len(yaw_points) // 3):])
        transition = 1.0 if neutral_before and returned else (0.45 if neutral_before else 0.0)
        return _weighted_mean([(_normalize(peak, lower=8.0, upper=22.0), 0.55), (_normalize(hold, lower=self.HEAD_TURN_MIN_HOLD_SECONDS, upper=0.35), 0.20), (transition, 0.25)])

    def compute_mouth_open_evidence(self, frames: Sequence[ReactionSignalFrame], mar_baseline: Optional[float]) -> Optional[float]:
        mar_points = [(f.timestamp, f.mar_current) for f in frames if f.mar_current is not None]
        mar_values = [value for _, value in mar_points]
        if not mar_values:
            return None
        mar_peak = max(mar_values)
        baseline = mar_baseline if mar_baseline is not None and mar_baseline > 1e-6 else float(np.mean(mar_values))
        if baseline <= 1e-6:
            return None
        peak_ratio = mar_peak / baseline
        hold = _longest_run_duration(mar_points, lambda value: value >= baseline * self.MOUTH_OPEN_RATIO_THRESHOLD)
        open_threshold = baseline * self.MOUTH_OPEN_RATIO_THRESHOLD
        return_threshold = baseline * (1.0 + (self.MOUTH_OPEN_RATIO_THRESHOLD - 1.0) * self.MOUTH_OPEN_RETURN_RATIO)
        neutral_before = any(value <= return_threshold for _, value in mar_points[: max(2, len(mar_points) // 3)])
        returned = any(value <= return_threshold for _, value in mar_points[-max(2, len(mar_points) // 3):])
        rise_ratio = (mar_peak - baseline) / baseline
        transition = 1.0 if neutral_before and returned else (0.45 if neutral_before else 0.0)
        if not any(value >= open_threshold for _, value in mar_points):
            transition = 0.0
        return _weighted_mean([(_normalize(peak_ratio, lower=1.35, upper=1.95), 0.35), (_normalize(rise_ratio, lower=0.28, upper=0.90), 0.25), (_normalize(hold, lower=self.MOUTH_OPEN_MIN_HOLD_SECONDS, upper=0.40), 0.20), (transition, 0.20)])


def _first_available(values: Sequence[Optional[float]]) -> Optional[float]:
    for value in reversed(values):
        if value is not None:
            return float(value)
    return None


def _maybe_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _normalize(value: Optional[float], *, lower: float, upper: float) -> Optional[float]:
    if value is None or upper <= lower:
        return None
    return _clamp01((float(value) - lower) / (upper - lower))


def _inverse_normalize(value: Optional[float], *, lower: float, upper: float) -> Optional[float]:
    normalized = _normalize(value, lower=lower, upper=upper)
    return None if normalized is None else _clamp01(1.0 - normalized)


def _weighted_mean(values: Sequence[tuple[Optional[float], float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    return 0.0 if denominator <= 1e-6 else numerator / denominator


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return None if not filtered else float(np.mean(filtered))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_sigmoid_score(
    evidence: float,
    *,
    midpoint: float,
    steepness: float,
    scale: float,
) -> float:
    clipped_evidence = _clamp01(evidence)

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-steepness * (x - midpoint))))

    low = _sigmoid(0.0)
    high = _sigmoid(1.0)
    if high - low <= 1e-6:
        return 0.0
    normalized = (_sigmoid(clipped_evidence) - low) / (high - low)
    return max(0.0, min(float(scale), float(scale) * normalized))


def _longest_run_duration(points: Sequence[tuple[float, Optional[float]]], predicate) -> float:
    longest = 0.0
    run_start: Optional[float] = None
    previous_timestamp: Optional[float] = None
    for timestamp, value in points:
        if value is not None and predicate(value):
            if run_start is None:
                run_start = timestamp
            previous_timestamp = timestamp
            continue
        if run_start is not None and previous_timestamp is not None:
            longest = max(longest, previous_timestamp - run_start)
        run_start = None
        previous_timestamp = None
    if run_start is not None and previous_timestamp is not None:
        longest = max(longest, previous_timestamp - run_start)
    return max(0.0, longest)
