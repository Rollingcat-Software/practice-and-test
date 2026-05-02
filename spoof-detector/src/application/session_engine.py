"""Session-based spoof detection engine.

The core architectural component. Accumulates per-frame evidence
into a session state and produces verdicts that improve over time.

Timeline of what becomes available:
  Frame 1 (0s):     MiniFASNet score, device boundary, texture
  Frame 30 (1s):    Temporal consistency baseline established
  Frame 90 (3s):    Blink detection active, rPPG warming up
  Frame 150 (5s):   First reliable session verdict (amispoof.com)
  Frame 900 (30s):  rPPG pulse confirmed, blink rate stable
  Frame 5400 (3m):  High-confidence behavioral baseline
  Frame 54000 (30m): Identity consistency across long session
"""

from __future__ import annotations

import time
import logging
from collections import deque
from typing import Optional

import numpy as np

from src.domain.models import (
    SpoofCategory, SpoofClassification, FrameAnalysis,
    AnalyzerResult, FaceROI,
)
from src.domain.session import (
    SessionState, SessionVerdict, Incident, Severity,
    TemporalSignals,
)

logger = logging.getLogger(__name__)


class SessionEngine:
    """Accumulates frame-level evidence into session-level verdicts.

    Usage:
        engine = SessionEngine()
        engine.start()

        while running:
            analysis = pipeline.process(frame)
            engine.ingest(analysis, frame)
            verdict = engine.get_verdict()
            print(verdict.summary)

        final = engine.conclude()
    """

    # Thresholds
    WARMUP_FRAMES = 30          # 1 second at 30fps
    MIN_VERDICT_FRAMES = 60     # 2 seconds minimum for any verdict
    BLINK_EAR_THRESHOLD = 0.21  # Eye Aspect Ratio below this = eye closed
    BLINK_CONSECUTIVE = 3       # Frames eye must be closed to count as blink
    NO_BLINK_ALERT_SEC = 15.0   # Alert if no blink for this long
    FACE_MISSING_ALERT_SEC = 5.0
    IDENTITY_CHANGE_THRESHOLD = 0.35  # MiniFASNet score swing threshold

    def __init__(self, session_id: str | None = None):
        self._session_id = session_id or f"session_{int(time.time())}"
        self._state = SessionState.WARMING_UP
        self._start_time: float = 0.0
        self._frame_count = 0
        self._face_present_count = 0

        # Per-face temporal tracking
        self._signals: dict[int, TemporalSignals] = {}
        self._primary_face_id: Optional[int] = None

        # Session-level accumulators
        self._incidents: list[Incident] = []
        self._category_evidence: dict[SpoofCategory, deque] = {
            cat: deque(maxlen=1800)  # 60 seconds of evidence
            for cat in SpoofCategory
        }

        # Frame-level score history for rolling average
        self._recent_verdicts: deque[SpoofClassification] = deque(maxlen=300)  # 10s

        # Consecutive counters
        self._face_missing_frames = 0
        self._last_face_time: float = 0.0
        self._consecutive_spoof_frames = 0

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def elapsed_sec(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def start(self):
        """Start the session clock."""
        self._start_time = time.time()
        self._state = SessionState.WARMING_UP
        logger.info(f"Session {self._session_id} started")

    def ingest(self, analysis: FrameAnalysis, frame: Optional[np.ndarray] = None):
        """Ingest a frame analysis into the session.

        This is called every frame (~30fps). It:
        1. Updates temporal signals per face
        2. Checks for incidents (no blink, face missing, identity change)
        3. Accumulates category evidence
        """
        self._frame_count += 1
        elapsed = self.elapsed_sec

        # Transition from warmup
        if self._state == SessionState.WARMING_UP and self._frame_count >= self.WARMUP_FRAMES:
            self._state = SessionState.ANALYZING

        # Track face presence
        if analysis.faces:
            self._face_present_count += 1
            self._face_missing_frames = 0
            self._last_face_time = elapsed
        else:
            self._face_missing_frames += 1
            # Alert on prolonged face absence
            if (self._face_missing_frames > self.FACE_MISSING_ALERT_SEC * 30
                    and elapsed > 5.0):
                self._add_incident(
                    analysis.frame_id, Severity.MEDIUM, SpoofCategory.REAL,
                    f"Face missing for {self._face_missing_frames / 30:.0f}s",
                    {"missing_frames": self._face_missing_frames},
                )

        # Process each face
        for face in analysis.faces:
            fid = face.face_id
            cls = analysis.classifications.get(fid)
            if cls is None:
                continue

            # Ensure temporal signals exist
            if fid not in self._signals:
                self._signals[fid] = TemporalSignals(face_id=fid)

            signals = self._signals[fid]

            # Set primary face (the one we see most)
            if self._primary_face_id is None:
                self._primary_face_id = fid

            # Accumulate per-frame classification
            signals.frame_verdicts.append(cls)

            # Track MiniFASNet specifically (our best discriminator)
            mfn = cls.analyzer_results.get("minifasnet")
            if mfn:
                signals.minifasnet_scores.append(mfn.score)

            # Track position for micro-movement
            signals.position_history.append(
                (face.bbox.center[0], face.bbox.center[1], face.bbox.area)
            )

            # Accumulate category evidence
            for cat, prob in cls.probabilities.items():
                self._category_evidence[cat].append(prob)

            self._recent_verdicts.append(cls)

            # === Incident Detection ===
            self._check_spoof_incident(cls, analysis.frame_id, elapsed)
            self._check_motion_naturalness(signals, analysis.frame_id, elapsed)

    def _check_spoof_incident(self, cls: SpoofClassification, frame_id: int, elapsed: float):
        """Check if current frame shows spoof evidence.

        Uses P(real) as the primary signal — when it drops below threshold,
        that's a spoof event regardless of which specific spoof category leads.
        """
        p_real = cls.probabilities.get(SpoofCategory.REAL, 1.0)

        if p_real < 0.45:
            # Determine severity based on how low P(real) drops
            if p_real < 0.20:
                severity = Severity.HIGH
            elif p_real < 0.35:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            # Find the dominant spoof category
            spoof_cats = {k: v for k, v in cls.probabilities.items() if k != SpoofCategory.REAL}
            dominant_spoof = max(spoof_cats, key=spoof_cats.get) if spoof_cats else SpoofCategory.STATIC_IMAGE

            # Track consecutive spoof frames for burst detection
            self._consecutive_spoof_frames += 1

            # Log incidents: first occurrence, or after 2s gap, or on severity escalation
            should_log = (
                not self._incidents
                or elapsed - self._incidents[-1].timestamp > 2.0
                or (self._consecutive_spoof_frames == 15  # 0.5s burst
                    and (not self._incidents or self._incidents[-1].severity != Severity.HIGH))
            )

            if should_log:
                burst_note = f" (burst: {self._consecutive_spoof_frames} frames)" if self._consecutive_spoof_frames > 10 else ""
                self._add_incident(
                    frame_id, severity, dominant_spoof,
                    f"P(real)={p_real:.0%}, dominant={dominant_spoof.value}{burst_note}",
                    {"p_real": round(p_real, 3),
                     "probabilities": {c.value: round(p, 3) for c, p in cls.probabilities.items()},
                     "consecutive_spoof_frames": self._consecutive_spoof_frames},
                )
        else:
            self._consecutive_spoof_frames = 0

    def _check_motion_naturalness(self, signals: TemporalSignals, frame_id: int, elapsed: float):
        """Check if face motion looks natural or frozen."""
        if len(signals.position_history) < 60:  # Need 2s of data
            return

        positions = list(signals.position_history)[-60:]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        areas = [p[2] for p in positions]

        mean_area = sum(areas) / len(areas) if areas else 1
        pos_std = float(np.sqrt(np.var(xs) + np.var(ys))) / max(np.sqrt(mean_area), 1)

        # Update naturalness
        if pos_std < 0.0001:
            signals.motion_naturalness = 0.1  # Frozen
            if elapsed > 3.0 and (not self._incidents or elapsed - self._incidents[-1].timestamp > 5.0):
                self._add_incident(
                    frame_id, Severity.MEDIUM, SpoofCategory.STATIC_IMAGE,
                    f"Face is unnaturally static (motion_std={pos_std:.6f})",
                    {"pos_std": pos_std},
                )
        elif pos_std < 0.001:
            signals.motion_naturalness = 0.4  # Suspiciously still
        else:
            signals.motion_naturalness = min(1.0, 0.5 + pos_std * 100)

    def _add_incident(self, frame_id: int, severity: Severity,
                      category: SpoofCategory, description: str,
                      evidence: dict):
        self._incidents.append(Incident(
            timestamp=self.elapsed_sec,
            frame_id=frame_id,
            severity=severity,
            category=category,
            confidence=0.0,
            description=description,
            evidence=evidence,
        ))

    def get_verdict(self) -> SessionVerdict:
        """Get current session verdict based on accumulated evidence.

        Uses a peak-sensitive approach: the session verdict is NOT just
        the average. If ANY sustained window shows spoof evidence, the
        session is flagged even if the majority of frames are live.

        This is critical for proctoring: a student who is real 90% of
        the time but shows a phone screen for 10% is still cheating.
        """
        elapsed = self.elapsed_sec
        face_ratio = self._face_present_count / max(self._frame_count, 1)

        # Aggregate category scores from accumulated evidence
        category_scores = {}
        for cat in SpoofCategory:
            evidence = self._category_evidence[cat]
            if evidence:
                category_scores[cat] = sum(evidence) / len(evidence)
            else:
                category_scores[cat] = 1.0 / len(SpoofCategory)

        # Average P(real) across session
        avg_real = category_scores.get(SpoofCategory.REAL, 0.5)

        # Peak spoof detection: what's the WORST window we've seen?
        real_evidence = list(self._category_evidence[SpoofCategory.REAL])
        worst_window_real = avg_real
        if len(real_evidence) >= 3:
            # Sliding window: find the worst 3-frame window
            window_size = min(5, len(real_evidence))
            for i in range(len(real_evidence) - window_size + 1):
                window_avg = sum(real_evidence[i:i + window_size]) / window_size
                worst_window_real = min(worst_window_real, window_avg)

        # Find dominant threat
        spoof_cats = {k: v for k, v in category_scores.items() if k != SpoofCategory.REAL}
        dominant_threat = max(spoof_cats, key=spoof_cats.get) if spoof_cats else None

        # Confidence increases with more data
        data_confidence = min(1.0, self._frame_count / 150.0)  # Full at 5s

        # Temporal signals boost confidence
        temporal_boost = self._compute_temporal_confidence()

        # Incident severity affects verdict
        incident_penalty = self._compute_incident_penalty()

        # Combined real score: blend average with worst-window (peak-sensitive)
        # The worst window drags the score down significantly
        blended_real = 0.50 * avg_real + 0.50 * worst_window_real

        # Apply modifiers
        adjusted_real = blended_real * data_confidence
        adjusted_real = adjusted_real * (1.0 - incident_penalty * 0.4)
        adjusted_real += temporal_boost * 0.15

        is_live = adjusted_real > 0.45
        confidence = min(1.0, data_confidence * (0.5 + abs(adjusted_real - 0.5)))

        # Get blink count and BPM from analyzers and primary face signals
        blink_count = 0
        estimated_bpm = None
        identity_changes = 0

        # Extract blink/rppg data from recent verdicts' analyzer results
        for cls in self._recent_verdicts:
            blink_result = cls.analyzer_results.get("blink")
            if blink_result and "blinks" in blink_result.details:
                blink_count = max(blink_count, blink_result.details["blinks"])
            rppg_result = cls.analyzer_results.get("rppg")
            if rppg_result and rppg_result.details.get("bpm") is not None:
                estimated_bpm = rppg_result.details["bpm"]

        return SessionVerdict(
            is_live=is_live,
            confidence=confidence,
            dominant_threat=dominant_threat if not is_live else None,
            category_scores=category_scores,
            incidents=self._incidents.copy(),
            session_duration_sec=elapsed,
            frames_analyzed=self._frame_count,
            face_detected_ratio=face_ratio,
            blink_count=blink_count,
            estimated_bpm=estimated_bpm,
            identity_changes=identity_changes,
        )

    def _compute_temporal_confidence(self) -> float:
        """Boost from temporal signals (blink, motion naturalness)."""
        if not self._signals:
            return 0.0

        boosts = []
        for sig in self._signals.values():
            if len(sig.frame_verdicts) < 30:
                continue
            # Motion naturalness contributes
            boosts.append(sig.motion_naturalness)

        return sum(boosts) / max(len(boosts), 1) if boosts else 0.0

    def _compute_incident_penalty(self) -> float:
        """Penalty from accumulated incidents.

        A single HIGH incident is enough to flag a session.
        3+ MEDIUM incidents in quick succession is definitive.
        Penalty also considers incident DENSITY (incidents per minute).
        """
        if not self._incidents:
            return 0.0

        severity_weights = {
            Severity.LOW: 0.15,
            Severity.MEDIUM: 0.40,
            Severity.HIGH: 0.80,
            Severity.CRITICAL: 1.00,
        }
        total = sum(severity_weights.get(i.severity, 0) for i in self._incidents)

        # Density bonus: many incidents in short time = stronger signal
        elapsed = self.elapsed_sec
        if elapsed > 5.0:
            incidents_per_min = len(self._incidents) / (elapsed / 60.0)
            if incidents_per_min > 10:  # More than 1 every 6 seconds
                total *= 1.5
            elif incidents_per_min > 5:
                total *= 1.2

        return min(1.0, total / 2.0)  # Caps at 2 medium-equivalent incidents

    def conclude(self) -> SessionVerdict:
        """Conclude the session and return final verdict."""
        self._state = SessionState.CONCLUDED
        verdict = self.get_verdict()
        logger.info(f"Session concluded: {verdict.summary}")
        return verdict

    def get_timeline(self) -> list[dict]:
        """Get incident timeline for reporting."""
        return [
            {
                "time_sec": round(i.timestamp, 1),
                "frame": i.frame_id,
                "severity": i.severity.value,
                "category": i.category.value,
                "description": i.description,
            }
            for i in self._incidents
        ]
