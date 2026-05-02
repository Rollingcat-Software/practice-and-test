"""Session domain models for time-based spoof detection.

A session runs from 5 seconds (amispoof.com quick test) to 3 hours
(exam proctoring). It accumulates evidence from every frame and
produces a verdict that improves over time.

Temporal signal hierarchy:
  - Per-frame (0-33ms): MiniFASNet, device boundary, texture
  - Short-term (1-5s): blink detection, rPPG pulse, micro-motion
  - Medium-term (5-30s): blink rate, movement naturalness, gaze variation
  - Long-term (30s-3hr): identity consistency, behavior patterns, attention

The session verdict is NOT a single frame's classification. It is the
time-weighted aggregate of ALL evidence collected during the session.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import SpoofCategory, SpoofClassification, FrameAnalysis


class SessionState(Enum):
    """Session lifecycle states."""
    WARMING_UP = "warming_up"     # First 1-2 seconds, accumulating baseline
    ANALYZING = "analyzing"       # Active analysis, verdict refining
    CONCLUDED = "concluded"       # Session ended, final verdict available


class Severity(Enum):
    """Spoof detection severity for incidents."""
    LOW = "low"           # Suspicious but not conclusive
    MEDIUM = "medium"     # Likely spoof, needs attention
    HIGH = "high"         # Strong spoof evidence
    CRITICAL = "critical" # Definitive spoof detected


@dataclass
class Incident:
    """A detected spoofing incident within a session."""
    timestamp: float          # Seconds since session start
    frame_id: int
    severity: Severity
    category: SpoofCategory
    confidence: float         # 0-1
    description: str
    evidence: dict = field(default_factory=dict)


@dataclass
class SessionVerdict:
    """Final or interim session verdict."""
    is_live: bool
    confidence: float                         # 0-1, how confident are we
    dominant_threat: Optional[SpoofCategory]  # Most likely spoof type if not live
    category_scores: dict[SpoofCategory, float]  # Accumulated evidence per category
    incidents: list[Incident]                 # Timeline of detected incidents
    session_duration_sec: float
    frames_analyzed: int
    face_detected_ratio: float               # % of frames where face was present
    blink_count: int
    estimated_bpm: Optional[float]           # From rPPG, None if not enough data
    identity_changes: int                    # Number of suspected identity switches

    @property
    def summary(self) -> str:
        verdict = "LIVE" if self.is_live else "SPOOF"
        threat = f" ({self.dominant_threat.value})" if self.dominant_threat else ""
        return (
            f"{verdict}{threat} | conf={self.confidence:.0%} | "
            f"{self.session_duration_sec:.1f}s | {self.frames_analyzed} frames | "
            f"blinks={self.blink_count} | incidents={len(self.incidents)}"
        )


@dataclass
class TemporalSignals:
    """Accumulated temporal signals for a tracked face."""
    face_id: int

    # Blink detection
    blink_count: int = 0
    last_blink_time: float = 0.0
    ear_history: deque = field(default_factory=lambda: deque(maxlen=90))  # 3s at 30fps

    # rPPG pulse
    green_channel_history: deque = field(default_factory=lambda: deque(maxlen=300))  # 10s
    estimated_bpm: Optional[float] = None
    pulse_confidence: float = 0.0

    # Micro-movement
    position_history: deque = field(default_factory=lambda: deque(maxlen=150))  # 5s
    motion_naturalness: float = 0.5  # 0=robotic/frozen, 1=natural

    # Head pose variation
    pose_history: deque = field(default_factory=lambda: deque(maxlen=300))  # 10s
    pose_variance: float = 0.0

    # Per-frame classification accumulator
    frame_verdicts: deque = field(default_factory=lambda: deque(maxlen=900))  # 30s of verdicts
    minifasnet_scores: deque = field(default_factory=lambda: deque(maxlen=300))

    @property
    def blink_rate_per_min(self) -> float:
        """Estimated blink rate. Normal: 15-20/min."""
        if not self.frame_verdicts:
            return 0.0
        duration_sec = len(self.frame_verdicts) / 30.0  # Assume 30fps
        if duration_sec < 2.0:
            return 0.0
        return self.blink_count / (duration_sec / 60.0)

    @property
    def avg_minifasnet(self) -> float:
        if not self.minifasnet_scores:
            return 50.0
        return sum(self.minifasnet_scores) / len(self.minifasnet_scores)
