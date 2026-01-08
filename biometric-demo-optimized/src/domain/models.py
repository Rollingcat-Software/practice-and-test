"""Core domain models for biometric processing."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class FaceRegion:
    """Bounding box for a detected face."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.9

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> Dict[str, Any]:
        return {'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h}


@dataclass
class Pose:
    """Head pose estimation (Euler angles)."""
    yaw: float = 0.0    # Left/right rotation
    pitch: float = 0.0  # Up/down tilt
    roll: float = 0.0   # Head tilt

    def is_within_tolerance(self, target_yaw: float, target_pitch: float, tolerance: float) -> bool:
        return (abs(self.yaw - target_yaw) < tolerance and
                abs(self.pitch - target_pitch) < tolerance)


@dataclass
class Landmarks:
    """Facial landmarks (468 points from MediaPipe)."""
    points: List[Tuple[int, int]] = field(default_factory=list)

    # MediaPipe Face Mesh indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    NOSE_TIP = 1
    CHIN = 152

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.points[idx]

    def is_valid(self) -> bool:
        return len(self.points) >= 468

    def get_point(self, idx: int) -> Optional[Tuple[int, int]]:
        if 0 <= idx < len(self.points):
            return self.points[idx]
        return None


@dataclass
class Quality:
    """Face image quality assessment."""
    score: float = 0.0
    blur_score: float = 0.0
    size_score: float = 0.0
    brightness_ok: bool = True
    issues: List[str] = field(default_factory=list)

    def is_acceptable(self, min_score: float = 65.0) -> bool:
        return self.score >= min_score


@dataclass
class LivenessResult:
    """Passive liveness detection result."""
    is_live: bool = False
    score: float = 0.0
    texture_score: float = 0.0
    color_score: float = 0.0
    skin_tone_score: float = 0.0
    moire_score: float = 0.0
    local_var_score: float = 0.0


@dataclass
class Demographics:
    """Demographic analysis result."""
    age: int = 0
    gender: str = '?'  # 'M', 'F', '?'
    emotion: str = '?'
    timestamp: float = 0.0


@dataclass
class Face:
    """Complete face detection result."""
    id: int
    region: FaceRegion
    landmarks: Optional[Landmarks] = None
    pose: Optional[Pose] = None
    quality: Optional[Quality] = None
    liveness: Optional[LivenessResult] = None
    demographics: Optional[Demographics] = None
    embedding: Optional[np.ndarray] = None
    match_name: Optional[str] = None
    match_confidence: float = 0.0


@dataclass
class EnrolledFace:
    """Enrolled face in database."""
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    thumbnail: Optional[np.ndarray] = None
    enrolled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_embedding(self, emb: np.ndarray, max_embeddings: int = 5):
        if len(self.embeddings) >= max_embeddings:
            self.embeddings.pop(0)
        self.embeddings.append(emb)
