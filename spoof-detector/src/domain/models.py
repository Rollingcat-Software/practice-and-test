"""Domain models for spoof detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SpoofCategory(Enum):
    """Multi-class spoof taxonomy.

    Covers all known face presentation attack types:
    - Static: printed photo or digital still on screen
    - Replay: pre-recorded video shown on a display
    - Mask: realistic 3D silicone/latex mask
    - Makeup: heavy contouring or prosthetics
    - AR Filter: live AR overlay (Snapchat, Instagram, OBS)
    - Deepfake: virtual webcam injection (DeepFaceLive etc.)
    - Real: genuine live person
    """
    REAL = "real"
    STATIC_IMAGE = "static_image"
    VIDEO_REPLAY = "video_replay"
    MASK_3D = "mask_3d"
    HEAVY_MAKEUP = "heavy_makeup"
    AR_FILTER = "ar_filter"
    DEEPFAKE_INJECT = "deepfake_inject"


# Display labels for overlay
CATEGORY_LABELS: dict[SpoofCategory, str] = {
    SpoofCategory.REAL: "Real",
    SpoofCategory.STATIC_IMAGE: "Static Image",
    SpoofCategory.VIDEO_REPLAY: "Video Replay",
    SpoofCategory.MASK_3D: "Mask",
    SpoofCategory.HEAVY_MAKEUP: "Makeup",
    SpoofCategory.AR_FILTER: "AR Filter",
    SpoofCategory.DEEPFAKE_INJECT: "Deepfake",
}

# BGR colors for each category
CATEGORY_COLORS: dict[SpoofCategory, tuple[int, int, int]] = {
    SpoofCategory.REAL: (0, 255, 0),           # Green
    SpoofCategory.STATIC_IMAGE: (0, 0, 255),    # Red
    SpoofCategory.VIDEO_REPLAY: (0, 0, 200),    # Dark red
    SpoofCategory.MASK_3D: (0, 100, 255),        # Orange
    SpoofCategory.HEAVY_MAKEUP: (180, 0, 255),   # Pink
    SpoofCategory.AR_FILTER: (255, 0, 180),      # Magenta
    SpoofCategory.DEEPFAKE_INJECT: (0, 0, 180),  # Dark red
}


@dataclass
class BBox:
    """Face bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: BBox) -> float:
        """Intersection over Union with another bbox."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / max(union, 1)


@dataclass
class FaceROI:
    """Detected face region of interest."""
    face_id: int
    bbox: BBox
    confidence: float
    landmarks: Optional[np.ndarray] = None  # Nx2 array of landmark points
    crop: Optional[np.ndarray] = None       # BGR face crop


@dataclass
class AnalyzerResult:
    """Result from a single analyzer."""
    name: str
    score: float            # 0-100, higher = more live-like
    details: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class SpoofClassification:
    """Final multi-class spoof classification for a face."""
    face_id: int
    probabilities: dict[SpoofCategory, float]  # Should sum to ~1.0
    dominant_category: SpoofCategory
    confidence: float       # Confidence of dominant category (0-1)
    analyzer_results: dict[str, AnalyzerResult] = field(default_factory=dict)

    @staticmethod
    def from_probabilities(
        face_id: int,
        probs: dict[SpoofCategory, float],
        analyzer_results: dict[str, AnalyzerResult] | None = None,
    ) -> SpoofClassification:
        """Create classification from probability distribution."""
        total = sum(probs.values())
        if total > 0:
            normalized = {k: v / total for k, v in probs.items()}
        else:
            normalized = {cat: 1.0 / len(SpoofCategory) for cat in SpoofCategory}

        dominant = max(normalized, key=normalized.get)  # type: ignore[arg-type]
        return SpoofClassification(
            face_id=face_id,
            probabilities=normalized,
            dominant_category=dominant,
            confidence=normalized[dominant],
            analyzer_results=analyzer_results or {},
        )


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame."""
    frame_id: int
    faces: list[FaceROI]
    classifications: dict[int, SpoofClassification]  # face_id -> classification
    frame_signals: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
