"""Protocol interfaces for spoof detection components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .models import FaceROI, AnalyzerResult, BBox


@runtime_checkable
class IFaceDetector(Protocol):
    """Face detection interface."""

    def detect(self, frame: np.ndarray) -> list[FaceROI]:
        """Detect faces in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detected face regions
        """
        ...


@runtime_checkable
class IFaceAnalyzer(Protocol):
    """Per-face spoof analysis interface.

    Analyzers receive a face crop and return a score (0-100)
    where higher values indicate more live-like appearance.
    """

    @property
    def name(self) -> str:
        """Analyzer name for logging and fusion."""
        ...

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        """Analyze a single face crop for spoofing signals.

        Args:
            face_crop: BGR face crop
            face_roi: Face region info (for temporal context via face_id)

        Returns:
            Analysis result with score and details
        """
        ...


@runtime_checkable
class IFrameAnalyzer(Protocol):
    """Whole-frame analysis interface.

    Analyzes the entire frame without face cropping,
    detecting environment-level spoof signals.
    """

    @property
    def name(self) -> str:
        """Analyzer name."""
        ...

    def analyze(self, frame: np.ndarray) -> AnalyzerResult:
        """Analyze the full frame for spoof signals.

        Args:
            frame: BGR full frame

        Returns:
            Analysis result
        """
        ...
