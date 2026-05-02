"""Spoof detection pipeline orchestrator.

Coordinates: detect → track → crop → analyze → fuse.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np

from src.domain.models import FaceROI, FrameAnalysis, SpoofClassification, AnalyzerResult
from src.domain.interfaces import IFaceDetector, IFaceAnalyzer, IFrameAnalyzer
from src.application.face_tracker import FaceTracker
from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser

logger = logging.getLogger(__name__)


class SpoofDetectionPipeline:
    """Main pipeline: detect → track → crop → analyze → fuse.

    Supports both per-face analyzers (operate on face crop) and
    whole-frame analyzers (operate on full frame).
    """

    def __init__(
        self,
        detector: IFaceDetector,
        tracker: FaceTracker,
        face_analyzers: list[IFaceAnalyzer],
        frame_analyzers: list[IFrameAnalyzer],
        fuser: MultiClassFuser,
    ):
        self._detector = detector
        self._tracker = tracker
        self._face_analyzers = face_analyzers
        self._frame_analyzers = frame_analyzers
        self._fuser = fuser
        self._frame_counter = 0

    def process(self, frame: np.ndarray) -> FrameAnalysis:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image

        Returns:
            FrameAnalysis with all face classifications
        """
        total_start = time.perf_counter()
        self._frame_counter += 1

        # Stage 1: Detect faces
        raw_faces = self._detector.detect(frame)

        # Stage 1b: Track (assign persistent IDs)
        tracked_faces = self._tracker.update(raw_faces)

        # Ensure crops exist
        for face in tracked_faces:
            if face.crop is None:
                bbox = face.bbox
                face.crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()

        # Stage 2: Per-face analysis
        classifications: dict[int, SpoofClassification] = {}
        for face in tracked_faces:
            if face.crop is None or face.crop.size == 0:
                continue
            analyzer_results = self._analyze_face(face.crop, face, frame)
            classification = self._fuser.fuse(face.face_id, analyzer_results)
            classifications[face.face_id] = classification

        # Stage 3: Whole-frame analysis
        frame_signals: dict[str, float] = {}
        for analyzer in self._frame_analyzers:
            result = analyzer.analyze(frame)
            frame_signals[f"{analyzer.name}_score"] = result.score
            frame_signals[f"{analyzer.name}_ms"] = result.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000

        return FrameAnalysis(
            frame_id=self._frame_counter,
            faces=tracked_faces,
            classifications=classifications,
            frame_signals=frame_signals,
            total_ms=total_ms,
        )

    def _analyze_face(
        self, crop: np.ndarray, face: FaceROI, frame: np.ndarray,
    ) -> dict[str, AnalyzerResult]:
        results: dict[str, AnalyzerResult] = {}
        for analyzer in self._face_analyzers:
            try:
                # Pass original frame to analyzers that need context
                if hasattr(analyzer, "set_frame"):
                    analyzer.set_frame(frame)
                # Share landmarks from blink analyzer to landmark_variance analyzer
                if hasattr(analyzer, "set_landmarks"):
                    # Find the blink analyzer instance and get its landmarks
                    for a in self._face_analyzers:
                        if hasattr(a, "_last_landmarks") and a._last_landmarks is not None:
                            analyzer.set_landmarks(a._last_landmarks)
                            break
                result = analyzer.analyze(crop, face)
                results[analyzer.name] = result
            except Exception as e:
                logger.debug(f"Analyzer {analyzer.name} failed: {e}")
                results[analyzer.name] = AnalyzerResult(
                    name=analyzer.name, score=50.0,
                    details={"error": str(e)},
                )
        return results
