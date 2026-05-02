"""MediaPipe-based face detector.

Uses MediaPipe Tasks API (0.10.9+) for fast, lightweight face detection.
"""

import logging

import cv2
import numpy as np
import mediapipe as mp

from src.domain.models import FaceROI, BBox

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediaPipeFaceDetector:
    """Face detector using MediaPipe Tasks API.

    Fast (~2ms) and lightweight. Suitable for real-time desktop use.
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize detector.

        Args:
            min_confidence: Minimum detection confidence (0-1)
        """
        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=self._find_model_path(),
            ),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = FaceDetector.create_from_options(options)
        self._face_id_counter = 0
        logger.info(f"MediaPipe face detector initialized: confidence={min_confidence}")

    @staticmethod
    def _find_model_path() -> str:
        """Find or download the BlazeFace model."""
        import os
        from pathlib import Path

        # Check local models/ directory first
        local_model = Path(__file__).parent.parent.parent.parent / "models" / "blaze_face_short_range.tflite"
        if local_model.exists():
            return str(local_model)

        # Check biometric-demo-optimized for shared model
        demo_model = (
            Path(__file__).parent.parent.parent.parent.parent
            / "biometric-demo-optimized"
            / "blaze_face_short_range.tflite"
        )
        if demo_model.exists():
            return str(demo_model)

        # Download from MediaPipe
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        local_model.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading BlazeFace model to {local_model}...")
        urllib.request.urlretrieve(url, str(local_model))
        logger.info("Download complete")
        return str(local_model)

    @property
    def name(self) -> str:
        return "mediapipe"

    def detect(self, frame: np.ndarray) -> list[FaceROI]:
        """Detect faces in a BGR frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)

        faces = []
        for detection in result.detections:
            bbox = detection.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)

            if x2 - x1 < 30 or y2 - y1 < 30:
                continue

            # Extract keypoints as landmarks
            landmarks = None
            if detection.keypoints:
                landmarks = np.array(
                    [[int(kp.x * w), int(kp.y * h)] for kp in detection.keypoints],
                    dtype=np.int32,
                )

            confidence = detection.categories[0].score if detection.categories else 0.0
            self._face_id_counter += 1

            crop = frame[y1:y2, x1:x2].copy()

            faces.append(
                FaceROI(
                    face_id=self._face_id_counter,
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=confidence,
                    landmarks=landmarks,
                    crop=crop,
                )
            )

        return faces

    def close(self):
        self._detector.close()
