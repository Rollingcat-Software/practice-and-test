"""Fast face detection using MediaPipe Tasks API with fallbacks."""

import os
import sys
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ...domain.models import FaceRegion

logger = logging.getLogger(__name__)


def get_resource_path(filename: str) -> str:
    """Get full path to a resource file (works in frozen exe)."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(base, filename)
    if os.path.exists(path):
        return path
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # Try parent directories
    original_base = base
    for _ in range(5):
        base = os.path.dirname(base)
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path

    # Try biometric-processor sibling directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(original_base)))))
    biometric_path = os.path.join(project_root, "biometric-processor", filename)
    if os.path.exists(biometric_path):
        return biometric_path

    return filename


class FaceDetector:
    """Fast face detection with MediaPipe Tasks API.

    Supports:
    - MediaPipe Tasks API (preferred, handles rotated faces)
    - MediaPipe Solutions API (fallback)
    - OpenCV Haar Cascade (final fallback, frontal only)
    """

    _instance = None
    _detector = None
    _mp = None
    _use_tasks = False
    _use_haar = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_detector()
        return cls._instance

    @classmethod
    def _load_detector(cls):
        """Load the best available face detector."""
        try:
            import mediapipe as mp
            cls._mp = mp

            # Try Tasks API first (best quality)
            if hasattr(mp, 'tasks'):
                try:
                    from mediapipe.tasks import python as tasks
                    from mediapipe.tasks.python import vision
                    import urllib.request

                    model_path = get_resource_path("blaze_face_short_range.tflite")
                    if not os.path.exists(model_path):
                        model_path = "blaze_face_short_range.tflite"
                        logger.info("Downloading BlazeFace model...")
                        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
                        urllib.request.urlretrieve(url, model_path)

                    opts = vision.FaceDetectorOptions(
                        base_options=tasks.BaseOptions(model_asset_path=model_path),
                        min_detection_confidence=0.5
                    )
                    cls._detector = vision.FaceDetector.create_from_options(opts)
                    cls._use_tasks = True
                    cls._use_haar = False
                    logger.info(f"MediaPipe Tasks Face Detector loaded")
                    return
                except Exception as e:
                    logger.debug(f"Tasks API failed: {e}")

            # Fallback to Solutions API
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                cls._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                cls._use_tasks = False
                cls._use_haar = False
                logger.info("MediaPipe Solutions Face Detection loaded")
                return

        except Exception as e:
            logger.warning(f"MediaPipe unavailable: {e}")

        # Final fallback to Haar Cascade
        cls._load_haar_fallback()

    @classmethod
    def _load_haar_fallback(cls):
        """Load Haar Cascade as final fallback."""
        cascade_name = "haarcascade_frontalface_alt2.xml"
        cascade_path = get_resource_path(cascade_name)

        if not os.path.exists(cascade_path):
            try:
                opencv_data = os.path.join(cv2.__path__[0], "data")
                cascade_path = os.path.join(opencv_data, cascade_name)
            except:
                pass

        if os.path.exists(cascade_path):
            cls._detector = cv2.CascadeClassifier(cascade_path)
            if not cls._detector.empty():
                cls._mp = None
                cls._use_tasks = False
                cls._use_haar = True
                logger.info(f"Haar Cascade fallback loaded: {cascade_path}")
                return

        raise RuntimeError("No face detector available")

    def detect(self, frame: np.ndarray) -> List[FaceRegion]:
        """Detect faces in frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            List of FaceRegion objects
        """
        h, w = frame.shape[:2]
        results = []

        if self._mp is not None and self._use_tasks:
            results = self._detect_tasks(frame, h, w)
        elif self._mp is not None and not self._use_tasks:
            results = self._detect_solutions(frame, h, w)
        elif self._use_haar:
            results = self._detect_haar(frame, h, w)

        return results

    def _detect_tasks(self, frame: np.ndarray, h: int, w: int) -> List[FaceRegion]:
        """Detect using MediaPipe Tasks API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        detections = self._detector.detect(mp_img)

        results = []
        if detections.detections:
            for detection in detections.detections:
                bbox = detection.bounding_box
                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                fw = min(bbox.width, w - x)
                fh = min(bbox.height, h - y)

                if fw > 30 and fh > 30:
                    conf = detection.categories[0].score if detection.categories else 0.9
                    results.append(FaceRegion(x=x, y=y, w=fw, h=fh, confidence=conf))

        return results

    def _detect_solutions(self, frame: np.ndarray, h: int, w: int) -> List[FaceRegion]:
        """Detect using MediaPipe Solutions API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._detector.process(rgb)

        results = []
        if detections.detections:
            for detection in detections.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                fw = min(int(bbox.width * w), w - x)
                fh = min(int(bbox.height * h), h - y)

                if fw > 30 and fh > 30:
                    conf = detection.score[0] if detection.score else 0.9
                    results.append(FaceRegion(x=x, y=y, w=fw, h=fh, confidence=conf))

        return results

    def _detect_haar(self, frame: np.ndarray, h: int, w: int) -> List[FaceRegion]:
        """Detect using Haar Cascade (frontal faces only)."""
        max_width = 640
        if w > max_width:
            scale = max_width / w
            small = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for (x, y, fw, fh) in faces:
            results.append(FaceRegion(
                x=int(x / scale), y=int(y / scale),
                w=int(fw / scale), h=int(fh / scale),
                confidence=0.95
            ))

        return results
