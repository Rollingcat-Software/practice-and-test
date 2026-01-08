"""Facial landmark detection with pose estimation."""

import os
import logging
import time
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from ...domain.models import Landmarks, Pose
from ...domain.filters import OneEuroFilter2D
from .face_detector import get_resource_path

logger = logging.getLogger(__name__)


class LandmarkDetector:
    """MediaPipe Face Mesh landmark detector with pose estimation.

    Detects 468 facial landmarks and estimates head pose (yaw, pitch, roll).
    Supports both Tasks API and Solutions API.

    Features:
    - One Euro Filter smoothing for stable landmark tracking
    - Reduces jitter while maintaining responsiveness
    """

    def __init__(self, max_faces: int = 5, cache_interval: float = 0.05,
                 smooth: bool = True, smooth_beta: float = 0.01):
        """Initialize landmark detector.

        Args:
            max_faces: Maximum faces to detect
            cache_interval: Minimum time between detections (seconds)
            smooth: Enable One Euro Filter smoothing
            smooth_beta: Speed coefficient (higher = more responsive, lower = smoother)
        """
        self._mp = None
        self._mesh = None
        self._use_tasks = False
        self._max_faces = max_faces
        self._cache_interval = cache_interval
        self._cache: List[Landmarks] = []
        self._last_detection = 0

        # One Euro Filter smoothing for each landmark
        self._smooth = smooth
        self._smooth_beta = smooth_beta
        self._filters: Dict[int, List[OneEuroFilter2D]] = {}  # face_idx -> filters per landmark

    def _load(self) -> bool:
        """Lazy load MediaPipe Face Mesh."""
        if self._mesh is not None:
            return True

        # Only attempt load once
        if hasattr(self, '_load_attempted'):
            return False
        self._load_attempted = True

        try:
            import mediapipe as mp
            self._mp = mp
            logger.info(f"MediaPipe version: {mp.__version__}")

            # Try Tasks API first
            if hasattr(mp, 'tasks'):
                try:
                    from mediapipe.tasks import python as tasks
                    from mediapipe.tasks.python import vision
                    import urllib.request

                    model_path = get_resource_path("face_landmarker.task")
                    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

                    # Check if file exists and is valid (at least 1MB)
                    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
                        model_path = "face_landmarker.task"
                        # Delete corrupted file if exists
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        logger.info("Downloading face landmarker model (4.1MB)...")
                        try:
                            urllib.request.urlretrieve(model_url, model_path)
                            logger.info(f"Downloaded {os.path.getsize(model_path)} bytes")
                        except Exception as dl_error:
                            logger.error(f"Download failed: {dl_error}")
                            raise

                    logger.info(f"Loading face landmarker from: {model_path} ({os.path.getsize(model_path)} bytes)")
                    opts = vision.FaceLandmarkerOptions(
                        base_options=tasks.BaseOptions(model_asset_path=model_path),
                        num_faces=self._max_faces
                    )
                    self._mesh = vision.FaceLandmarker.create_from_options(opts)
                    self._use_tasks = True
                    logger.info("MediaPipe Tasks Face Landmarker loaded")
                    return True
                except Exception as e:
                    logger.warning(f"Tasks API failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Fallback to Solutions API
            logger.info("Trying MediaPipe Solutions API fallback...")
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=self._max_faces,
                    refine_landmarks=True, min_detection_confidence=0.5
                )
                self._use_tasks = False
                logger.info("MediaPipe Solutions Face Mesh loaded")
                return True
            else:
                logger.warning("MediaPipe solutions.face_mesh not available")

        except Exception as e:
            logger.warning(f"MediaPipe unavailable: {e}")
            import traceback
            traceback.print_exc()

        logger.error("Failed to load any landmark detector!")
        return False

    def _smooth_landmarks(self, face_idx: int, points: List[Tuple[int, int]], t: float) -> List[Tuple[int, int]]:
        """Apply One Euro Filter smoothing to landmarks.

        Args:
            face_idx: Index of face (for multi-face tracking)
            points: Raw landmark points
            t: Current timestamp

        Returns:
            Smoothed landmark points
        """
        if not self._smooth:
            return points

        # Initialize filters for this face if needed
        if face_idx not in self._filters:
            self._filters[face_idx] = [
                OneEuroFilter2D(min_cutoff=0.5, beta=self._smooth_beta)
                for _ in range(len(points))
            ]
            # First frame: just return raw points (filters need initialization)
            for i, pt in enumerate(points):
                self._filters[face_idx][i](pt[0], pt[1], t)
            return points

        # Ensure we have enough filters
        while len(self._filters[face_idx]) < len(points):
            self._filters[face_idx].append(
                OneEuroFilter2D(min_cutoff=0.5, beta=self._smooth_beta)
            )

        # Apply smoothing
        smoothed = []
        for i, pt in enumerate(points):
            sx, sy = self._filters[face_idx][i](pt[0], pt[1], t)
            smoothed.append((int(sx), int(sy)))

        return smoothed

    def detect(self, frame: np.ndarray) -> List[Landmarks]:
        """Detect facial landmarks in frame.

        Uses caching to avoid redundant detections.

        Args:
            frame: BGR image

        Returns:
            List of Landmarks objects (one per detected face)
        """
        now = time.time()
        if now - self._last_detection < self._cache_interval:
            return self._cache

        if not self._load():
            # Warning already logged in _load(), don't spam
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            raw_landmarks = []
            if self._use_tasks:
                mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                results = self._mesh.detect(mp_img)
                if results.face_landmarks:
                    raw_landmarks = [
                        [(int(lm.x * w), int(lm.y * h)) for lm in face]
                        for face in results.face_landmarks
                    ]
            else:
                results = self._mesh.process(rgb)
                if results.multi_face_landmarks:
                    raw_landmarks = [
                        [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
                        for face in results.multi_face_landmarks
                    ]

            # Apply One Euro Filter smoothing
            if raw_landmarks:
                self._cache = [
                    Landmarks(points=self._smooth_landmarks(i, pts, now))
                    for i, pts in enumerate(raw_landmarks)
                ]
                logger.debug(f"Landmarks: {len(self._cache)} faces, {len(self._cache[0].points) if self._cache else 0} points")
            else:
                self._cache = []

            self._last_detection = now
        except Exception as e:
            logger.error(f"Landmark detection error: {e}")
            import traceback
            traceback.print_exc()

        return self._cache

    def estimate_pose(self, landmarks: Landmarks, frame_size: Tuple[int, int]) -> Pose:
        """Estimate head pose from landmarks.

        Uses key facial landmarks to estimate yaw (left/right) and pitch (up/down).

        Args:
            landmarks: Facial landmarks
            frame_size: (width, height) of the frame

        Returns:
            Pose object with yaw, pitch, roll
        """
        if not landmarks.is_valid():
            return Pose()

        try:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]

            # Yaw (left/right rotation)
            eye_cx = (left_eye[0] + right_eye[0]) / 2
            eye_dist = abs(right_eye[0] - left_eye[0])
            yaw = ((nose[0] - eye_cx) / eye_dist * 60) if eye_dist > 0 else 0

            # Pitch (up/down tilt)
            eye_cy = (left_eye[1] + right_eye[1]) / 2
            mouth_cy = (left_mouth[1] + right_mouth[1]) / 2
            face_h = mouth_cy - eye_cy
            mid_y = (eye_cy + mouth_cy) / 2
            pitch = ((nose[1] - mid_y) / face_h * 60) if face_h > 0 else 0

            # Clamp to reasonable ranges
            yaw = max(-45, min(45, yaw))
            pitch = max(-35, min(35, pitch))

            return Pose(yaw=yaw, pitch=pitch)
        except (IndexError, ValueError):
            return Pose()

    def calculate_ear(self, landmarks: Landmarks, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio (EAR).

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Low EAR = eye closed, High EAR = eye open

        Args:
            landmarks: Facial landmarks
            eye_indices: 6 landmark indices for the eye

        Returns:
            EAR value (typically 0.2-0.4 for open eye, <0.2 for closed)
        """
        try:
            p1 = np.array(landmarks[eye_indices[0]])
            p2 = np.array(landmarks[eye_indices[1]])
            p3 = np.array(landmarks[eye_indices[2]])
            p4 = np.array(landmarks[eye_indices[3]])
            p5 = np.array(landmarks[eye_indices[4]])
            p6 = np.array(landmarks[eye_indices[5]])

            vertical_1 = np.linalg.norm(p2 - p6)
            vertical_2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)

            if horizontal == 0:
                return 0.3  # Default open

            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        except (IndexError, ValueError):
            return 0.3

    def calculate_mar(self, landmarks: Landmarks) -> float:
        """Calculate Mouth Aspect Ratio (MAR).

        MAR = vertical / horizontal
        High MAR = mouth open

        Args:
            landmarks: Facial landmarks

        Returns:
            MAR value
        """
        try:
            left = np.array(landmarks[Landmarks.MOUTH_LEFT])
            right = np.array(landmarks[Landmarks.MOUTH_RIGHT])
            upper = np.array(landmarks[Landmarks.UPPER_LIP])
            lower = np.array(landmarks[Landmarks.LOWER_LIP])

            horizontal = np.linalg.norm(right - left)
            vertical = np.linalg.norm(lower - upper)

            if horizontal == 0:
                return 0.0

            return vertical / horizontal
        except (IndexError, ValueError):
            return 0.0

    def calculate_smile(self, landmarks: Landmarks) -> Tuple[float, float]:
        """Calculate smile metrics.

        Returns: (corner_raise_ratio, width_ratio) relative to face height
        """
        try:
            left_corner = np.array(landmarks[Landmarks.MOUTH_LEFT])
            right_corner = np.array(landmarks[Landmarks.MOUTH_RIGHT])
            upper_lip = np.array(landmarks[Landmarks.UPPER_LIP])
            lower_lip = np.array(landmarks[Landmarks.LOWER_LIP])
            nose_tip = np.array(landmarks[Landmarks.NOSE_TIP])
            chin = np.array(landmarks[Landmarks.CHIN])

            face_height = np.linalg.norm(chin - nose_tip)
            if face_height == 0:
                return 0.0, 0.0

            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
            left_raise = mouth_center_y - left_corner[1]
            right_raise = mouth_center_y - right_corner[1]
            avg_raise = (left_raise + right_raise) / 2
            corner_raise_ratio = avg_raise / face_height

            mouth_width = np.linalg.norm(right_corner - left_corner)
            width_ratio = mouth_width / face_height

            return corner_raise_ratio, width_ratio
        except (IndexError, ValueError):
            return 0.0, 0.0

    def calculate_eyebrow_raise(self, landmarks: Landmarks,
                                 baseline: Optional[dict] = None) -> Tuple[float, float, float, dict]:
        """Calculate eyebrow raise ratio compared to baseline.

        Returns: (both_ratio, left_ratio, right_ratio, new_baseline)
        """
        try:
            left_brow_y = np.mean([landmarks[i][1] for i in Landmarks.LEFT_EYEBROW])
            right_brow_y = np.mean([landmarks[i][1] for i in Landmarks.RIGHT_EYEBROW])
            left_eye_y = np.mean([landmarks[i][1] for i in Landmarks.LEFT_EYE])
            right_eye_y = np.mean([landmarks[i][1] for i in Landmarks.RIGHT_EYE])

            left_dist = left_eye_y - left_brow_y
            right_dist = right_eye_y - right_brow_y
            avg_dist = (left_dist + right_dist) / 2

            if baseline is None:
                baseline = {'left': left_dist, 'right': right_dist, 'avg': avg_dist}
                return 1.0, 1.0, 1.0, baseline

            both_ratio = avg_dist / baseline['avg'] if baseline['avg'] > 0 else 1.0
            left_ratio = left_dist / baseline['left'] if baseline['left'] > 0 else 1.0
            right_ratio = right_dist / baseline['right'] if baseline['right'] > 0 else 1.0

            return both_ratio, left_ratio, right_ratio, baseline
        except (IndexError, ValueError):
            return 1.0, 1.0, 1.0, baseline or {}
