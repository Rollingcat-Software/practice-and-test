"""MediaPipe-based facial landmark detector implementation."""

import logging
from typing import List, Optional

import numpy as np

from app.domain.entities.face_landmarks import HeadPose, Landmark, LandmarkResult
from app.domain.exceptions.feature_errors import LandmarkError

logger = logging.getLogger(__name__)


class MediaPipeLandmarkDetector:
    """Facial landmark detector using MediaPipe Face Mesh.

    Detects 468 facial landmarks with optional 3D coordinates.
    """

    # Facial region indices for MediaPipe Face Mesh
    REGIONS = {
        "left_eye": [33, 133, 160, 159, 158, 144, 145, 153],
        "right_eye": [362, 263, 387, 386, 385, 373, 374, 380],
        "nose": [1, 2, 3, 4, 5, 6, 168, 197, 195],
        "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],
        "left_eyebrow": [70, 63, 105, 66, 107, 55, 65],
        "right_eyebrow": [300, 293, 334, 296, 336, 285, 295],
        "face_oval": [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        ],
    }

    def __init__(self) -> None:
        """Initialize MediaPipe landmark detector."""
        self._face_mesh = None
        logger.info("MediaPipeLandmarkDetector initialized")

    def _get_face_mesh(self):
        """Lazy load MediaPipe Face Mesh."""
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._face_mesh

    def detect(
        self, image: np.ndarray, include_3d: bool = False
    ) -> LandmarkResult:
        """Detect facial landmarks in image.

        Args:
            image: Face image as numpy array (RGB format)
            include_3d: Whether to include 3D coordinates

        Returns:
            LandmarkResult with detected landmarks

        Raises:
            LandmarkError: If detection fails
        """
        logger.debug(f"Starting landmark detection (include_3d={include_3d})")

        try:
            face_mesh = self._get_face_mesh()
            results = face_mesh.process(image)

            if not results.multi_face_landmarks:
                raise LandmarkError("No face landmarks detected")

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            height, width = image.shape[:2]

            # Extract landmarks
            landmarks = []
            for idx, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = lm.z if include_3d else None
                vis = float(lm.visibility) if hasattr(lm, "visibility") else None

                landmarks.append(Landmark(id=idx, x=x, y=y, z=z, visibility=vis))

            # Estimate head pose
            head_pose = self._estimate_head_pose(landmarks, width, height)

            result = LandmarkResult(
                model="mediapipe_468",
                landmark_count=len(landmarks),
                landmarks=landmarks,
                regions=self.REGIONS,
                head_pose=head_pose,
            )

            logger.info(f"Landmark detection complete: {len(landmarks)} landmarks")
            return result

        except LandmarkError:
            raise
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            raise LandmarkError(f"Landmark detection failed: {str(e)}")

    def _estimate_head_pose(
        self, landmarks: List[Landmark], width: int, height: int
    ) -> Optional[HeadPose]:
        """Estimate head pose from landmarks."""
        try:
            # Use nose tip and key facial points for estimation
            nose = landmarks[1]  # Nose tip
            left_eye = landmarks[33]  # Left eye corner
            right_eye = landmarks[263]  # Right eye corner
            chin = landmarks[152]  # Chin

            # Simple angle estimation
            # Yaw (left/right)
            eye_center_x = (left_eye.x + right_eye.x) / 2
            yaw = (nose.x - eye_center_x) / (width / 2) * 45

            # Pitch (up/down)
            face_height = chin.y - (left_eye.y + right_eye.y) / 2
            expected_nose_y = (left_eye.y + right_eye.y) / 2 + face_height * 0.4
            pitch = (nose.y - expected_nose_y) / face_height * 30

            # Roll (tilt)
            dy = right_eye.y - left_eye.y
            dx = right_eye.x - left_eye.x
            roll = np.degrees(np.arctan2(dy, dx))

            return HeadPose(
                pitch=round(pitch, 1),
                yaw=round(yaw, 1),
                roll=round(roll, 1),
            )

        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
            return None

    def get_landmark_count(self) -> int:
        """Get number of landmarks this detector provides."""
        return 468

    def get_model_name(self) -> str:
        """Get name of the landmark model."""
        return "mediapipe_468"
