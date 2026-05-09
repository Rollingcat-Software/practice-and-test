"""DeepFace-based face detector implementation."""

import logging
from typing import Optional, Tuple

import numpy as np
from deepface import DeepFace

from app.domain.entities.face_detection import FaceDetectionResult
from app.domain.exceptions.face_errors import FaceNotDetectedError

logger = logging.getLogger(__name__)


class DeepFaceDetector:
    """Face detector using DeepFace library.

    Implements IFaceDetector interface using DeepFace's face detection.
    Supports multiple detection backends and built-in anti-spoofing (DeepFace 0.0.98+).

    Following Open/Closed Principle: Can be replaced with different detector
    without changing client code.
    """

    def __init__(
        self,
        detector_backend: str = "opencv",
        align: bool = True,
        anti_spoofing: bool = False,
        anti_spoofing_threshold: float = 0.5,
    ) -> None:
        """Initialize DeepFace detector.

        Args:
            detector_backend: Detection backend to use
                Options: "opencv", "ssd", "mtcnn", "retinaface", "mediapipe",
                         "yolov8", "yolov11n", "yolov11s", "yolov12n", "centerface"
            align: Whether to align detected faces
            anti_spoofing: Enable DeepFace built-in anti-spoofing
            anti_spoofing_threshold: Minimum score to accept face as real (0.0-1.0)
        """
        self._detector_backend = detector_backend
        self._align = align
        self._anti_spoofing = anti_spoofing
        self._anti_spoofing_threshold = anti_spoofing_threshold

        logger.info(
            f"Initialized DeepFaceDetector with backend: {detector_backend}, "
            f"align: {align}, anti_spoofing: {anti_spoofing}"
        )

    def detect_sync(self, image: np.ndarray) -> FaceDetectionResult:
        """Synchronous face detection for thread pool execution.

        This method contains the actual blocking DeepFace call.
        Called by AsyncFaceDetector via thread pool for non-blocking execution.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            FaceDetectionResult with detection information

        Raises:
            FaceNotDetectedError: When no face is detected
            MultipleFacesError: When multiple faces are detected
        """
        try:
            # Extract faces using DeepFace (blocking operation)
            face_objs = self._extract_faces(image, anti_spoofing=self._anti_spoofing)

            if not face_objs or len(face_objs) == 0:
                logger.warning("No face detected in image")
                raise FaceNotDetectedError()

            if len(face_objs) > 1:
                # Pick the largest face (most likely the intended subject)
                # Client-side MediaPipe already confirmed a single face — extra detections are false positives
                face_objs = self._filter_implausible_multi_face_candidates(face_objs, image.shape)
                logger.info(f"Multiple faces detected ({len(face_objs)}), selecting largest face")
                face_objs.sort(
                    key=lambda f: f.get('facial_area', {}).get('w', 0) * f.get('facial_area', {}).get('h', 0),
                    reverse=True
                )

            candidate_bounding_boxes = self._extract_bounding_boxes(face_objs)

            # Get the largest/primary detected face
            face_obj = face_objs[0]

            antispoof_score = None
            antispoof_label = None

            # Anti-spoofing check (DeepFace 0.0.98+)
            if self._anti_spoofing:
                is_real = face_obj.get("is_real", True)
                antispoof_score = float(face_obj.get("antispoof_score", 1.0))
                antispoof_label = "real" if is_real else "spoof"

                logger.info(
                    f"Anti-spoofing result: is_real={is_real}, "
                    f"antispoof_score={antispoof_score:.3f}, "
                    f"threshold={self._anti_spoofing_threshold}"
                )

            # Extract facial area (bounding box)
            facial_area = face_obj.get("facial_area", {})
            x = facial_area.get("x", 0)
            y = facial_area.get("y", 0)
            w = facial_area.get("w", 0)
            h = facial_area.get("h", 0)

            bounding_box: Optional[Tuple[int, int, int, int]] = (x, y, w, h)
            additional_bounding_boxes = tuple(
                candidate_bbox
                for candidate_bbox in candidate_bounding_boxes
                if candidate_bbox != bounding_box
                and not self._is_nested_false_positive(candidate_bbox, bounding_box)
            )

            # Extract confidence
            confidence = float(face_obj.get("confidence", 0.99))

            # Landmarks (not provided by all backends)
            landmarks = None  # DeepFace doesn't expose landmarks in extract_faces

            logger.info(
                f"Face detected successfully: bbox=({x},{y},{w},{h}), confidence={confidence:.2f}"
            )

            return FaceDetectionResult(
                found=True,
                bounding_box=bounding_box,
                landmarks=landmarks,
                confidence=confidence,
                antispoof_score=antispoof_score,
                antispoof_label=antispoof_label,
                additional_bounding_boxes=additional_bounding_boxes,
            )

        except ValueError as e:
            err_str = str(e).lower()
            if "face could not be detected" in err_str or "no face" in err_str:
                logger.warning(f"No face detected: {e}")
                raise FaceNotDetectedError()
            elif "spoof" in err_str or "real face" in err_str:
                logger.warning(
                    "DeepFace anti-spoofing raised spoof exception; "
                    "falling back to plain face extraction and tagging detection as spoof: %s",
                    e,
                )
                return self._detect_with_spoof_fallback(image)
            else:
                logger.error(f"Face detection error: {e}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Unexpected error during face detection: {e}", exc_info=True)
            raise

    async def detect(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect face in image (async wrapper).

        This method delegates to detect_sync for backward compatibility.
        For truly non-blocking execution, use AsyncFaceDetector wrapper.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            FaceDetectionResult with detection information

        Raises:
            FaceNotDetectedError: When no face is detected
            MultipleFacesError: When multiple faces are detected
        """
        return self.detect_sync(image)

    def get_detector_name(self) -> str:
        """Get the name of the detector backend.

        Returns:
            Detector backend name
        """
        return self._detector_backend

    def _extract_faces(self, image: np.ndarray, anti_spoofing: bool) -> list[dict]:
        """Call DeepFace.extract_faces with the current detector settings."""
        return DeepFace.extract_faces(
            img_path=image,
            detector_backend=self._detector_backend,
            enforce_detection=True,
            align=self._align,
            anti_spoofing=anti_spoofing,
        )

    def _detect_with_spoof_fallback(self, image: np.ndarray) -> FaceDetectionResult:
        """Recover face metadata when DeepFace anti-spoofing throws before returning a face.

        Some DeepFace versions surface spoof decisions as exceptions instead of a
        normal face object. In that case, run a second pass without anti-spoofing
        so the use case can apply the veto policy without losing the face crop.
        """
        face_objs = self._extract_faces(image, anti_spoofing=False)

        if not face_objs:
            logger.warning("Spoof fallback could not recover a face")
            raise FaceNotDetectedError()

        if len(face_objs) > 1:
            logger.info(
                "Spoof fallback detected multiple faces (%d), selecting largest face",
                len(face_objs),
            )
            face_objs.sort(
                key=lambda f: f.get("facial_area", {}).get("w", 0)
                * f.get("facial_area", {}).get("h", 0),
                reverse=True,
            )

        face_obj = face_objs[0]
        facial_area = face_obj.get("facial_area", {})
        x = facial_area.get("x", 0)
        y = facial_area.get("y", 0)
        w = facial_area.get("w", 0)
        h = facial_area.get("h", 0)

        logger.info(
            "Recovered face metadata after spoof exception: bbox=(%s,%s,%s,%s)",
            x,
            y,
            w,
            h,
        )

        return FaceDetectionResult(
            found=True,
            bounding_box=(x, y, w, h),
            landmarks=None,
            confidence=float(face_obj.get("confidence", 0.99)),
            antispoof_score=1.0,
            antispoof_label="spoof",
        )

    @staticmethod
    def _extract_bounding_boxes(face_objs: list[dict]) -> list[Tuple[int, int, int, int]]:
        bounding_boxes: list[Tuple[int, int, int, int]] = []
        for face_obj in face_objs:
            facial_area = face_obj.get("facial_area", {})
            x = int(facial_area.get("x", 0))
            y = int(facial_area.get("y", 0))
            w = int(facial_area.get("w", 0))
            h = int(facial_area.get("h", 0))
            if w > 0 and h > 0:
                bounding_boxes.append((x, y, w, h))
        return bounding_boxes

    @staticmethod
    def _filter_implausible_multi_face_candidates(
        face_objs: list[dict],
        image_shape: tuple[int, ...],
    ) -> list[dict]:
        frame_height, frame_width = image_shape[:2]
        frame_area = max(frame_width * frame_height, 1)
        plausible: list[dict] = []

        for face_obj in face_objs:
            facial_area = face_obj.get("facial_area", {})
            x = int(facial_area.get("x", 0))
            y = int(facial_area.get("y", 0))
            w = int(facial_area.get("w", 0))
            h = int(facial_area.get("h", 0))
            if w <= 0 or h <= 0:
                continue

            area_ratio = (w * h) / frame_area
            touches_left = x <= max(4, int(frame_width * 0.01))
            touches_top = y <= max(4, int(frame_height * 0.01))
            touches_right = (x + w) >= min(frame_width - 4, int(frame_width * 0.99))
            touches_bottom = (y + h) >= min(frame_height - 4, int(frame_height * 0.99))
            edge_touch_count = sum((touches_left, touches_top, touches_right, touches_bottom))

            if area_ratio >= 0.30 and edge_touch_count >= 2:
                continue
            plausible.append(face_obj)

        if plausible and len(plausible) != len(face_objs):
            logger.info(
                "Filtered %d implausible multi-face candidate(s) before primary-face selection",
                len(face_objs) - len(plausible),
            )
        return plausible or face_objs

    @staticmethod
    def _is_nested_false_positive(
        candidate_bbox: Tuple[int, int, int, int],
        primary_bbox: Tuple[int, int, int, int],
    ) -> bool:
        cx, cy, cw, ch = candidate_bbox
        px, py, pw, ph = primary_bbox
        candidate_area = max(cw * ch, 1)
        primary_area = max(pw * ph, 1)

        inter_left = max(cx, px)
        inter_top = max(cy, py)
        inter_right = min(cx + cw, px + pw)
        inter_bottom = min(cy + ch, py + ph)
        inter_width = max(0, inter_right - inter_left)
        inter_height = max(0, inter_bottom - inter_top)
        intersection_area = inter_width * inter_height
        candidate_cover = intersection_area / candidate_area
        primary_cover = intersection_area / primary_area

        candidate_center_x = cx + (cw / 2.0)
        candidate_center_y = cy + (ch / 2.0)
        inside_primary = (
            px <= candidate_center_x <= (px + pw)
            and py <= candidate_center_y <= (py + ph)
        )
        lower_face_region = candidate_center_y >= (py + ph * 0.48)
        small_relative_candidate = candidate_area <= (primary_area * 0.45)

        return bool(
            (inside_primary and candidate_cover >= 0.78 and small_relative_candidate)
            or (inside_primary and lower_face_region and primary_cover >= 0.10 and small_relative_candidate)
        )
