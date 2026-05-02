"""Device boundary (bezel) detector for phone/tablet screen attacks.

Ported from biometric-processor's DeviceBoundaryDetector.
Detects rectangular frames (phone/tablet bezels) surrounding faces
using Canny edge detection + Hough line segments + contour analysis.

This is the most direct physical signal for screen-based attacks:
if there's a phone bezel around the face, it's a screen replay.
"""

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

# Common phone/tablet aspect ratios
DEVICE_ASPECT_RATIOS = (16.0 / 9.0, 19.5 / 9.0, 18.0 / 9.0, 4.0 / 3.0)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class DeviceBoundaryAnalyzer:
    """Detects phone/tablet screen bezels around tracked faces.

    Searches for rectangular contours and straight line segments
    in a padded region around each face. Validates against common
    device aspect ratios (16:9, 19.5:9, 18:9, 4:3).

    This analyzer needs the FULL FRAME (not just face crop) because
    it looks for bezel edges AROUND the face. Use set_frame() before
    analyze(), or pass via the pipeline's frame context.
    """

    def __init__(self, padding_ratio: float = 0.55, spoof_threshold: float = 0.50):
        self._padding = max(0.05, padding_ratio)
        self._spoof_threshold = spoof_threshold
        self._original_frame: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "device_boundary"

    def set_frame(self, frame: np.ndarray):
        self._original_frame = frame

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        if self._original_frame is None:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"error": "no_frame"}, elapsed_ms=0)

        frame = self._original_frame
        bbox = face_roi.bbox
        fh, fw = frame.shape[:2]

        # Extract padded ROI around face
        pad_x = int(bbox.width * self._padding)
        pad_y = int(bbox.height * self._padding)
        rx1 = max(0, bbox.x1 - pad_x)
        ry1 = max(0, bbox.y1 - pad_y)
        rx2 = min(fw, bbox.x2 + pad_x)
        ry2 = min(fh, bbox.y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return AnalyzerResult(name=self.name, score=100.0,
                                  details={"no_roi": True},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        roi_h, roi_w = roi.shape[:2]

        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)

        # Hough lines
        min_line = max(18, int(min(roi_h, roi_w) * 0.18))
        lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180, 36,
                                 minLineLength=min_line,
                                 maxLineGap=max(10, int(min(roi_h, roi_w) * 0.05)))

        line_score = self._analyze_lines(lines, roi_h, roi_w)
        contour_score = self._analyze_contours(edges, roi_h, roi_w, bbox, rx1, ry1)

        # Combined boundary score (0-1, higher = more bezel-like)
        boundary_score = 0.50 * contour_score + 0.50 * line_score

        # Convert to liveness score (0-100, higher = more live-like)
        # High boundary score = spoof, so invert
        if boundary_score >= self._spoof_threshold:
            score = max(0.0, 30.0 * (1.0 - boundary_score))  # Strong spoof signal
        else:
            score = 70.0 + 30.0 * (1.0 - boundary_score / self._spoof_threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "boundary_score": boundary_score,
                "line_score": line_score,
                "contour_score": contour_score,
                "bezel_detected": boundary_score >= self._spoof_threshold,
                "n_lines": len(lines) if lines is not None else 0,
            },
            elapsed_ms=elapsed_ms,
        )

    def _analyze_lines(self, lines: Optional[np.ndarray], roi_h: int, roi_w: int) -> float:
        if lines is None or len(lines) == 0:
            return 0.0

        horizontal = []
        vertical = []
        total_length = 0.0

        for raw in lines[:, 0]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < max(18.0, min(roi_w, roi_h) * 0.18):
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            total_length += length

            if min(angle, abs(angle - 180.0)) <= 18.0:
                horizontal.append(length)
            elif abs(angle - 90.0) <= 18.0:
                vertical.append(length)

        parallel = min(1.0, 0.5 * min(len(horizontal), 2) + 0.5 * min(len(vertical), 2))
        orthogonal = 1.0 if horizontal and vertical else 0.0
        density = min(1.0, total_length / max(float((roi_w + roi_h) * 2), 1.0))

        return min(1.0, 0.45 * parallel + 0.35 * orthogonal + 0.20 * density)

    def _analyze_contours(self, edges: np.ndarray, roi_h: int, roi_w: int,
                          face_bbox, roi_x: int, roi_y: int) -> float:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(roi_h * roi_w)
        face_cx = face_bbox.x1 + face_bbox.width / 2
        face_cy = face_bbox.y1 + face_bbox.height / 2

        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < roi_area * 0.04:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1.0:
                continue
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            abs_x, abs_y = roi_x + x, roi_y + y

            # Must be bigger than face
            if w < face_bbox.width * 1.1 or h < face_bbox.height * 1.1:
                continue
            # Face center must be inside rectangle
            if not (abs_x <= face_cx <= abs_x + w and abs_y <= face_cy <= abs_y + h):
                continue

            rectangularity = area / max(float(w * h), 1.0)
            aspect = max(float(w / max(h, 1)), float(h / max(w, 1)))
            aspect_score = max(0.0, 1.0 - min(abs(aspect - t) for t in DEVICE_ASPECT_RATIOS) / 0.55)
            face_cover = min(1.0, min(w / max(float(face_bbox.width), 1.0),
                                       h / max(float(face_bbox.height), 1.0)) / 2.1)
            area_ratio = float(w * h) / max(roi_area, 1.0)

            score = min(1.0,
                        0.35 * _clamp01((rectangularity - 0.60) / 0.35)
                        + 0.30 * aspect_score
                        + 0.20 * face_cover
                        + 0.15 * _clamp01((area_ratio - 0.20) / 0.45))

            best_score = max(best_score, score)

        return best_score
