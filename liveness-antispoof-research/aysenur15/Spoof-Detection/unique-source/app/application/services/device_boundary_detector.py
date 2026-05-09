"""Device boundary and bezel detector for replay-attack preview checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DeviceBoundaryDetection:
    """Structured output for device-boundary analysis."""

    boundary_score: float
    is_spoof_confirmed: bool
    details: dict[str, float] = field(default_factory=dict)


class DeviceBoundaryDetector:
    """Detect likely phone/tablet screen boundaries around a tracked face.

    The detector focuses on a padded region around the face, searches for
    straight line segments and rectangular contours, validates common device
    aspect ratios, and adds a temporal bonus when the detected frame moves in
    sync with the face across recent frames.
    """

    COMMON_DEVICE_ASPECT_RATIOS = (
        16.0 / 9.0,
        19.5 / 9.0,
        18.0 / 9.0,
        4.0 / 3.0,
    )

    def __init__(
        self,
        *,
        padding_ratio: float = 0.55,
        history_size: int = 5,
        spoof_threshold: float = 0.72,
        debug: bool = False,
    ) -> None:
        self._padding_ratio = max(0.05, padding_ratio)
        self._spoof_threshold = spoof_threshold
        self._debug = debug
        self._history: deque[dict[str, tuple[float, float] | float]] = deque(maxlen=max(2, history_size))

    def analyze(
        self,
        *,
        frame_bgr: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        debug_frame_bgr: Optional[np.ndarray] = None,
    ) -> DeviceBoundaryDetection:
        """Analyze one frame and estimate whether a device border surrounds the face."""
        if frame_bgr is None or frame_bgr.size == 0:
            return DeviceBoundaryDetection(boundary_score=0.0, is_spoof_confirmed=False, details={})

        roi, roi_rect = self._extract_roi(frame_bgr, face_bbox)
        if roi.size == 0:
            return DeviceBoundaryDetection(boundary_score=0.0, is_spoof_confirmed=False, details={})

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=36,
            minLineLength=max(18, int(min(roi.shape[:2]) * 0.18)),
            maxLineGap=max(10, int(min(roi.shape[:2]) * 0.05)),
        )

        geometry_details = self._analyze_lines(lines, roi.shape[:2], face_bbox, roi_rect)
        contour_details = self._analyze_contours(edges, roi.shape[:2], face_bbox, roi_rect)
        candidate_center = contour_details.get("candidate_center") or geometry_details.get("candidate_center")
        temporal_details = self._analyze_temporal_sync(face_bbox, candidate_center)

        boundary_score = self._combine_scores(
            line_score=float(geometry_details["line_score"]),
            partial_score=float(geometry_details["partial_score"]),
            contour_score=float(contour_details["contour_score"]),
            temporal_score=float(temporal_details["temporal_sync_score"]),
        )
        is_spoof_confirmed = boundary_score >= self._spoof_threshold

        if self._debug and debug_frame_bgr is not None:
            self._draw_debug(
                debug_frame_bgr=debug_frame_bgr,
                roi_rect=roi_rect,
                lines=lines,
                contour_rect=contour_details.get("candidate_rect") or geometry_details.get("candidate_rect"),
            )

        details = {
            "boundary_score": boundary_score,
            "boundary_line_score": float(geometry_details["line_score"]),
            "boundary_parallel_score": float(geometry_details["parallel_score"]),
            "boundary_orthogonal_score": float(geometry_details["orthogonal_score"]),
            "boundary_line_density": float(geometry_details["line_density"]),
            "boundary_partial_score": float(geometry_details["partial_score"]),
            "boundary_partial_aspect_score": float(geometry_details["partial_aspect_score"]),
            "boundary_partial_face_cover_score": float(geometry_details["partial_face_cover_score"]),
            "boundary_partial_candidate_area_ratio": float(geometry_details["partial_candidate_area_ratio"]),
            "boundary_partial_candidate_found": float(geometry_details["partial_candidate_found"]),
            "boundary_contour_score": float(contour_details["contour_score"]),
            "boundary_rectangularity": float(contour_details["rectangularity"]),
            "boundary_aspect_score": float(contour_details["aspect_score"]),
            "boundary_face_coverage_score": float(contour_details["face_coverage_score"]),
            "boundary_temporal_sync_score": float(temporal_details["temporal_sync_score"]),
            "boundary_motion_sync_ratio": float(temporal_details["motion_sync_ratio"]),
            "boundary_roi_area_ratio": float(contour_details["roi_area_ratio"]),
            "boundary_candidate_area_ratio": float(contour_details["candidate_area_ratio"]),
            "boundary_candidate_found": float(contour_details["candidate_found"]),
        }
        return DeviceBoundaryDetection(
            boundary_score=boundary_score,
            is_spoof_confirmed=is_spoof_confirmed,
            details=details,
        )

    def _extract_roi(
        self,
        frame_bgr: np.ndarray,
        face_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        x, y, w, h = face_bbox
        pad_x = int(round(w * self._padding_ratio))
        pad_y = int(round(h * self._padding_ratio))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    def _analyze_lines(
        self,
        lines: Optional[np.ndarray],
        roi_shape: tuple[int, int],
        face_bbox: tuple[int, int, int, int],
        roi_rect: tuple[int, int, int, int],
    ) -> dict[str, float | tuple[float, float] | tuple[int, int, int, int] | None]:
        if lines is None or len(lines) == 0:
            return {
                "line_score": 0.0,
                "parallel_score": 0.0,
                "orthogonal_score": 0.0,
                "line_density": 0.0,
                "partial_score": 0.0,
                "partial_aspect_score": 0.0,
                "partial_face_cover_score": 0.0,
                "partial_candidate_area_ratio": 0.0,
                "partial_candidate_found": 0.0,
                "candidate_rect": None,
                "candidate_center": None,
            }

        roi_h, roi_w = roi_shape
        roi_x, roi_y, _, _ = roi_rect
        fx, fy, fw, fh = face_bbox
        face_center = (fx + 0.5 * fw, fy + 0.5 * fh)
        line_segments: list[tuple[float, float, float, float, float, float]] = []
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < max(18.0, min(roi_w, roi_h) * 0.18):
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = angle % 180.0
            line_segments.append((x1, y1, x2, y2, angle, length))

        if not line_segments:
            return {
                "line_score": 0.0,
                "parallel_score": 0.0,
                "orthogonal_score": 0.0,
                "line_density": 0.0,
                "partial_score": 0.0,
                "partial_aspect_score": 0.0,
                "partial_face_cover_score": 0.0,
                "partial_candidate_area_ratio": 0.0,
                "partial_candidate_found": 0.0,
                "candidate_rect": None,
                "candidate_center": None,
            }

        horizontal = [seg for seg in line_segments if min(seg[4], abs(seg[4] - 180.0)) <= 18.0]
        vertical = [seg for seg in line_segments if abs(seg[4] - 90.0) <= 18.0]
        parallel_score = min(1.0, 0.5 * min(len(horizontal), 2) + 0.5 * min(len(vertical), 2))
        orthogonal_score = 1.0 if horizontal and vertical else 0.0
        total_length = sum(seg[5] for seg in line_segments)
        line_density = min(1.0, total_length / max(float((roi_w + roi_h) * 2), 1.0))
        line_score = min(1.0, 0.45 * parallel_score + 0.35 * orthogonal_score + 0.20 * line_density)
        partial_details = self._build_partial_boundary_candidate(
            horizontal=horizontal,
            vertical=vertical,
            roi_rect=roi_rect,
            face_bbox=face_bbox,
            face_center=face_center,
        )
        return {
            "line_score": line_score,
            "parallel_score": parallel_score,
            "orthogonal_score": orthogonal_score,
            "line_density": line_density,
            "partial_score": float(partial_details["partial_score"]),
            "partial_aspect_score": float(partial_details["partial_aspect_score"]),
            "partial_face_cover_score": float(partial_details["partial_face_cover_score"]),
            "partial_candidate_area_ratio": float(partial_details["partial_candidate_area_ratio"]),
            "partial_candidate_found": float(partial_details["partial_candidate_found"]),
            "candidate_rect": partial_details["candidate_rect"],
            "candidate_center": partial_details["candidate_center"],
        }

    def _build_partial_boundary_candidate(
        self,
        *,
        horizontal: list[tuple[float, float, float, float, float, float]],
        vertical: list[tuple[float, float, float, float, float, float]],
        roi_rect: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
        face_center: tuple[float, float],
    ) -> dict[str, float | tuple[int, int, int, int] | tuple[float, float] | None]:
        roi_x, roi_y, roi_w, roi_h = roi_rect
        fx, fy, fw, fh = face_bbox

        partial_score = 0.0
        partial_aspect_score = 0.0
        partial_face_cover_score = 0.0
        partial_candidate_area_ratio = 0.0
        partial_candidate_found = 0.0
        candidate_rect = None
        candidate_center = None

        for family, is_vertical in ((horizontal, False), (vertical, True)):
            if len(family) < 2:
                continue

            sorted_family = sorted(family, key=lambda item: item[5], reverse=True)
            first = sorted_family[0]
            second = sorted_family[1]

            if is_vertical:
                x_left = min(first[0], first[2], second[0], second[2])
                x_right = max(first[0], first[2], second[0], second[2])
                center_x = 0.5 * (x_left + x_right)
                width = max(1.0, x_right - x_left)
                height = max(fh * 1.45, width * 1.75)
                abs_rect = (
                    int(round(center_x + roi_x - width / 2.0)),
                    int(round((fy + 0.5 * fh) - height / 2.0)),
                    int(round(width)),
                    int(round(height)),
                )
            else:
                y_top = min(first[1], first[3], second[1], second[3])
                y_bottom = max(first[1], first[3], second[1], second[3])
                center_y = 0.5 * (y_top + y_bottom)
                height = max(1.0, y_bottom - y_top)
                width = max(fw * 1.45, height * 0.55)
                abs_rect = (
                    int(round((fx + 0.5 * fw) - width / 2.0)),
                    int(round(center_y + roi_y - height / 2.0)),
                    int(round(width)),
                    int(round(height)),
                )

            ax, ay, aw, ah = abs_rect
            if aw <= 0 or ah <= 0:
                continue
            if not (ax <= face_center[0] <= ax + aw and ay <= face_center[1] <= ay + ah):
                continue

            aspect_ratio = max(float(ah / max(aw, 1)), float(aw / max(ah, 1)))
            aspect_score = max(
                0.0,
                1.0 - min(abs(aspect_ratio - target) for target in self.COMMON_DEVICE_ASPECT_RATIOS) / 0.70,
            )
            face_cover_score = min(
                1.0,
                min(aw / max(float(fw), 1.0), ah / max(float(fh), 1.0)) / 2.2,
            )
            candidate_area_ratio = float(aw * ah) / max(float(roi_w * roi_h), 1.0)
            orientation_bonus = 0.15 if is_vertical else 0.10
            family_score = min(
                1.0,
                0.40 * aspect_score
                + 0.35 * face_cover_score
                + 0.15 * _clamp01((candidate_area_ratio - 0.14) / 0.42)
                + orientation_bonus,
            )
            if family_score <= partial_score:
                continue

            partial_score = family_score
            partial_aspect_score = aspect_score
            partial_face_cover_score = face_cover_score
            partial_candidate_area_ratio = candidate_area_ratio
            partial_candidate_found = 1.0
            candidate_rect = abs_rect
            candidate_center = (ax + 0.5 * aw, ay + 0.5 * ah)

        return {
            "partial_score": partial_score,
            "partial_aspect_score": partial_aspect_score,
            "partial_face_cover_score": partial_face_cover_score,
            "partial_candidate_area_ratio": partial_candidate_area_ratio,
            "partial_candidate_found": partial_candidate_found,
            "candidate_rect": candidate_rect,
            "candidate_center": candidate_center,
        }

    def _analyze_contours(
        self,
        edges: np.ndarray,
        roi_shape: tuple[int, int],
        face_bbox: tuple[int, int, int, int],
        roi_rect: tuple[int, int, int, int],
    ) -> dict[str, float | tuple[float, float] | tuple[int, int, int, int] | None]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_h, roi_w = roi_shape
        roi_area = float(roi_h * roi_w)
        roi_x, roi_y, _, _ = roi_rect
        fx, fy, fw, fh = face_bbox
        face_center = (fx + 0.5 * fw, fy + 0.5 * fh)

        best_score = 0.0
        best_rect = None
        best_center = None
        best_metrics = {
            "contour_score": 0.0,
            "rectangularity": 0.0,
            "aspect_score": 0.0,
            "face_coverage_score": 0.0,
            "roi_area_ratio": roi_area / max(float((roi_x + roi_w) * (roi_y + roi_h)), 1.0),
            "candidate_area_ratio": 0.0,
            "candidate_found": 0.0,
        }

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
            if w < fw * 1.10 or h < fh * 1.10:
                continue

            absolute_rect = (roi_x + x, roi_y + y, w, h)
            candidate_center = (absolute_rect[0] + 0.5 * w, absolute_rect[1] + 0.5 * h)
            if not (
                absolute_rect[0] <= face_center[0] <= absolute_rect[0] + w
                and absolute_rect[1] <= face_center[1] <= absolute_rect[1] + h
            ):
                continue

            rectangularity = area / max(float(w * h), 1.0)
            aspect_ratio = max(float(w / max(h, 1)), float(h / max(w, 1)))
            aspect_score = max(
                0.0,
                1.0 - min(abs(aspect_ratio - target) for target in self.COMMON_DEVICE_ASPECT_RATIOS) / 0.55,
            )
            face_coverage_score = min(
                1.0,
                min(w / max(float(fw), 1.0), h / max(float(fh), 1.0)) / 2.1,
            )
            area_ratio = float(w * h) / max(roi_area, 1.0)
            contour_score = min(
                1.0,
                0.35 * _clamp01((rectangularity - 0.60) / 0.35)
                + 0.30 * aspect_score
                + 0.20 * face_coverage_score
                + 0.15 * _clamp01((area_ratio - 0.20) / 0.45),
            )
            if contour_score <= best_score:
                continue

            best_score = contour_score
            best_rect = absolute_rect
            best_center = candidate_center
            best_metrics = {
                "contour_score": contour_score,
                "rectangularity": rectangularity,
                "aspect_score": aspect_score,
                "face_coverage_score": face_coverage_score,
                "roi_area_ratio": 1.0,
                "candidate_area_ratio": area_ratio,
                "candidate_found": 1.0,
            }

        best_metrics["candidate_rect"] = best_rect
        best_metrics["candidate_center"] = best_center
        return best_metrics

    def _analyze_temporal_sync(
        self,
        face_bbox: tuple[int, int, int, int],
        candidate_center: Optional[tuple[float, float]],
    ) -> dict[str, float]:
        fx, fy, fw, fh = face_bbox
        face_center = (fx + 0.5 * fw, fy + 0.5 * fh)
        entry = {
            "face_center": face_center,
            "candidate_center": candidate_center,
        }
        self._history.append(entry)
        if len(self._history) < 3:
            return {
                "temporal_sync_score": 0.0,
                "motion_sync_ratio": 0.0,
            }

        sync_samples = 0
        valid_samples = 0
        for prev, curr in zip(list(self._history)[:-1], list(self._history)[1:]):
            prev_candidate = prev["candidate_center"]
            curr_candidate = curr["candidate_center"]
            if prev_candidate is None or curr_candidate is None:
                continue
            valid_samples += 1
            face_delta = np.asarray(curr["face_center"], dtype=np.float32) - np.asarray(prev["face_center"], dtype=np.float32)
            candidate_delta = np.asarray(curr_candidate, dtype=np.float32) - np.asarray(prev_candidate, dtype=np.float32)
            norm = float(np.linalg.norm(face_delta))
            if norm < 1.0:
                continue
            error = float(np.linalg.norm(face_delta - candidate_delta))
            if error <= max(6.0, norm * 0.35):
                sync_samples += 1

        motion_sync_ratio = sync_samples / max(valid_samples, 1)
        temporal_sync_score = motion_sync_ratio if valid_samples >= 2 else 0.0
        return {
            "temporal_sync_score": temporal_sync_score,
            "motion_sync_ratio": motion_sync_ratio,
        }

    def _combine_scores(
        self,
        *,
        line_score: float,
        partial_score: float,
        contour_score: float,
        temporal_score: float,
    ) -> float:
        return min(
            1.0,
            0.20 * line_score
            + 0.30 * partial_score
            + 0.30 * contour_score
            + 0.20 * temporal_score,
        )

    def _draw_debug(
        self,
        *,
        debug_frame_bgr: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        lines: Optional[np.ndarray],
        contour_rect: Optional[tuple[int, int, int, int]],
    ) -> None:
        x, y, w, h = roi_rect
        cv2.rectangle(debug_frame_bgr, (x, y), (x + w, y + h), (255, 180, 0), 1)
        if lines is not None:
            for raw in lines[:, 0]:
                x1, y1, x2, y2 = [int(v) for v in raw]
                cv2.line(debug_frame_bgr, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 255), 1)
        if contour_rect is not None:
            cx, cy, cw, ch = contour_rect
            cv2.rectangle(debug_frame_bgr, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
