"""Professional OpenCV overlay for spoof detection visualization.

Renders:
- Top-left: face count, FPS, resolution stats
- Per-face: colored bounding box + category label
- Bottom-left: detailed probability bars per face
- Bottom-right: profiler timings (optional)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.domain.models import (
    FrameAnalysis,
    SpoofCategory,
    SpoofClassification,
    CATEGORY_LABELS,
    CATEGORY_COLORS,
    FaceROI,
)


class Colors:
    """BGR color constants."""
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    ORANGE = (0, 165, 255)
    GRAY = (100, 100, 100)
    DARK_GRAY = (50, 50, 50)


# Ordered categories for display
DISPLAY_ORDER = [
    SpoofCategory.REAL,
    SpoofCategory.STATIC_IMAGE,
    SpoofCategory.VIDEO_REPLAY,
    SpoofCategory.MASK_3D,
    SpoofCategory.HEAVY_MAKEUP,
    SpoofCategory.AR_FILTER,
    SpoofCategory.DEEPFAKE_INJECT,
]


class Overlay:
    """Professional HUD overlay for spoof detection results."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN
    BAR_WIDTH = 100
    BAR_HEIGHT = 10
    LINE_HEIGHT = 18
    PANEL_PADDING = 10

    def __init__(self, show_detail: bool = True, show_profiler: bool = False):
        self._show_detail = show_detail
        self._show_profiler = show_profiler
        self._fps_buffer: deque[float] = deque(maxlen=30)
        self._last_time = time.perf_counter()

    def render(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Render all overlays on the frame (in-place)."""
        self._update_fps()
        self._draw_stats_panel(frame, analysis)
        self._draw_face_boxes(frame, analysis)
        if self._show_detail:
            self._draw_detail_panel(frame, analysis)

    def toggle_detail(self):
        self._show_detail = not self._show_detail

    def toggle_profiler(self):
        self._show_profiler = not self._show_profiler

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_buffer.append(1.0 / dt)

    @property
    def fps(self) -> float:
        return sum(self._fps_buffer) / max(len(self._fps_buffer), 1)

    def _draw_stats_panel(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Top-left stats: face count, FPS, resolution, pipeline time."""
        h, w = frame.shape[:2]
        lines = [
            f"Faces: {len(analysis.faces)}",
            f"FPS: {self.fps:.1f}",
            f"Res: {w}x{h}",
            f"Pipeline: {analysis.total_ms:.1f}ms",
        ]

        panel_h = len(lines) * self.LINE_HEIGHT + self.PANEL_PADDING * 2
        panel_w = 200

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "FIVUCSAS SPOOF DETECTOR", (10, 20),
                    self.FONT, 0.4, Colors.CYAN, 1, cv2.LINE_AA)

        y = 38
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        self.FONT, 0.38, Colors.WHITE, 1, cv2.LINE_AA)
            y += self.LINE_HEIGHT

    def _draw_face_boxes(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Draw bounding boxes colored by dominant category with labels."""
        for face in analysis.faces:
            cls = analysis.classifications.get(face.face_id)
            if cls:
                color = CATEGORY_COLORS.get(cls.dominant_category, Colors.YELLOW)
                label = CATEGORY_LABELS.get(cls.dominant_category, "?")
                conf = cls.confidence * 100
                text = f"#{face.face_id} {label} {conf:.0f}%"
            else:
                color = Colors.GRAY
                text = f"#{face.face_id}"

            bbox = face.bbox
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.5, 1)
            label_y = max(bbox.y1 - 8, th + 5)
            cv2.rectangle(frame, (bbox.x1, label_y - th - 4),
                          (bbox.x1 + tw + 8, label_y + 4), color, -1)
            cv2.putText(frame, text, (bbox.x1 + 4, label_y),
                        self.FONT, 0.5, Colors.BLACK, 1, cv2.LINE_AA)

    def _draw_detail_panel(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Bottom-left panel with per-face probability breakdown."""
        if not analysis.classifications:
            return

        h, w = frame.shape[:2]
        n_faces = len(analysis.classifications)
        n_categories = len(DISPLAY_ORDER)
        panel_h = n_faces * (n_categories * self.LINE_HEIGHT + 30) + self.PANEL_PADDING
        panel_w = 280
        panel_y = max(0, h - panel_h - 10)

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, panel_y), (5 + panel_w, h - 5), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = panel_y + 15
        for face_id, cls in analysis.classifications.items():
            # Face header
            dominant_label = CATEGORY_LABELS.get(cls.dominant_category, "?")
            cv2.putText(frame, f"Face #{face_id} [{dominant_label}]",
                        (10, y), self.FONT, 0.4, Colors.CYAN, 1, cv2.LINE_AA)
            y += self.LINE_HEIGHT + 2

            # Category probability bars
            for cat in DISPLAY_ORDER:
                prob = cls.probabilities.get(cat, 0.0)
                label = f"{CATEGORY_LABELS[cat]:>14s}: {prob * 100:5.1f}%"
                color = CATEGORY_COLORS.get(cat, Colors.WHITE)

                cv2.putText(frame, label, (12, y),
                            self.FONT, 0.32, Colors.WHITE, 1, cv2.LINE_AA)

                # Progress bar
                bar_x = 175
                bar_y_top = y - self.BAR_HEIGHT + 2
                bar_fill = int(self.BAR_WIDTH * prob)

                cv2.rectangle(frame, (bar_x, bar_y_top),
                              (bar_x + self.BAR_WIDTH, bar_y_top + self.BAR_HEIGHT),
                              Colors.DARK_GRAY, -1)
                if bar_fill > 0:
                    cv2.rectangle(frame, (bar_x, bar_y_top),
                                  (bar_x + bar_fill, bar_y_top + self.BAR_HEIGHT),
                                  color, -1)

                y += self.LINE_HEIGHT

            y += 8  # Spacing between faces
