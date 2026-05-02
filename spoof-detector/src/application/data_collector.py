"""Research data collection service.

Captures frames with metadata and optional manual labels
for building the AR-filter detection training dataset.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.domain.models import FrameAnalysis, SpoofCategory

logger = logging.getLogger(__name__)


class DataCollector:
    """Opt-in frame and label capture for research dataset building.

    Usage:
        collector = DataCollector(output_dir="data/captures")
        collector.capture(frame, analysis, label=SpoofCategory.AR_FILTER)
    """

    def __init__(self, output_dir: str = "data/captures"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def capture(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        label: Optional[SpoofCategory] = None,
        notes: str = "",
    ) -> Path:
        """Capture a frame with metadata for the dataset.

        Args:
            frame: BGR full frame
            analysis: Current frame analysis
            label: Ground truth label (manual annotation)
            notes: Optional notes

        Returns:
            Path to saved image
        """
        self._counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"sample_{ts}_{self._counter:06d}"

        # Save full frame
        img_path = self._output_dir / f"{base}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save per-face crops
        for face in analysis.faces:
            if face.crop is not None:
                crop_path = self._output_dir / f"{base}_face{face.face_id}.jpg"
                cv2.imwrite(str(crop_path), face.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save metadata
        meta = {
            "timestamp": ts,
            "frame_id": analysis.frame_id,
            "ground_truth": label.value if label else None,
            "notes": notes,
            "face_count": len(analysis.faces),
            "faces": [],
        }
        for face in analysis.faces:
            face_meta = {
                "face_id": face.face_id,
                "bbox": [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2],
                "confidence": round(face.confidence, 4),
            }
            cls = analysis.classifications.get(face.face_id)
            if cls:
                face_meta["predicted"] = cls.dominant_category.value
                face_meta["probabilities"] = {
                    cat.value: round(prob, 4)
                    for cat, prob in cls.probabilities.items()
                }
            meta["faces"].append(face_meta)

        meta_path = self._output_dir / f"{base}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Captured {img_path} (label={label})")
        return img_path
