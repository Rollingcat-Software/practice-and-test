"""Structured JSON logger for research data collection."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.domain.models import FrameAnalysis, SpoofCategory

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Logs frame analysis results as JSONL for research.

    Each line is a self-contained JSON object with frame metadata,
    per-face classifications, analyzer signals, and timing data.
    """

    def __init__(
        self,
        output_dir: str = "logs",
        log_every_n: int = 30,
        session_id: str | None = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_every_n = max(1, log_every_n)
        self._session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file: Optional[object] = None
        self._path = self._output_dir / f"session_{self._session_id}.jsonl"
        self._enabled = True
        self._frame_counter = 0

    def start(self):
        self._file = open(self._path, "a", encoding="utf-8")
        logger.info(f"Structured logging to {self._path}")

    def log(self, analysis: FrameAnalysis):
        if not self._enabled or self._file is None:
            return

        self._frame_counter += 1
        if self._frame_counter % self._log_every_n != 0:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "frame_id": analysis.frame_id,
            "face_count": len(analysis.faces),
            "faces": [],
            "frame_signals": analysis.frame_signals,
            "total_ms": round(analysis.total_ms, 2),
        }

        for face in analysis.faces:
            fid = face.face_id
            cls = analysis.classifications.get(fid)
            face_entry = {
                "face_id": fid,
                "bbox": [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2],
                "confidence": round(face.confidence, 4),
            }
            if cls:
                face_entry["classification"] = {
                    cat.value: round(prob, 4)
                    for cat, prob in cls.probabilities.items()
                }
                face_entry["dominant"] = cls.dominant_category.value
                face_entry["dominant_confidence"] = round(cls.confidence, 4)
                face_entry["analyzers"] = {}
                for name, result in cls.analyzer_results.items():
                    face_entry["analyzers"][name] = {
                        "score": round(result.score, 2),
                        "ms": round(result.elapsed_ms, 2),
                        **{k: round(v, 4) if isinstance(v, float) else v
                           for k, v in result.details.items()},
                    }
            entry["faces"].append(face_entry)

        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
        self._file.flush()  # type: ignore[union-attr]

    def toggle(self):
        self._enabled = not self._enabled
        state = "enabled" if self._enabled else "disabled"
        logger.info(f"Structured logging {state}")

    def stop(self):
        if self._file:
            self._file.close()  # type: ignore[union-attr]
            self._file = None
