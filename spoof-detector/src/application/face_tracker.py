"""Multi-face tracker with persistent IDs via IoU matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.domain.models import FaceROI, BBox

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """A tracked face with persistent identity."""
    face_id: int
    bbox: BBox
    lost_frames: int = 0
    total_frames: int = 0


class FaceTracker:
    """IoU-based multi-face tracker.

    Assigns persistent IDs to faces across frames by matching
    detections to existing tracks using Intersection over Union.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
    ):
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._tracks: dict[int, TrackedFace] = {}
        self._next_id = 1

    def update(self, detections: list[FaceROI]) -> list[FaceROI]:
        """Match detections to existing tracks, assign persistent IDs.

        Args:
            detections: Newly detected faces (with temporary IDs)

        Returns:
            Detections with persistent face_ids assigned
        """
        if not detections:
            self._age_tracks()
            return []

        if not self._tracks:
            return self._init_tracks(detections)

        # Build cost matrix (IoU scores)
        track_ids = list(self._tracks.keys())
        matched_det = set()
        matched_track = set()

        # Greedy matching: assign each detection to best matching track
        pairs: list[tuple[int, int, float]] = []
        for di, det in enumerate(detections):
            for ti, tid in enumerate(track_ids):
                iou = det.bbox.iou(self._tracks[tid].bbox)
                if iou >= self._iou_threshold:
                    pairs.append((di, ti, iou))

        # Sort by IoU descending, greedily assign
        pairs.sort(key=lambda p: p[2], reverse=True)
        result = []

        for di, ti, iou in pairs:
            if di in matched_det or ti in matched_track:
                continue
            matched_det.add(di)
            matched_track.add(ti)

            tid = track_ids[ti]
            track = self._tracks[tid]
            track.bbox = detections[di].bbox
            track.lost_frames = 0
            track.total_frames += 1

            det = detections[di]
            det.face_id = tid
            result.append(det)

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_det:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = TrackedFace(
                    face_id=new_id, bbox=det.bbox, total_frames=1
                )
                det.face_id = new_id
                result.append(det)

        # Age unmatched tracks
        for ti, tid in enumerate(track_ids):
            if ti not in matched_track:
                self._tracks[tid].lost_frames += 1

        # Remove stale tracks
        stale = [
            tid
            for tid, track in self._tracks.items()
            if track.lost_frames > self._max_lost_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return result

    def _init_tracks(self, detections: list[FaceROI]) -> list[FaceROI]:
        result = []
        for det in detections:
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = TrackedFace(
                face_id=new_id, bbox=det.bbox, total_frames=1
            )
            det.face_id = new_id
            result.append(det)
        return result

    def _age_tracks(self):
        for track in self._tracks.values():
            track.lost_frames += 1
        stale = [
            tid
            for tid, track in self._tracks.items()
            if track.lost_frames > self._max_lost_frames
        ]
        for tid in stale:
            del self._tracks[tid]

    @property
    def active_count(self) -> int:
        return sum(
            1 for t in self._tracks.values() if t.lost_frames == 0
        )

    @property
    def track_ids(self) -> list[int]:
        return [
            tid for tid, t in self._tracks.items() if t.lost_frames == 0
        ]
