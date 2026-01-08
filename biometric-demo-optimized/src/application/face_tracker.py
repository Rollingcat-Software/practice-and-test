"""Face tracking service - Assigns persistent IDs to detected faces."""

from typing import Dict, List
import numpy as np

from ..domain.models import FaceRegion


class FaceTracker:
    """Simple centroid-based face tracker.

    Assigns persistent IDs to faces across frames using
    nearest-neighbor matching based on face center positions.
    """

    def __init__(self, max_gone: int = 15, max_distance: int = 120):
        """Initialize tracker.

        Args:
            max_gone: Frames before a track is removed
            max_distance: Maximum pixel distance for matching
        """
        self._next_id = 0
        self._tracks: Dict[int, Dict] = {}  # {id: {'centroid': (x,y), 'gone': int}}
        self._max_gone = max_gone
        self._max_distance = max_distance

    def update(self, detections: List[FaceRegion]) -> Dict[int, FaceRegion]:
        """Update tracks with new detections.

        Args:
            detections: List of detected face regions

        Returns:
            Dict mapping track IDs to face regions
        """
        # Handle no detections
        if not detections:
            for tid in list(self._tracks.keys()):
                self._tracks[tid]['gone'] += 1
                if self._tracks[tid]['gone'] > self._max_gone:
                    del self._tracks[tid]
            return {}

        # Calculate centroids
        centroids = [d.center for d in detections]

        # Initialize tracks if empty
        if not self._tracks:
            result = {}
            for i, det in enumerate(detections):
                self._tracks[self._next_id] = {'centroid': centroids[i], 'gone': 0}
                result[self._next_id] = det
                self._next_id += 1
            return result

        # Match existing tracks to detections
        used_detections = set()
        result = {}

        for tid, track in list(self._tracks.items()):
            tc = track['centroid']
            best_j, best_dist = -1, float('inf')

            for j, nc in enumerate(centroids):
                if j not in used_detections:
                    dist = np.sqrt((tc[0] - nc[0])**2 + (tc[1] - nc[1])**2)
                    if dist < best_dist and dist < self._max_distance:
                        best_dist, best_j = dist, j

            if best_j >= 0:
                # Match found
                self._tracks[tid]['centroid'] = centroids[best_j]
                self._tracks[tid]['gone'] = 0
                result[tid] = detections[best_j]
                used_detections.add(best_j)
            else:
                # No match - increment gone counter
                self._tracks[tid]['gone'] += 1
                if self._tracks[tid]['gone'] > self._max_gone:
                    del self._tracks[tid]

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in used_detections:
                self._tracks[self._next_id] = {'centroid': centroids[j], 'gone': 0}
                result[self._next_id] = det
                self._next_id += 1

        return result

    def reset(self):
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 0
