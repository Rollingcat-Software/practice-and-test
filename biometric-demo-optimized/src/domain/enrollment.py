"""Enrollment domain models."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from collections import deque
import numpy as np


class EnrollmentPhase(Enum):
    """Enrollment process phases."""
    IDLE = auto()
    LIVENESS_PUZZLE = auto()  # Phase 1: Active liveness check
    FACE_CAPTURE = auto()     # Phase 2: Multi-angle capture


@dataclass
class EnrollmentPose:
    """Target pose for enrollment capture."""
    name: str
    target_yaw: float
    target_pitch: float
    tolerance: float

    @staticmethod
    def get_capture_poses() -> List['EnrollmentPose']:
        """Get the 5 capture poses for enrollment."""
        return [
            EnrollmentPose("STRAIGHT", 0, 0, 12),
            EnrollmentPose("LEFT", -25, 0, 15),
            EnrollmentPose("RIGHT", 25, 0, 15),
            EnrollmentPose("UP", 0, 18, 15),
            EnrollmentPose("DOWN", 0, -18, 15),
        ]


@dataclass
class EnrollmentState:
    """State of the enrollment process."""
    name: str = ""
    phase: EnrollmentPhase = EnrollmentPhase.IDLE
    step: int = 0  # Current capture step (0-4)
    embeddings: List[np.ndarray] = field(default_factory=list)
    thumbnail: Optional[np.ndarray] = None

    # Timing
    hold_start: float = 0.0
    hold_duration: float = 0.8  # Seconds to hold pose

    # Current pose tracking
    current_yaw: float = 0.0
    current_pitch: float = 0.0

    # Stability tracking
    face_positions: deque = field(default_factory=lambda: deque(maxlen=10))
    stability_threshold: float = 15.0  # Max pixels of movement
    stability_score: float = 0.0
    is_stable: bool = False

    @property
    def is_active(self) -> bool:
        return self.phase != EnrollmentPhase.IDLE

    @property
    def current_pose(self) -> Optional[EnrollmentPose]:
        poses = EnrollmentPose.get_capture_poses()
        if 0 <= self.step < len(poses):
            return poses[self.step]
        return None

    @property
    def is_complete(self) -> bool:
        return self.step >= 5

    def reset(self):
        self.name = ""
        self.phase = EnrollmentPhase.IDLE
        self.step = 0
        self.embeddings = []
        self.thumbnail = None
        self.hold_start = 0.0
        self.face_positions.clear()
        self.stability_score = 0.0
        self.is_stable = False

    def update_stability(self, cx: int, cy: int) -> bool:
        """Update stability tracking. Returns True if stable."""
        self.face_positions.append((cx, cy))

        if len(self.face_positions) < 5:
            self.is_stable = False
            self.stability_score = 0.0
            return False

        positions = list(self.face_positions)
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        movement = max(x_range, y_range)

        self.stability_score = max(0, min(100, 100 - (movement / self.stability_threshold) * 100))
        self.is_stable = movement < self.stability_threshold

        return self.is_stable

    def advance_step(self) -> bool:
        """Move to next capture step. Returns True if enrollment complete."""
        self.step += 1
        self.hold_start = 0.0
        self.face_positions.clear()
        return self.step >= 5
