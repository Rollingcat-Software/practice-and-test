"""Enrollment service - Two-phase face enrollment with liveness verification."""

import time
import logging
from typing import Optional, List

import numpy as np

from ..domain.models import FaceRegion, Landmarks, Pose, Quality
from ..domain.enrollment import EnrollmentPhase, EnrollmentPose, EnrollmentState
from ..infrastructure.analysis.quality_assessor import QualityAssessor
from ..infrastructure.analysis.embedding_extractor import EmbeddingExtractor
from ..infrastructure.persistence.face_database import FaceDatabase
from .puzzle_service import BiometricPuzzleService

logger = logging.getLogger(__name__)


class EnrollmentService:
    """Two-phase face enrollment service.

    Phase 1: Active liveness verification (Biometric Puzzle)
    Phase 2: Multi-angle face capture (5 poses)

    Features:
    - Face stability tracking
    - Quality threshold enforcement
    - Pose-based capture for robustness
    """

    def __init__(self, puzzle_service: BiometricPuzzleService,
                 face_db: FaceDatabase,
                 quality_assessor: QualityAssessor,
                 embedding_extractor: Optional[EmbeddingExtractor] = None,
                 min_quality: float = 65.0):
        self._puzzle = puzzle_service
        self._face_db = face_db
        self._quality = quality_assessor
        self._embedding = embedding_extractor
        self._min_quality = min_quality
        self._state = EnrollmentState()
        self._capture_poses = EnrollmentPose.get_capture_poses()

    @property
    def state(self) -> EnrollmentState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def phase(self) -> EnrollmentPhase:
        return self._state.phase

    def start(self, name: Optional[str] = None):
        """Start enrollment process.

        Args:
            name: Optional name for the identity (auto-generated if None)
        """
        self._state.reset()
        self._state.name = name or f"Person_{len(self._face_db) + 1}"
        self._state.phase = EnrollmentPhase.LIVENESS_PUZZLE
        self._puzzle.start()
        logger.info(f"Enrollment started: {self._state.name} (Phase 1: Liveness)")

    def cancel(self):
        """Cancel enrollment process."""
        self._puzzle.stop()
        self._state.reset()
        logger.info("Enrollment cancelled")

    def update_pose(self, pose: Pose):
        """Update current head pose tracking."""
        self._state.current_yaw = pose.yaw
        self._state.current_pitch = pose.pitch

    def check_stability(self, face: FaceRegion) -> bool:
        """Update and check face stability.

        Args:
            face: Current face detection

        Returns:
            True if face is stable
        """
        return self._state.update_stability(*face.center)

    def process_phase1(self, landmarks: Landmarks, pose: Pose) -> bool:
        """Process Phase 1: Liveness puzzle.

        Args:
            landmarks: Facial landmarks
            pose: Head pose

        Returns:
            True if phase completed successfully
        """
        if self._state.phase != EnrollmentPhase.LIVENESS_PUZZLE:
            return False

        result = self._puzzle.check(landmarks, pose)

        if self._puzzle.is_complete:
            if self._puzzle.passed:
                self._state.phase = EnrollmentPhase.FACE_CAPTURE
                self._state.hold_start = time.time()
                logger.info("Liveness verified! Moving to Phase 2: Face Capture")
                return True
            else:
                self.cancel()
                logger.warning("Liveness check failed - enrollment cancelled")
                return False

        return False

    def process_phase2(self, frame: np.ndarray, face: FaceRegion) -> Optional[bool]:
        """Process Phase 2: Multi-angle face capture.

        Args:
            frame: Full BGR frame
            face: Detected face region

        Returns:
            True if capture completed, False if step failed, None if in progress
        """
        if self._state.phase != EnrollmentPhase.FACE_CAPTURE:
            return None

        if self._state.is_complete:
            return self._finalize(frame, face)

        # Get current target pose
        target = self._state.current_pose
        if target is None:
            return None

        # Check pose alignment
        yaw_ok = abs(self._state.current_yaw - target.target_yaw) < target.tolerance
        pitch_ok = abs(self._state.current_pitch - target.target_pitch) < target.tolerance

        if not (yaw_ok and pitch_ok):
            self._state.hold_start = time.time()
            return None

        # Check stability
        if not self._state.is_stable:
            self._state.hold_start = time.time()
            return None

        # Check hold time
        if time.time() - self._state.hold_start < self._state.hold_duration:
            return None

        # Capture!
        return self._capture(frame, face, target)

    def _capture(self, frame: np.ndarray, face: FaceRegion, pose: EnrollmentPose) -> Optional[bool]:
        """Capture face embedding at current pose."""
        # Extract face image with padding
        padding = 20
        y1 = max(0, face.y - padding)
        y2 = min(frame.shape[0], face.y + face.h + padding)
        x1 = max(0, face.x - padding)
        x2 = min(frame.shape[1], face.x + face.w + padding)
        face_img = frame[y1:y2, x1:x2]

        # Check quality
        quality = self._quality.assess(face_img)
        if quality.score < self._min_quality:
            self._state.hold_start = time.time()
            return None

        # Extract embedding
        if self._embedding is None:
            self._embedding = EmbeddingExtractor()

        emb = self._embedding.extract(face_img)
        if emb is None:
            self._state.hold_start = time.time()
            return None

        # Store embedding
        self._state.embeddings.append(emb)

        # Save thumbnail on first capture
        if self._state.step == 0:
            self._state.thumbnail = face_img.copy()

        # Advance to next step
        is_complete = self._state.advance_step()
        logger.info(f"Captured angle {self._state.step}/5: {pose.name}")

        if is_complete:
            return self._finalize(frame, face)

        return None

    def _finalize(self, frame: np.ndarray, face: FaceRegion) -> bool:
        """Finalize enrollment and save to database."""
        if not self._state.embeddings:
            self.cancel()
            return False

        # Get thumbnail if not set
        if self._state.thumbnail is None:
            y1 = max(0, face.y)
            y2 = min(frame.shape[0], face.y + face.h)
            x1 = max(0, face.x)
            x2 = min(frame.shape[1], face.x + face.w)
            self._state.thumbnail = frame[y1:y2, x1:x2].copy()

        # Save to database
        self._face_db.enroll(
            self._state.name,
            self._state.embeddings[0],
            self._state.thumbnail
        )

        # Add additional embeddings
        for emb in self._state.embeddings[1:]:
            self._face_db.add_embedding(self._state.name, emb)

        logger.info(f"Enrolled {self._state.name} with {len(self._state.embeddings)} angles")

        # Reset state
        self._state.reset()
        return True

    def get_pose_hint(self) -> str:
        """Get hint for current pose adjustment."""
        if self._state.phase != EnrollmentPhase.FACE_CAPTURE:
            return ""

        target = self._state.current_pose
        if target is None:
            return ""

        hints = []
        if self._state.current_yaw < target.target_yaw - target.tolerance:
            hints.append("Turn RIGHT")
        elif self._state.current_yaw > target.target_yaw + target.tolerance:
            hints.append("Turn LEFT")

        if self._state.current_pitch < target.target_pitch - target.tolerance:
            hints.append("Tilt DOWN")
        elif self._state.current_pitch > target.target_pitch + target.tolerance:
            hints.append("Tilt UP")

        return " & ".join(hints) if hints else "Adjust"

    def get_hold_progress(self) -> float:
        """Get current hold progress (0-100)."""
        if self._state.hold_start == 0:
            return 0
        elapsed = time.time() - self._state.hold_start
        return min(100, (elapsed / self._state.hold_duration) * 100)
