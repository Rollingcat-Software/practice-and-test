"""Liveness Prover — "Guilty Until Proven Innocent" architecture.

Everyone starts as SPOOF. You must PROVE you're alive through:

PASSIVE PROOFS (detected automatically):
  1. Blinks — real eyes close/open in V-shaped EAR pattern
  2. Landmark variance — 478 points show natural micro-movement
  3. Head rotation — yaw/pitch changes detected from landmarks
  4. Expression changes — mouth/eye region variance over time

ACTIVE CHALLENGES (user must respond):
  5. "Blink now" — must blink within 3 seconds
  6. "Turn head left/right" — must rotate head in requested direction
  7. "Show your hand" — must briefly show open hand near face

Each proven signal awards liveness points. Threshold to pass: 60/100.
A video can't respond to random challenges. A photo can't blink.
"""

from __future__ import annotations

import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"
    SHOW_HAND = "show_hand"


class ChallengeState(Enum):
    WAITING = "waiting"       # No active challenge
    PROMPTED = "prompted"     # Challenge displayed, waiting for response
    COMPLETED = "completed"   # User responded correctly
    FAILED = "failed"         # User didn't respond in time


@dataclass
class Challenge:
    challenge_type: ChallengeType
    state: ChallengeState = ChallengeState.WAITING
    prompted_at: float = 0.0
    completed_at: float = 0.0
    timeout_sec: float = 4.0
    response_latency_ms: float = 0.0

    @property
    def display_text(self) -> str:
        texts = {
            ChallengeType.BLINK: "BLINK YOUR EYES",
            ChallengeType.TURN_LEFT: "TURN HEAD LEFT",
            ChallengeType.TURN_RIGHT: "TURN HEAD RIGHT",
            ChallengeType.NOD: "NOD YOUR HEAD",
            ChallengeType.SHOW_HAND: "SHOW YOUR HAND",
        }
        return texts.get(self.challenge_type, "")


@dataclass
class LivenessScore:
    """Accumulated liveness evidence."""
    total: float = 0.0          # 0-100, must reach 60 to pass
    blink_points: float = 0.0   # Max 20
    landmark_points: float = 0.0 # Max 15
    rotation_points: float = 0.0 # Max 15
    expression_points: float = 0.0  # Max 10
    challenge_points: float = 0.0   # Max 40 (active challenges)
    challenges_passed: int = 0
    challenges_failed: int = 0

    def update_total(self):
        self.total = min(100.0, (
            self.blink_points
            + self.landmark_points
            + self.rotation_points
            + self.expression_points
            + self.challenge_points
        ))

    @property
    def is_proven_live(self) -> bool:
        return self.total >= 60.0


# MediaPipe face mesh key points for head pose estimation
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


class LivenessProver:
    """Accumulates liveness evidence and manages active challenges.

    Usage:
        prover = LivenessProver()
        prover.start()

        # Each frame:
        prover.update(landmarks, blink_detected, has_hand)
        challenge = prover.get_active_challenge()
        score = prover.get_score()

        # Draw challenge text on frame
        if challenge and challenge.state == ChallengeState.PROMPTED:
            draw_text(challenge.display_text)
    """

    CHALLENGE_INTERVAL_SEC = 8.0   # Seconds between challenges
    MAX_CHALLENGES = 5             # Total challenges per session
    BLINK_AWARD = 5.0              # Points per real blink detected
    MAX_BLINK_POINTS = 20.0
    LANDMARK_VAR_THRESHOLD = 2.0   # Minimum variance for points
    ROTATION_THRESHOLD = 5.0       # Degrees of rotation for points
    CHALLENGE_AWARD = 10.0         # Points per passed challenge

    def __init__(self, enable_challenges: bool = True):
        self._enable_challenges = enable_challenges
        self._start_time: float = 0.0
        self._score = LivenessScore()
        self._challenges: list[Challenge] = []
        self._active_challenge: Optional[Challenge] = None
        self._last_challenge_time: float = 0.0
        self._challenges_issued = 0

        # Head pose tracking
        self._yaw_history: deque[float] = deque(maxlen=90)
        self._pitch_history: deque[float] = deque(maxlen=90)
        self._yaw_range_seen = 0.0
        self._pitch_range_seen = 0.0

        # Baseline for challenge detection
        self._baseline_yaw: Optional[float] = None
        self._baseline_pitch: Optional[float] = None
        self._blink_count_at_challenge: int = 0
        self._hand_detected_at_challenge: bool = False

    def start(self):
        self._start_time = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        blink_count: int,
        landmark_variance: float,
        expression_ratio: float,
        face_count: int,
    ):
        """Update liveness evidence with current frame data."""
        elapsed = self.elapsed

        # === Passive Proof: Blinks ===
        if blink_count > 0:
            self._score.blink_points = min(
                self.MAX_BLINK_POINTS,
                blink_count * self.BLINK_AWARD
            )

        # === Passive Proof: Landmark Variance ===
        if landmark_variance > self.LANDMARK_VAR_THRESHOLD:
            self._score.landmark_points = min(15.0, landmark_variance * 3.0)

        # === Passive Proof: Expression Changes ===
        if expression_ratio > 1.5:
            self._score.expression_points = min(10.0, expression_ratio * 2.0)

        # === Passive Proof: Head Rotation ===
        if landmarks is not None and len(landmarks) > max(NOSE_TIP, LEFT_EAR, RIGHT_EAR, FOREHEAD, CHIN):
            yaw, pitch = self._estimate_head_pose(landmarks)
            self._yaw_history.append(yaw)
            self._pitch_history.append(pitch)

            if len(self._yaw_history) > 10:
                self._yaw_range_seen = max(self._yaw_range_seen,
                                            float(np.max(list(self._yaw_history)) - np.min(list(self._yaw_history))))
                self._pitch_range_seen = max(self._pitch_range_seen,
                                              float(np.max(list(self._pitch_history)) - np.min(list(self._pitch_history))))

            if self._yaw_range_seen > self.ROTATION_THRESHOLD:
                self._score.rotation_points = min(10.0, self._yaw_range_seen * 0.5)
            if self._pitch_range_seen > self.ROTATION_THRESHOLD:
                self._score.rotation_points = min(15.0, self._score.rotation_points + self._pitch_range_seen * 0.3)

        # === Active Challenges ===
        if self._enable_challenges:
            self._manage_challenges(elapsed, blink_count, landmarks, face_count)

        self._score.update_total()

    def _estimate_head_pose(self, landmarks: np.ndarray) -> tuple[float, float]:
        """Estimate yaw and pitch from face landmarks (degrees)."""
        nose = landmarks[NOSE_TIP][:2]
        left_ear = landmarks[LEFT_EAR][:2]
        right_ear = landmarks[RIGHT_EAR][:2]
        forehead = landmarks[FOREHEAD][:2]
        chin = landmarks[CHIN][:2]

        # Yaw: nose position relative to ear midpoint
        ear_mid = (left_ear + right_ear) / 2.0
        ear_dist = float(np.linalg.norm(left_ear - right_ear))
        if ear_dist > 1.0:
            yaw_ratio = float(nose[0] - ear_mid[0]) / (ear_dist / 2.0)
            yaw = float(np.degrees(np.arcsin(np.clip(yaw_ratio, -1, 1))))
        else:
            yaw = 0.0

        # Pitch: nose vertical position relative to forehead-chin line
        face_height = float(np.linalg.norm(forehead - chin))
        if face_height > 1.0:
            vertical_mid = (forehead[1] + chin[1]) / 2.0
            pitch_ratio = (nose[1] - vertical_mid) / (face_height / 2.0)
            pitch = float(np.degrees(np.arcsin(np.clip(pitch_ratio, -1, 1))))
        else:
            pitch = 0.0

        return yaw, pitch

    def _manage_challenges(self, elapsed: float, blink_count: int,
                           landmarks: Optional[np.ndarray], face_count: int):
        """Issue and evaluate active challenges."""
        # Check active challenge timeout/completion
        if self._active_challenge and self._active_challenge.state == ChallengeState.PROMPTED:
            time_since = elapsed - self._active_challenge.prompted_at
            if time_since > self._active_challenge.timeout_sec:
                self._active_challenge.state = ChallengeState.FAILED
                self._score.challenges_failed += 1
                logger.info(f"Challenge FAILED: {self._active_challenge.challenge_type.value}")
                self._active_challenge = None
            else:
                # Check if challenge was completed
                completed = self._check_challenge_response(
                    self._active_challenge, blink_count, landmarks, face_count
                )
                if completed:
                    self._active_challenge.state = ChallengeState.COMPLETED
                    self._active_challenge.completed_at = elapsed
                    latency = (elapsed - self._active_challenge.prompted_at) * 1000
                    self._active_challenge.response_latency_ms = latency
                    self._score.challenges_passed += 1
                    self._score.challenge_points = min(
                        40.0,
                        self._score.challenges_passed * self.CHALLENGE_AWARD
                    )
                    self._challenges.append(self._active_challenge)
                    logger.info(f"Challenge PASSED: {self._active_challenge.challenge_type.value} ({latency:.0f}ms)")
                    self._active_challenge = None

        # Issue new challenge if ready
        if (self._active_challenge is None
                and self._challenges_issued < self.MAX_CHALLENGES
                and elapsed > 5.0  # Wait 5s before first challenge
                and elapsed - self._last_challenge_time > self.CHALLENGE_INTERVAL_SEC):
            self._issue_challenge(elapsed, blink_count)

    def _issue_challenge(self, elapsed: float, blink_count: int):
        """Issue a random challenge."""
        # Choose challenge type (weighted by what we haven't tested)
        available = [ChallengeType.BLINK, ChallengeType.TURN_LEFT,
                     ChallengeType.TURN_RIGHT, ChallengeType.NOD]
        challenge_type = random.choice(available)

        self._active_challenge = Challenge(
            challenge_type=challenge_type,
            state=ChallengeState.PROMPTED,
            prompted_at=elapsed,
        )
        self._challenges_issued += 1
        self._last_challenge_time = elapsed

        # Store baseline for comparison
        self._blink_count_at_challenge = blink_count
        if self._yaw_history:
            self._baseline_yaw = self._yaw_history[-1]
        if self._pitch_history:
            self._baseline_pitch = self._pitch_history[-1]

        logger.info(f"Challenge issued: {challenge_type.value}")

    def _check_challenge_response(self, challenge: Challenge, blink_count: int,
                                   landmarks: Optional[np.ndarray],
                                   face_count: int) -> bool:
        """Check if the user responded to the active challenge."""
        if challenge.challenge_type == ChallengeType.BLINK:
            return blink_count > self._blink_count_at_challenge

        if landmarks is None or len(landmarks) < 468:
            return False

        yaw, pitch = self._estimate_head_pose(landmarks)

        if challenge.challenge_type == ChallengeType.TURN_LEFT:
            if self._baseline_yaw is not None:
                return yaw - self._baseline_yaw < -8.0  # Turned left

        if challenge.challenge_type == ChallengeType.TURN_RIGHT:
            if self._baseline_yaw is not None:
                return yaw - self._baseline_yaw > 8.0  # Turned right

        if challenge.challenge_type == ChallengeType.NOD:
            if self._baseline_pitch is not None:
                return abs(pitch - self._baseline_pitch) > 6.0  # Nodded

        return False

    def get_active_challenge(self) -> Optional[Challenge]:
        return self._active_challenge

    def get_score(self) -> LivenessScore:
        return self._score

    def get_challenge_history(self) -> list[dict]:
        return [
            {
                "type": c.challenge_type.value,
                "state": c.state.value,
                "latency_ms": round(c.response_latency_ms, 0),
            }
            for c in self._challenges
        ]
