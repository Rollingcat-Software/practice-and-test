"""Biometric Puzzle Service - Active liveness challenge detection."""

import random
import time
import logging
from typing import Optional, List
from collections import deque

from ..domain.models import Landmarks, Pose
from ..domain.challenges import (
    ChallengeType, Challenge, ChallengeResult, PuzzleState, ChallengeThresholds
)
from ..infrastructure.detection.landmark_detector import LandmarkDetector

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Rolling average smoother for reducing measurement noise."""

    def __init__(self, window_size: int = 5):
        self._window = window_size
        self._history: deque = deque(maxlen=window_size)

    def smooth(self, value: float) -> float:
        """Add value and return smoothed average."""
        self._history.append(value)
        return sum(self._history) / len(self._history)

    def reset(self):
        """Clear history."""
        self._history.clear()


class BiometricPuzzleService:
    """Active liveness detection through challenge-response puzzles.

    Supports 14 challenge types:
    - Eye actions: Blink, Close Left, Close Right
    - Mouth actions: Smile, Open Mouth
    - Head movements: Turn Left/Right, Look Up/Down
    - Eyebrow actions: Raise Both/Left/Right
    - Dynamic: Nod, Shake Head
    """

    def __init__(self, landmark_detector: LandmarkDetector, num_challenges: int = 3):
        self._landmark_detector = landmark_detector
        self._num_challenges = num_challenges
        self._state = PuzzleState()
        self._challenges = Challenge.get_all()
        self._thresholds = ChallengeThresholds

        # Temporal smoothers for stable detection (window=5 frames)
        self._left_ear_smoother = TemporalSmoother(window_size=5)
        self._right_ear_smoother = TemporalSmoother(window_size=5)
        self._mar_smoother = TemporalSmoother(window_size=5)
        self._smile_smoother = TemporalSmoother(window_size=5)

    @property
    def state(self) -> PuzzleState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def is_complete(self) -> bool:
        return self._state.is_complete

    @property
    def passed(self) -> bool:
        return self._state.passed

    def start(self, challenge_types: Optional[List[ChallengeType]] = None):
        """Start a new puzzle with random or specified challenges."""
        self._state.reset()

        # Reset temporal smoothers for fresh start
        self._left_ear_smoother.reset()
        self._right_ear_smoother.reset()
        self._mar_smoother.reset()
        self._smile_smoother.reset()

        if challenge_types:
            self._state.challenges = challenge_types[:self._num_challenges]
        else:
            pool = Challenge.get_simple_pool()
            self._state.challenges = random.sample(pool, min(self._num_challenges, len(pool)))

        self._state.is_active = True
        logger.info(f"Puzzle started: {[c.name for c in self._state.challenges]}")

    def stop(self):
        """Stop the puzzle."""
        self._state.is_active = False
        self._state.is_complete = True
        logger.info(f"Puzzle stopped. Results: {self._state.results}")

    def get_current_challenge(self) -> Optional[Challenge]:
        """Get current challenge definition."""
        current_type = self._state.current_challenge
        if current_type:
            return self._challenges.get(current_type)
        return None

    def check(self, landmarks: Landmarks, pose: Pose) -> ChallengeResult:
        """Check if current challenge is being performed.

        Args:
            landmarks: Facial landmarks
            pose: Head pose

        Returns:
            ChallengeResult with detection status and progress
        """
        if not self._state.is_active or self._state.current_challenge is None:
            return ChallengeResult(detected=False, message='No active challenge')

        # Verify landmarks are valid
        if not landmarks or not landmarks.is_valid():
            return ChallengeResult(detected=False, message='No valid landmarks detected')

        challenge = self._state.current_challenge

        # Calculate metrics for eye/mouth challenges
        left_ear_raw = self._landmark_detector.calculate_ear(landmarks, Landmarks.LEFT_EYE)
        right_ear_raw = self._landmark_detector.calculate_ear(landmarks, Landmarks.RIGHT_EYE)
        mar_raw = self._landmark_detector.calculate_mar(landmarks)
        smile_raise, smile_width_raw = self._landmark_detector.calculate_smile(landmarks)

        # Apply temporal smoothing for stable detection
        left_ear = self._left_ear_smoother.smooth(left_ear_raw)
        right_ear = self._right_ear_smoother.smooth(right_ear_raw)
        avg_ear = (left_ear + right_ear) / 2
        mar = self._mar_smoother.smooth(mar_raw)
        smile_width = self._smile_smoother.smooth(smile_width_raw)
        brow_both, brow_left, brow_right, new_baseline = self._landmark_detector.calculate_eyebrow_raise(
            landmarks, self._state.baseline_eyebrow_dist
        )

        # Log challenge detection values (throttled to reduce spam)
        now = time.time()
        if not hasattr(self, '_last_log_time') or now - self._last_log_time > 0.5:
            self._last_log_time = now
            # Include brow ratios for eyebrow challenges
            brow_info = f", Brow L:{brow_left:.2f} R:{brow_right:.2f}" if 'BROW' in challenge.name else ""
            logger.info(f"[{challenge.name}] EAR L:{left_ear:.2f} R:{right_ear:.2f}, MAR:{mar:.2f}, Smile:{smile_width:.2f}, Yaw:{pose.yaw:.0f} Pitch:{pose.pitch:.0f}{brow_info}")
        if self._state.baseline_eyebrow_dist is None:
            self._state.baseline_eyebrow_dist = new_baseline

        # Track motion for dynamic challenges
        self._state.motion_history.append((pose.yaw, pose.pitch, time.time()))

        # Check based on challenge type
        detected, message = self._check_challenge(
            challenge, left_ear, right_ear, avg_ear, mar,
            smile_raise, smile_width, brow_both, brow_left, brow_right, pose
        )

        # Handle hold timer with hysteresis to prevent flickering
        if detected:
            # Reset miss counter on detection
            self._miss_count = 0

            if not self._state.action_detected:
                self._state.hold_start = time.time()
                self._state.action_detected = True

            hold_time = time.time() - self._state.hold_start
            progress = min(100, (hold_time / self._state.hold_duration) * 100)

            if hold_time >= self._state.hold_duration:
                # Challenge completed!
                is_puzzle_complete = self._state.advance()
                if is_puzzle_complete:
                    logger.info("Puzzle PASSED - All challenges completed!")
                return ChallengeResult(detected=True, progress=100, message='Completed!', completed=True)

            return ChallengeResult(detected=True, progress=progress, message=message)
        else:
            # Require 3 consecutive misses before resetting (prevents flickering)
            self._miss_count = getattr(self, '_miss_count', 0) + 1
            if self._miss_count >= 3:
                self._state.action_detected = False
                self._state.hold_start = time.time()
                return ChallengeResult(detected=False, progress=0, message=message)
            else:
                # Keep previous progress during brief detection gaps
                if self._state.action_detected:
                    hold_time = time.time() - self._state.hold_start
                    progress = min(100, (hold_time / self._state.hold_duration) * 100)
                    return ChallengeResult(detected=True, progress=progress, message="Keep holding...")
                return ChallengeResult(detected=False, progress=0, message=message)

    def _check_challenge(self, challenge: ChallengeType, left_ear: float, right_ear: float,
                         avg_ear: float, mar: float, smile_raise: float, smile_width: float,
                         brow_both: float, brow_left: float, brow_right: float,
                         pose: Pose) -> tuple:
        """Check specific challenge type.

        Note: In mirrored camera, user's left = MediaPipe's right (appears on left of screen)
        """
        T = self._thresholds

        # Swap for user perspective in mirrored camera
        user_left_ear = right_ear   # User's LEFT eye = MediaPipe RIGHT
        user_right_ear = left_ear   # User's RIGHT eye = MediaPipe LEFT

        if challenge == ChallengeType.BLINK:
            detected = avg_ear < T.EAR_CLOSED
            if detected:
                return True, "✓ Both eyes closed!"
            elif user_left_ear < T.EAR_CLOSED and user_right_ear > T.EAR_OPEN:
                return False, "Only LEFT closed! Close RIGHT eye too!"
            elif user_right_ear < T.EAR_CLOSED and user_left_ear > T.EAR_OPEN:
                return False, "Only RIGHT closed! Close LEFT eye too!"
            return False, f"EAR: {avg_ear:.2f} - Close BOTH eyes!"

        elif challenge == ChallengeType.CLOSE_LEFT:
            detected = user_left_ear < T.EAR_CLOSED and user_right_ear > T.EAR_OPEN
            if detected:
                return True, "✓ Left eye closed!"
            elif user_right_ear < T.EAR_CLOSED and user_left_ear > T.EAR_OPEN:
                return False, "WRONG! That's your RIGHT eye! Close LEFT!"
            elif user_left_ear < T.EAR_CLOSED and user_right_ear < T.EAR_CLOSED:
                return False, "OPEN your RIGHT eye! Only close LEFT!"
            return False, f"L:{user_left_ear:.2f} - Close your LEFT eye!"

        elif challenge == ChallengeType.CLOSE_RIGHT:
            detected = user_right_ear < T.EAR_CLOSED and user_left_ear > T.EAR_OPEN
            if detected:
                return True, "✓ Right eye closed!"
            elif user_left_ear < T.EAR_CLOSED and user_right_ear > T.EAR_OPEN:
                return False, "WRONG! That's your LEFT eye! Close RIGHT!"
            elif user_left_ear < T.EAR_CLOSED and user_right_ear < T.EAR_CLOSED:
                return False, "OPEN your LEFT eye! Only close RIGHT!"
            return False, f"R:{user_right_ear:.2f} - Close your RIGHT eye!"

        elif challenge == ChallengeType.SMILE:
            is_corners_raised = smile_raise > T.SMILE_CORNER
            is_mouth_wide = smile_width > T.SMILE_WIDTH
            detected = is_corners_raised and is_mouth_wide
            if detected:
                return True, "✓ Great smile!"
            elif is_mouth_wide and not is_corners_raised:
                return False, "Lift the corners of your mouth! Show teeth!"
            elif is_corners_raised and not is_mouth_wide:
                return False, f"Smile WIDER! W:{smile_width:.2f} need >{T.SMILE_WIDTH}"
            return False, f"SMILE! Show your teeth! W:{smile_width:.2f}"

        elif challenge == ChallengeType.OPEN_MOUTH:
            detected = mar > T.MOUTH_OPEN
            if detected:
                return True, "✓ Mouth open!"
            return False, f"Open: {mar:.2f} - Open WIDER! Need >{T.MOUTH_OPEN}"

        elif challenge == ChallengeType.TURN_LEFT:
            detected = pose.yaw < -T.YAW
            if detected:
                return True, "✓ Turned left!"
            elif pose.yaw > T.YAW:
                return False, f"WRONG WAY! Turn LEFT, not right! Yaw: {pose.yaw:.0f}°"
            return False, f"Yaw: {pose.yaw:.0f}° - Turn LEFT more!"

        elif challenge == ChallengeType.TURN_RIGHT:
            detected = pose.yaw > T.YAW
            if detected:
                return True, "✓ Turned right!"
            elif pose.yaw < -T.YAW:
                return False, f"WRONG WAY! Turn RIGHT, not left! Yaw: {pose.yaw:.0f}°"
            return False, f"Yaw: {pose.yaw:.0f}° - Turn RIGHT more!"

        elif challenge == ChallengeType.LOOK_UP:
            detected = pose.pitch < -T.PITCH
            if detected:
                return True, "✓ Looking up!"
            elif pose.pitch > T.PITCH:
                return False, f"WRONG WAY! Chin UP, not down! Pitch: {pose.pitch:.0f}°"
            return False, f"Pitch: {pose.pitch:.0f}° - Tilt chin UP more!"

        elif challenge == ChallengeType.LOOK_DOWN:
            detected = pose.pitch > T.PITCH
            if detected:
                return True, "✓ Looking down!"
            elif pose.pitch < -T.PITCH:
                return False, f"WRONG WAY! Chin DOWN, not up! Pitch: {pose.pitch:.0f}°"
            return False, f"Pitch: {pose.pitch:.0f}° - Tilt chin DOWN more!"

        elif challenge == ChallengeType.RAISE_BOTH_BROWS:
            detected = brow_both > T.EYEBROW_BOTH
            if detected:
                return True, "✓ Both brows raised!"
            elif brow_left > T.EYEBROW_SINGLE and brow_right < T.EYEBROW_BOTH:
                return False, "Only LEFT raised! Raise RIGHT brow too!"
            elif brow_right > T.EYEBROW_SINGLE and brow_left < T.EYEBROW_BOTH:
                return False, "Only RIGHT raised! Raise LEFT brow too!"
            return False, f"Both: {brow_both:.2f}x - Raise BOTH eyebrows!"

        elif challenge == ChallengeType.RAISE_LEFT_BROW:
            detected = brow_left > T.EYEBROW_SINGLE and brow_right < T.EYEBROW_BOTH
            if detected:
                return True, "✓ Left brow raised!"
            elif brow_right > T.EYEBROW_SINGLE and brow_left < T.EYEBROW_BOTH:
                return False, "WRONG! That's your RIGHT brow! Raise LEFT!"
            elif brow_left > T.EYEBROW_SINGLE and brow_right > T.EYEBROW_BOTH:
                return False, "LOWER your RIGHT brow! Only raise LEFT!"
            return False, f"L:{brow_left:.2f} - Raise your LEFT eyebrow!"

        elif challenge == ChallengeType.RAISE_RIGHT_BROW:
            detected = brow_right > T.EYEBROW_SINGLE and brow_left < T.EYEBROW_BOTH
            if detected:
                return True, "✓ Right brow raised!"
            elif brow_left > T.EYEBROW_SINGLE and brow_right < T.EYEBROW_BOTH:
                return False, "WRONG! That's your LEFT brow! Raise RIGHT!"
            elif brow_right > T.EYEBROW_SINGLE and brow_left > T.EYEBROW_BOTH:
                return False, "LOWER your LEFT brow! Only raise RIGHT!"
            return False, f"R:{brow_right:.2f} - Raise your RIGHT eyebrow!"

        elif challenge == ChallengeType.NOD:
            detected = self._check_nod()
            return detected, "Nod your head up and down" + (" ✓" if detected else "")

        elif challenge == ChallengeType.SHAKE_HEAD:
            detected = self._check_shake()
            return detected, "Shake your head left and right" + (" ✓" if detected else "")

        return False, "Unknown challenge"

    def _check_nod(self) -> bool:
        """Check for nodding motion (pitch oscillation)."""
        if len(self._state.motion_history) < 20:
            return False
        pitches = [p for _, p, _ in self._state.motion_history]
        return (max(pitches) - min(pitches)) > self._thresholds.NOD_RANGE

    def _check_shake(self) -> bool:
        """Check for head shake motion (yaw oscillation)."""
        if len(self._state.motion_history) < 20:
            return False
        yaws = [y for y, _, _ in self._state.motion_history]
        return (max(yaws) - min(yaws)) > self._thresholds.SHAKE_RANGE
