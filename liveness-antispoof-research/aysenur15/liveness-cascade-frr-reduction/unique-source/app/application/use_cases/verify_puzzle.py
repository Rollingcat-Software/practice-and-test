"""Verify liveness puzzle use case.

This module provides the use case for verifying puzzle completion
with anti-replay protection and confidence validation.
"""
import base64
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional

from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.domain.entities.puzzle import Puzzle, VerificationResult
from app.domain.interfaces.puzzle_repository import IPuzzleRepository

logger = logging.getLogger(__name__)


class VerifyPuzzleUseCase:
    """Use case for verifying liveness puzzle completion.

    This use case handles:
    - Puzzle existence and expiration checks
    - Step sequence validation
    - Timestamp monotonicity (anti-replay)
    - Confidence scoring
    - Marking puzzle as completed

    Following Single Responsibility Principle: Only handles puzzle verification.
    Dependencies are injected for testability (Dependency Inversion Principle).
    """

    # Minimum confidence required per step
    MIN_STEP_CONFIDENCE = 0.6

    # Minimum step duration (anti-replay)
    MIN_STEP_DURATION_SECONDS = 0.5

    # Maximum clock skew allowed
    MAX_CLOCK_SKEW_SECONDS = 5.0

    # Overall pass threshold
    PASS_THRESHOLD = 0.6  # 60% overall score

    def __init__(
            self,
            puzzle_repository: IPuzzleRepository,
            spot_check_detector: Optional[ILivenessDetector] = None,
    ):
        self._repository = puzzle_repository
        self._spot_check_detector = spot_check_detector
        logger.info("VerifyPuzzleUseCase initialized")

    def _validate_timestamps(
        self,
        results: List[Dict],
        puzzle: Puzzle,
    ) -> List[str]:
        """Validate timestamp sequence for anti-replay.

        Args:
            results: List of step results with timestamps
            puzzle: Original puzzle

        Returns:
            List of reason codes for failures
        """
        reasons = []

        if not results:
            reasons.append("NO_RESULTS")
            return reasons

        puzzle_ts = puzzle.created_at.timestamp()

        # Check first timestamp is after puzzle creation
        first_start = results[0].get("start_timestamp", 0)
        if first_start < puzzle_ts - self.MAX_CLOCK_SKEW_SECONDS:
            reasons.append("TIMESTAMP_BEFORE_PUZZLE")

        # Check each step's timestamps
        for i, result in enumerate(results):
            start = result.get("start_timestamp", 0)
            end = result.get("end_timestamp", 0)

            # End must be after start
            if end <= start:
                reasons.append(f"STEP_{i}_END_BEFORE_START")
                continue

            # Minimum duration check
            duration = end - start
            if duration < self.MIN_STEP_DURATION_SECONDS:
                reasons.append(f"STEP_{i}_TOO_SHORT")

            # Next step must start after previous ends (with small tolerance)
            if i > 0:
                prev_end = results[i - 1].get("end_timestamp", 0)
                if start < prev_end - 0.1:  # 100ms tolerance for timing
                    reasons.append("TIMESTAMPS_OVERLAP")

        return reasons

    def _validate_confidence(
        self,
        results: List[Dict],
    ) -> tuple[int, List[str], float]:
        """Validate step confidence values.

        Args:
            results: List of step results with confidence

        Returns:
            Tuple of (steps_passed, reason_codes, total_confidence)
        """
        reasons = []
        steps_passed = 0
        total_confidence = 0.0

        for i, result in enumerate(results):
            confidence = result.get("confidence", 0)
            total_confidence += confidence

            if confidence >= self.MIN_STEP_CONFIDENCE:
                steps_passed += 1
            else:
                reasons.append(
                    f"STEP_{i}_LOW_CONFIDENCE:got={confidence:.2f},"
                    f"min={self.MIN_STEP_CONFIDENCE}"
                )

        return steps_passed, reasons, total_confidence

    def _calculate_completion_time(
        self,
        results: List[Dict],
    ) -> float:
        """Calculate total completion time.

        Args:
            results: List of step results with timestamps

        Returns:
            Completion time in seconds
        """
        if not results:
            return 0.0

        first_start = results[0].get("start_timestamp", 0)
        last_end = results[-1].get("end_timestamp", 0)

        return max(0.0, last_end - first_start)

    async def _run_spot_check(
            self,
            spot_frames: List[str],
    ) -> tuple[bool, str]:
        """Spot-check frames with passive liveness detector.

        Args:
            spot_frames: List of base64 encoded frames

        Returns:
            Tuple of (passed, reason_code)
        """
        if not self._spot_check_detector or not spot_frames:
            return True, ""

        failed_count = 0

        for i, frame_b64 in enumerate(spot_frames[:3]):  # max 3 frame
            try:
                frame_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    logger.warning(f"Spot-check frame {i} could not be decoded")
                    continue

                result = await self._spot_check_detector.check_liveness(frame)

                if not result.is_live:
                    failed_count += 1
                    logger.warning(
                        f"Spot-check frame {i} failed: score={result.liveness_score}"
                    )

            except Exception as e:
                logger.warning(f"Spot-check frame {i} error: {e}")
                continue

        if failed_count >= 2:
            return False, "SPOT_CHECK_FAILED"

        return True, ""

    async def execute(
            self,
            puzzle_id: str,
            results: List[Dict],
            final_frame: Optional[str] = None,
            spot_frames: Optional[List[str]] = None,
            client_meta: Optional[Dict] = None,
    ) -> VerificationResult:
        """Execute puzzle verification.
        Args:
            puzzle_id: Puzzle ID to verify
            results: List of step results from client
            final_frame: Optional base64 encoded final frame
            client_meta: Optional client metadata

        Returns:
            VerificationResult with outcome
        """
        logger.info(f"Verifying puzzle {puzzle_id}")
        reason_codes: List[str] = []

        # Get puzzle from repository
        puzzle = await self._repository.get(puzzle_id)

        if puzzle is None:
            logger.warning(f"Puzzle {puzzle_id} not found")
            return VerificationResult(
                success=False,
                liveness_confirmed=False,
                steps_completed=0,
                total_steps=0,
                completion_time_seconds=0,
                reason_codes=["PUZZLE_NOT_FOUND"],
                overall_score=0.0,
            )

        total_steps = len(puzzle.steps)

        # Check expiration
        if puzzle.is_expired():
            logger.warning(f"Puzzle {puzzle_id} expired")
            reason_codes.append("PUZZLE_EXPIRED")
            return VerificationResult(
                success=False,
                liveness_confirmed=False,
                steps_completed=0,
                total_steps=total_steps,
                completion_time_seconds=0,
                reason_codes=reason_codes,
                overall_score=0.0,
            )

        # Check already completed
        if puzzle.completed:
            logger.warning(f"Puzzle {puzzle_id} already completed")
            reason_codes.append("PUZZLE_ALREADY_COMPLETED")
            return VerificationResult(
                success=False,
                liveness_confirmed=False,
                steps_completed=total_steps,
                total_steps=total_steps,
                completion_time_seconds=puzzle.completion_time or 0,
                reason_codes=reason_codes,
                overall_score=0.0,
            )

        # Validate step sequence matches puzzle
        valid, step_reasons = puzzle.validate_steps(results)
        reason_codes.extend(step_reasons)

        # Validate timestamps (anti-replay)
        timestamp_reasons = self._validate_timestamps(results, puzzle)
        reason_codes.extend(timestamp_reasons)

        # Validate confidence
        steps_passed, confidence_reasons, total_confidence = self._validate_confidence(
            results
        )
        reason_codes.extend(confidence_reasons)

        # Calculate completion time
        completion_time = self._calculate_completion_time(results)

        # Calculate overall score
        if results:
            avg_confidence = total_confidence / len(results)
            overall_score = avg_confidence * 100  # Convert to 0-100
        else:
            overall_score = 0.0

        # Spot-check — optional server-side frame verification
        spot_check_passed, spot_reason = await self._run_spot_check(spot_frames or [])
        if not spot_check_passed:
            reason_codes.append(spot_reason)

        # Determine liveness
        liveness_confirmed = (
            steps_passed == total_steps
            and len(reason_codes) == 0
            and overall_score >= self.PASS_THRESHOLD * 100
        )

        success = len(reason_codes) == 0

        # Mark puzzle as completed if successful
        if liveness_confirmed:
            await self._repository.mark_completed(puzzle_id, completion_time)
            logger.info(
                f"Puzzle {puzzle_id} verified successfully: "
                f"score={overall_score:.1f}, time={completion_time:.2f}s"
            )
        else:
            logger.info(
                f"Puzzle {puzzle_id} verification failed: "
                f"reasons={reason_codes}, score={overall_score:.1f}"
            )

        return VerificationResult(
            success=success,
            liveness_confirmed=liveness_confirmed,
            steps_completed=steps_passed,
            total_steps=total_steps,
            completion_time_seconds=completion_time,
            reason_codes=reason_codes,
            overall_score=overall_score,
        )
