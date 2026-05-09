"""Preview-only biometric puzzle controller for the dev liveness window."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.api.schemas.active_liveness import ChallengeType
from app.application.use_cases.generate_puzzle import GeneratePuzzleUseCase
from app.domain.entities.puzzle import Puzzle, PuzzleDifficulty
from app.domain.interfaces.puzzle_repository import IPuzzleRepository


@dataclass(frozen=True)
class PreviewPuzzleSummary:
    status: str
    current_step: str
    progress: float
    completed_steps: int
    total_steps: int
    active_evidence: float
    confidence: float
    success: bool
    fusion_active: bool
    sequence_label: str


class _InMemoryPuzzleRepository(IPuzzleRepository):
    def __init__(self) -> None:
        self._items: dict[str, Puzzle] = {}

    async def save(self, puzzle: Puzzle) -> None:
        self._items[puzzle.puzzle_id] = puzzle

    async def get(self, puzzle_id: str) -> Optional[Puzzle]:
        return self._items.get(puzzle_id)

    async def delete(self, puzzle_id: str) -> bool:
        return self._items.pop(puzzle_id, None) is not None

    async def exists(self, puzzle_id: str) -> bool:
        return puzzle_id in self._items

    async def mark_completed(self, puzzle_id: str, completion_time: float) -> bool:
        puzzle = self._items.get(puzzle_id)
        if puzzle is None:
            return False
        puzzle.mark_completed(completion_time)
        return True


class PreviewBiometricPuzzleController:
    """Manage biometric puzzle state for the developer live preview."""

    SUPPORTED_ACTIONS: Sequence[ChallengeType] = (
        ChallengeType.BLINK,
        ChallengeType.SMILE,
        ChallengeType.TURN_LEFT,
        ChallengeType.TURN_RIGHT,
        ChallengeType.OPEN_MOUTH,
    )

    STEP_COMPLETE_THRESHOLD = 0.72
    STEP_TIMEOUT_GRACE_SECONDS = 0.2

    def __init__(self) -> None:
        self._repository = _InMemoryPuzzleRepository()
        self._generate_use_case = GeneratePuzzleUseCase(self._repository)
        self._generate_use_case._get_available_challenges = lambda: list(self.SUPPORTED_ACTIONS)
        self._puzzle: Optional[Puzzle] = None
        self._status = "idle"
        self._current_step_index = 0
        self._current_step_started_at: Optional[float] = None
        self._session_started_at: Optional[float] = None
        self._session_timeout_seconds = 60.0
        self._last_step_confidence = 0.0
        self._step_support_history: list[float] = []

    def start_session(
        self,
        *,
        difficulty: str = "standard",
        min_steps: int = 3,
        max_steps: int = 4,
        timeout_seconds: int = 60,
    ) -> PreviewPuzzleSummary:
        self.reset()
        self._puzzle = asyncio.run(
            self._generate_use_case.execute(
                difficulty=difficulty,
                min_steps=min_steps,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        )
        self._status = "running"
        self._session_started_at = time.time()
        self._session_timeout_seconds = float(timeout_seconds)
        self._current_step_started_at = self._session_started_at
        return self._build_summary(active_evidence=0.0, confidence=0.0)

    def reset(self) -> None:
        self._puzzle = None
        self._status = "idle"
        self._current_step_index = 0
        self._current_step_started_at = None
        self._session_started_at = None
        self._session_timeout_seconds = 60.0
        self._last_step_confidence = 0.0
        self._step_support_history = []

    def evaluate(
        self,
        *,
        frame_timestamp: float,
        current_frame_details: dict[str, Any],
        temporal_signal_summary: dict[str, Any],
    ) -> PreviewPuzzleSummary:
        if self._puzzle is None:
            return self._build_summary(active_evidence=0.0, confidence=0.0)
        if self._status in {"completed", "failed", "timed_out"}:
            evidence = 1.0 if self._status == "completed" else 0.0
            confidence = 1.0 if self._status == "completed" else 0.0
            return self._build_summary(active_evidence=evidence, confidence=confidence)

        current_step = self._get_current_step()
        if current_step is None:
            self._status = "completed"
            return self._build_summary(active_evidence=1.0, confidence=1.0)

        if self._is_session_timed_out(frame_timestamp):
            self._status = "timed_out"
            return self._build_summary(active_evidence=0.0, confidence=0.0)

        if self._is_step_timed_out(frame_timestamp, current_step.duration_seconds):
            self._status = "timed_out"
            return self._build_summary(active_evidence=0.0, confidence=0.0)

        step_confidence = self._evaluate_step_confidence(
            action=current_step.action,
            current_frame_details=current_frame_details,
            temporal_signal_summary=temporal_signal_summary,
        )
        self._last_step_confidence = step_confidence
        self._step_support_history.append(step_confidence)
        if step_confidence >= self.STEP_COMPLETE_THRESHOLD:
            self._current_step_index += 1
            self._current_step_started_at = frame_timestamp
            if self._current_step_index >= len(self._puzzle.steps):
                self._status = "completed"

        progress = self._current_step_index / max(len(self._puzzle.steps), 1)
        puzzle_active_evidence = 1.0 if self._status == "completed" else _clamp01(0.75 * progress + 0.25 * step_confidence)
        return self._build_summary(active_evidence=puzzle_active_evidence, confidence=step_confidence)

    def _get_current_step(self):
        if self._puzzle is None or self._current_step_index >= len(self._puzzle.steps):
            return None
        return self._puzzle.steps[self._current_step_index]

    def _is_step_timed_out(self, frame_timestamp: float, step_duration_seconds: float) -> bool:
        if self._current_step_started_at is None:
            return False
        return frame_timestamp - self._current_step_started_at > step_duration_seconds + self.STEP_TIMEOUT_GRACE_SECONDS

    def _is_session_timed_out(self, frame_timestamp: float) -> bool:
        if self._session_started_at is None:
            return False
        return frame_timestamp - self._session_started_at > self._session_timeout_seconds

    def _evaluate_step_confidence(
        self,
        *,
        action: str,
        current_frame_details: dict[str, Any],
        temporal_signal_summary: dict[str, Any],
    ) -> float:
        if action == ChallengeType.BLINK.value:
            return _safe_float(temporal_signal_summary.get("blink_evidence"))
        if action == ChallengeType.SMILE.value:
            return _safe_float(temporal_signal_summary.get("smile_evidence"))
        if action == ChallengeType.OPEN_MOUTH.value:
            return _safe_float(temporal_signal_summary.get("mouth_open_evidence"))
        if action == ChallengeType.TURN_LEFT.value:
            return _safe_float(temporal_signal_summary.get("head_turn_left_evidence"))
        if action == ChallengeType.TURN_RIGHT.value:
            return _safe_float(temporal_signal_summary.get("head_turn_right_evidence"))
        return 0.0

    def _build_summary(self, *, active_evidence: float, confidence: float) -> PreviewPuzzleSummary:
        total_steps = len(self._puzzle.steps) if self._puzzle is not None else 0
        current_step = self._get_current_step()
        sequence_label = " -> ".join(step.action for step in self._puzzle.steps) if self._puzzle is not None else "-"
        return PreviewPuzzleSummary(
            status=self._status,
            current_step=current_step.action if current_step is not None else "-",
            progress=self._current_step_index / max(total_steps, 1) if total_steps else 0.0,
            completed_steps=self._current_step_index,
            total_steps=total_steps,
            active_evidence=_clamp01(active_evidence),
            confidence=_clamp01(confidence),
            success=self._status == "completed",
            fusion_active=self._status in {"running", "completed"},
            sequence_label=sequence_label,
        )


def _safe_float(value: Any) -> float:
    try:
        return _clamp01(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
