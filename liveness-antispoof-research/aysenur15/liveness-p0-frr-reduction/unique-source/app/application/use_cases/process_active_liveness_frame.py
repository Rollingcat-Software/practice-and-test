"""Use case for processing active liveness frames."""

import logging
import time

import cv2

from app.api.schemas.active_liveness import ActiveLivenessResponse
from app.application.services.active_liveness_manager import ActiveLivenessManager
from app.domain.interfaces.active_liveness_session_repository import IActiveLivenessSessionRepository

logger = logging.getLogger(__name__)


class ActiveLivenessSessionNotFoundError(Exception):
    """Raised when an active liveness session cannot be found."""


class ActiveLivenessSessionExpiredError(Exception):
    """Raised when an active liveness session has expired."""


class ProcessActiveLivenessFrameUseCase:
    """Process a frame for an existing active liveness session."""

    def __init__(
        self,
        manager: ActiveLivenessManager,
        session_repository: IActiveLivenessSessionRepository,
    ) -> None:
        self._manager = manager
        self._session_repository = session_repository
        logger.info("ProcessActiveLivenessFrameUseCase initialized")

    async def execute(
        self,
        session_id: str,
        image_path: str,
        frame_timestamp: float | None = None,
    ) -> ActiveLivenessResponse:
        normalized_frame_timestamp = self._normalize_and_validate_frame_timestamp(frame_timestamp)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load the uploaded image")

        async def handle_session(session):
            current_time = time.time()

            if self._manager.is_expired(session, now=current_time):
                raise ActiveLivenessSessionExpiredError("Active liveness session has expired")

            if session.is_complete:
                return self._manager.build_response(session=session)

            session.last_activity_at = current_time
            return await self._manager.process_frame(
                session=session,
                image=image,
                frame_timestamp=normalized_frame_timestamp,
            )

        try:
            response = await self._session_repository.mutate(session_id, handle_session)
        except ActiveLivenessSessionExpiredError:
            await self._session_repository.delete(session_id)
            raise
        if response is None:
            raise ActiveLivenessSessionNotFoundError("Active liveness session not found")
        if response.session_complete and response.session_passed:
            await self._session_repository.delete(session_id)

        return response

    @staticmethod
    def _normalize_and_validate_frame_timestamp(frame_timestamp: float | None) -> float:
        if frame_timestamp is None:
            raise ValueError("frame_timestamp is required")

        normalized = frame_timestamp / 1000.0 if frame_timestamp > 1_000_000_000_000 else frame_timestamp
        current_time = time.time()
        if normalized <= 0:
            raise ValueError("frame_timestamp must be a positive Unix timestamp")
        if normalized < 946684800:
            raise ValueError("frame_timestamp is too old")
        if normalized > current_time + 30:
            raise ValueError("frame_timestamp is too far in the future")
        return normalized
