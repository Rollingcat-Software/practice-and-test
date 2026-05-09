"""Liveness check API routes."""

import logging

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from app.api.schemas.active_liveness import ActiveLivenessResponse, ActiveLivenessStartRequest
from app.api.schemas.liveness import LivenessResponse
from app.application.use_cases.process_active_liveness_frame import (
    ActiveLivenessSessionExpiredError,
    ActiveLivenessSessionNotFoundError,
    ProcessActiveLivenessFrameUseCase,
)
from app.application.use_cases.start_active_liveness import StartActiveLivenessUseCase
from app.application.use_cases.check_liveness import CheckLivenessUseCase
from app.core.container import (
    get_check_liveness_use_case,
    get_file_storage,
    get_process_active_liveness_frame_use_case,
    get_start_active_liveness_use_case,
)
from app.core.config import get_settings
from app.core.validation import ValidationError, validate_image_file
from app.domain.interfaces.file_storage import IFileStorage

settings = get_settings()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Liveness"])


@router.post("/liveness", response_model=LivenessResponse, status_code=200)
async def check_liveness(
    file: UploadFile = File(..., description="Face image file"),
    use_case: CheckLivenessUseCase = Depends(get_check_liveness_use_case),
    storage: IFileStorage = Depends(get_file_storage),
) -> LivenessResponse:
    """Check liveness of a face.

    This endpoint:
    1. Detects face in image
    2. Performs liveness check
    3. Returns liveness result

    Args:
        file: Face image file (JPEG/PNG)
        use_case: Injected liveness check use case
        storage: Injected file storage

    Returns:
        LivenessResponse with liveness check result

    Raises:
        HTTPException 400: Bad request (no face, multiple faces)
        HTTPException 500: Internal server error

    Note:
        Currently uses stub liveness detector.
        Will be updated in Sprint 3 with real smile/blink detection.
    """
    image_path = None

    try:
        logger.info("Liveness check request")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file temporarily
        image_path = await storage.save_temp(file)

        # SECURITY: Validate actual file type using magic bytes (not just Content-Type header)
        try:
            detected_format = validate_image_file(image_path, allowed_formats=settings.ALLOWED_IMAGE_FORMATS)
            logger.debug(f"File type validated: {detected_format}")
        except ValidationError as e:
            logger.warning(f"File type validation failed: {str(e)}")
            await storage.cleanup(image_path)
            raise HTTPException(status_code=400, detail=str(e))

        # Execute liveness check use case
        result = await use_case.execute(image_path=image_path)

        message = "Liveness check passed" if result.is_live else "Liveness check failed"

        return LivenessResponse(
            is_live=result.is_live,
            score=result.score,
            confidence=result.confidence,
            challenge=result.challenge,
            challenge_completed=result.challenge_completed,
            message=message,
        )

    finally:
        # Cleanup temporary file
        if image_path:
            await storage.cleanup(image_path)


@router.post("/liveness/active/start", response_model=ActiveLivenessResponse, status_code=200)
async def start_active_liveness(
    request: ActiveLivenessStartRequest | None = Body(default=None),
    use_case: StartActiveLivenessUseCase = Depends(get_start_active_liveness_use_case),
) -> ActiveLivenessResponse:
    """Start a new active liveness session."""

    return await use_case.execute(config=request)


@router.post("/liveness/active/frame", response_model=ActiveLivenessResponse, status_code=200)
async def process_active_liveness_frame(
    session_id: str = Form(..., description="Active liveness session ID"),
    frame_timestamp: float = Form(..., description="Client frame capture time as Unix seconds or milliseconds"),
    image: UploadFile = File(..., description="Frame image file"),
    use_case: ProcessActiveLivenessFrameUseCase = Depends(get_process_active_liveness_frame_use_case),
    storage: IFileStorage = Depends(get_file_storage),
) -> ActiveLivenessResponse:
    """Process a frame for an existing active liveness session."""

    image_path = None
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_path = await storage.save_temp(image)

        try:
            validate_image_file(image_path, allowed_formats=settings.ALLOWED_IMAGE_FORMATS)
        except ValidationError as exc:
            await storage.cleanup(image_path)
            raise HTTPException(status_code=400, detail=str(exc))

        return await use_case.execute(
            session_id=session_id,
            image_path=image_path,
            frame_timestamp=frame_timestamp,
        )
    except ActiveLivenessSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ActiveLivenessSessionExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if image_path:
            await storage.cleanup(image_path)
