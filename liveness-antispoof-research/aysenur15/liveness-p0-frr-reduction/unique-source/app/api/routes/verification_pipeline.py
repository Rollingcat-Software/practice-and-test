"""Identity verification pipeline API routes.

Phase 8B: Document Processing (document-scan, data-extract)
Phase 8C: Face-to-Document Matching (face-match, liveness-check, pipeline test)
Phase 8D: Video Interview (upload and store for manual admin review)

These endpoints compose existing infrastructure (YOLO card detection, DeepFace,
liveness detection) into a sequential verification pipeline.
"""

import asyncio
import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.core.config import get_settings

from app.core.container import (
    get_check_liveness_use_case,
    get_detect_card_type_use_case,
    get_embedding_extractor,
    get_face_detector,
    get_file_storage,
    get_similarity_calculator,
)
from app.domain.services.document_ocr import DocumentOCR
from app.domain.services.mrz_parser import (
    MRZData,
    detect_and_parse_mrz,
    format_date,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/verification",
    tags=["Verification Pipeline"],
)


# ============================================================================
# Response Models
# ============================================================================


class BoundingBoxModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DocumentScanResponse(BaseModel):
    """Response from document scan step."""

    detected: bool
    card_type: Optional[str] = None
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBoxModel] = None
    cropped_document_image_base64: Optional[str] = None


class ExtractedData(BaseModel):
    """Extracted data from a document."""

    name: Optional[str] = None
    surname: Optional[str] = None
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    sex: Optional[str] = None
    given_names: Optional[str] = None
    mrz_line1: Optional[str] = None
    mrz_line2: Optional[str] = None
    mrz_line3: Optional[str] = None
    mrz_valid: Optional[bool] = None


class DataExtractResponse(BaseModel):
    """Response from data extraction step."""

    document_type: Optional[str] = None
    extracted_data: ExtractedData
    confidence: float
    method: str  # "mrz_parse" | "tesseract_ocr" | "card_type_only"


class FaceMatchResponse(BaseModel):
    """Response from face matching step."""

    match: bool
    confidence: float
    threshold: float
    distance: float
    similarity: float


class LivenessCheckResponse(BaseModel):
    """Response from liveness check step."""

    is_live: bool
    confidence: float
    method: str


class PipelineStepResult(BaseModel):
    """Result of a single pipeline step."""

    step: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class PipelineTestResponse(BaseModel):
    """Response from the full pipeline test endpoint."""

    overall_success: bool
    steps: list[PipelineStepResult]


class VideoInterviewResponse(BaseModel):
    """Response from video interview upload."""

    stored: bool
    filename: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "PENDING_REVIEW"
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


async def _read_image_from_upload(file: UploadFile) -> tuple[Image.Image, np.ndarray]:
    """Read an uploaded file into PIL Image and numpy array.

    Args:
        file: Uploaded image file

    Returns:
        Tuple of (PIL Image, numpy array in RGB)

    Raises:
        HTTPException: If the file is not a valid image
    """
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize if too large
    max_dim = 1280
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))

    img_np = np.array(image)
    return image, img_np


def _decode_base64_image(b64_string: str) -> tuple[Image.Image, np.ndarray]:
    """Decode a base64 string into PIL Image and numpy array.

    Args:
        b64_string: Base64-encoded image data (may include data URI prefix)

    Returns:
        Tuple of (PIL Image, numpy array in RGB)

    Raises:
        HTTPException: If decoding fails
    """
    # Strip data URI prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    max_dim = 1280
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))

    img_np = np.array(image)
    return image, img_np


def _image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL Image as base64 string.

    Args:
        image: PIL Image
        format: Image format (JPEG, PNG)

    Returns:
        Base64-encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================================
# 1. DOCUMENT_SCAN Step
# ============================================================================


@router.post(
    "/document-scan",
    response_model=DocumentScanResponse,
    summary="Scan document and detect card type",
    description=(
        "Accepts a document image, runs YOLO card detection, "
        "and returns the detected card type with a cropped document image."
    ),
)
async def document_scan(
    file: UploadFile = File(..., description="Document image file"),
) -> DocumentScanResponse:
    """Scan a document image to detect and classify the card type.

    Uses the existing YOLO card detection model to identify the document,
    then crops the detected region for subsequent processing steps.
    """
    image, img_np = await _read_image_from_upload(file)

    use_case = get_detect_card_type_use_case()

    try:
        result = await asyncio.wait_for(use_case.execute_from_array(img_np), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Document scan timed out")

    if not result.detected:
        return DocumentScanResponse(detected=False)

    # Get bounding box from YOLO and crop
    # Re-run YOLO directly to get bounding box coordinates
    from app.infrastructure.ml.card_type.yolo_card_type_detector import _get_yolo_model, DEFAULT_MODEL_PATH

    model = _get_yolo_model(str(DEFAULT_MODEL_PATH))
    yolo_results = model(img_np, conf=0.5, verbose=False)
    yolo_result = yolo_results[0]

    bounding_box = None
    cropped_b64 = None

    if len(yolo_result.boxes) > 0:
        best_box = max(yolo_result.boxes, key=lambda b: float(b.conf[0]))
        # xyxy format: [x1, y1, x2, y2]
        coords = best_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        w = x2 - x1
        h = y2 - y1

        bounding_box = BoundingBoxModel(x=x1, y=y1, width=w, height=h)

        # Crop the document region
        cropped_img = image.crop((x1, y1, x2, y2))
        cropped_b64 = _image_to_base64(cropped_img)

    return DocumentScanResponse(
        detected=True,
        card_type=result.class_name,
        confidence=round(result.confidence, 4) if result.confidence else None,
        bounding_box=bounding_box,
        cropped_document_image_base64=cropped_b64,
    )


# ============================================================================
# 2. DATA_EXTRACT Step (MRZ-based)
# ============================================================================


@router.post(
    "/data-extract",
    response_model=DataExtractResponse,
    summary="Extract data from document image",
    description=(
        "Extracts identity data from a document image. "
        "For documents with MRZ (passports, some ID cards): parses MRZ lines via regex. "
        "For Turkish ID cards (TC Kimlik) without MRZ: returns document type for frontend field capture."
    ),
)
async def data_extract(
    file: Optional[UploadFile] = File(default=None, description="Document image file"),
    image_base64: Optional[str] = Form(default=None, description="Document image as base64"),
    mrz_text: Optional[str] = Form(
        default=None,
        description="Raw MRZ text (if already extracted by client/OCR). 2-3 lines separated by newline.",
    ),
) -> DataExtractResponse:
    """Extract identity data from a document.

    Supports three input modes:
    1. MRZ text provided directly (most reliable -- client-side OCR or manual input)
    2. Image file upload (runs card detection to determine document type)
    3. Base64 image (same as file upload)

    For MRZ parsing: standardized regex-based extraction of TD1/TD3 formats.
    For non-MRZ documents: returns document type so the frontend can handle field capture.
    """
    # If MRZ text is provided directly, parse it
    if mrz_text:
        mrz_data = detect_and_parse_mrz(mrz_text)
        if mrz_data:
            return _mrz_to_response(mrz_data)
        else:
            raise HTTPException(
                status_code=400,
                detail="Could not parse MRZ from provided text. Expected 2 lines x 44 chars (passport) or 3 lines x 30 chars (ID card).",
            )

    # Determine document type from image
    if file:
        pil_img, img_np = await _read_image_from_upload(file)
    elif image_base64:
        pil_img, img_np = _decode_base64_image(image_base64)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload, image_base64, or mrz_text",
        )

    # Run card detection to identify the document type
    use_case = get_detect_card_type_use_case()
    try:
        result = await asyncio.wait_for(use_case.execute_from_array(img_np), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Card detection timed out")

    if not result.detected:
        return DataExtractResponse(
            document_type=None,
            extracted_data=ExtractedData(),
            confidence=0.0,
            method="card_type_only",
        )

    doc_type_map = {
        "tc_kimlik": "turkish_id",
        "pasaport": "passport",
        "ehliyet": "drivers_license",
        "ogrenci_karti": "student_id",
        "akademisyen_karti": "academic_id",
    }

    document_type = doc_type_map.get(result.class_name, result.class_name)

    # For TC Kimlik (no MRZ): run Tesseract OCR to extract fields
    if document_type == "turkish_id":
        try:
            ocr = DocumentOCR()
            ocr_result = ocr.extract_tc_kimlik(pil_img)
            if ocr_result.confidence > 0.0:
                f = ocr_result.fields
                extracted = ExtractedData(
                    id_number=f.get("tc_number"),
                    surname=f.get("surname"),
                    name=f.get("name"),
                    given_names=f.get("name"),
                    date_of_birth=f.get("date_of_birth"),
                    expiry_date=f.get("expiry_date"),
                    sex=f.get("gender"),
                    nationality=f.get("nationality"),
                )
                return DataExtractResponse(
                    document_type=document_type,
                    extracted_data=extracted,
                    confidence=ocr_result.confidence,
                    method="tesseract_ocr",
                )
        except Exception as e:
            logger.warning(f"OCR extraction failed for TC Kimlik, falling back: {e}")

    # Fallback: return document type only (frontend handles field capture)
    return DataExtractResponse(
        document_type=document_type,
        extracted_data=ExtractedData(),
        confidence=round(result.confidence, 4) if result.confidence else 0.0,
        method="card_type_only",
    )


def _mrz_to_response(mrz: MRZData) -> DataExtractResponse:
    """Convert parsed MRZ data to API response.

    Args:
        mrz: Parsed MRZ data

    Returns:
        DataExtractResponse
    """
    doc_type_map = {
        "P": "passport",
        "I": "id_card",
        "ID": "id_card",
        "A": "id_card",
        "C": "id_card",
    }

    document_type = doc_type_map.get(mrz.document_type, f"unknown_{mrz.document_type}")

    extracted = ExtractedData(
        surname=mrz.surname or None,
        given_names=mrz.given_names or None,
        name=f"{mrz.given_names} {mrz.surname}".strip() or None,
        id_number=mrz.document_number or None,
        date_of_birth=format_date(mrz.date_of_birth) if mrz.date_of_birth else None,
        expiry_date=format_date(mrz.expiry_date) if mrz.expiry_date else None,
        nationality=mrz.nationality or None,
        sex=mrz.sex or None,
        mrz_valid=mrz.check_digits_valid,
    )

    # Include raw MRZ lines
    if mrz.raw_mrz:
        if len(mrz.raw_mrz) >= 1:
            extracted.mrz_line1 = mrz.raw_mrz[0]
        if len(mrz.raw_mrz) >= 2:
            extracted.mrz_line2 = mrz.raw_mrz[1]
        if len(mrz.raw_mrz) >= 3:
            extracted.mrz_line3 = mrz.raw_mrz[2]

    # Confidence: 1.0 if all check digits pass, 0.7 otherwise
    confidence = 1.0 if mrz.check_digits_valid else 0.7

    return DataExtractResponse(
        document_type=document_type,
        extracted_data=extracted,
        confidence=confidence,
        method="mrz_parse",
    )


# ============================================================================
# 3. FACE_MATCH Step
# ============================================================================


@router.post(
    "/face-match",
    response_model=FaceMatchResponse,
    summary="Match a live face against a document face",
    description=(
        "Computes face similarity between a live selfie and a document photo. "
        "Uses DeepFace embeddings and cosine distance."
    ),
)
async def face_match(
    live_face_image: str = Form(..., description="Live face image as base64"),
    document_face_image: str = Form(..., description="Document face image as base64"),
    threshold: float = Query(default=0.6, ge=0.0, le=1.0, description="Match threshold"),
) -> FaceMatchResponse:
    """Compare a live face with a document face photo.

    Both images should contain a single face. Uses the existing DeepFace
    embedding extractor and cosine similarity calculator.
    """
    # Decode images
    _, live_np = _decode_base64_image(live_face_image)
    _, doc_np = _decode_base64_image(document_face_image)

    detector = get_face_detector()
    extractor = get_embedding_extractor()
    similarity_calc = get_similarity_calculator()

    # Detect faces in both images
    live_detection = await detector.detect(live_np)
    if not live_detection.found:
        raise HTTPException(status_code=400, detail="No face detected in live image")

    doc_detection = await detector.detect(doc_np)
    if not doc_detection.found:
        raise HTTPException(status_code=400, detail="No face detected in document image")

    # Extract face regions
    live_face = live_detection.get_face_region(live_np)
    doc_face = doc_detection.get_face_region(doc_np)

    # Compute embeddings
    live_embedding = await extractor.extract(live_face)
    doc_embedding = await extractor.extract(doc_face)

    # Calculate distance (lower = more similar)
    distance = similarity_calc.calculate(live_embedding, doc_embedding)
    similarity = max(0.0, 1.0 - distance)
    match = distance < threshold

    logger.info(
        f"Face match result: match={match}, similarity={similarity:.4f}, "
        f"distance={distance:.4f}, threshold={threshold}"
    )

    return FaceMatchResponse(
        match=match,
        confidence=round(similarity, 4),
        threshold=threshold,
        distance=round(distance, 4),
        similarity=round(similarity, 4),
    )


# ============================================================================
# 4. LIVENESS_CHECK as Pipeline Step
# ============================================================================


@router.post(
    "/liveness-check",
    response_model=LivenessCheckResponse,
    summary="Quick liveness assessment for pipeline",
    description=(
        "Single-frame liveness check for the verification pipeline. "
        "Uses the existing EnhancedLivenessDetector with DeepFace anti-spoofing."
    ),
)
async def liveness_check(
    file: UploadFile = File(..., description="Face image file"),
) -> LivenessCheckResponse:
    """Perform a single-frame liveness check.

    Uses the existing liveness detection infrastructure (texture analysis
    + optional DeepFace anti-spoofing) to assess whether the image is
    of a live person.
    """
    storage = get_file_storage()
    image_path = None

    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save to temp for the liveness use case (expects file path)
        image_path = await storage.save_temp(file)

        use_case = get_check_liveness_use_case()
        result = await use_case.execute(image_path=image_path)

        return LivenessCheckResponse(
            is_live=result.is_live,
            confidence=round(result.confidence, 4),
            method="enhanced_liveness" + (
                "+deepface_antispoof"
                if result.details.get("antispoof_label") is not None
                else ""
            ),
        )
    finally:
        if image_path:
            await storage.cleanup(image_path)


# ============================================================================
# 5. Full Pipeline Test Endpoint
# ============================================================================


@router.post(
    "/pipeline/test",
    response_model=PipelineTestResponse,
    summary="Run the full verification pipeline",
    description=(
        "Runs the complete identity verification pipeline in sequence: "
        "document scan, data extract, face match, liveness check. "
        "Intended for testing the pipeline end-to-end in one call."
    ),
)
async def pipeline_test(
    document_image: UploadFile = File(..., description="Document image file"),
    face_image: UploadFile = File(..., description="Live face image file"),
    mrz_text: Optional[str] = Form(default=None, description="Optional MRZ text from client-side OCR"),
    threshold: float = Query(default=0.6, ge=0.0, le=1.0, description="Face match threshold"),
) -> PipelineTestResponse:
    """Run the full verification pipeline in one call.

    Steps executed in sequence:
    1. Document Scan - detect and classify the document
    2. Data Extract - extract identity data (MRZ or type-only)
    3. Face Match - compare live face with document face
    4. Liveness Check - verify the face image is of a live person

    Each step reports success/failure independently. The pipeline continues
    even if individual steps fail, so the caller gets a complete picture.
    """
    steps: list[PipelineStepResult] = []
    overall_success = True

    # Read both images upfront
    doc_content = await document_image.read()
    face_content = await face_image.read()

    try:
        doc_pil = Image.open(BytesIO(doc_content)).convert("RGB")
    except Exception:
        return PipelineTestResponse(
            overall_success=False,
            steps=[PipelineStepResult(step="document_scan", success=False, error="Invalid document image")],
        )

    try:
        face_pil = Image.open(BytesIO(face_content)).convert("RGB")
    except Exception:
        return PipelineTestResponse(
            overall_success=False,
            steps=[PipelineStepResult(step="face_match", success=False, error="Invalid face image")],
        )

    max_dim = 1280
    if max(doc_pil.size) > max_dim:
        doc_pil.thumbnail((max_dim, max_dim))
    if max(face_pil.size) > max_dim:
        face_pil.thumbnail((max_dim, max_dim))

    doc_np = np.array(doc_pil)
    face_np = np.array(face_pil)

    # ---- Step 1: Document Scan ----
    card_type = None
    try:
        use_case = get_detect_card_type_use_case()
        card_result = await asyncio.wait_for(use_case.execute_from_array(doc_np), timeout=30)

        if card_result.detected:
            card_type = card_result.class_name

            # Attempt to get cropped document
            from app.infrastructure.ml.card_type.yolo_card_type_detector import _get_yolo_model, DEFAULT_MODEL_PATH
            model = _get_yolo_model(str(DEFAULT_MODEL_PATH))
            yolo_results = model(doc_np, conf=0.5, verbose=False)
            if len(yolo_results[0].boxes) > 0:
                best_box = max(yolo_results[0].boxes, key=lambda b: float(b.conf[0]))
                coords = best_box.xyxy[0].cpu().numpy().astype(int)
                doc_pil.crop((int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])))

            steps.append(PipelineStepResult(
                step="document_scan",
                success=True,
                data={"card_type": card_type, "confidence": round(card_result.confidence, 4)},
            ))
        else:
            overall_success = False
            steps.append(PipelineStepResult(
                step="document_scan",
                success=False,
                error="No document detected in image",
            ))
    except Exception as e:
        overall_success = False
        steps.append(PipelineStepResult(step="document_scan", success=False, error=str(e)))

    # ---- Step 2: Data Extract ----
    try:
        if mrz_text:
            mrz_data = detect_and_parse_mrz(mrz_text)
            if mrz_data:
                resp = _mrz_to_response(mrz_data)
                steps.append(PipelineStepResult(
                    step="data_extract",
                    success=True,
                    data={
                        "document_type": resp.document_type,
                        "method": resp.method,
                        "mrz_valid": mrz_data.check_digits_valid,
                        "surname": mrz_data.surname,
                        "given_names": mrz_data.given_names,
                    },
                ))
            else:
                steps.append(PipelineStepResult(
                    step="data_extract",
                    success=False,
                    error="Could not parse MRZ text",
                ))
                overall_success = False
        else:
            # No MRZ provided - try OCR for TC Kimlik, otherwise report type only
            doc_type_map = {
                "tc_kimlik": "turkish_id",
                "pasaport": "passport",
                "ehliyet": "drivers_license",
                "ogrenci_karti": "student_id",
            }
            doc_type = doc_type_map.get(card_type, card_type) if card_type else None

            ocr_success = False
            if doc_type == "turkish_id":
                try:
                    ocr = DocumentOCR()
                    ocr_result = ocr.extract_tc_kimlik(doc_pil)
                    if ocr_result.confidence > 0.0:
                        steps.append(PipelineStepResult(
                            step="data_extract",
                            success=True,
                            data={
                                "document_type": doc_type,
                                "method": "tesseract_ocr",
                                "confidence": ocr_result.confidence,
                                **{k: v for k, v in ocr_result.fields.items() if v},
                            },
                        ))
                        ocr_success = True
                except Exception as ocr_err:
                    logger.warning(f"Pipeline OCR failed: {ocr_err}")

            if not ocr_success:
                steps.append(PipelineStepResult(
                    step="data_extract",
                    success=bool(doc_type),
                    data={"document_type": doc_type, "method": "card_type_only"},
                ))
            if not doc_type and not ocr_success:
                overall_success = False
    except Exception as e:
        overall_success = False
        steps.append(PipelineStepResult(step="data_extract", success=False, error=str(e)))

    # ---- Step 3: Face Match ----
    try:
        detector = get_face_detector()
        extractor = get_embedding_extractor()
        sim_calc = get_similarity_calculator()

        # Detect face in live image
        live_detection = await detector.detect(face_np)
        if not live_detection.found:
            raise ValueError("No face detected in live image")

        # Detect face in document image
        doc_detection = await detector.detect(doc_np)
        if not doc_detection.found:
            raise ValueError("No face detected in document image")

        live_face = live_detection.get_face_region(face_np)
        doc_face = doc_detection.get_face_region(doc_np)

        live_emb = await extractor.extract(live_face)
        doc_emb = await extractor.extract(doc_face)

        distance = sim_calc.calculate(live_emb, doc_emb)
        similarity = max(0.0, 1.0 - distance)
        match = distance < threshold

        steps.append(PipelineStepResult(
            step="face_match",
            success=match,
            data={
                "match": match,
                "similarity": round(similarity, 4),
                "distance": round(distance, 4),
                "threshold": threshold,
            },
        ))
        if not match:
            overall_success = False
    except Exception as e:
        overall_success = False
        steps.append(PipelineStepResult(step="face_match", success=False, error=str(e)))

    # ---- Step 4: Liveness Check ----
    try:
        # Write face image to temp file for liveness use case
        face_buffer = BytesIO()
        face_pil.save(face_buffer, format="JPEG")
        face_buffer.seek(0)

        # Create a temporary UploadFile-like object
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(face_buffer.getvalue())
            tmp_path = tmp.name

        liveness_uc = get_check_liveness_use_case()
        liveness_result = await liveness_uc.execute(image_path=tmp_path)

        # Cleanup temp file
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        steps.append(PipelineStepResult(
            step="liveness_check",
            success=liveness_result.is_live,
            data={
                "is_live": liveness_result.is_live,
                "confidence": round(liveness_result.confidence, 4),
                "score": round(liveness_result.score, 1),
            },
        ))
        if not liveness_result.is_live:
            overall_success = False
    except Exception as e:
        overall_success = False
        steps.append(PipelineStepResult(step="liveness_check", success=False, error=str(e)))

    return PipelineTestResponse(
        overall_success=overall_success,
        steps=steps,
    )


# ============================================================================
# 6. VIDEO_INTERVIEW Step
# ============================================================================

# Max video file size: 50MB
_MAX_VIDEO_SIZE = 50 * 1024 * 1024
_ACCEPTED_VIDEO_TYPES = {"video/webm", "video/mp4"}


@router.post(
    "/video-interview",
    response_model=VideoInterviewResponse,
    summary="Upload a video interview recording",
    description=(
        "Accepts a short video recording (webm/mp4) from the user's webcam. "
        "Stores the file for manual admin review. Max 50MB."
    ),
)
async def video_interview(
    file: UploadFile = File(..., description="Video file (webm or mp4)"),
) -> VideoInterviewResponse:
    """Upload a video interview recording for manual review.

    The video is stored on disk with a unique filename. No AI analysis
    is performed -- an admin reviews the video and approves or rejects it.
    """
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in _ACCEPTED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {content_type}. Accepted: video/webm, video/mp4",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > _MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large ({len(content)} bytes). Maximum: {_MAX_VIDEO_SIZE} bytes (50MB)",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty video file")

    # Determine file extension
    extension = "mp4" if content_type == "video/mp4" else "webm"

    # Generate unique filename
    unique_name = f"interview_{uuid.uuid4().hex}.{extension}"

    # Store in uploads directory
    settings = get_settings()
    video_dir = Path(settings.UPLOAD_FOLDER) / "video_interviews"
    video_dir.mkdir(parents=True, exist_ok=True)

    file_path = video_dir / unique_name
    with open(file_path, "wb") as f:
        f.write(content)

    # Estimate duration from file size (rough: ~100KB/s for webm, ~150KB/s for mp4)
    bytes_per_second = 150_000 if extension == "mp4" else 100_000
    estimated_duration = round(len(content) / bytes_per_second, 1)

    logger.info(
        "Video interview stored: %s (%d bytes, ~%.1fs)",
        unique_name,
        len(content),
        estimated_duration,
    )

    return VideoInterviewResponse(
        stored=True,
        filename=unique_name,
        duration_seconds=estimated_duration,
        status="PENDING_REVIEW",
    )
