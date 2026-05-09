"""Dependency injection container.

This module provides factory functions for creating and wiring up
all application dependencies.

Following Dependency Inversion Principle:
- High-level modules (use cases) depend on abstractions (interfaces)
- Low-level modules (infrastructure) implement abstractions
- This container wires them together
"""

import logging
from functools import lru_cache

from app.application.services.event_publisher import EventPublisher
from app.application.services.active_liveness_manager import ActiveLivenessManager
from app.application.use_cases.batch_process import BatchEnrollmentUseCase, BatchVerificationUseCase
from app.application.use_cases.check_liveness import CheckLivenessUseCase
from app.application.use_cases.generate_puzzle import GeneratePuzzleUseCase
from app.application.use_cases.process_active_liveness_frame import ProcessActiveLivenessFrameUseCase
from app.application.use_cases.start_active_liveness import StartActiveLivenessUseCase
from app.application.use_cases.verify_puzzle import VerifyPuzzleUseCase

# Application use cases
from app.application.use_cases.delete_enrollment import DeleteEnrollmentUseCase
from app.application.use_cases.detect_card_type import DetectCardTypeUseCase
from app.application.use_cases.enroll_face import EnrollFaceUseCase
from app.application.use_cases.enroll_multi_image import EnrollMultiImageUseCase
from app.application.use_cases.search_face import SearchFaceUseCase
from app.application.use_cases.verify_face import VerifyFaceUseCase
from app.core.config import settings
from app.domain.interfaces.card_type_detector import ICardTypeDetector
from app.domain.interfaces.embedding_extractor import IEmbeddingExtractor
from app.domain.interfaces.embedding_repository import IEmbeddingRepository
from app.domain.interfaces.event_bus import IEventBus
from app.domain.interfaces.active_liveness_session_repository import IActiveLivenessSessionRepository

# Domain interfaces (imported for type hints)
from app.domain.interfaces.face_detector import IFaceDetector
from app.domain.interfaces.file_storage import IFileStorage
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.domain.interfaces.quality_assessor import IQualityAssessor
from app.domain.interfaces.similarity_calculator import ISimilarityCalculator

# Infrastructure implementations
from app.infrastructure.ml.card_type.yolo_card_type_detector import YOLOCardTypeDetector
from app.infrastructure.ml.factories.detector_factory import FaceDetectorFactory
from app.infrastructure.ml.factories.extractor_factory import EmbeddingExtractorFactory
from app.infrastructure.ml.factories.similarity_factory import SimilarityCalculatorFactory
from app.infrastructure.ml.liveness.enhanced_liveness_detector import EnhancedLivenessDetector
from app.infrastructure.ml.liveness.hybrid_liveness_detector import HybridLivenessDetector
from app.infrastructure.ml.liveness.texture_liveness_detector import TextureLivenessDetector
from app.infrastructure.ml.liveness.uniface_liveness_detector import UniFaceLivenessDetector
from app.infrastructure.ml.quality.quality_assessor import QualityAssessor
from app.infrastructure.idempotency import IdempotencyStore
from app.infrastructure.messaging.event_handlers import BiometricEventHandler, EventRouter
from app.infrastructure.messaging.redis_event_bus import RedisEventBus
from app.infrastructure.persistence.repositories.pgvector_embedding_repository import (
    PgVectorEmbeddingRepository,
)
from app.infrastructure.persistence.repositories.in_memory_active_liveness_session_repository import (
    InMemoryActiveLivenessSessionRepository,
)
from app.infrastructure.persistence.repositories.redis_active_liveness_session_repository import (
    RedisActiveLivenessSessionRepository,
)
from app.infrastructure.storage.local_file_storage import LocalFileStorage
from app.infrastructure.async_execution.thread_pool_manager import ThreadPoolManager
from app.infrastructure.ml.voice.speaker_embedder import SpeakerEmbedder
from app.infrastructure.persistence.repositories.pgvector_voice_repository import PgVectorVoiceRepository
# Fingerprint infrastructure is retained for potential future use but
# endpoints return 501 (WebAuthn is the production path).
from app.infrastructure.ml.fingerprint.hash_embedder import FingerprintHashEmbedder
from app.infrastructure.persistence.repositories.pgvector_fingerprint_repository import PgVectorFingerprintRepository

logger = logging.getLogger(__name__)


# ============================================================================
# Infrastructure Layer Dependencies (Singletons)
# ============================================================================


@lru_cache()
def get_thread_pool() -> ThreadPoolManager:
    """Get thread pool manager instance (singleton).

    Returns:
        Thread pool manager for async ML operations

    Note:
        Thread pool is used to execute CPU-bound ML operations (DeepFace)
        without blocking the async event loop. Optimized for ML workloads.
    """
    # AUTO-DETECTION FIX: Use auto-detected pool size
    pool_size = settings.get_thread_pool_size()
    logger.info(f"Creating thread pool manager with {pool_size} workers (auto-detected: {settings.ML_THREAD_POOL_SIZE == 0})")
    return ThreadPoolManager(
        max_workers=pool_size,
        thread_name_prefix="ml-worker",
    )


@lru_cache()
def get_face_detector() -> IFaceDetector:
    """Get face detector instance (singleton) with async support.

    Returns:
        Face detector implementation with async execution if enabled
    """
    logger.info(
        f"Creating face detector: {settings.FACE_DETECTION_BACKEND} "
        f"(anti_spoofing={settings.ANTI_SPOOFING_ENABLED})"
    )
    return FaceDetectorFactory.create(
        detector_type=settings.FACE_DETECTION_BACKEND,
        align=True,
        anti_spoofing=settings.ANTI_SPOOFING_ENABLED,
        anti_spoofing_threshold=settings.ANTI_SPOOFING_THRESHOLD,
        async_enabled=settings.ASYNC_ML_ENABLED,
        thread_pool=get_thread_pool() if settings.ASYNC_ML_ENABLED else None,
    )


@lru_cache()
def get_embedding_extractor() -> IEmbeddingExtractor:
    """Get embedding extractor instance (singleton) with async support.

    Returns:
        Embedding extractor implementation with async execution if enabled
    """
    logger.info(f"Creating embedding extractor: {settings.FACE_RECOGNITION_MODEL}")
    return EmbeddingExtractorFactory.create(
        model_name=settings.FACE_RECOGNITION_MODEL,
        detector_backend=settings.FACE_DETECTION_BACKEND,
        enforce_detection=False,
        async_enabled=settings.ASYNC_ML_ENABLED,
        thread_pool=get_thread_pool() if settings.ASYNC_ML_ENABLED else None,
    )


@lru_cache()
def get_quality_assessor() -> IQualityAssessor:
    """Get quality assessor instance (singleton).

    Returns:
        Quality assessor implementation
    """
    logger.info("Creating quality assessor")
    return QualityAssessor(
        blur_threshold=settings.BLUR_THRESHOLD,
        min_face_size=settings.MIN_FACE_SIZE,
        quality_threshold=settings.QUALITY_THRESHOLD,
    )


@lru_cache()
def get_similarity_calculator() -> ISimilarityCalculator:
    """Get similarity calculator instance (singleton).

    Returns:
        Similarity calculator implementation
    """
    logger.info("Creating similarity calculator")
    return SimilarityCalculatorFactory.create(
        metric="cosine", threshold=settings.VERIFICATION_THRESHOLD
    )


@lru_cache()
def get_file_storage() -> IFileStorage:
    """Get file storage instance (singleton).

    Returns:
        File storage implementation
    """
    logger.info(f"Creating file storage: {settings.UPLOAD_FOLDER}")
    return LocalFileStorage(storage_path=settings.UPLOAD_FOLDER)


@lru_cache()
def get_active_liveness_manager() -> ActiveLivenessManager:
    """Get active liveness manager instance."""

    return ActiveLivenessManager()


@lru_cache()
def get_active_liveness_session_repository() -> IActiveLivenessSessionRepository:
    """Get active liveness session repository instance."""

    if settings.TESTING:
        return InMemoryActiveLivenessSessionRepository()

    return RedisActiveLivenessSessionRepository(
        redis_url=settings.redis_url,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
    )


@lru_cache()
def get_idempotency_store() -> IdempotencyStore:
    """Get idempotency store instance (singleton).

    Returns:
        Idempotency store for preventing duplicate operations

    Note:
        The store uses a 24-hour TTL for idempotency keys.
        This means duplicate requests with the same key will be
        detected and prevented for 24 hours after the original request.
    """
    logger.info("Creating idempotency store (TTL: 24h)")
    return IdempotencyStore(ttl_hours=24)


@lru_cache()
def get_embedding_repository() -> IEmbeddingRepository:
    """Get embedding repository instance (singleton).

    Returns:
        PostgreSQL pgvector embedding repository (production-ready)

    Raises:
        ValueError: If DATABASE_URL is not configured

    Note:
        Always uses PgVectorEmbeddingRepository with efficient vector similarity search.
        In-memory repositories have been removed - only real database allowed.
    """
    if not settings.DATABASE_URL:
        raise ValueError(
            "DATABASE_URL must be set. In-memory repositories are not allowed. "
            "Please configure a PostgreSQL database with pgvector extension."
        )

    # AUTO-DETECTION FIX: Use auto-detected pool sizes
    pool_config = settings.get_database_pool_config()

    logger.info(
        f"Creating embedding repository (pgvector) - "
        f"dimension={settings.EMBEDDING_DIMENSION}, "
        f"pool={pool_config['min_size']}-{pool_config['max_size']} "
        f"(auto-detected: {settings.DATABASE_POOL_MIN_SIZE == 0})"
    )
    return PgVectorEmbeddingRepository(
        database_url=settings.DATABASE_URL,
        pool_min_size=pool_config['min_size'],
        pool_max_size=pool_config['max_size'],
        embedding_dimension=settings.EMBEDDING_DIMENSION,
    )


def get_liveness_detector() -> ILivenessDetector:
    """Get liveness detector instance.

    Returns:
        Liveness detector implementation based on effective liveness config.

    Supported backends:
        - 'enhanced' (default): Multi-modal detector combining LBP texture analysis,
          blink detection, smile detection, and color/frequency analysis.
        - 'texture': Texture-only passive liveness detection using Laplacian variance,
          color distribution, frequency domain, and moire pattern analysis.
        - 'uniface': UniFace MiniFASNet ONNX model for deep learning based
          anti-spoofing (print, replay, mask attack detection).
        - 'hybrid': Enhanced heuristics gated by UniFace as a second opinion.

    Configuration:
        LIVENESS_MODE is the canonical configuration source.
        LIVENESS_BACKEND is treated as a backwards-compatible override.

    Important:
        This dependency is intentionally NOT cached. The enhanced liveness detector
        maintains blink-related state internally, and sharing a singleton instance
        across requests can leak state between users and produce false positives.
        Creating a fresh detector per request/session isolates that state.
    """
    backend = settings.get_liveness_backend()
    logger.info(
        "Resolving liveness detector: mode=%s, backend_override=%s, effective_backend=%s",
        settings.LIVENESS_MODE,
        settings.LIVENESS_BACKEND,
        backend,
    )

    if backend == "uniface":
        logger.info("Creating liveness detector (UniFace MiniFASNet)")
        return UniFaceLivenessDetector(
            liveness_threshold=settings.LIVENESS_THRESHOLD,
        )
    elif backend == "hybrid":
        logger.info("Creating liveness detector (hybrid enhanced + UniFace)")
        return HybridLivenessDetector(
            liveness_threshold=settings.LIVENESS_THRESHOLD,
        )
    elif backend == "texture":
        logger.info("Creating liveness detector (texture analysis)")
        return TextureLivenessDetector(
            texture_threshold=100.0,
            liveness_threshold=settings.LIVENESS_THRESHOLD,
        )
    else:
        logger.info("Creating liveness detector (enhanced multi-modal)")
        return EnhancedLivenessDetector(
            texture_threshold=100.0,
            liveness_threshold=settings.LIVENESS_THRESHOLD,
            enable_blink_detection=True,
            enable_smile_detection=True,
            blink_frames_required=2,
        )


@lru_cache()
def get_puzzle_spot_check_liveness_detector() -> ILivenessDetector:
    """Get the server-side spot-check detector for puzzle verification.

    Puzzle spot-checks are intentionally pinned to UniFace so the server-side
    replay gate is stable regardless of the main runtime liveness backend.
    """
    logger.info("Creating puzzle spot-check liveness detector (UniFace MiniFASNet)")
    return UniFaceLivenessDetector(
        liveness_threshold=settings.LIVENESS_THRESHOLD,
    )


@lru_cache()
def get_card_type_detector() -> ICardTypeDetector:
    """Get card type detector instance (singleton).

    Returns:
        Card type detector implementation (YOLO-based)

    Note:
        Uses YOLOv8 for detecting Turkish identity cards:
        - TC Kimlik (National ID)
        - Ehliyet (Driver's License)
        - Pasaport (Passport)
        - Ogrenci Karti (Student ID)
    """
    logger.info("Creating card type detector (YOLO-based)")
    return YOLOCardTypeDetector(confidence_threshold=settings.CARD_DETECTION_THRESHOLD)


@lru_cache()
def get_demographics_analyzer():
    """Get demographics analyzer instance (singleton).

    Returns:
        Demographics analyzer implementation (DeepFace-based)

    Note:
        Analyzes age, gender, and optionally emotion from faces.
        Race estimation is disabled by default for ethical reasons.
    """
    from app.infrastructure.ml.factories.demographics_factory import DemographicsAnalyzerFactory

    logger.info("Creating demographics analyzer (DeepFace-based)")
    return DemographicsAnalyzerFactory.create(
        backend="deepface",
        include_race=settings.DEMOGRAPHICS_INCLUDE_RACE,
        include_emotion=settings.DEMOGRAPHICS_INCLUDE_EMOTION,
        min_image_size=settings.DEMOGRAPHICS_MIN_IMAGE_SIZE,
        age_margin=settings.DEMOGRAPHICS_AGE_MARGIN,
        age_confidence=settings.DEMOGRAPHICS_AGE_CONFIDENCE,
    )


@lru_cache()
def get_landmark_detector():
    """Get landmark detector instance (singleton).

    Returns:
        Landmark detector implementation (MediaPipe 468-point by default)

    Note:
        Detects facial landmarks for various applications:
        - Face mesh (468 points with MediaPipe)
        - Eye tracking
        - Gaze estimation
        - Expression analysis
    """
    from app.infrastructure.ml.factories.landmark_factory import LandmarkDetectorFactory

    logger.info("Creating landmark detector (MediaPipe 468-point)")
    return LandmarkDetectorFactory.create(model="mediapipe_468")


@lru_cache()
def get_webhook_sender():
    """Get webhook sender instance (singleton).

    Returns:
        Webhook sender implementation (HTTP-based)

    Note:
        Sends event notifications to external systems via HTTP webhooks.
        Includes automatic retry logic with exponential backoff.
    """
    from app.infrastructure.webhooks.webhook_factory import WebhookSenderFactory

    logger.info("Creating webhook sender (HTTP)")
    return WebhookSenderFactory.create(
        transport="http",
        timeout=10,
        retry_count=3,
    )


@lru_cache()
def get_event_bus() -> IEventBus:
    """Get event bus instance (singleton).

    Returns:
        Event bus implementation (Redis-based)

    Note:
        - Uses Redis Pub/Sub for real-time event distribution
        - Async/non-blocking operations
        - Automatic reconnection handling
        - Configurable via environment variables
    """
    if not settings.EVENT_BUS_ENABLED:
        logger.warning("Event bus is disabled in configuration")
        # Return a null/no-op implementation if needed
        # For now, we'll still create it but won't use it

    logger.info(f"Creating Redis event bus: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    return RedisEventBus(
        redis_url=settings.redis_url,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
        retry_attempts=settings.EVENT_BUS_RETRY_ATTEMPTS,
        retry_delay=settings.EVENT_BUS_RETRY_DELAY,
    )


@lru_cache()
def get_event_handler() -> BiometricEventHandler:
    """Get event handler instance (singleton).

    Returns:
        Biometric event handler for processing incoming events
    """
    logger.info("Creating biometric event handler")
    return BiometricEventHandler()


@lru_cache()
def get_event_router() -> EventRouter:
    """Get event router instance (singleton).

    Returns:
        Event router for dispatching events to handlers
    """
    logger.info("Creating event router")
    return EventRouter(handler=get_event_handler())


@lru_cache()
def get_event_publisher() -> EventPublisher:
    """Get event publisher instance (singleton).

    Returns:
        Event publisher for use cases to publish events

    Note:
        Returns publisher with or without event bus depending on configuration
    """
    if settings.EVENT_BUS_ENABLED:
        logger.info("Creating event publisher (enabled)")
        return EventPublisher(event_bus=get_event_bus())
    else:
        logger.info("Creating event publisher (disabled)")
        return EventPublisher(event_bus=None)


# ============================================================================
# Application Layer Dependencies (Use Cases)
# ============================================================================


def get_enroll_face_use_case() -> EnrollFaceUseCase:
    """Get enroll face use case instance.

    Returns:
        EnrollFaceUseCase with all dependencies injected
    """
    return EnrollFaceUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        quality_assessor=get_quality_assessor(),
        repository=get_embedding_repository(),
    )


def get_delete_enrollment_use_case() -> DeleteEnrollmentUseCase:
    """Get delete enrollment use case instance.

    Returns:
        DeleteEnrollmentUseCase with all dependencies injected
    """
    return DeleteEnrollmentUseCase(
        repository=get_embedding_repository(),
    )


def get_enroll_multi_image_use_case() -> EnrollMultiImageUseCase:
    """Get multi-image enrollment use case instance.

    Returns:
        EnrollMultiImageUseCase with all dependencies injected
    """
    return EnrollMultiImageUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        quality_assessor=get_quality_assessor(),
        repository=get_embedding_repository(),
    )


def get_verify_face_use_case() -> VerifyFaceUseCase:
    """Get verify face use case instance.

    Returns:
        VerifyFaceUseCase with all dependencies injected
    """
    return VerifyFaceUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        similarity_calculator=get_similarity_calculator(),
        repository=get_embedding_repository(),
        quality_assessor=get_quality_assessor(),
    )


def get_check_liveness_use_case() -> CheckLivenessUseCase:
    """Get check liveness use case instance.

    Returns:
        CheckLivenessUseCase with all dependencies injected
    """
    return CheckLivenessUseCase(
        detector=get_face_detector(),
        liveness_detector=get_liveness_detector(),
        landmark_detector=get_landmark_detector(),
    )


def get_start_active_liveness_use_case() -> StartActiveLivenessUseCase:
    """Get start active liveness use case instance."""

    return StartActiveLivenessUseCase(
        manager=get_active_liveness_manager(),
        session_repository=get_active_liveness_session_repository(),
    )


def get_process_active_liveness_frame_use_case() -> ProcessActiveLivenessFrameUseCase:
    """Get process active liveness frame use case instance."""

    return ProcessActiveLivenessFrameUseCase(
        manager=get_active_liveness_manager(),
        session_repository=get_active_liveness_session_repository(),
    )


def get_detect_card_type_use_case() -> DetectCardTypeUseCase:
    """Get card type detection use case instance.

    Returns:
        DetectCardTypeUseCase with all dependencies injected
    """
    return DetectCardTypeUseCase(
        detector=get_card_type_detector(),
    )


def get_search_face_use_case() -> SearchFaceUseCase:
    """Get search face use case instance.

    Returns:
        SearchFaceUseCase with all dependencies injected
    """
    return SearchFaceUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        repository=get_embedding_repository(),
        similarity_calculator=get_similarity_calculator(),
    )


def get_batch_enrollment_use_case() -> BatchEnrollmentUseCase:
    """Get batch enrollment use case instance.

    Returns:
        BatchEnrollmentUseCase with all dependencies injected
    """
    return BatchEnrollmentUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        quality_assessor=get_quality_assessor(),
        repository=get_embedding_repository(),
        max_concurrent=5,
    )


def get_batch_verification_use_case() -> BatchVerificationUseCase:
    """Get batch verification use case instance.

    Returns:
        BatchVerificationUseCase with all dependencies injected
    """
    return BatchVerificationUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        repository=get_embedding_repository(),
        similarity_calculator=get_similarity_calculator(),
        max_concurrent=5,
        default_threshold=settings.VERIFICATION_THRESHOLD,
    )


def get_analyze_quality_use_case():
    """Get analyze quality use case instance.

    Returns:
        AnalyzeQualityUseCase with all dependencies injected
    """
    from app.application.use_cases.analyze_quality import AnalyzeQualityUseCase

    return AnalyzeQualityUseCase(
        detector=get_face_detector(),
        quality_assessor=get_quality_assessor(),
    )


def get_detect_multi_face_use_case():
    """Get detect multi-face use case instance.

    Returns:
        DetectMultiFaceUseCase with all dependencies injected
    """
    from app.application.use_cases.detect_multi_face import DetectMultiFaceUseCase

    return DetectMultiFaceUseCase(
        detector=get_face_detector(),
        quality_assessor=get_quality_assessor(),
    )


def get_analyze_demographics_use_case():
    """Get analyze demographics use case instance.

    Returns:
        AnalyzeDemographicsUseCase with all dependencies injected
    """
    from app.application.use_cases.analyze_demographics import AnalyzeDemographicsUseCase

    return AnalyzeDemographicsUseCase(
        detector=get_face_detector(),
        demographics_analyzer=get_demographics_analyzer(),
    )


def get_compare_faces_use_case():
    """Get compare faces use case instance.

    Returns:
        CompareFacesUseCase with all dependencies injected
    """
    from app.application.use_cases.compare_faces import CompareFacesUseCase

    return CompareFacesUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        similarity_calculator=get_similarity_calculator(),
        quality_assessor=get_quality_assessor(),
    )


def get_compute_similarity_matrix_use_case():
    """Get compute similarity matrix use case instance.

    Returns:
        ComputeSimilarityMatrixUseCase with all dependencies injected
    """
    from app.application.use_cases.compute_similarity_matrix import ComputeSimilarityMatrixUseCase

    return ComputeSimilarityMatrixUseCase(
        detector=get_face_detector(),
        extractor=get_embedding_extractor(),
        similarity_calculator=get_similarity_calculator(),
    )


def get_detect_landmarks_use_case():
    """Get detect landmarks use case instance.

    Returns:
        DetectLandmarksUseCase with all dependencies injected
    """
    from app.application.use_cases.detect_landmarks import DetectLandmarksUseCase

    return DetectLandmarksUseCase(
        detector=get_face_detector(),
        landmark_detector=get_landmark_detector(),
    )


def get_export_embeddings_use_case():
    """Get export embeddings use case instance.

    Returns:
        ExportEmbeddingsUseCase with all dependencies injected
    """
    from app.application.use_cases.export_embeddings import ExportEmbeddingsUseCase

    return ExportEmbeddingsUseCase(
        repository=get_embedding_repository(),
    )


def get_import_embeddings_use_case():
    """Get import embeddings use case instance.

    Returns:
        ImportEmbeddingsUseCase with all dependencies injected
    """
    from app.application.use_cases.import_embeddings import ImportEmbeddingsUseCase

    return ImportEmbeddingsUseCase(
        repository=get_embedding_repository(),
    )


def get_send_webhook_use_case():
    """Get send webhook use case instance.

    Returns:
        SendWebhookUseCase with all dependencies injected
    """
    from app.application.use_cases.send_webhook import SendWebhookUseCase

    return SendWebhookUseCase(
        webhook_sender=get_webhook_sender(),
    )


# ============================================================================
# Voice Biometric Dependencies
# ============================================================================


@lru_cache()
def get_speaker_embedder() -> SpeakerEmbedder:
    """Get speaker embedder instance (singleton).

    Returns:
        SpeakerEmbedder using Resemblyzer (256-dim speaker embeddings)
    """
    logger.info("Creating speaker embedder (Resemblyzer)")
    return SpeakerEmbedder()


@lru_cache()
def get_voice_repository() -> PgVectorVoiceRepository:
    """Get voice embedding repository instance (singleton).

    Returns:
        PgVectorVoiceRepository for voice enrollment storage
    """
    if not settings.DATABASE_URL:
        raise ValueError(
            "DATABASE_URL must be set for voice enrollment storage."
        )

    logger.info("Creating voice embedding repository (pgvector, dim=256)")
    return PgVectorVoiceRepository(
        database_url=settings.DATABASE_URL,
        pool_min_size=2,
        pool_max_size=5,
        embedding_dimension=256,
    )


# ============================================================================
# Fingerprint Biometric Dependencies
# ============================================================================


@lru_cache()
def get_fingerprint_embedder() -> FingerprintHashEmbedder:
    """Get fingerprint embedder instance (singleton).

    Returns:
        FingerprintHashEmbedder using SHA-256 hash-based embedding (256-dim)
    """
    logger.info("Creating fingerprint embedder (hash-based)")
    return FingerprintHashEmbedder()


@lru_cache()
def get_fingerprint_repository() -> PgVectorFingerprintRepository:
    """Get fingerprint embedding repository instance (singleton).

    Returns:
        PgVectorFingerprintRepository for fingerprint enrollment storage
    """
    if not settings.DATABASE_URL:
        raise ValueError(
            "DATABASE_URL must be set for fingerprint enrollment storage."
        )

    logger.info("Creating fingerprint embedding repository (pgvector, dim=256)")
    return PgVectorFingerprintRepository(
        database_url=settings.DATABASE_URL,
        pool_min_size=2,
        pool_max_size=5,
        embedding_dimension=256,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def initialize_dependencies() -> None:
    """Initialize all singleton dependencies.

    This pre-loads ML models and creates thread pool at application startup
    for better first-request performance.

    Critical Performance Optimization:
        - Creates thread pool for async ML operations
        - Pre-loads DeepFace models to avoid first-request delay
        - Initializes database connection pool
    """
    logger.info("Initializing dependencies...")

    # CRITICAL: Initialize thread pool first (required for async ML operations)
    if settings.ASYNC_ML_ENABLED:
        logger.info("Initializing thread pool for async ML operations...")
        get_thread_pool()

    # Pre-load expensive ML models (with async wrappers if enabled)
    logger.info("Pre-loading ML models...")
    get_face_detector()
    get_embedding_extractor()
    get_quality_assessor()
    get_similarity_calculator()
    get_liveness_detector()

    # Pre-load voice model
    logger.info("Pre-loading Resemblyzer speaker encoder...")
    get_speaker_embedder()

    # Initialize storage and repositories
    logger.info("Initializing storage and database...")
    get_file_storage()
    get_embedding_repository()

    # Initialize event bus and handlers (if enabled)
    if settings.EVENT_BUS_ENABLED:
        logger.info("Initializing event bus...")
        get_event_bus()
        get_event_handler()
        get_event_router()
        get_event_publisher()

    logger.info(
        f"Dependencies initialized successfully "
        f"(async_ml={settings.ASYNC_ML_ENABLED}, "
        f"thread_pool_size={settings.ML_THREAD_POOL_SIZE})"
    )


async def shutdown_dependencies(wait: bool = True) -> None:
    """Shutdown all dependencies gracefully.

    This function should be called during application shutdown to ensure
    proper cleanup of resources (thread pools, database connections, etc.).

    Args:
        wait: If True, wait for pending operations to complete

    Critical for Production:
        - Prevents resource leaks
        - Ensures graceful shutdown
        - Closes database connection pools
        - Shuts down thread pool workers
    """
    logger.info("Shutting down dependencies...")

    # Shutdown thread pool first (prevents new ML tasks)
    if settings.ASYNC_ML_ENABLED:
        try:
            thread_pool = get_thread_pool()
            logger.info("Shutting down thread pool...")
            thread_pool.shutdown(wait=wait, cancel_futures=not wait)
            logger.info("Thread pool shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}", exc_info=True)

    # Close database connection pool
    try:
        repository = get_embedding_repository()
        if hasattr(repository, 'close'):
            logger.info("Closing database connection pool...")
            await repository.close()
            logger.info("Database connection pool closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}", exc_info=True)

    # Close voice repository connection pool
    try:
        voice_repo = get_voice_repository()
        if hasattr(voice_repo, 'close'):
            logger.info("Closing voice repository connection pool...")
            await voice_repo.close()
            logger.info("Voice repository connection pool closed")
    except Exception as e:
        logger.error(f"Error closing voice repository: {e}", exc_info=True)

    # Close fingerprint repository connection pool
    try:
        fingerprint_repo = get_fingerprint_repository()
        if hasattr(fingerprint_repo, 'close'):
            logger.info("Closing fingerprint repository connection pool...")
            await fingerprint_repo.close()
            logger.info("Fingerprint repository connection pool closed")
    except Exception as e:
        logger.error(f"Error closing fingerprint repository: {e}", exc_info=True)

    # Close active liveness session repository
    try:
        active_liveness_repository = get_active_liveness_session_repository()
        if hasattr(active_liveness_repository, 'close'):
            logger.info("Closing active liveness session repository...")
            await active_liveness_repository.close()
            logger.info("Active liveness session repository closed")
    except Exception as e:
        logger.error(f"Error closing active liveness session repository: {e}", exc_info=True)

    # Close event bus connections
    if settings.EVENT_BUS_ENABLED:
        try:
            event_bus = get_event_bus()
            if hasattr(event_bus, 'close'):
                logger.info("Closing event bus connections...")
                await event_bus.close()
                logger.info("Event bus closed")
        except Exception as e:
            logger.error(f"Error closing event bus: {e}", exc_info=True)

    logger.info("Dependencies shutdown complete")


def shutdown_thread_pool(wait: bool = True) -> None:
    """Shutdown thread pool gracefully (sync wrapper).

    Args:
        wait: If True, wait for pending operations to complete

    Note:
        This is a synchronous wrapper for use in non-async contexts.
        For async code, use shutdown_dependencies() instead.
    """
    if not settings.ASYNC_ML_ENABLED:
        logger.debug("Thread pool not enabled, skipping shutdown")
        return

    try:
        thread_pool = get_thread_pool()
        logger.info(f"Shutting down thread pool (wait={wait})...")
        thread_pool.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("Thread pool shutdown complete")
    except Exception as e:
        logger.error(f"Error during thread pool shutdown: {e}", exc_info=True)


@lru_cache()
def get_puzzle_repository():
    """Get puzzle repository instance (singleton)."""
    from app.infrastructure.persistence.repositories.redis_puzzle_repository import InMemoryPuzzleRepository
    logger.info("Creating puzzle repository (in-memory)")
    return InMemoryPuzzleRepository()


def get_generate_puzzle_use_case() -> GeneratePuzzleUseCase:
    """Get generate puzzle use case instance."""
    return GeneratePuzzleUseCase(puzzle_repository=get_puzzle_repository())


def get_verify_puzzle_use_case() -> VerifyPuzzleUseCase:
    return VerifyPuzzleUseCase(
        puzzle_repository=get_puzzle_repository(),
        spot_check_detector=get_puzzle_spot_check_liveness_detector(),
    )

def clear_cache() -> None:
    """Clear dependency cache (for testing).

    Warning:
        This will cause all dependencies to be recreated.
        Only use in tests or during development.
    """
    logger.warning("Clearing dependency cache")

    get_thread_pool.cache_clear()
    get_face_detector.cache_clear()
    get_embedding_extractor.cache_clear()
    get_quality_assessor.cache_clear()
    get_similarity_calculator.cache_clear()
    get_file_storage.cache_clear()
    get_active_liveness_manager.cache_clear()
    get_active_liveness_session_repository.cache_clear()
    get_embedding_repository.cache_clear()
    get_event_bus.cache_clear()
    get_event_handler.cache_clear()
    get_event_router.cache_clear()
    get_event_publisher.cache_clear()
    get_puzzle_repository.cache_clear()
    get_puzzle_spot_check_liveness_detector.cache_clear()
    get_speaker_embedder.cache_clear()
    get_voice_repository.cache_clear()
    get_fingerprint_embedder.cache_clear()
    get_fingerprint_repository.cache_clear()
