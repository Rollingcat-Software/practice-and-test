"""Application configuration with Pydantic validation."""

import os
from typing import List, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation.

    All settings are validated using Pydantic to ensure correctness
    at startup time, preventing runtime errors from misconfiguration.

    Following best practices:
    - Type hints for all fields
    - Validation for acceptable ranges
    - Sensible defaults
    - Environment variable support
    """

    # Application
    APP_NAME: str = Field(default="FIVUCSAS Biometric Processor")
    VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(default="development")
    DEBUG: bool = Field(default=False)

    # API Settings
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8001, ge=1024, le=65535)
    API_WORKERS: int = Field(default=4, ge=1, le=32)

    # CORS Settings (NO WILDCARD!)
    # Frontend is now served from same origin (port 8001), no need for separate ports
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:8001", "http://localhost:8080"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # File Upload Settings
    UPLOAD_FOLDER: str = Field(default="./temp_uploads")
    MAX_FILE_SIZE: int = Field(
        default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024
    )  # 1KB to 50MB
    ALLOWED_IMAGE_FORMATS: List[str] = Field(default=["jpg", "jpeg", "png"])

    # ML Model Settings
    FACE_DETECTION_BACKEND: Literal[
        "opencv", "ssd", "mtcnn", "retinaface", "mediapipe", "yolov8",
        "yolov11n", "yolov11s", "yolov12n", "centerface",
    ] = Field(default="opencv")

    FACE_RECOGNITION_MODEL: Literal[
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
        "GhostFaceNet",
    ] = Field(default="Facenet")

    MODEL_DEVICE: Literal["cpu", "cuda"] = Field(default="cpu")

    # Anti-Spoofing (DeepFace 0.0.98+ built-in)
    ANTI_SPOOFING_ENABLED: bool = Field(
        default=False,
        description="Enable DeepFace built-in anti-spoofing on face detection",
        validation_alias=AliasChoices("ANTI_SPOOFING_ENABLED", "DEEPFACE_ANTI_SPOOFING"),
    )
    ANTI_SPOOFING_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum antispoof score to accept a face as real"
    )

    # Thresholds
    VERIFICATION_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)
    LIVENESS_THRESHOLD: float = Field(default=70.0, ge=0.0, le=100.0)
    QUALITY_THRESHOLD: float = Field(default=70.0, ge=0.0, le=100.0)

    # ML Model Timeouts (prevents hung requests)
    ML_MODEL_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=120, description="Timeout for ML model operations")

    # CRITICAL PERFORMANCE: Async ML Execution
    ASYNC_ML_ENABLED: bool = Field(
        default=True,
        description="Enable async ML operations using thread pool (CRITICAL for performance - 10-25x improvement)"
    )
    ML_THREAD_POOL_SIZE: int = Field(
        default=0,  # 0 = auto-detect CPU count
        ge=0,
        le=32,
        description="Thread pool size for async ML operations (0 = auto-detect, recommended: number of CPU cores)"
    )

    def get_thread_pool_size(self) -> int:
        """Get optimal thread pool size.

        AUTO-DETECTION FIX: Automatically detects CPU count if not explicitly set.
        This ensures optimal performance across different deployment environments.
        """
        if self.ML_THREAD_POOL_SIZE == 0:
            import os
            cpu_count = os.cpu_count() or 4
            # Use CPU count but cap at 8 for safety
            return min(cpu_count, 8)
        return self.ML_THREAD_POOL_SIZE

    def get_liveness_backend(self) -> Literal["enhanced", "texture", "uniface", "hybrid"]:
        """Get the effective liveness backend.

        LIVENESS_MODE is the canonical configuration source. LIVENESS_BACKEND is
        kept only for backwards compatibility and explicit backend overrides.
        """
        if self.LIVENESS_BACKEND is not None:
            return self.LIVENESS_BACKEND

        mode_to_backend = {
            "passive": "texture",
            "active": "enhanced",
            "combined": "hybrid",
        }
        return mode_to_backend[self.LIVENESS_MODE]

    # Request Timeouts (prevents hung requests)
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum time for entire request processing (prevents resource exhaustion)"
    )

    # Batch Processing Limits (prevents DoS attacks)
    BATCH_MAX_SIZE: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of images in a batch operation (prevents memory exhaustion)"
    )
    BATCH_MAX_TOTAL_SIZE_MB: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum total size of all files in a batch (MB)"
    )

    # Liveness Detection Mode
    LIVENESS_MODE: Literal["passive", "active", "combined"] = Field(
        default="combined",
        description=(
            "Canonical liveness configuration. "
            "'passive' maps to texture analysis, "
            "'active' maps to enhanced active checks, "
            "'combined' maps to enhanced multi-modal checks."
        ),
    )
    LIVENESS_SECURITY_PROFILE: Literal["standard"] = Field(
        default="standard",
        description=(
            "Security posture for liveness decisions. "
            "'standard' preserves the baseline behavior."
        ),
    )

    # Liveness Detection Backend
    LIVENESS_BACKEND: Optional[Literal["enhanced", "texture", "uniface", "hybrid"]] = Field(
        default=None,
        description=(
            "Deprecated compatibility alias for backend selection. "
            "Prefer LIVENESS_MODE. When set, this value overrides the backend "
            "derived from LIVENESS_MODE."
        ),
    )
    LIVENESS_UNIFACE_DEFAULT_ENABLED: bool = Field(
        default=False,
        description=(
            "Feature flag for rolling out UniFace as the default backend for "
            "combined liveness mode when no explicit LIVENESS_BACKEND override is set."
        ),
    )
    LIVENESS_CALIBRATION_LOG_PATH: str = Field(
        default="logs/liveness_calibration.jsonl",
        description="Dedicated JSONL sink for liveness calibration events.",
    )

    # Quality Assessment
    MIN_IMAGE_SIZE: int = Field(default=100, ge=50, le=1000)
    MAX_IMAGE_SIZE: int = Field(default=4000, ge=1000, le=10000)
    MIN_FACE_SIZE: int = Field(default=80, ge=40, le=500)
    BLUR_THRESHOLD: float = Field(default=100.0, ge=0.0)

    # Database (PostgreSQL with pgvector)
    # CRITICAL: In-memory repositories removed - only real database allowed
    # SECURITY: No default credentials - must be provided via environment variable
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL database URL with pgvector extension (REQUIRED - set via DATABASE_URL env var)"
    )
    DATABASE_POOL_MIN_SIZE: int = Field(default=0, ge=0, le=100)  # 0 = auto-detect
    DATABASE_POOL_MAX_SIZE: int = Field(default=0, ge=0, le=100)  # 0 = auto-detect

    def get_database_pool_config(self) -> dict:
        """Get optimal database pool configuration.

        AUTO-DETECTION FIX: Automatically calculates pool size based on environment.
        Formula: min_size = workers * 2, max_size = workers * 4
        """
        if self.DATABASE_POOL_MIN_SIZE == 0 or self.DATABASE_POOL_MAX_SIZE == 0:
            # Auto-detect based on API workers
            workers = self.API_WORKERS
            return {
                "min_size": max(workers * 2, 5),    # At least 5, typically 8-16
                "max_size": max(workers * 4, 10),   # At least 10, typically 16-32
            }
        return {
            "min_size": self.DATABASE_POOL_MIN_SIZE,
            "max_size": self.DATABASE_POOL_MAX_SIZE,
        }
    EMBEDDING_DIMENSION: int = Field(default=512, ge=128, le=4096)  # FaceNet: 512, VGG-Face: 2622

    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1024, le=65535)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_MAX_CONNECTIONS: int = Field(default=10, ge=1, le=100)
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, ge=1, le=60)
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(default=5, ge=1, le=60)

    # Event Bus Configuration
    EVENT_BUS_ENABLED: bool = Field(default=True)
    EVENT_BUS_RETRY_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    EVENT_BUS_RETRY_DELAY: float = Field(default=1.0, ge=0.1, le=10.0)

    @property
    def redis_url(self) -> str:
        """Build Redis URL from components.

        Returns:
            Redis connection URL in the format:
            redis://:password@host:port/db or redis://host:port/db
        """
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Webhook
    WEBHOOK_TIMEOUT: int = Field(default=10, ge=1, le=60)
    WEBHOOK_MAX_RETRIES: int = Field(default=3, ge=0, le=10)

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    LOG_FORMAT: Literal["json", "text"] = Field(default="json")

    # Rate Limiting
    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable API rate limiting")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1, description="Default requests per minute")
    RATE_LIMIT_STORAGE: Literal["memory", "redis"] = Field(default="memory", description="Rate limit storage backend")
    RATE_LIMIT_DEFAULT: int = Field(default=60, ge=1, description="Default rate limit for free tier")
    RATE_LIMIT_PREMIUM: int = Field(default=300, ge=1, description="Rate limit for premium tier")

    # Per-Endpoint Rate Limiting (PERFORMANCE FIX: Cost-based throttling)
    # Different endpoints have different computational costs
    ENROLLMENT_RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Rate limit for enrollment endpoints (lower than general API to prevent abuse)"
    )
    VERIFICATION_RATE_LIMIT_PER_MINUTE: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Rate limit for verification endpoints (moderate cost)"
    )
    SEARCH_RATE_LIMIT_PER_MINUTE: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Rate limit for search endpoints (expensive operation)"
    )
    LIVENESS_RATE_LIMIT_PER_MINUTE: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Rate limit for liveness detection (expensive processing)"
    )
    BATCH_RATE_LIMIT_PER_MINUTE: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Rate limit for batch operations (very expensive)"
    )
    HEALTH_CHECK_RATE_LIMIT_PER_MINUTE: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Rate limit for health checks (cheap operation)"
    )

    def get_endpoint_rate_limit(self, endpoint_type: str) -> int:
        """Get rate limit for specific endpoint type.

        PERFORMANCE FIX: Returns cost-based rate limits per endpoint.
        Expensive operations get lower limits to prevent resource exhaustion.

        Args:
            endpoint_type: Type of endpoint (enrollment, verification, search, etc.)

        Returns:
            Rate limit (requests per minute)
        """
        limits = {
            "enrollment": self.ENROLLMENT_RATE_LIMIT_PER_MINUTE,
            "verification": self.VERIFICATION_RATE_LIMIT_PER_MINUTE,
            "search": self.SEARCH_RATE_LIMIT_PER_MINUTE,
            "liveness": self.LIVENESS_RATE_LIMIT_PER_MINUTE,
            "batch": self.BATCH_RATE_LIMIT_PER_MINUTE,
            "health": self.HEALTH_CHECK_RATE_LIMIT_PER_MINUTE,
        }
        return limits.get(endpoint_type, self.RATE_LIMIT_PER_MINUTE)

    # Demographics Analysis
    DEMOGRAPHICS_ENABLED: bool = Field(default=True)
    DEMOGRAPHICS_INCLUDE_RACE: bool = Field(default=False)  # Privacy consideration
    DEMOGRAPHICS_INCLUDE_EMOTION: bool = Field(default=True)

    # Demographics Quality Settings
    # DeepFace age estimation has MAE ~10 years, these settings help manage expectations
    # Note: Lowered minimum to 48px to support webcam captures and small thumbnails
    DEMOGRAPHICS_MIN_IMAGE_SIZE: int = Field(default=64, ge=48, le=1024)  # Minimum image size for accuracy
    DEMOGRAPHICS_MIN_CONFIDENCE: float = Field(default=0.5, ge=0.0, le=1.0)  # Minimum confidence threshold
    DEMOGRAPHICS_AGE_MARGIN: int = Field(default=10, ge=5, le=30)  # Age range margin (±years) based on known MAE
    DEMOGRAPHICS_AGE_CONFIDENCE: float = Field(default=0.65, ge=0.1, le=0.95)  # Age estimation confidence (conservative)

    # Landmarks
    LANDMARK_MODEL: Literal["mediapipe_468", "dlib_68"] = Field(default="mediapipe_468")

    # Image Preprocessing
    PREPROCESS_AUTO_ROTATE: bool = Field(default=True)
    PREPROCESS_MAX_SIZE: int = Field(default=1920, ge=640, le=4096)
    PREPROCESS_NORMALIZE: bool = Field(default=True)

    # Webhooks
    WEBHOOK_ENABLED: bool = Field(default=False)
    WEBHOOK_URL: Optional[str] = Field(default=None)
    WEBHOOK_SECRET: Optional[str] = Field(default=None)
    WEBHOOK_EVENTS: List[str] = Field(
        default=["enrollment", "verification", "liveness"]
    )
    WEBHOOK_RETRY_COUNT: int = Field(default=3, ge=0, le=10)

    # Export/Import
    EXPORT_FORMAT: Literal["json", "msgpack"] = Field(default="json")
    EXPORT_INCLUDE_METADATA: bool = Field(default=True)

    # Multi-Image Enrollment
    MULTI_IMAGE_ENROLLMENT_ENABLED: bool = Field(default=True)
    MULTI_IMAGE_MIN_IMAGES: int = Field(default=2, ge=2, le=5)
    MULTI_IMAGE_MAX_IMAGES: int = Field(default=5, ge=2, le=5)
    MULTI_IMAGE_FUSION_STRATEGY: Literal["weighted_average", "simple_average"] = Field(
        default="weighted_average"
    )
    MULTI_IMAGE_NORMALIZATION: Literal["l2", "none"] = Field(default="l2")
    MULTI_IMAGE_MIN_QUALITY_PER_IMAGE: float = Field(default=60.0, ge=0.0, le=100.0)

    # Embedding Cache Settings
    EMBEDDING_CACHE_ENABLED: bool = Field(default=True, description="Enable LRU cache for embedding lookups")
    EMBEDDING_CACHE_TTL_SECONDS: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds (1 min to 1 hour)")
    EMBEDDING_CACHE_MAX_SIZE: int = Field(default=1000, ge=100, le=10000, description="Maximum cached embeddings (100 to 10000)")

    # API Key Authentication
    # SECURITY: In production, API key auth is mandatory
    API_KEY_ENABLED: bool = Field(default=False)
    API_KEY_REQUIRE_AUTH: bool = Field(default=False)
    API_KEY_HEADER: str = Field(default="X-API-Key")

    # JWT Configuration (must match Identity Core API)
    JWT_SECRET: str = Field(
        default="",
        description="JWT secret key for token verification (REQUIRED in production)"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_ENABLED: bool = Field(default=True, description="Enable JWT authentication")
    JWT_ISSUER: Optional[str] = Field(default=None, description="Expected JWT issuer")
    JWT_AUDIENCE: Optional[str] = Field(default=None, description="Expected JWT audience")

    # Database Connection Pool (Phase 7 optimization)
    DB_POOL_SIZE: int = Field(default=20, ge=5, le=100, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50, description="Maximum pool overflow connections")
    DB_POOL_TIMEOUT: int = Field(default=30, ge=5, le=120, description="Pool connection timeout in seconds")
    DB_POOL_RECYCLE: int = Field(default=1800, ge=300, le=7200, description="Connection recycle time in seconds")
    DB_ECHO: bool = Field(default=False, description="Echo SQL statements for debugging")
    TESTING: bool = Field(default=False, description="Enable testing mode (disables connection pooling)")

    def get_api_key_config(self) -> dict:
        """Get API key configuration with production enforcement.

        SECURITY FIX: In production, API key authentication is mandatory.
        This prevents accidental deployment without authentication.
        """
        require_auth = self.API_KEY_REQUIRE_AUTH
        enabled = self.API_KEY_ENABLED

        # CRITICAL: Enforce authentication in production
        if self.is_production():
            if not enabled:
                raise ValueError(
                    "SECURITY ERROR: API_KEY_ENABLED must be True in production. "
                    "Set API_KEY_ENABLED=true environment variable."
                )
            require_auth = True  # Always require in production

        return {
            "enabled": enabled,
            "require_auth": require_auth,
            "header": self.API_KEY_HEADER,
        }

    # Metrics (disabled for PaaS deployment)
    METRICS_ENABLED: bool = Field(default=False)
    METRICS_PATH: str = Field(default="/metrics")

    # ============================================================================
    # Proctoring Service Configuration
    # ============================================================================

    # Feature Flags
    PROCTOR_ENABLED: bool = Field(default=True)
    PROCTOR_GAZE_ENABLED: bool = Field(default=True)
    PROCTOR_OBJECT_DETECTION_ENABLED: bool = Field(default=True)
    PROCTOR_DEEPFAKE_ENABLED: bool = Field(default=True)
    PROCTOR_AUDIO_ENABLED: bool = Field(default=False)

    # Session Management
    PROCTOR_MAX_SESSIONS_PER_USER: int = Field(default=1, ge=1, le=5)
    PROCTOR_SESSION_TIMEOUT_MINUTES: int = Field(default=180, ge=30, le=480)
    PROCTOR_STORAGE_TYPE: Literal["memory", "postgres"] = Field(default="memory")

    # Verification Settings
    PROCTOR_VERIFICATION_INTERVAL_SEC: int = Field(default=60, ge=10, le=300)
    PROCTOR_VERIFICATION_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)
    PROCTOR_LIVENESS_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    # Gaze Tracking
    PROCTOR_GAZE_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)
    PROCTOR_GAZE_AWAY_THRESHOLD_SEC: float = Field(default=5.0, ge=1.0, le=30.0)
    PROCTOR_HEAD_PITCH_THRESHOLD: float = Field(default=20.0, ge=5.0, le=45.0)
    PROCTOR_HEAD_YAW_THRESHOLD: float = Field(default=30.0, ge=10.0, le=60.0)

    # Object Detection
    PROCTOR_OBJECT_MODEL_SIZE: Literal["nano", "small", "medium", "large"] = Field(default="nano")
    PROCTOR_OBJECT_CONFIDENCE_THRESHOLD: float = Field(default=0.5, ge=0.1, le=0.9)
    PROCTOR_MAX_PERSONS_ALLOWED: int = Field(default=1, ge=1, le=3)

    # Deepfake Detection
    PROCTOR_DEEPFAKE_THRESHOLD: float = Field(default=0.6, ge=0.3, le=0.9)
    PROCTOR_DEEPFAKE_TEMPORAL_WINDOW: int = Field(default=10, ge=3, le=30)

    # Audio Analysis
    PROCTOR_AUDIO_SAMPLE_RATE: int = Field(default=16000, ge=8000, le=48000)
    PROCTOR_AUDIO_VAD_THRESHOLD: float = Field(default=0.5, ge=0.1, le=0.9)

    # Risk Management
    PROCTOR_RISK_THRESHOLD_WARNING: float = Field(default=0.5, ge=0.1, le=0.9)
    PROCTOR_RISK_THRESHOLD_CRITICAL: float = Field(default=0.8, ge=0.5, le=1.0)
    PROCTOR_AUTO_TERMINATE_ON_CRITICAL: bool = Field(default=False)

    # Rate Limiting (per-session)
    PROCTOR_RATE_LIMIT_ENABLED: bool = Field(default=True)
    PROCTOR_MAX_FRAMES_PER_SECOND: int = Field(default=5, ge=1, le=30)
    PROCTOR_MAX_FRAMES_PER_MINUTE: int = Field(default=120, ge=30, le=600)
    PROCTOR_RATE_LIMIT_BURST_ALLOWANCE: int = Field(default=10, ge=0, le=30)

    # Circuit Breaker
    PROCTOR_CIRCUIT_BREAKER_ENABLED: bool = Field(default=True)
    PROCTOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=3, ge=1, le=10)
    PROCTOR_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(default=2, ge=1, le=5)
    PROCTOR_CIRCUIT_BREAKER_TIMEOUT_SEC: float = Field(default=30.0, ge=5.0, le=120.0)

    # ============================================================================
    # Performance Optimization Settings
    # ============================================================================
    # Note: ASYNC_ML_ENABLED and ML_THREAD_POOL_SIZE are defined above (line ~95)
    # Note: EMBEDDING_CACHE_ENABLED/TTL/MAX_SIZE are defined above (line ~327)

    # Repository Settings
    REPOSITORY_MAX_CAPACITY: int = Field(default=100000, ge=1000, le=10000000)
    REPOSITORY_ENABLE_VECTORIZED_SEARCH: bool = Field(default=True)

    # Rate Limit Memory Cleanup
    RATE_LIMIT_MAX_ENTRIES: int = Field(default=100000, ge=1000, le=1000000)
    RATE_LIMIT_CLEANUP_INTERVAL: int = Field(default=60, ge=10, le=3600)

    # Liveness Detection Optimization
    LIVENESS_ENABLE_OPTIMIZED: bool = Field(default=True)
    LIVENESS_FFT_DOWNSAMPLE_SIZE: int = Field(default=192, ge=64, le=512)

    # Card Detection
    CARD_DETECTION_THRESHOLD: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="YOLO card detection confidence threshold (lowered from 0.5 for better recall on non-passport cards)",
    )

    # Development-only live liveness preview
    DEV_LIVENESS_PREVIEW: bool = Field(
        default=False,
        description="Open the developer-only live liveness webcam preview on startup.",
    )
    DEV_LIVENESS_PREVIEW_CAMERA_INDEX: int = Field(
        default=0,
        ge=0,
        le=8,
        description="OpenCV camera index for the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_WINDOW_SECONDS: float = Field(
        default=2.0,
        ge=1.0,
        le=3.0,
        description="Time-based rolling window for live liveness temporal aggregation.",
    )
    DEV_LIVENESS_PREVIEW_BASELINE_SECONDS: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Neutral baseline calibration duration for background active reaction detection.",
    )
    DEV_LIVENESS_PREVIEW_BUFFER_SIZE: int = Field(
        default=45,
        ge=5,
        le=300,
        description="Maximum retained frame entries inside the rolling time window.",
    )
    DEV_LIVENESS_PREVIEW_EMA_ALPHA: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="EMA smoothing factor for live liveness preview metrics.",
    )
    DEV_LIVENESS_PREVIEW_SHOW_DEBUG: bool = Field(
        default=True,
        description="Show extended per-frame debug metrics in the live liveness preview overlay.",
    )
    DEV_LIVENESS_PREVIEW_FLASH_REPLAY_ENABLED: bool = Field(
        default=True,
        description="Drive the developer preview with debug-only screen flash challenges and compute flash replay metrics.",
    )
    DEV_LIVENESS_PREVIEW_FLASH_INTERVAL_SECONDS: float = Field(
        default=2.5,
        ge=0.0,
        le=10.0,
        description="Seconds between debug flash challenges in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_FLASH_HISTORY_SIZE: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of recent flash challenges averaged into the flash replay metrics.",
    )
    DEV_LIVENESS_PREVIEW_PUZZLE_ENABLED: bool = Field(
        default=True,
        description="Enable Biometric Puzzle controls and active-score fusion in the developer live preview.",
    )
    DEV_LIVENESS_PREVIEW_PUZZLE_DIFFICULTY: Literal["easy", "standard", "hard"] = Field(
        default="standard",
        description="Default Biometric Puzzle difficulty used when starting a preview challenge session.",
    )
    DEV_LIVENESS_PREVIEW_PUZZLE_MIN_STEPS: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Minimum number of puzzle steps generated for the developer live preview.",
    )
    DEV_LIVENESS_PREVIEW_PUZZLE_MAX_STEPS: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Maximum number of puzzle steps generated for the developer live preview.",
    )
    DEV_LIVENESS_PREVIEW_PUZZLE_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Overall timeout for a preview Biometric Puzzle session.",
    )
    DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_BACKGROUND_WEIGHT: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Default fusion weight for background active evidence when a preview puzzle session is active.",
    )
    DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_PUZZLE_WEIGHT: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Default fusion weight for puzzle active evidence when a preview puzzle session is active.",
    )
    DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_MOIRE: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Replay fusion weight for moire risk in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_REFLECTION: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Replay fusion weight for reflection risk in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLICKER: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Replay fusion weight for flicker risk in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLASH: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Replay fusion weight for flash replay risk in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_SCREEN_FRAME: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional replay fusion weight for screen-frame risk in the live liveness preview.",
    )
    DEV_LIVENESS_PREVIEW_LOG_EVERY_N_FRAMES: int = Field(
        default=0,
        ge=0,
        le=600,
        description="Optional console logging cadence for preview metrics. 0 disables logging.",
    )
    DEV_LIVENESS_PREVIEW_INFERENCE_MAX_SIDE: int = Field(
        default=960,
        ge=0,
        le=2160,
        description="Resize preview frames so the largest inference side does not exceed this value. 0 disables resizing.",
    )
    DEV_LIVENESS_PREVIEW_DETECT_EVERY_N_FRAMES: int = Field(
        default=2,
        ge=1,
        le=30,
        description="Run full face detection every N preview frames and reuse the last face box in between.",
    )
    DEV_LIVENESS_PREVIEW_LANDMARK_EVERY_N_FRAMES: int = Field(
        default=2,
        ge=1,
        le=30,
        description="Run landmark extraction every N preview frames and reuse the last landmark metrics in between.",
    )
    DEV_LIVENESS_PREVIEW_LIVENESS_EVERY_N_FRAMES: int = Field(
        default=2,
        ge=1,
        le=30,
        description="Run full liveness inference every N preview frames and reuse the last liveness result in between.",
    )
    DEV_LIVENESS_PREVIEW_HOLD_LAST_SUCCESS_FRAMES: int = Field(
        default=8,
        ge=0,
        le=60,
        description="Hold the last successful face/liveness result for a few preview frames when detection fails temporarily.",
    )
    DEV_LIVENESS_PREVIEW_NO_FACE_CONSECUTIVE_THRESHOLD: int = Field(
        default=4,
        ge=1,
        le=30,
        description="Number of consecutive no-face frames required before the preview enters NO_FACE state.",
    )
    DEV_LIVENESS_PREVIEW_FACE_RETURN_GRACE_SECONDS: float = Field(
        default=1.0,
        ge=0.2,
        le=5.0,
        description="Warm-recovery grace period that retains recent valid temporal context after a brief face loss.",
    )
    DEV_LIVENESS_PREVIEW_FACE_LOSS_RESET_SECONDS: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="If face is lost longer than this, preview baseline and active temporal state may fully reset.",
    )
    DEV_LIVENESS_PREVIEW_ACTIVE_DECAY_SECONDS: float = Field(
        default=1.25,
        ge=0.2,
        le=5.0,
        description="Decay time constant for persisted background active evidence after an event occurs.",
    )
    DEV_LIVENESS_PREVIEW_MIN_TRUSTED_FACE_SIZE_RATIO: float = Field(
        default=0.06,
        ge=0.01,
        le=0.5,
        description="Minimum face area ratio before active landmark evidence is trusted strongly.",
    )
    DEV_LIVENESS_PREVIEW_FACE_BOX_SIDE_PADDING_RATIO: float = Field(
        default=0.07,
        ge=0.0,
        le=0.6,
        description="Extra horizontal padding ratio added to preview face boxes on each side.",
    )
    DEV_LIVENESS_PREVIEW_FACE_BOX_TOP_PADDING_RATIO: float = Field(
        default=0.10,
        ge=0.0,
        le=0.6,
        description="Extra top padding ratio added to preview face boxes.",
    )
    DEV_LIVENESS_PREVIEW_FACE_BOX_BOTTOM_PADDING_RATIO: float = Field(
        default=0.12,
        ge=0.0,
        le=0.8,
        description="Extra bottom padding ratio added to preview face boxes to avoid chin clipping.",
    )

    # Batch Processing
    BATCH_MAX_CONCURRENT: int = Field(default=5, ge=1, le=20)
    BATCH_ADAPTIVE_CONCURRENCY: bool = Field(default=False)

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Ensure JWT_SECRET is set when JWT is enabled."""
        jwt_enabled = info.data.get("JWT_ENABLED", True)
        environment = info.data.get("ENVIRONMENT", "development")
        if jwt_enabled and not v:
            if environment == "production":
                raise ValueError(
                    "SECURITY ERROR: JWT_SECRET must be set when JWT_ENABLED=True in production. "
                    "Set JWT_SECRET environment variable with a secure 256-bit key."
                )
            import warnings
            warnings.warn(
                "JWT_SECRET is empty while JWT_ENABLED=True. "
                "Set JWT_SECRET environment variable for secure JWT validation.",
                stacklevel=2,
            )
        return v

    @field_validator("PROCTOR_RISK_THRESHOLD_CRITICAL")
    @classmethod
    def validate_risk_thresholds(cls, v, info):
        """Ensure critical threshold is higher than warning."""
        warning = info.data.get("PROCTOR_RISK_THRESHOLD_WARNING", 0.5)
        if v <= warning:
            raise ValueError("critical threshold must be greater than warning threshold")
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v):
        """Additional validation for production environment."""
        # Can add production-specific checks here
        return v

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"

    def should_run_dev_liveness_preview(self) -> bool:
        """Return whether the dev-only live liveness preview may run."""
        return self.is_development() and self.DEV_LIVENESS_PREVIEW

    def is_strict_exam_security_profile(self) -> bool:
        """Return whether strict exam anti-spoof behavior is enabled."""
        return False

    def get_liveness_security_profile(self) -> str:
        """Return the configured liveness security profile name."""
        return "standard"

    def get_dev_preview_device_replay_weights(self) -> dict[str, float]:
        """Return effective replay-fusion weights for the developer preview."""
        return {
            "moire": self.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_MOIRE,
            "reflection": self.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_REFLECTION,
            "flicker": self.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLICKER,
            "flash": self.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_FLASH,
            "screen_frame": self.DEV_LIVENESS_PREVIEW_DEVICE_REPLAY_WEIGHT_SCREEN_FRAME,
        }

    def get_dev_preview_active_fusion_weights(self) -> dict[str, float]:
        """Return normalized active-evidence fusion weights for preview puzzle sessions."""
        background = max(0.0, float(self.DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_BACKGROUND_WEIGHT))
        puzzle = max(0.0, float(self.DEV_LIVENESS_PREVIEW_ACTIVE_FUSION_PUZZLE_WEIGHT))
        total = background + puzzle
        if total <= 1e-6:
            return {"background": 0.40, "puzzle": 0.60}
        return {
            "background": background / total,
            "puzzle": puzzle / total,
        }

    def get_strict_sigmoid_config(self) -> dict[str, float]:
        """Return the normalized sigmoid parameters used by strict liveness scoring."""
        return {
            "midpoint": 0.62,
            "steepness": 12.0,
            "scale": 100.0,
        }

    def get_strict_micro_texture_config(self) -> dict[str, float]:
        """Return strict-mode micro-texture and moire spoof-support weights."""
        return {
            "micro_texture_weight": 1.0,
            "moire_support_weight": 1.0,
            "cutout_support_weight": 1.0,
        }

    def get_strict_exam_decision_config(self) -> dict[str, float]:
        """Return strict-exam decision-layer penalties and escalation thresholds."""
        return {
            "replay_penalty_max": 0.0,
            "spoof_support_penalty_max": 0.0,
            "challenge_penalty": 0.0,
            "hard_block_replay_risk": 1.0,
            "hard_block_spoof_support": 1.0,
        }

    def get_cors_config(self) -> dict:
        """Get CORS configuration.

        SECURITY FIX: Validates that wildcard CORS is never used in production.
        This prevents accidental misconfiguration that could expose the API to attacks.
        """
        # CRITICAL: Never allow wildcard in production
        if self.is_production() and ("*" in self.CORS_ORIGINS or len(self.CORS_ORIGINS) == 0):
            raise ValueError(
                "SECURITY ERROR: Wildcard CORS ('*') or empty CORS_ORIGINS is not allowed in production. "
                "Please configure specific allowed origins in CORS_ORIGINS environment variable."
            )

        # In development, allow all origins for easier testing
        # In production, use the configured CORS_ORIGINS
        origins = ["*"] if self.is_development() else self.CORS_ORIGINS

        return {
            "allow_origins": origins,
            "allow_credentials": False if origins == ["*"] else True,  # Can't use credentials with wildcard
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    def get_proctor_session_config(self) -> dict:
        """Get proctoring session configuration for session creation."""
        return {
            "verification_interval_sec": self.PROCTOR_VERIFICATION_INTERVAL_SEC,
            "verification_threshold": self.PROCTOR_VERIFICATION_THRESHOLD,
            "liveness_threshold": self.PROCTOR_LIVENESS_THRESHOLD,
            "gaze_away_threshold_sec": self.PROCTOR_GAZE_AWAY_THRESHOLD_SEC,
            "risk_threshold_warning": self.PROCTOR_RISK_THRESHOLD_WARNING,
            "risk_threshold_critical": self.PROCTOR_RISK_THRESHOLD_CRITICAL,
            "enable_gaze_tracking": self.PROCTOR_GAZE_ENABLED,
            "enable_object_detection": self.PROCTOR_OBJECT_DETECTION_ENABLED,
            "enable_audio_monitoring": self.PROCTOR_AUDIO_ENABLED,
        }

    def get_proctor_ml_config(self) -> dict:
        """Get proctoring ML component configuration."""
        return {
            "gaze_threshold": self.PROCTOR_GAZE_THRESHOLD,
            "head_pose_threshold": (self.PROCTOR_HEAD_PITCH_THRESHOLD, self.PROCTOR_HEAD_YAW_THRESHOLD),
            "object_model_size": self.PROCTOR_OBJECT_MODEL_SIZE,
            "object_confidence_threshold": self.PROCTOR_OBJECT_CONFIDENCE_THRESHOLD,
            "max_persons_allowed": self.PROCTOR_MAX_PERSONS_ALLOWED,
            "deepfake_threshold": self.PROCTOR_DEEPFAKE_THRESHOLD,
            "deepfake_temporal_window": self.PROCTOR_DEEPFAKE_TEMPORAL_WINDOW,
            "audio_sample_rate": self.PROCTOR_AUDIO_SAMPLE_RATE,
            "audio_vad_threshold": self.PROCTOR_AUDIO_VAD_THRESHOLD,
        }

    def get_proctor_rate_limit_config(self) -> dict:
        """Get proctoring rate limit configuration."""
        return {
            "enabled": self.PROCTOR_RATE_LIMIT_ENABLED,
            "max_frames_per_second": self.PROCTOR_MAX_FRAMES_PER_SECOND,
            "max_frames_per_minute": self.PROCTOR_MAX_FRAMES_PER_MINUTE,
            "burst_allowance": self.PROCTOR_RATE_LIMIT_BURST_ALLOWANCE,
        }

    def get_proctor_circuit_breaker_config(self) -> dict:
        """Get proctoring circuit breaker configuration."""
        return {
            "enabled": self.PROCTOR_CIRCUIT_BREAKER_ENABLED,
            "failure_threshold": self.PROCTOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "success_threshold": self.PROCTOR_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
            "timeout_seconds": self.PROCTOR_CIRCUIT_BREAKER_TIMEOUT_SEC,
        }

    @property
    def port(self) -> int:
        """Get port from PORT env var (PaaS standard) or API_PORT.

        PaaS platforms (Railway, Render, Heroku) typically provide a PORT
        environment variable. This property checks for PORT first, then
        falls back to API_PORT.
        """
        return int(os.getenv("PORT", self.API_PORT))


# Singleton settings instance
settings = Settings()


def get_settings() -> Settings:
    """Return the singleton settings instance."""
    return settings


# Create upload folder on import
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
