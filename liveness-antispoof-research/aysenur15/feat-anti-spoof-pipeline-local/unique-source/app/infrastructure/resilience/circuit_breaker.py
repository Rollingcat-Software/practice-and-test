"""Circuit breaker pattern for ML service resilience."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Optional, TypeVar

from prometheus_client import Counter, Gauge

from app.domain.exceptions.face_errors import FaceNotDetectedError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation, requests pass through
    OPEN = "open"           # Failures exceeded, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    excluded_exceptions: tuple = field(default_factory=tuple)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open", component: str = None):
        super().__init__(message)
        self.component = component


T = TypeVar("T")


# Prometheus metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "biometric_proctor_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["component"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "biometric_proctor_circuit_breaker_trips_total",
    "Circuit breaker trip count",
    ["component"],
)

CIRCUIT_BREAKER_FALLBACKS = Counter(
    "biometric_proctor_circuit_breaker_fallbacks_total",
    "Circuit breaker fallback invocations",
    ["component"],
)


class CircuitBreaker(Generic[T]):
    """Circuit breaker for ML service resilience."""

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        name: str = "default",
    ):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        self._update_metrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try half-open."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._update_metrics()
        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
        )

    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2,
        }
        CIRCUIT_BREAKER_STATE.labels(component=self.name).set(state_value[self._state])

    def call(
        self,
        func: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        """Execute function with circuit breaker protection."""
        state = self.state

        if state == CircuitState.OPEN:
            if fallback:
                CIRCUIT_BREAKER_FALLBACKS.labels(component=self.name).inc()
                logger.warning(
                    f"Circuit breaker '{self.name}' is open, using fallback"
                )
                return fallback()
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                component=self.name,
            )

        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.excluded_exceptions):
                raise
            self._on_failure()
            if fallback:
                CIRCUIT_BREAKER_FALLBACKS.labels(component=self.name).inc()
                return fallback()
            raise

    async def call_async(
        self,
        func: Callable,
        fallback: Optional[Callable] = None,
    ) -> T:
        """Execute async function with circuit breaker protection."""
        state = self.state

        if state == CircuitState.OPEN:
            if fallback:
                CIRCUIT_BREAKER_FALLBACKS.labels(component=self.name).inc()
                logger.warning(
                    f"Circuit breaker '{self.name}' is open, using fallback"
                )
                return await fallback() if callable(fallback) else fallback
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                component=self.name,
            )

        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.excluded_exceptions):
                raise
            self._on_failure()
            if fallback:
                CIRCUIT_BREAKER_FALLBACKS.labels(component=self.name).inc()
                return await fallback() if callable(fallback) else fallback
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0
                CIRCUIT_BREAKER_TRIPS.labels(component=self.name).inc()
            elif self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                CIRCUIT_BREAKER_TRIPS.labels(component=self.name).inc()

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            }


# Pre-configured circuit breakers for ML components
FACE_DETECTOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout_seconds=30.0,
        excluded_exceptions=(FaceNotDetectedError,),
    ),
    name="face_detector",
)

EMBEDDING_EXTRACTOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="embedding_extractor",
)

QUALITY_ASSESSOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="quality_assessor",
)

FACE_VERIFIER_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0),
    name="face_verifier",
)

LIVENESS_DETECTOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="liveness_detector",
)

DEEPFAKE_DETECTOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=60.0),
    name="deepfake_detector",
)

GAZE_TRACKER_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="gaze_tracker",
)

OBJECT_DETECTOR_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="object_detector",
)

AUDIO_ANALYZER_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=30.0),
    name="audio_analyzer",
)
