"""Comprehensive logging configuration with structured logging and correlation IDs.

This module provides:
- Structured JSON logging for production
- Request correlation IDs for distributed tracing
- Security event logging
- Performance monitoring
- Error tracking
"""

import atexit
import logging
import logging.config
import logging.handlers
import queue
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import json
import contextvars

from app.core.config import settings

# Context variable for request correlation ID
request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to log record if available."""
        record.request_id = request_id_context.get() or "N/A"
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, 'request_id', 'N/A'),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'tenant_id'):
            log_data["tenant_id"] = record.tenant_id
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, 'event_type'):
            log_data["event_type"] = record.event_type
        if hasattr(record, 'payload') and isinstance(record.payload, dict):
            log_data.update(record.payload)

        return json.dumps(log_data)


class SecurityEventLogger:
    """Dedicated logger for security events."""

    def __init__(self):
        self.logger = logging.getLogger("security")

    def log_authentication_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log authentication attempt."""
        extra = {
            "event_type": "authentication_attempt",
            "user_id": user_id,
            "success": success,
            "ip_address": ip_address or "unknown",
            "user_agent": user_agent or "unknown",
        }
        
        if success:
            self.logger.info("Authentication successful", extra=extra)
        else:
            self.logger.warning("Authentication failed", extra=extra)

    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
    ):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization denied",
            extra={
                "event_type": "authorization_failure",
                "user_id": user_id,
                "resource": resource,
                "action": action,
            }
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ):
        """Log sensitive data access."""
        self.logger.info(
            "Data access",
            extra={
                "event_type": "data_access",
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
            }
        )

    def log_suspicious_activity(
        self,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log suspicious activity."""
        extra = {
            "event_type": "suspicious_activity",
            "description": description,
        }
        if user_id:
            extra["user_id"] = user_id
        if ip_address:
            extra["ip_address"] = ip_address
        if details:
            extra.update(details)

        self.logger.warning("Suspicious activity detected", extra=extra)


# Global queue listener for async-safe logging
_queue_listener: Optional[logging.handlers.QueueListener] = None


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "correlation_id": {
            "()": CorrelationIdFilter,
        },
    },
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(request_id)s] %(levelname)s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "structured": {
            "()": StructuredFormatter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG" if settings.DEBUG else "INFO",
            "formatter": "structured" if settings.ENVIRONMENT == "production" else "default",
            "stream": sys.stdout,
            "filters": ["correlation_id"],
        },
        # QueueHandler for async-safe file logging (non-blocking)
        "file_queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": None,  # Will be set in setup_logging()
            "filters": ["correlation_id"],
        },
        "security_queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": None,  # Will be set in setup_logging()
            "filters": ["correlation_id"],
        },
        "error_queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": None,  # Will be set in setup_logging()
            "filters": ["correlation_id"],
        },
        "calibration_queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": None,  # Will be set in setup_logging()
            "filters": ["correlation_id"],
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file_queue", "error_queue"],
            "level": "DEBUG" if settings.DEBUG else "INFO",
            "propagate": False,
        },
        "security": {
            "handlers": ["security_queue", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["console", "file_queue"],
            "level": "INFO",
            "propagate": False,
        },
        "liveness_calibration": {
            "handlers": ["calibration_queue"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Setup async-safe logging configuration with QueueHandler.

    This implementation uses QueueHandler + QueueListener to ensure that
    file I/O operations don't block the async event loop. All log writes
    are handled in a separate thread.

    Architecture:
        - Loggers write to QueueHandler (non-blocking, ~1μs overhead)
        - QueueListener runs in background thread
        - Listener writes to RotatingFileHandler (blocking, but separate thread)
        - Event loop remains responsive during log writes

    Thread Safety:
        - Queue operations are thread-safe
        - Listener runs in dedicated thread
        - No blocking of async event loop
    """
    import os
    global _queue_listener

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    calibration_log_dir = os.path.dirname(settings.LIVENESS_CALIBRATION_LOG_PATH)
    if calibration_log_dir:
        os.makedirs(calibration_log_dir, exist_ok=True)

    # Create log queues (unbounded for reliability)
    file_queue = queue.Queue(-1)  # -1 = unbounded
    security_queue = queue.Queue(-1)
    error_queue = queue.Queue(-1)
    calibration_queue = queue.Queue(-1)

    # Update config with queue instances
    LOGGING_CONFIG["handlers"]["file_queue"]["queue"] = file_queue
    LOGGING_CONFIG["handlers"]["security_queue"]["queue"] = security_queue
    LOGGING_CONFIG["handlers"]["error_queue"]["queue"] = error_queue
    LOGGING_CONFIG["handlers"]["calibration_queue"]["queue"] = calibration_queue

    # Apply logging configuration
    logging.config.dictConfig(LOGGING_CONFIG)

    # Create actual file handlers (will run in separate thread via QueueListener)
    correlation_filter = CorrelationIdFilter()
    structured_formatter = StructuredFormatter()

    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(structured_formatter)
    file_handler.addFilter(correlation_filter)

    security_handler = logging.handlers.RotatingFileHandler(
        filename="logs/security.log",
        maxBytes=10485760,  # 10MB
        backupCount=10,
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(structured_formatter)
    security_handler.addFilter(correlation_filter)

    error_handler = logging.handlers.RotatingFileHandler(
        filename="logs/error.log",
        maxBytes=10485760,  # 10MB
        backupCount=10,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(structured_formatter)
    error_handler.addFilter(correlation_filter)

    calibration_handler = logging.handlers.RotatingFileHandler(
        filename=settings.LIVENESS_CALIBRATION_LOG_PATH,
        maxBytes=10485760,  # 10MB
        backupCount=10,
    )
    calibration_handler.setLevel(logging.INFO)
    calibration_handler.setFormatter(structured_formatter)
    calibration_handler.addFilter(correlation_filter)

    # Create QueueListeners to process log records in separate threads
    # Each listener handles file I/O operations off the main event loop
    file_listener = logging.handlers.QueueListener(
        file_queue,
        file_handler,
        respect_handler_level=True
    )

    security_listener = logging.handlers.QueueListener(
        security_queue,
        security_handler,
        respect_handler_level=True
    )

    error_listener = logging.handlers.QueueListener(
        error_queue,
        error_handler,
        respect_handler_level=True
    )

    calibration_listener = logging.handlers.QueueListener(
        calibration_queue,
        calibration_handler,
        respect_handler_level=True
    )

    # Start all queue listener threads
    file_listener.start()
    security_listener.start()
    error_listener.start()
    calibration_listener.start()

    # Store listeners for cleanup
    global _queue_listener
    _queue_listener = (file_listener, security_listener, error_listener, calibration_listener)

    # Register cleanup handler to stop listeners on shutdown
    atexit.register(_shutdown_logging)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Async-safe logging initialized with QueueHandler (3 listeners): "
        f"environment={settings.ENVIRONMENT}, debug={settings.DEBUG}"
    )


def _shutdown_logging():
    """Stop all queue listeners on application shutdown."""
    global _queue_listener
    if _queue_listener:
        for listener in _queue_listener:
            listener.stop()
        _queue_listener = None


# Global security event logger instance
security_logger = SecurityEventLogger()
