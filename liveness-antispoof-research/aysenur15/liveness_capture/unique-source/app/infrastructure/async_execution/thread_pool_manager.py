"""Thread pool manager for CPU-bound ML operations.

This module provides a centralized thread pool for executing blocking
ML operations without blocking the async event loop.

Following:
- Singleton Pattern: Single instance via container.py @lru_cache
- Single Responsibility: Only manages thread pool lifecycle
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

from app.domain.exceptions.face_errors import FaceNotDetectedError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadPoolManager:
    """Centralized thread pool for CPU-bound ML operations.

    This manager provides a shared thread pool for executing blocking
    operations (like DeepFace inference) without blocking the event loop.

    Thread Safety:
        The ThreadPoolExecutor is internally thread-safe. This manager
        simply wraps it with async-friendly methods.

    Usage:
        pool = ThreadPoolManager(max_workers=4)

        # In async context:
        result = await pool.run_blocking(blocking_func, arg1, arg2)

        # Cleanup on shutdown:
        pool.shutdown()

    Attributes:
        max_workers: Maximum number of worker threads
        thread_name_prefix: Prefix for thread names (for debugging)
    """

    def __init__(
        self,
        max_workers: int = None,
        thread_name_prefix: str = "bio-ml",
    ) -> None:
        """Initialize thread pool manager.

        Args:
            max_workers: Maximum worker threads. Defaults to CPU count.
            thread_name_prefix: Prefix for worker thread names.

        Note:
            For ML workloads, recommended workers = CPU cores.
            Too many workers can cause memory issues with large models.
        """
        self._max_workers = max_workers or (os.cpu_count() or 4)
        self._thread_name_prefix = thread_name_prefix
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._shutdown = False

        logger.info(
            f"ThreadPoolManager initialized: max_workers={self._max_workers}, "
            f"prefix={thread_name_prefix}"
        )

    async def run_blocking(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a blocking function in the thread pool.

        This method offloads CPU-bound operations to the thread pool,
        allowing the event loop to remain responsive.

        Args:
            func: The blocking function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            RuntimeError: If the pool has been shut down
            Exception: Any exception raised by the function

        Example:
            # Blocking DeepFace call:
            result = await pool.run_blocking(
                DeepFace.extract_faces,
                img_path=image,
                detector_backend="opencv"
            )
        """
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager has been shut down")

        loop = asyncio.get_running_loop()

        # Use partial to bind kwargs since run_in_executor only takes args
        if kwargs:
            func = partial(func, **kwargs)

        try:
            result = await loop.run_in_executor(self._executor, func, *args)
            return result
        except FaceNotDetectedError as e:
            logger.info(f"Thread pool execution returned no-face result: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in thread pool execution: {e}", exc_info=True)
            raise

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shutdown the thread pool gracefully.

        Args:
            wait: If True, wait for pending futures to complete
            cancel_futures: If True, cancel pending futures (Python 3.9+)

        Note:
            Call this during application shutdown to ensure clean exit.
            After shutdown, run_blocking will raise RuntimeError.
        """
        if self._shutdown:
            logger.warning("ThreadPoolManager already shut down")
            return

        self._shutdown = True

        logger.info(
            f"Shutting down ThreadPoolManager: wait={wait}, "
            f"cancel_futures={cancel_futures}"
        )

        try:
            # Python 3.9+ supports cancel_futures parameter
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            # Python 3.8 fallback
            self._executor.shutdown(wait=wait)

        logger.info("ThreadPoolManager shut down complete")

    @property
    def max_workers(self) -> int:
        """Get maximum number of worker threads."""
        return self._max_workers

    @property
    def is_shutdown(self) -> bool:
        """Check if the pool has been shut down."""
        return self._shutdown

    def __repr__(self) -> str:
        status = "shutdown" if self._shutdown else "active"
        return (
            f"ThreadPoolManager(max_workers={self._max_workers}, "
            f"prefix='{self._thread_name_prefix}', status={status})"
        )
