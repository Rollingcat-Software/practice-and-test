"""Compatibility factory for creating liveness detectors.

This module keeps the public ``LivenessDetectorFactory`` symbol available while
aligning detector selection with the current application architecture:

- Canonical configuration comes from ``LIVENESS_MODE``
- Explicit backend override comes from ``LIVENESS_BACKEND``
- Effective runtime backends are ``enhanced``, ``texture``, and ``uniface``

Legacy mode names (``passive``, ``active``, ``combined``, ``stub``) are still
accepted to avoid breaking older imports and scripts.
"""

import logging
import os
from typing import Literal, Optional

from app.core.config import settings
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.infrastructure.ml.liveness.enhanced_liveness_detector import EnhancedLivenessDetector
from app.infrastructure.ml.liveness.hybrid_liveness_detector import HybridLivenessDetector
from app.infrastructure.ml.liveness.optimized_texture_liveness import OptimizedTextureLivenessDetector
from app.infrastructure.ml.liveness.stub_liveness_detector import StubLivenessDetector
from app.infrastructure.ml.liveness.texture_liveness_detector import TextureLivenessDetector
from app.infrastructure.ml.liveness.uniface_liveness_detector import UniFaceLivenessDetector

logger = logging.getLogger(__name__)

LivenessMode = Literal["passive", "active", "combined", "stub"]
LivenessBackend = Literal["enhanced", "texture", "uniface", "optimized", "hybrid"]
SupportedLivenessSelection = Literal[
    "passive",
    "active",
    "combined",
    "stub",
    "enhanced",
    "texture",
    "uniface",
    "optimized",
    "hybrid",
]


class LivenessDetectorFactory:
    """Factory for creating liveness detector instances.

    Prefer the dependency container for runtime use. This factory remains as a
    backwards-compatible construction helper for tests, scripts, and imports
    that still expect ``LivenessDetectorFactory`` to exist.
    """

    @staticmethod
    def create(
        mode: Optional[SupportedLivenessSelection] = None,
        liveness_threshold: Optional[float] = None,
        **kwargs,
    ) -> ILivenessDetector:
        """Create a liveness detector instance.

        Args:
            mode: Optional selection value. Supports both legacy modes
                (``passive``, ``active``, ``combined``, ``stub``) and current
                backends (``enhanced``, ``texture``, ``uniface``). When omitted,
                the current application settings are used.
            liveness_threshold: Optional threshold override. Falls back to the
                configured application threshold when omitted.
            **kwargs: Detector-specific compatibility arguments.

        Returns:
            A liveness detector implementation.
        """
        selection = (mode or settings.LIVENESS_BACKEND or settings.LIVENESS_MODE).lower()
        threshold = settings.LIVENESS_THRESHOLD if liveness_threshold is None else liveness_threshold

        logger.info("Creating liveness detector via factory: selection=%s", selection)

        if selection == "stub":
            return LivenessDetectorFactory._create_stub(threshold, **kwargs)

        backend = LivenessDetectorFactory._resolve_backend(selection)

        if backend == "uniface":
            return UniFaceLivenessDetector(
                liveness_threshold=threshold,
            )

        if backend == "hybrid":
            return HybridLivenessDetector(
                liveness_threshold=threshold,
            )

        if backend == "texture":
            return TextureLivenessDetector(
                texture_threshold=kwargs.get("texture_threshold", 100.0),
                color_threshold=kwargs.get("color_threshold", 0.3),
                frequency_threshold=kwargs.get("frequency_threshold", 0.5),
                liveness_threshold=threshold,
            )

        if backend == "optimized":
            fft_w = kwargs.get("fft_width", 192)
            fft_size = (fft_w, fft_w * 108 // 192)
            return OptimizedTextureLivenessDetector(
                texture_threshold=kwargs.get("texture_threshold", 100.0),
                color_threshold=kwargs.get("color_threshold", 0.3),
                frequency_threshold=kwargs.get("frequency_threshold", 0.5),
                liveness_threshold=threshold,
                fft_downsample_size=fft_size,
            )

        return EnhancedLivenessDetector(
            texture_threshold=kwargs.get("texture_threshold", 100.0),
            liveness_threshold=threshold,
            enable_blink_detection=kwargs.get("enable_blink_detection", True),
            enable_smile_detection=kwargs.get("enable_smile_detection", True),
            blink_frames_required=kwargs.get("blink_frames_required", 2),
        )

    @staticmethod
    def _resolve_backend(selection: str) -> LivenessBackend:
        """Resolve legacy mode/backend names to an effective backend."""
        if selection in ("enhanced", "texture", "uniface", "optimized", "hybrid"):
            return selection

        legacy_mode_to_backend: dict[str, LivenessBackend] = {
            "passive": "texture",
            "active": "enhanced",
            "combined": "enhanced",
        }
        if selection in legacy_mode_to_backend:
            logger.warning(
                "Legacy liveness mode '%s' used with LivenessDetectorFactory; "
                "mapping to backend '%s'. Prefer explicit backends.",
                selection,
                legacy_mode_to_backend[selection],
            )
            return legacy_mode_to_backend[selection]

        raise ValueError(
            f"Unsupported liveness selection: {selection!r}. "
            f"Supported values: {', '.join(LivenessDetectorFactory.get_available_modes())}"
        )

    @staticmethod
    def _create_stub(liveness_threshold: float, **kwargs) -> ILivenessDetector:
        """Create a stub detector only in non-production environments."""
        env = os.getenv("APP_ENV", "production").lower()
        if env not in ("development", "test", "testing", "ci"):
            logger.error(
                "StubLivenessDetector requested in non-test environment '%s'. "
                "Falling back to enhanced detector for safety.",
                env,
            )
            return EnhancedLivenessDetector(
                texture_threshold=kwargs.get("texture_threshold", 100.0),
                liveness_threshold=liveness_threshold,
                enable_blink_detection=kwargs.get("enable_blink_detection", True),
                enable_smile_detection=kwargs.get("enable_smile_detection", True),
                blink_frames_required=kwargs.get("blink_frames_required", 2),
            )

        logger.warning("Using StubLivenessDetector - only for testing environments")
        return StubLivenessDetector(default_score=kwargs.get("default_score", 85.0))

    @staticmethod
    def get_available_modes() -> list[str]:
        """Get supported legacy and current selection values."""
        modes = ["passive", "active", "combined", "enhanced", "texture", "uniface", "optimized", "hybrid"]
        env = os.getenv("APP_ENV", "production").lower()
        if env in ("development", "test", "testing", "ci"):
            modes.append("stub")
        return modes

    @staticmethod
    def get_recommended_mode() -> str:
        """Get the recommended selection value for current architecture."""
        return "enhanced"
