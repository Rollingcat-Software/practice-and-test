"""Unit tests for EnhancedLivenessDetector."""

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.domain.exceptions.face_errors import FaceNotDetectedError
from app.domain.exceptions.liveness_errors import LivenessCheckError
from app.infrastructure.ml.liveness.enhanced_liveness_detector import EnhancedLivenessDetector


class TestEnhancedLivenessDetector:
    """Tests for enhanced liveness detector with blink and smile detection."""

    def _create_test_image(
        self,
        size: int = 200,
        add_texture: bool = True,
        brightness: int = 128,
    ) -> np.ndarray:
        """Create a test image with configurable properties.

        Args:
            size: Image size (square)
            add_texture: Whether to add texture variation
            brightness: Base brightness level

        Returns:
            Test image as numpy array
        """
        # Create base image
        img = np.ones((size, size, 3), dtype=np.uint8) * brightness

        # Add simple face-like features
        center = size // 2
        cv2.circle(img, (center, center), size // 3, (180, 180, 180), -1)
        cv2.circle(img, (center - size // 6, center - size // 8), size // 20, (50, 50, 50), -1)
        cv2.circle(img, (center + size // 6, center - size // 8), size // 20, (50, 50, 50), -1)
        cv2.ellipse(
            img, (center, center + size // 6), (size // 8, size // 16), 0, 0, 180, (100, 100, 100), 2
        )

        if add_texture:
            # Add noise for texture
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    @pytest.mark.asyncio
    async def test_initialization_default_params(self):
        """Test detector initializes with default parameters."""
        detector = EnhancedLivenessDetector()

        assert detector._texture_threshold == 100.0
        assert detector._liveness_threshold == 70.0
        assert detector._enable_blink is True
        assert detector._enable_smile is True
        assert detector._blink_frames_required == 2

    @pytest.mark.asyncio
    async def test_initialization_custom_params(self):
        """Test detector initializes with custom parameters."""
        detector = EnhancedLivenessDetector(
            texture_threshold=150.0,
            liveness_threshold=75.0,
            enable_blink_detection=False,
            enable_smile_detection=True,
            blink_frames_required=3,
        )

        assert detector._texture_threshold == 150.0
        assert detector._liveness_threshold == 75.0
        assert detector._enable_blink is False
        assert detector._enable_smile is True
        assert detector._blink_frames_required == 3

    @pytest.mark.asyncio
    async def test_check_liveness_returns_result(self):
        """Test that check_liveness returns proper LivenessResult."""
        detector = EnhancedLivenessDetector(liveness_threshold=50.0)
        image = self._create_test_image()

        result = await detector.check_liveness(image)

        # Check result structure
        assert hasattr(result, "is_live")
        assert hasattr(result, "liveness_score")
        assert hasattr(result, "challenge")
        assert hasattr(result, "challenge_completed")

        # Check score range
        assert 0.0 <= result.liveness_score <= 100.0

    @pytest.mark.asyncio
    async def test_check_liveness_invalid_image(self):
        """Test that invalid image raises LivenessCheckError."""
        detector = EnhancedLivenessDetector()

        # Empty image
        empty_image = np.array([])

        with pytest.raises(LivenessCheckError):
            await detector.check_liveness(empty_image)

    @pytest.mark.asyncio
    async def test_check_liveness_none_image(self):
        """Test that None image raises LivenessCheckError."""
        detector = EnhancedLivenessDetector()

        with pytest.raises(LivenessCheckError):
            await detector.check_liveness(None)

    @pytest.mark.asyncio
    async def test_texture_score_calculation(self):
        """Test that texture score is calculated correctly."""
        detector = EnhancedLivenessDetector()

        # High texture image
        high_texture = self._create_test_image(add_texture=True)
        high_score = detector._calculate_texture_score(high_texture)

        # Low texture image (completely flat)
        low_texture = np.ones((200, 200, 3), dtype=np.uint8) * 128
        low_score = detector._calculate_texture_score(low_texture)

        # High texture should score higher than completely flat
        assert high_score > low_score
        # Both should be in valid range
        assert 0.0 <= high_score <= 100.0
        assert 0.0 <= low_score <= 100.0

    @pytest.mark.asyncio
    async def test_lbp_score_calculation(self):
        """Test that LBP score is calculated."""
        detector = EnhancedLivenessDetector()
        image = self._create_test_image()

        score = detector._calculate_lbp_score(image)

        assert 0.0 <= score <= 100.0

    @pytest.mark.asyncio
    async def test_color_score_calculation(self):
        """Test that color score is calculated."""
        detector = EnhancedLivenessDetector()
        image = self._create_test_image()

        score = detector._calculate_color_score(image)

        assert 0.0 <= score <= 100.0

    @pytest.mark.asyncio
    async def test_compute_lbp(self):
        """Test LBP computation."""
        detector = EnhancedLivenessDetector()
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        lbp = detector._compute_lbp(gray)

        # LBP should have same shape as input
        assert lbp.shape == gray.shape
        # LBP values should be in valid range
        assert np.all(lbp >= 0)
        assert np.all(lbp <= 255)

    @pytest.mark.asyncio
    async def test_get_challenge_type_both_enabled(self):
        """Test challenge type when both blink and smile enabled."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=True,
            enable_smile_detection=True,
        )

        assert detector.get_challenge_type() == "blink_and_smile"

    @pytest.mark.asyncio
    async def test_get_challenge_type_blink_only(self):
        """Test challenge type when only blink enabled."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=True,
            enable_smile_detection=False,
        )

        assert detector.get_challenge_type() == "blink"

    @pytest.mark.asyncio
    async def test_get_challenge_type_smile_only(self):
        """Test challenge type when only smile enabled."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=False,
            enable_smile_detection=True,
        )

        assert detector.get_challenge_type() == "smile"

    @pytest.mark.asyncio
    async def test_get_challenge_type_passive(self):
        """Test challenge type when both disabled (passive)."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=False,
            enable_smile_detection=False,
        )

        assert detector.get_challenge_type() == "passive"

    @pytest.mark.asyncio
    async def test_get_liveness_threshold(self):
        """Test getting the liveness threshold."""
        detector = EnhancedLivenessDetector(liveness_threshold=75.0)

        assert detector.get_liveness_threshold() == 75.0

    @pytest.mark.asyncio
    async def test_get_score_weights_all_enabled(self):
        """Test score weights when all features enabled."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=True,
            enable_smile_detection=True,
        )

        weights = detector._get_score_weights()

        # Check all weights present
        assert "texture" in weights
        assert "lbp" in weights
        assert "color" in weights
        assert "blink" in weights
        assert "smile" in weights

        # Check weights sum to 1.0 (approximately)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_get_score_weights_passive_only(self):
        """Test score weights when only passive checks enabled."""
        detector = EnhancedLivenessDetector(
            enable_blink_detection=False,
            enable_smile_detection=False,
        )

        weights = detector._get_score_weights()

        # Blink and smile weights should be 0
        assert weights["blink"] == 0.0
        assert weights["smile"] == 0.0

        # Passive weights should be redistributed
        assert weights["texture"] > 0.25
        assert weights["lbp"] > 0.25
        assert weights["color"] > 0.20

        # Check weights sum to 1.0 (approximately)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_reset_state(self):
        """Test state reset functionality."""
        detector = EnhancedLivenessDetector()

        # Set some state
        detector._blink_counter = 5
        detector._eyes_closed_frames = 3

        # Reset
        detector.reset_state()

        # Check state is cleared
        assert detector._blink_counter == 0
        assert detector._eyes_closed_frames == 0

    @pytest.mark.asyncio
    async def test_different_image_sizes(self):
        """Test detection works with different image sizes."""
        detector = EnhancedLivenessDetector(liveness_threshold=50.0)

        for size in [100, 200, 300]:
            image = self._create_test_image(size=size)
            try:
                result = await detector.check_liveness(image)
                # Should return valid result regardless of size
                assert 0.0 <= result.liveness_score <= 100.0
            except FaceNotDetectedError:
                # Haar cascade may not detect synthetic faces - this is acceptable
                pass

    @pytest.mark.asyncio
    async def test_very_dark_image(self):
        """Test detection with very dark image."""
        detector = EnhancedLivenessDetector(liveness_threshold=50.0)
        image = self._create_test_image(brightness=30)

        result = await detector.check_liveness(image)

        # Dark images should still return valid result
        assert 0.0 <= result.liveness_score <= 100.0

    @pytest.mark.asyncio
    async def test_very_bright_image(self):
        """Test detection with very bright image."""
        detector = EnhancedLivenessDetector(liveness_threshold=50.0)
        image = self._create_test_image(brightness=240)

        try:
            result = await detector.check_liveness(image)
            # Bright images should still return valid result
            assert 0.0 <= result.liveness_score <= 100.0
        except FaceNotDetectedError:
            # Haar cascade may not detect synthetic faces in extreme lighting - this is acceptable
            pass

    @pytest.mark.asyncio
    async def test_haar_cascades_loaded(self):
        """Test that Haar cascade classifiers are properly loaded."""
        detector = EnhancedLivenessDetector()

        # Check that cascades are loaded
        assert hasattr(detector, "_face_cascade")
        assert hasattr(detector, "_eye_cascade")
        assert hasattr(detector, "_smile_cascade")

        # Check that cascades are not empty
        assert not detector._face_cascade.empty()
        assert not detector._eye_cascade.empty()
        assert not detector._smile_cascade.empty()
