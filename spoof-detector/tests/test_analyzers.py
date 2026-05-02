"""Unit tests for each analyzer.

Tests each analyzer independently with synthetic and real images.
Verifies score ranges, timing, and expected behavior for known inputs.
"""

import time
import pytest
import numpy as np
import cv2

from src.domain.models import FaceROI, BBox, SpoofCategory, AnalyzerResult


def make_face_roi(face_id: int = 1, w: int = 200, h: int = 200) -> FaceROI:
    """Create a dummy FaceROI for testing."""
    return FaceROI(
        face_id=face_id,
        bbox=BBox(0, 0, w, h),
        confidence=0.95,
    )


def make_solid_image(w: int = 200, h: int = 200, color=(128, 128, 128)) -> np.ndarray:
    """Create a solid-color BGR image (very suspicious — no texture)."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def make_noisy_image(w: int = 200, h: int = 200) -> np.ndarray:
    """Create a noisy BGR image (natural texture)."""
    rng = np.random.RandomState(42)
    return rng.randint(50, 200, (h, w, 3), dtype=np.uint8)


def make_gradient_image(w: int = 200, h: int = 200) -> np.ndarray:
    """Create a gradient image (moderate texture)."""
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.merge([grad, grad, grad])


def make_moire_image(w: int = 200, h: int = 200, freq: float = 0.15) -> np.ndarray:
    """Create an image with moire-like periodic patterns."""
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pattern = ((np.sin(freq * xx) * np.sin(freq * yy) + 1) * 127).astype(np.uint8)
    return cv2.merge([pattern, pattern, pattern])


# ─── Texture Analyzer ─────────────────────────────────────────

class TestTextureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
        return TextureAnalyzer()

    def test_score_range(self, analyzer):
        img = make_noisy_image()
        result = analyzer.analyze(img, make_face_roi())
        assert 0 <= result.score <= 100

    def test_solid_image_low_score(self, analyzer):
        """A perfectly solid image should get a low texture score."""
        img = make_solid_image()
        result = analyzer.analyze(img, make_face_roi())
        assert result.score < 50, f"Solid image scored {result.score}, expected < 50"

    def test_noisy_image_higher_score(self, analyzer):
        """A noisy image should score higher than solid."""
        solid = analyzer.analyze(make_solid_image(), make_face_roi())
        noisy = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert noisy.score > solid.score

    def test_performance(self, analyzer):
        """Texture analysis should complete in < 15ms."""
        img = make_noisy_image(300, 300)
        result = analyzer.analyze(img, make_face_roi(w=300, h=300))
        assert result.elapsed_ms < 15, f"Texture took {result.elapsed_ms:.1f}ms"

    def test_result_details(self, analyzer):
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert "texture_score" in result.details
        assert "color_score" in result.details
        assert "frequency_score" in result.details


# ─── Moire Analyzer ───────────────────────────────────────────

class TestMoireAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
        return MoireAnalyzer()

    def test_score_range(self, analyzer):
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert 0 <= result.score <= 100

    def test_moire_pattern_detected(self, analyzer):
        """An image with strong periodic patterns should get lower score."""
        clean = analyzer.analyze(make_noisy_image(), make_face_roi())
        moire = analyzer.analyze(make_moire_image(freq=0.3), make_face_roi())
        # Moire image should score lower (more spoof-like)
        assert moire.score <= clean.score, (
            f"Moire scored {moire.score} vs clean {clean.score}"
        )

    def test_performance(self, analyzer):
        """Moire analysis should complete in < 40ms for 200x200."""
        img = make_noisy_image(200, 200)
        result = analyzer.analyze(img, make_face_roi())
        assert result.elapsed_ms < 40, f"Moire took {result.elapsed_ms:.1f}ms"

    def test_details_contain_risk(self, analyzer):
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert "moire_risk" in result.details
        assert 0 <= result.details["moire_risk"] <= 1


# ─── Screen Replay Analyzer ───────────────────────────────────

class TestScreenReplayAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
        return ScreenReplayAnalyzer()

    def test_score_range(self, analyzer):
        result = analyzer.analyze(make_noisy_image(400, 300))
        assert 0 <= result.score <= 100

    def test_performance(self, analyzer):
        """Screen replay analysis should complete in < 20ms."""
        img = make_noisy_image(640, 480)
        result = analyzer.analyze(img)
        assert result.elapsed_ms < 20, f"Screen replay took {result.elapsed_ms:.1f}ms"

    def test_solid_image_suspicious(self, analyzer):
        """Solid images should be treated as blur (indeterminate)."""
        result = analyzer.analyze(make_solid_image(400, 300))
        # Blur floor triggers — should return 50 (indeterminate)
        assert result.details.get("blur_floor") is True


# ─── Temporal Analyzer ─────────────────────────────────────────

class TestTemporalAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
        return TemporalAnalyzer(warmup_frames=5, min_motion_std=0.001)

    def test_warmup_returns_50(self, analyzer):
        """Before warmup, score should be neutral (50)."""
        img = make_noisy_image()
        roi = make_face_roi()
        result = analyzer.analyze(img, roi)
        assert result.score == 50.0
        assert result.details.get("warmup") is True

    def test_static_face_low_score(self, analyzer):
        """A face that never moves should eventually get a low score."""
        img = make_noisy_image()
        roi = make_face_roi()
        roi.bbox = BBox(100, 100, 200, 200)  # Fixed position
        for _ in range(20):
            result = analyzer.analyze(img, roi)
        assert result.score < 30, f"Static face scored {result.score}"

    def test_moving_face_high_score(self, analyzer):
        """A face that moves naturally should get a high score."""
        img = make_noisy_image()
        rng = np.random.RandomState(42)
        for i in range(20):
            roi = make_face_roi(face_id=1)
            x = 100 + int(rng.normal(0, 5))
            y = 100 + int(rng.normal(0, 5))
            roi.bbox = BBox(x, y, x + 100, y + 100)
            result = analyzer.analyze(img, roi)
        assert result.score > 50, f"Moving face scored {result.score}"

    def test_multiple_faces_independent(self, analyzer):
        """Each face ID should have independent tracking."""
        img = make_noisy_image()
        roi1 = make_face_roi(face_id=1)
        roi2 = make_face_roi(face_id=2)
        roi1.bbox = BBox(100, 100, 200, 200)
        roi2.bbox = BBox(300, 300, 400, 400)
        for _ in range(10):
            analyzer.analyze(img, roi1)
            analyzer.analyze(img, roi2)
        # Both should have independent history
        assert 1 in analyzer._histories
        assert 2 in analyzer._histories


# ─── MiniFASNet Analyzer ───────────────────────────────────────

class TestMiniFASNetAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
        return MiniFASNetAnalyzer()

    def test_score_range(self, analyzer):
        img = make_noisy_image()
        result = analyzer.analyze(img, make_face_roi())
        assert 0 <= result.score <= 100

    def test_returns_result_on_any_input(self, analyzer):
        """Should never crash, even on garbage input."""
        img = make_solid_image(50, 50)
        result = analyzer.analyze(img, make_face_roi(w=50, h=50))
        assert isinstance(result, AnalyzerResult)

    def test_performance(self, analyzer):
        """MiniFASNet should complete in < 50ms after warmup."""
        img = make_noisy_image(200, 200)
        # First call loads model
        analyzer.analyze(img, make_face_roi())
        # Measure second call
        result = analyzer.analyze(img, make_face_roi())
        assert result.elapsed_ms < 50, f"MiniFASNet took {result.elapsed_ms:.1f}ms"


# ─── Face Tracker ──────────────────────────────────────────────

class TestFaceTracker:
    @pytest.fixture
    def tracker(self):
        from src.application.face_tracker import FaceTracker
        return FaceTracker(iou_threshold=0.3, max_lost_frames=3)

    def test_new_faces_get_ids(self, tracker):
        faces = [
            FaceROI(0, BBox(10, 10, 60, 60), 0.9),
            FaceROI(0, BBox(200, 200, 250, 250), 0.9),
        ]
        result = tracker.update(faces)
        assert len(result) == 2
        assert result[0].face_id != result[1].face_id

    def test_persistent_ids(self, tracker):
        f1 = [FaceROI(0, BBox(10, 10, 60, 60), 0.9)]
        r1 = tracker.update(f1)
        id1 = r1[0].face_id

        # Same position → same ID
        f2 = [FaceROI(0, BBox(12, 12, 62, 62), 0.9)]
        r2 = tracker.update(f2)
        assert r2[0].face_id == id1

    def test_lost_faces_removed(self, tracker):
        f1 = [FaceROI(0, BBox(10, 10, 60, 60), 0.9)]
        tracker.update(f1)

        # 4 empty frames → face should be removed (max_lost=3)
        for _ in range(4):
            tracker.update([])
        assert tracker.active_count == 0

    def test_new_face_at_different_position(self, tracker):
        f1 = [FaceROI(0, BBox(10, 10, 60, 60), 0.9)]
        r1 = tracker.update(f1)
        id1 = r1[0].face_id

        # Face at completely different position → new ID
        f2 = [FaceROI(0, BBox(500, 500, 550, 550), 0.9)]
        r2 = tracker.update(f2)
        assert r2[0].face_id != id1


# ─── Fusion ────────────────────────────────────────────────────

class TestMultiClassFuser:
    @pytest.fixture
    def fuser(self):
        from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
        return MultiClassFuser()

    def test_high_scores_favor_real(self, fuser):
        """All analyzers returning high scores → REAL dominant."""
        results = {
            "minifasnet": AnalyzerResult("minifasnet", 95.0),
            "texture": AnalyzerResult("texture", 90.0),
            "moire": AnalyzerResult("moire", 85.0),
        }
        cls = fuser.fuse(1, results)
        assert cls.dominant_category == SpoofCategory.REAL
        assert cls.probabilities[SpoofCategory.REAL] > 0.5

    def test_low_scores_favor_spoof(self, fuser):
        """All analyzers returning low scores → spoof category dominant."""
        results = {
            "minifasnet": AnalyzerResult("minifasnet", 10.0),
            "texture": AnalyzerResult("texture", 15.0),
            "moire": AnalyzerResult("moire", 5.0),
        }
        cls = fuser.fuse(1, results)
        assert cls.dominant_category != SpoofCategory.REAL

    def test_probabilities_sum_to_one(self, fuser):
        results = {
            "minifasnet": AnalyzerResult("minifasnet", 60.0),
            "texture": AnalyzerResult("texture", 70.0),
        }
        cls = fuser.fuse(1, results)
        total = sum(cls.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_moire_low_increases_screen_categories(self, fuser):
        """Low moire score should push toward video_replay/static_image."""
        results = {
            "moire": AnalyzerResult("moire", 10.0),
        }
        cls = fuser.fuse(1, results)
        screen_prob = (
            cls.probabilities[SpoofCategory.VIDEO_REPLAY]
            + cls.probabilities[SpoofCategory.STATIC_IMAGE]
        )
        assert screen_prob > 0.3, f"Screen categories only {screen_prob:.2f}"


# ─── Pipeline Integration ─────────────────────────────────────

class TestPipelineIntegration:
    """Integration test: full pipeline on a synthetic image."""

    def test_pipeline_processes_frame(self):
        from src.application.pipeline import SpoofDetectionPipeline
        from src.application.face_tracker import FaceTracker
        from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
        from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
        from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser

        # Use a simple stub detector that always returns one face
        class StubDetector:
            def detect(self, frame):
                h, w = frame.shape[:2]
                return [FaceROI(
                    face_id=0, bbox=BBox(10, 10, w // 2, h // 2),
                    confidence=0.95, crop=frame[10:h // 2, 10:w // 2].copy(),
                )]

        pipeline = SpoofDetectionPipeline(
            detector=StubDetector(),
            tracker=FaceTracker(),
            face_analyzers=[TextureAnalyzer(), MoireAnalyzer()],
            frame_analyzers=[],
            fuser=MultiClassFuser(),
        )

        frame = make_noisy_image(640, 480)
        analysis = pipeline.process(frame)

        assert analysis.frame_id == 1
        assert len(analysis.faces) == 1
        assert len(analysis.classifications) == 1
        fid = analysis.faces[0].face_id
        cls = analysis.classifications[fid]
        assert abs(sum(cls.probabilities.values()) - 1.0) < 0.01
