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


# ─── Blink Rhythm Analyzer ────────────────────────────────────

class TestBlinkRhythmAnalyzer:
    """Test blink rhythm periodicity detection for video replay loops."""

    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.blink_rhythm_analyzer import BlinkRhythmAnalyzer
        return BlinkRhythmAnalyzer()

    @staticmethod
    def _make_mock_blink_analyzer(timestamps: list[float]):
        """Create a mock BlinkAnalyzer that returns fixed timestamps."""
        class MockBlinkAnalyzer:
            def get_blink_timestamps(self, face_id: int) -> list[float]:
                return list(timestamps)
        return MockBlinkAnalyzer()

    def test_insufficient_data_returns_neutral(self, analyzer):
        """With fewer than 8 blinks, score should be neutral (50)."""
        mock = self._make_mock_blink_analyzer([1.0, 2.0, 3.0])
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert result.score == 50.0
        assert result.details["status"] == "insufficient_data"

    def test_regular_intervals_low_score(self, analyzer):
        """Perfectly regular blink intervals (loop) should score low."""
        # 10 blinks at exactly 2.0 second intervals
        timestamps = [2.0 * i for i in range(10)]
        mock = self._make_mock_blink_analyzer(timestamps)
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert result.score < 25, (
            f"Regular intervals scored {result.score}, expected < 25 (loop signature)"
        )
        assert result.details["cv"] < 0.01

    def test_irregular_intervals_high_score(self, analyzer):
        """Naturally irregular blink intervals should score high."""
        # Irregular intervals mimicking a real person
        rng = np.random.RandomState(42)
        timestamps = []
        t = 0.0
        for _ in range(12):
            t += rng.uniform(1.5, 5.0)  # Irregular 1.5-5.0 second intervals
            timestamps.append(t)
        mock = self._make_mock_blink_analyzer(timestamps)
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert result.score > 60, (
            f"Irregular intervals scored {result.score}, expected > 60 (live-like)"
        )
        assert result.details["cv"] > 0.3

    def test_no_blink_analyzer_returns_neutral(self, analyzer):
        """Without a blink analyzer reference, should return neutral."""
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert result.score == 50.0
        assert result.details.get("error") == "no_blink_analyzer"

    def test_moderately_regular_intervals_mid_score(self, analyzer):
        """Slightly irregular intervals should score in the middle range."""
        # Small variation around 2.5 seconds
        rng = np.random.RandomState(123)
        timestamps = []
        t = 0.0
        for _ in range(10):
            t += 2.5 + rng.normal(0, 0.3)  # Small CV ~0.12
            timestamps.append(t)
        mock = self._make_mock_blink_analyzer(timestamps)
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        # Should be in the suspicious-to-moderate range
        assert 10 < result.score < 60, (
            f"Moderately regular scored {result.score}, expected 10-60"
        )

    def test_score_range(self, analyzer):
        """Score should always be in [0, 100]."""
        timestamps = [float(i) * 2.0 for i in range(15)]
        mock = self._make_mock_blink_analyzer(timestamps)
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert 0 <= result.score <= 100

    def test_result_details_present(self, analyzer):
        """Result details should contain key diagnostic fields."""
        timestamps = [float(i) * 2.0 for i in range(10)]
        mock = self._make_mock_blink_analyzer(timestamps)
        analyzer.set_blink_analyzer(mock)
        result = analyzer.analyze(make_noisy_image(), make_face_roi())
        assert "cv" in result.details
        assert "autocorr_peak" in result.details
        assert "mean_ibi_sec" in result.details
        assert "blinks" in result.details


# ─── rPPG Analyzer (notch filter) ────────────────────────────

class TestRPPGAnalyzerNotchFilters:
    """Verify that 50/60Hz screen-flicker notch filters work correctly.

    - A synthetic 1.2 Hz (72 BPM) pulse signal must survive the filters.
    - A synthetic 10 Hz signal (50 Hz aliased to 10 Hz at 30 fps) must be
      suppressed so it does not inflate the SNR in the pulse band.
    """

    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.rppg_analyzer import RPPGAnalyzer
        return RPPGAnalyzer(fps=30.0)

    @staticmethod
    def _feed_synthetic_signal(analyzer, signal: np.ndarray) -> "AnalyzerResult":
        """Feed a pre-built green-channel time series into the analyzer.

        We bypass the image path by writing directly into the internal
        state and calling analyze() with a dummy face crop that yields the
        next value in the signal.  This avoids needing a real camera.
        """
        from src.infrastructure.analyzers.rppg_analyzer import PulseState
        from collections import deque

        fid = 99
        state = PulseState()
        state.green_values = deque(signal.tolist(), maxlen=300)
        state.frame_count = len(signal)
        analyzer._states[fid] = state

        # Build a tiny face crop whose forehead green mean equals the
        # last value — only matters for the append inside analyze().
        last_val = int(np.clip(signal[-1], 0, 255))
        face_crop = np.full((100, 100, 3), last_val, dtype=np.uint8)
        roi = make_face_roi(face_id=fid)
        return analyzer.analyze(face_crop, roi)

    def test_pulse_signal_survives_notch(self, analyzer):
        """A 1.2 Hz (72 BPM) pulse should pass through the notch filters."""
        fps = 30.0
        duration = 5.0  # seconds
        n = int(fps * duration)
        t = np.arange(n) / fps
        # Pure 1.2 Hz sinusoid (well within pulse band, far from notches)
        signal = 128.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t)

        result = self._feed_synthetic_signal(analyzer, signal)

        # Should detect a pulse (score > 50) with BPM near 72
        assert result.score > 50, (
            f"Pulse signal scored {result.score}, expected > 50 (pulse should survive)"
        )
        bpm = result.details.get("bpm")
        assert bpm is not None, "No BPM detected for a clean 1.2 Hz pulse"
        assert 60 <= bpm <= 84, f"BPM {bpm} too far from expected 72"

    def test_screen_artifact_suppressed(self, analyzer):
        """A 10 Hz signal (aliased 50 Hz screen flicker) should be killed."""
        fps = 30.0
        duration = 5.0
        n = int(fps * duration)
        t = np.arange(n) / fps
        # Pure 10 Hz sinusoid — screen artefact, no real pulse
        signal = 128.0 + 5.0 * np.sin(2 * np.pi * 10.0 * t)

        result = self._feed_synthetic_signal(analyzer, signal)

        # The 10 Hz energy should be notched out.  Without any real
        # pulse in the 0.75-4.0 Hz band the SNR should be low and
        # the score should NOT indicate a live person.
        snr = result.details.get("snr", 0)
        assert snr < 3.0, (
            f"10 Hz artefact SNR is {snr}, expected < 3 after notch filter"
        )
        assert result.score <= 50, (
            f"10 Hz artefact scored {result.score}, expected <= 50 (should not look live)"
        )

    def test_mixed_pulse_and_artifact(self, analyzer):
        """A 1.2 Hz pulse + 10 Hz artefact: pulse should dominate after filtering."""
        fps = 30.0
        duration = 5.0
        n = int(fps * duration)
        t = np.arange(n) / fps
        # Pulse (small) + screen flicker (large)
        signal = 128.0 + 3.0 * np.sin(2 * np.pi * 1.2 * t) + 8.0 * np.sin(2 * np.pi * 10.0 * t)

        result = self._feed_synthetic_signal(analyzer, signal)

        # After notch filtering the 10 Hz component, the 1.2 Hz pulse
        # should be the dominant frequency detected.
        bpm = result.details.get("bpm")
        peak_freq = result.details.get("peak_freq_hz", 0)
        # Peak should be near 1.2 Hz, not near 10 Hz
        assert peak_freq < 4.0, (
            f"Peak freq {peak_freq} Hz — artefact was not suppressed"
        )


# ─── Image Quality Analyzer ─────────────────────────────────

class TestImageQualityAnalyzer:
    """Verify quality scoring for various lighting/blur conditions."""

    def test_bright_image_scores_high(self):
        from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer(frame_area=1280 * 720)
        # Bright, contrasty face crop
        crop = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
        roi = make_face_roi(w=200, h=200)
        result = analyzer.analyze(crop, roi)
        assert result.score >= 50, f"Bright image scored {result.score}, expected >= 50"
        assert result.details["quality_grade"] in ("A", "B", "C")
        assert result.details["quality_weight"] >= 0.5

    def test_dark_image_scores_low(self):
        from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer(frame_area=1280 * 720)
        # Very dark crop (mean ~10) — random noise has high Laplacian variance,
        # so sharpness stays high; but brightness+contrast are terrible
        crop = np.random.randint(0, 20, (200, 200, 3), dtype=np.uint8)
        roi = make_face_roi(w=200, h=200)
        result = analyzer.analyze(crop, roi)
        assert result.score < 40, f"Dark image scored {result.score}, expected < 40"
        assert result.details["quality_grade"] in ("C", "D", "F")
        assert result.details["brightness"] < 20

    def test_blurry_image_penalized(self):
        from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer(frame_area=1280 * 720)
        # Completely uniform = zero sharpness AND zero contrast
        # but brightness is perfect (128), so overall score is moderate
        crop = np.full((200, 200, 3), 128, dtype=np.uint8)
        roi = make_face_roi(w=200, h=200)
        result = analyzer.analyze(crop, roi)
        assert result.details["sharpness"] < 1.0
        assert result.details["contrast"] < 1.0
        assert result.score < 55, f"Uniform/blurry image scored {result.score}"

    def test_quality_weight_range(self):
        from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        crop = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        roi = make_face_roi(w=200, h=200)
        result = analyzer.analyze(crop, roi)
        qw = result.details["quality_weight"]
        assert 0.0 <= qw <= 1.0, f"quality_weight {qw} out of range"

    def test_name(self):
        from src.infrastructure.analyzers.image_quality_analyzer import ImageQualityAnalyzer
        assert ImageQualityAnalyzer().name == "image_quality"


class TestFusionQualityGating:
    """Verify that low quality down-weights non-robust analyzers."""

    def test_dark_conditions_reduce_minifasnet_influence(self):
        from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
        fuser = MultiClassFuser()

        # Simulate dark conditions: quality_weight = 0.1
        quality_result = AnalyzerResult(
            name="image_quality", score=15.0,
            details={"quality_weight": 0.1, "quality_grade": "F"},
        )
        # MiniFASNet scores low (darkness causes false spoof)
        minifas_result = AnalyzerResult(name="minifasnet", score=5.0, details={})
        # Device boundary is fine (quality-robust)
        device_result = AnalyzerResult(name="device_boundary", score=90.0, details={})

        results = {
            "image_quality": quality_result,
            "minifasnet": minifas_result,
            "device_boundary": device_result,
        }

        cls = fuser.fuse(face_id=1, results=results)
        p_real = cls.probabilities[SpoofCategory.REAL]

        # Without quality gating, MiniFASNet's 5.0 would dominate → very low P(real)
        # With gating, MiniFASNet weight is 5.0*0.1=0.5, device_boundary stays 2.5
        # Device boundary's 90.0 score should pull P(real) up significantly
        assert p_real > 0.4, (
            f"P(real)={p_real:.3f} — quality gating should reduce MiniFASNet influence "
            f"so device_boundary (robust) dominates"
        )

    def test_good_conditions_full_weights(self):
        from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
        fuser = MultiClassFuser()

        quality_result = AnalyzerResult(
            name="image_quality", score=85.0,
            details={"quality_weight": 1.0, "quality_grade": "A"},
        )
        minifas_result = AnalyzerResult(name="minifasnet", score=95.0, details={})

        results = {
            "image_quality": quality_result,
            "minifasnet": minifas_result,
        }

        cls = fuser.fuse(face_id=1, results=results)
        p_real = cls.probabilities[SpoofCategory.REAL]
        assert p_real > 0.8, f"Good quality + high MiniFASNet should give P(real) > 0.8, got {p_real:.3f}"
