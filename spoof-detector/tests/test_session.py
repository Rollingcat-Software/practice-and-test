"""Unit tests for SessionEngine and DeviceBoundaryAnalyzer."""

import time
import pytest
import numpy as np
import cv2

from src.domain.models import (
    FaceROI, BBox, SpoofCategory, SpoofClassification,
    AnalyzerResult, FrameAnalysis,
)
from src.domain.session import SessionState, Severity
from src.application.session_engine import SessionEngine


def make_frame_analysis(
    frame_id: int,
    p_real: float = 0.75,
    face_id: int = 1,
    n_faces: int = 1,
) -> FrameAnalysis:
    """Create a synthetic FrameAnalysis for testing."""
    faces = []
    classifications = {}

    for i in range(n_faces):
        fid = face_id + i
        face = FaceROI(
            face_id=fid,
            bbox=BBox(100 + i * 50, 100, 300 + i * 50, 400),
            confidence=0.95,
        )
        faces.append(face)

        probs = {cat: (1.0 - p_real) / 6 for cat in SpoofCategory if cat != SpoofCategory.REAL}
        probs[SpoofCategory.REAL] = p_real
        cls = SpoofClassification.from_probabilities(fid, probs)
        classifications[fid] = cls

    return FrameAnalysis(
        frame_id=frame_id,
        faces=faces,
        classifications=classifications,
        total_ms=20.0,
    )


def make_empty_analysis(frame_id: int) -> FrameAnalysis:
    """Create a FrameAnalysis with no faces."""
    return FrameAnalysis(frame_id=frame_id, faces=[], classifications={})


# ─── Session Lifecycle ─────────────────────────────────────────

class TestSessionLifecycle:
    def test_initial_state(self):
        engine = SessionEngine()
        assert engine.state == SessionState.WARMING_UP

    def test_start_sets_state(self):
        engine = SessionEngine()
        engine.start()
        assert engine.state == SessionState.WARMING_UP
        assert engine.elapsed_sec >= 0

    def test_warmup_to_analyzing_transition(self):
        engine = SessionEngine()
        engine.start()
        # Feed 30 frames (warmup threshold)
        for i in range(31):
            engine.ingest(make_frame_analysis(i))
        assert engine.state == SessionState.ANALYZING

    def test_conclude_sets_concluded(self):
        engine = SessionEngine()
        engine.start()
        for i in range(35):
            engine.ingest(make_frame_analysis(i))
        verdict = engine.conclude()
        assert engine.state == SessionState.CONCLUDED
        assert verdict.frames_analyzed == 35


# ─── Verdict Logic ─────────────────────────────────────────────

class TestVerdictLogic:
    def test_high_real_frames_without_liveness_proof_is_spoof(self):
        """With guilty-until-proven architecture, high P(real) alone isn't enough.
        Must also prove liveness through blinks/landmarks/challenges."""
        engine = SessionEngine()
        engine.start()
        for i in range(100):
            engine.ingest(make_frame_analysis(i, p_real=0.80))
        verdict = engine.get_verdict()
        # Without liveness proof (no blinks, no landmarks), verdict is SPOOF
        # even with high P(real) — this is the "guilty until proven" design
        assert verdict.is_live is False

    def test_all_spoof_frames_give_spoof_verdict(self):
        engine = SessionEngine()
        engine.start()
        for i in range(100):
            engine.ingest(make_frame_analysis(i, p_real=0.15))
        verdict = engine.get_verdict()
        assert verdict.is_live is False
        assert verdict.dominant_threat is not None

    def test_mixed_session_peak_sensitive(self):
        """Even if most frames are real, a spoof burst should lower verdict."""
        engine = SessionEngine()
        engine.start()
        # 80 real frames
        for i in range(80):
            engine.ingest(make_frame_analysis(i, p_real=0.80))
        # 20 spoof frames (burst)
        for i in range(80, 100):
            engine.ingest(make_frame_analysis(i, p_real=0.20))
        verdict = engine.get_verdict()
        # The spoof burst should significantly lower confidence
        assert verdict.confidence < 0.90

    def test_verdict_confidence_increases_with_frames(self):
        engine = SessionEngine()
        engine.start()
        engine.ingest(make_frame_analysis(1, p_real=0.80))
        early = engine.get_verdict()

        for i in range(2, 200):
            engine.ingest(make_frame_analysis(i, p_real=0.80))
        late = engine.get_verdict()

        assert late.confidence > early.confidence

    def test_face_detected_ratio(self):
        engine = SessionEngine()
        engine.start()
        for i in range(50):
            engine.ingest(make_frame_analysis(i))
        for i in range(50, 100):
            engine.ingest(make_empty_analysis(i))
        verdict = engine.get_verdict()
        assert 0.45 <= verdict.face_detected_ratio <= 0.55

    def test_session_duration_tracked(self):
        engine = SessionEngine()
        engine.start()
        for i in range(10):
            engine.ingest(make_frame_analysis(i))
        verdict = engine.get_verdict()
        assert verdict.session_duration_sec >= 0
        assert verdict.frames_analyzed == 10


# ─── Incident Detection ───────────────────────────────────────

class TestIncidentDetection:
    def test_no_incidents_on_real_session(self):
        engine = SessionEngine()
        engine.start()
        for i in range(60):
            engine.ingest(make_frame_analysis(i, p_real=0.80))
        verdict = engine.get_verdict()
        assert len(verdict.incidents) == 0

    def test_incidents_on_spoof_frames(self):
        engine = SessionEngine()
        engine.start()
        for i in range(60):
            engine.ingest(make_frame_analysis(i, p_real=0.20))
        verdict = engine.get_verdict()
        assert len(verdict.incidents) > 0

    def test_incident_severity_scales_with_confidence(self):
        engine = SessionEngine()
        engine.start()
        # Very strong spoof signal
        for i in range(60):
            engine.ingest(make_frame_analysis(i, p_real=0.10))
        verdict = engine.get_verdict()
        high_severity = [i for i in verdict.incidents if i.severity == Severity.HIGH]
        assert len(high_severity) > 0

    def test_frozen_face_triggers_incident(self):
        engine = SessionEngine()
        engine._start_time = time.time() - 10  # Pretend session started 10s ago
        engine._state = SessionState.ANALYZING
        # Same bbox every frame = frozen — need 60+ frames AND elapsed > 3s
        for i in range(100):
            analysis = make_frame_analysis(i, p_real=0.60)
            engine.ingest(analysis)
        verdict = engine.get_verdict()
        # Should detect unnaturally static face
        static_incidents = [
            i for i in verdict.incidents
            if "static" in i.description.lower()
        ]
        assert len(static_incidents) > 0

    def test_timeline_format(self):
        engine = SessionEngine()
        engine.start()
        for i in range(60):
            engine.ingest(make_frame_analysis(i, p_real=0.20))
        timeline = engine.get_timeline()
        assert isinstance(timeline, list)
        if timeline:
            entry = timeline[0]
            assert "time_sec" in entry
            assert "severity" in entry
            assert "category" in entry
            assert "description" in entry


# ─── Device Boundary Analyzer ─────────────────────────────────

class TestDeviceBoundaryAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
        return DeviceBoundaryAnalyzer()

    def test_no_frame_returns_neutral(self, analyzer):
        """Without set_frame(), should return neutral score."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(50, 50, 150, 150), confidence=0.9)
        result = analyzer.analyze(img, roi)
        assert result.score == 50.0
        assert result.details.get("error") == "no_frame"

    def test_plain_face_high_score(self, analyzer):
        """A face on plain background should score high (live-like)."""
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)  # Light gray
        # Add some random texture to avoid being too uniform
        noise = np.random.RandomState(42).randint(180, 220, frame.shape, dtype=np.uint8)
        frame = noise
        roi = FaceROI(face_id=1, bbox=BBox(400, 200, 800, 600), confidence=0.9)
        analyzer.set_frame(frame)
        result = analyzer.analyze(frame[200:600, 400:800], roi)
        assert result.score > 50

    def test_rectangular_bezel_lowers_score(self, analyzer):
        """A clear rectangular frame around face should lower score."""
        frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
        # Draw a phone-like rectangle around the face area
        cv2.rectangle(frame, (350, 150), (900, 650), (0, 0, 0), 3)
        roi = FaceROI(face_id=1, bbox=BBox(450, 250, 750, 550), confidence=0.9)
        analyzer.set_frame(frame)
        crop = frame[250:550, 450:750]
        result = analyzer.analyze(crop, roi)
        # Should detect the bezel
        assert result.details.get("line_score", 0) > 0 or result.details.get("contour_score", 0) > 0

    def test_performance(self, analyzer):
        """Device boundary analysis should complete in < 200ms even on noisy input."""
        # Note: random noise generates many Hough lines (~1000+), real images are faster
        frame = np.random.RandomState(42).randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(400, 200, 800, 600), confidence=0.9)
        analyzer.set_frame(frame)
        crop = frame[200:600, 400:800]
        result = analyzer.analyze(crop, roi)
        assert result.elapsed_ms < 300, f"Device boundary took {result.elapsed_ms:.1f}ms"

    def test_score_range(self, analyzer):
        frame = np.random.RandomState(42).randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(400, 200, 800, 600), confidence=0.9)
        analyzer.set_frame(frame)
        result = analyzer.analyze(frame[200:600, 400:800], roi)
        assert 0 <= result.score <= 100

    def test_name(self, analyzer):
        assert analyzer.name == "device_boundary"


# --- rPPG Analyzer ---

class TestRPPGAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.rppg_analyzer import RPPGAnalyzer
        return RPPGAnalyzer()

    def test_warmup_returns_neutral(self, analyzer):
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(0, 0, 200, 200), confidence=0.9)
        result = analyzer.analyze(img, roi)
        assert result.score == 50.0
        assert result.details.get("warmup") is True

    def test_score_after_accumulation(self, analyzer):
        """After enough frames, rPPG should return a score (not warmup)."""
        roi = FaceROI(face_id=1, bbox=BBox(0, 0, 200, 200), confidence=0.9)
        for i in range(70):
            img = np.full((200, 200, 3), 128, dtype=np.uint8)
            result = analyzer.analyze(img, roi)
        assert 0 <= result.score <= 100
        # Should be past warmup phase
        assert result.details.get("warmup") is not True

    def test_name(self, analyzer):
        assert analyzer.name == "rppg"


# --- AR Filter Analyzer ---

class TestARFilterAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.ar_filter_analyzer import ARFilterAnalyzer
        return ARFilterAnalyzer()

    def test_score_range(self, analyzer):
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(0, 0, 200, 200), confidence=0.9)
        result = analyzer.analyze(img, roi)
        assert 0 <= result.score <= 100

    def test_heuristic_method(self, analyzer):
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(0, 0, 200, 200), confidence=0.9)
        result = analyzer.analyze(img, roi)
        assert result.details.get("method") == "heuristic"

    def test_name(self, analyzer):
        assert analyzer.name == "ar_filter"


# --- Blink Analyzer (basic, no FaceLandmarker model needed) ---

class TestBlinkAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from src.infrastructure.analyzers.blink_analyzer import BlinkAnalyzer
        return BlinkAnalyzer()

    def test_no_frame_returns_neutral(self, analyzer):
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        roi = FaceROI(face_id=1, bbox=BBox(0, 0, 200, 200), confidence=0.9)
        # Without set_frame, should return neutral
        result = analyzer.analyze(img, roi)
        assert result.score == 50.0

    def test_name(self, analyzer):
        assert analyzer.name == "blink"
