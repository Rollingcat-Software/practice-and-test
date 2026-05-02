"""Unit tests for domain models."""

import pytest
from src.domain.models import (
    BBox, FaceROI, SpoofCategory, SpoofClassification,
    AnalyzerResult, FrameAnalysis, CATEGORY_LABELS, CATEGORY_COLORS,
)


class TestBBox:
    def test_dimensions(self):
        bb = BBox(10, 20, 110, 170)
        assert bb.width == 100
        assert bb.height == 150
        assert bb.area == 15000
        assert bb.center == (60, 95)

    def test_iou_identical(self):
        bb = BBox(0, 0, 100, 100)
        assert bb.iou(bb) == 1.0

    def test_iou_no_overlap(self):
        a = BBox(0, 0, 50, 50)
        b = BBox(100, 100, 200, 200)
        assert a.iou(b) == 0.0

    def test_iou_partial(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 150, 150)
        iou = a.iou(b)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 0.001


class TestSpoofCategory:
    def test_all_categories_have_labels(self):
        for cat in SpoofCategory:
            assert cat in CATEGORY_LABELS

    def test_all_categories_have_colors(self):
        for cat in SpoofCategory:
            assert cat in CATEGORY_COLORS

    def test_seven_categories(self):
        assert len(SpoofCategory) == 7


class TestSpoofClassification:
    def test_from_probabilities_normalizes(self):
        probs = {
            SpoofCategory.REAL: 80.0,
            SpoofCategory.STATIC_IMAGE: 10.0,
            SpoofCategory.VIDEO_REPLAY: 5.0,
            SpoofCategory.MASK_3D: 2.0,
            SpoofCategory.HEAVY_MAKEUP: 1.0,
            SpoofCategory.AR_FILTER: 1.0,
            SpoofCategory.DEEPFAKE_INJECT: 1.0,
        }
        cls = SpoofClassification.from_probabilities(1, probs)
        total = sum(cls.probabilities.values())
        assert abs(total - 1.0) < 0.001

    def test_dominant_category(self):
        probs = {cat: 0.0 for cat in SpoofCategory}
        probs[SpoofCategory.AR_FILTER] = 1.0
        cls = SpoofClassification.from_probabilities(1, probs)
        assert cls.dominant_category == SpoofCategory.AR_FILTER

    def test_zero_probabilities(self):
        probs = {cat: 0.0 for cat in SpoofCategory}
        cls = SpoofClassification.from_probabilities(1, probs)
        # Should produce uniform distribution
        for prob in cls.probabilities.values():
            assert abs(prob - 1.0 / 7) < 0.001


class TestAnalyzerResult:
    def test_defaults(self):
        r = AnalyzerResult(name="test", score=75.0)
        assert r.name == "test"
        assert r.score == 75.0
        assert r.elapsed_ms == 0.0
        assert r.details == {}
