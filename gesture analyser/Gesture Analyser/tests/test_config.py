"""
tests/test_config.py
====================
Verify the centralized configuration system:
  • AppConfig loads without error and provides correct types.
  • All nested config sections are accessible.
  • Values match config.yaml defaults (or the hardcoded dataclass defaults
    when YAML is absent / field is missing).
"""

import unittest
from app_config import (
    AppConfig, TrackerConfig, GestureConfig, TouchConfig,
    AntiSpoofConfig, MathConfig, LivenessConfig, SequentialConfig,
    TouchTestConfig, ShapeTraceConfig, ShapeEvalConfig, LoggingConfig,
    load_config, cfg,
)


class TestConfigTypes(unittest.TestCase):
    """All fields must have the right Python type after loading."""

    def setUp(self):
        self.c = load_config()

    # ── Top-level structure ──────────────────────────────────────────

    def test_returns_app_config(self):
        self.assertIsInstance(self.c, AppConfig)

    def test_all_sections_present(self):
        sections = [
            "tracker", "gesture", "touch", "anti_spoof", "math",
            "liveness", "sequential", "touch_test", "shape_trace",
            "shape_eval", "logging",
        ]
        for s in sections:
            self.assertTrue(hasattr(self.c, s), f"Missing section: {s}")

    # ── TrackerConfig ────────────────────────────────────────────────

    def test_tracker_types(self):
        t = self.c.tracker
        self.assertIsInstance(t, TrackerConfig)
        self.assertIsInstance(t.detection_confidence, float)
        self.assertIsInstance(t.tracking_confidence, float)
        self.assertIsInstance(t.max_lost_frames, int)
        self.assertIsInstance(t.clahe_clip_limit, float)
        self.assertIsInstance(t.clahe_tile_size, int)

    def test_tracker_confidence_range(self):
        t = self.c.tracker
        self.assertGreater(t.detection_confidence, 0.0)
        self.assertLessEqual(t.detection_confidence, 1.0)
        self.assertGreater(t.tracking_confidence, 0.0)
        self.assertLessEqual(t.tracking_confidence, 1.0)

    # ── GestureConfig ────────────────────────────────────────────────

    def test_gesture_types(self):
        g = self.c.gesture
        self.assertIsInstance(g, GestureConfig)
        self.assertIsInstance(g.finger_open_threshold, float)
        self.assertIsInstance(g.finger_close_threshold, float)
        self.assertIsInstance(g.thumb_open_threshold, float)
        self.assertIsInstance(g.thumb_close_threshold, float)
        self.assertIsInstance(g.ewma_alpha, float)
        self.assertIsInstance(g.smoothing_window, int)

    def test_gesture_hysteresis_order(self):
        """Close threshold must be strictly below open threshold."""
        g = self.c.gesture
        self.assertLess(g.finger_close_threshold, g.finger_open_threshold)
        self.assertLess(g.thumb_close_threshold, g.thumb_open_threshold)

    # ── TouchConfig ──────────────────────────────────────────────────

    def test_touch_types(self):
        t = self.c.touch
        self.assertIsInstance(t, TouchConfig)
        self.assertIsInstance(t.verify_frames, int)
        self.assertIsInstance(t.threshold, float)
        self.assertIsInstance(t.z_max_diff, float)
        self.assertIsInstance(t.bystander_open_thresh, float)
        self.assertIsInstance(t.bystander_min_open, int)

    # ── AntiSpoofConfig ──────────────────────────────────────────────

    def test_anti_spoof_types(self):
        a = self.c.anti_spoof
        self.assertIsInstance(a, AntiSpoofConfig)
        self.assertIsInstance(a.tremor_min_std, float)
        self.assertIsInstance(a.tremor_buffer_size, int)
        self.assertIsInstance(a.tremor_warmup_frames, int)
        self.assertIsInstance(a.brightness_min_var, float)
        self.assertIsInstance(a.brightness_buffer, int)
        self.assertIsInstance(a.block_frames, int)

    def test_anti_spoof_block_frames_positive(self):
        self.assertGreater(self.c.anti_spoof.block_frames, 0)

    # ── LoggingConfig ────────────────────────────────────────────────

    def test_logging_types(self):
        l = self.c.logging
        self.assertIsInstance(l, LoggingConfig)
        self.assertIsInstance(l.enabled, bool)
        self.assertIsInstance(l.log_dir, str)

    def test_logging_log_dir_nonempty(self):
        self.assertTrue(len(self.c.logging.log_dir) > 0)

    # ── Module-level singleton ───────────────────────────────────────

    def test_module_singleton_is_app_config(self):
        self.assertIsInstance(cfg, AppConfig)

    def test_singleton_same_values(self):
        fresh = load_config()
        self.assertEqual(cfg.gesture.finger_open_threshold,
                         fresh.gesture.finger_open_threshold)
        self.assertEqual(cfg.anti_spoof.block_frames,
                         fresh.anti_spoof.block_frames)


if __name__ == "__main__":
    unittest.main()
