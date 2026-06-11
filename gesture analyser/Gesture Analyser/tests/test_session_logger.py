"""
tests/test_session_logger.py
=============================
Unit tests for SessionLogger (session_logger.py).

Covers:
  • A disabled logger silently ignores all log() calls.
  • A enabled logger creates the file and writes valid JSON-Lines.
  • Each log() call appends exactly one line.
  • Multiple calls produce multiple parseable records.
  • OSError on directory creation disables the logger gracefully.
  • Record schema (required keys are present).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from session_logger import SessionLogger


class TestSessionLoggerDisabled(unittest.TestCase):

    def test_disabled_logger_creates_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            logger = SessionLogger(log_dir=td, enabled=False)
            logger.log("Math", "GAME_OVER", 42.0, {"score": 7})
            log_path = Path(td) / "sessions.jsonl"
            self.assertFalse(log_path.exists())

    def test_disabled_logger_log_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            logger = SessionLogger(log_dir=td, enabled=False)
            # Should not raise
            for _ in range(10):
                logger.log("Liveness", "VERIFIED_100", 15.0, {"score": 100})
            self.assertFalse(logger.enabled)


class TestSessionLoggerEnabled(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logger = SessionLogger(log_dir=self.tmp, enabled=True)
        self.log_path = Path(self.tmp) / "sessions.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_file_created(self):
        self.logger.log("Math", "GAME_OVER", 60.0, {"score": 5})
        self.assertTrue(self.log_path.exists())

    def test_single_record_is_valid_json(self):
        self.logger.log("Liveness", "VERIFIED_100", 18.5, {"score": 100})
        with self.log_path.open() as fh:
            line = fh.readline().strip()
        record = json.loads(line)
        self.assertIsInstance(record, dict)

    def test_record_has_required_keys(self):
        self.logger.log("Sequential", "COMPLETE", 45.0, {"passed_count": 8})
        with self.log_path.open() as fh:
            record = json.loads(fh.readline())
        for key in ("timestamp", "mode", "result", "duration_s", "metrics"):
            self.assertIn(key, record, f"Missing key: {key}")

    def test_record_values_correct(self):
        self.logger.log("TouchTest", "COMPLETE", 30.1, {"passed_count": 5})
        with self.log_path.open() as fh:
            record = json.loads(fh.readline())
        self.assertEqual(record["mode"],       "TouchTest")
        self.assertEqual(record["result"],     "COMPLETE")
        self.assertAlmostEqual(record["duration_s"], 30.1, places=1)
        self.assertEqual(record["metrics"],    {"passed_count": 5})

    def test_each_call_appends_one_line(self):
        for i in range(5):
            self.logger.log("Math", "GAME_OVER", float(i), {"score": i})
        with self.log_path.open() as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        self.assertEqual(len(lines), 5)

    def test_all_appended_lines_are_valid_json(self):
        records = [
            ("Math",       "GAME_OVER",    60.0, {"score": 3}),
            ("Liveness",   "VERIFIED_100", 22.0, {"score": 100}),
            ("Sequential", "COMPLETE",     50.0, {"passed_count": 10}),
        ]
        for args in records:
            self.logger.log(*args)
        with self.log_path.open() as fh:
            for line in fh:
                if line.strip():
                    parsed = json.loads(line)
                    self.assertIn("timestamp", parsed)

    def test_duration_is_rounded_to_2dp(self):
        self.logger.log("Math", "GAME_OVER", 12.3456789, {"score": 1})
        with self.log_path.open() as fh:
            record = json.loads(fh.readline())
        # Should be rounded to 2 decimal places
        self.assertEqual(record["duration_s"], round(12.3456789, 2))

    def test_timestamp_is_iso8601_utc(self):
        """Timestamp string must end with +00:00 (UTC) and parse cleanly."""
        from datetime import datetime, timezone
        self.logger.log("Math", "GAME_OVER", 1.0, {})
        with self.log_path.open() as fh:
            record = json.loads(fh.readline())
        ts = record["timestamp"]
        self.assertTrue(ts.endswith("+00:00"), f"Not UTC: {ts}")
        dt = datetime.fromisoformat(ts)
        self.assertEqual(dt.tzinfo, timezone.utc)


class TestSessionLoggerOSError(unittest.TestCase):

    def test_invalid_log_dir_disables_logger(self):
        """When the directory cannot be created, logger degrades gracefully."""
        # Use a path whose parent is a file (guarantees OSError on mkdir).
        with tempfile.NamedTemporaryFile() as f:
            bad_dir = f.name + "/subdir_cannot_exist"
            logger = SessionLogger(log_dir=bad_dir, enabled=True)
        # Logger should have caught the error and self-disabled.
        self.assertFalse(logger.enabled)
        self.assertIsNone(logger._path)

    def test_disabled_after_oserror_log_is_noop(self):
        with tempfile.NamedTemporaryFile() as f:
            bad_dir = f.name + "/nope"
            logger = SessionLogger(log_dir=bad_dir, enabled=True)
        # Should not raise
        logger.log("Math", "GAME_OVER", 1.0, {"score": 0})


if __name__ == "__main__":
    unittest.main()
