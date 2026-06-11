"""
Session Audit Logger
====================
Writes one JSON-Lines record to ``logs/sessions.jsonl`` each time a
mode reaches a terminal state (game over, verified, complete).

Record schema
-------------
Every record is a single JSON object followed by a newline::

    {
        "timestamp":  "2026-04-14T10:23:01.456789+00:00",  # ISO-8601 UTC
        "mode":       "Liveness",
        "result":     "VERIFIED_100",
        "duration_s": 18.42,
        "metrics": {
            "challenges_completed": 5,
            "score": 100,
            "spoof_blocks": 0
        }
    }

JSON-Lines format was chosen because:
  • Each record is self-contained and human-readable.
  • Appending never requires reading or rewriting the file.
  • Any log-analytics tool (jq, pandas, Excel Power Query) can parse it
    with a single import statement.

Result values per mode
----------------------
  Math       : "GAME_OVER"     — 60 s elapsed regardless of score
  Liveness   : "VERIFIED_100"  — all 5 challenges passed
  Sequential : "COMPLETE"      — all steps attempted (some may be TIMED_OUT)
  Touch Test : "COMPLETE"      — all 5 touch commands passed

No external dependencies — stdlib only (json, pathlib, datetime).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionLogger:
    """Append-only JSON-Lines session recorder.

    Parameters
    ----------
    log_dir:
        Directory in which ``sessions.jsonl`` is created.  Created
        automatically if it does not exist.
    enabled:
        When ``False`` every ``log()`` call is a silent no-op.  Lets
        operators disable logging via config without code changes.
    """

    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        self.enabled = enabled
        self._path: Path | None = None

        if not enabled:
            return

        try:
            self._path = Path(log_dir) / "sessions.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("SessionLogger initialised — writing to %s", self._path)
        except OSError as exc:
            logger.error(
                "SessionLogger: cannot create log directory '%s' (%s). "
                "Logging disabled for this session.",
                log_dir, exc,
            )
            self.enabled = False
            self._path = None

    # ------------------------------------------------------------------

    def log(
        self,
        mode: str,
        result: str,
        duration_s: float,
        metrics: dict[str, Any],
    ) -> None:
        """Append one record to the log file.

        Parameters
        ----------
        mode:
            Human-readable mode name ("Math", "Liveness", etc.).
        result:
            Terminal state label ("GAME_OVER", "VERIFIED_100", etc.).
        duration_s:
            Seconds from session start to terminal state.
        metrics:
            Mode-specific key/value pairs (score, passed count, etc.).
        """
        if not self.enabled or self._path is None:
            return

        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "mode":       mode,
            "result":     result,
            "duration_s": round(duration_s, 2),
            "metrics":    metrics,
        }

        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug("Logged %s/%s (%.1fs)", mode, result, duration_s)
        except OSError as exc:
            logger.error(
                "SessionLogger: failed to write record (%s). Record lost: %s",
                exc, record,
            )
