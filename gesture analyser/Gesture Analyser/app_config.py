"""
Centralized Application Configuration
======================================
Single source of truth for all tunable parameters in the Liveness
Challenge System.

Usage
-----
Import the module-level ``cfg`` singleton anywhere in the project::

    from app_config import cfg

    threshold = cfg.touch.threshold
    time_limit = cfg.liveness.time_limit

Loading order
-------------
1. Every field starts from a safe built-in default (defined in the
   dataclass declarations below).
2. ``config.yaml`` is loaded from the project root (next to main.py).
3. YAML values are merged on top of defaults — a missing or partial YAML
   is always safe; unrecognised keys are ignored with a warning.
4. If ``pyyaml`` is not installed, defaults are used and a RuntimeWarning
   is emitted.  Install with:  pip install pyyaml

Custom config path (testing / CI)
----------------------------------
Call ``load_config(Path("path/to/other.yaml"))`` and reassign ``cfg``::

    import app_config
    app_config.cfg = app_config.load_config(Path("tests/strict.yaml"))

No project-internal imports — this module depends only on the stdlib so
every other module can safely import it without circular dependency risk.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location: same directory as this file (project root).
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── Typed configuration sections ──────────────────────────────────────────────

@dataclass
class TrackerConfig:
    """MediaPipe hand-tracking parameters and CLAHE contrast settings."""
    detection_confidence: float = 0.70
    tracking_confidence:  float = 0.50
    max_lost_frames:      int   = 2
    clahe_clip_limit:     float = 2.0
    clahe_tile_size:      int   = 8


@dataclass
class GestureConfig:
    """Finger-open / finger-close thresholds and EWMA smoothing."""
    finger_open_threshold:  float = 0.20
    finger_close_threshold: float = 0.12
    thumb_open_threshold:   float = 0.75
    thumb_close_threshold:  float = 0.60
    ewma_alpha:             float = 0.35
    smoothing_window:       int   = 7


@dataclass
class TouchConfig:
    """Fingertip touch detection geometry and temporal confirmation."""
    threshold:             float = 0.28
    z_max_diff:            float = 0.04
    verify_frames:         int   = 10
    bystander_open_thresh: float = 0.05
    bystander_min_open:    int   = 1


@dataclass
class AntiSpoofConfig:
    """Static-image and frozen-video detection parameters."""
    enabled:               bool  = True  # master switch for the liveness hard-block gate
    tremor_min_std:        float = 0.0003
    tremor_buffer_size:    int   = 30
    tremor_warmup_frames:  int   = 90
    brightness_min_var:    float = 0.05
    brightness_buffer:     int   = 60
    block_frames:          int   = 45   # sustained suspicious frames → hard FAILED block


@dataclass
class MathConfig:
    """Math challenge timing."""
    stability_seconds:   float = 2.0
    pause_after_success: float = 1.5
    game_duration:       float = 60.0
    smoothing_window:    int   = 7


@dataclass
class LivenessConfig:
    """Liveness challenge timing, scoring, and gesture smoothing."""
    time_limit:            float = 4.0
    debounce_seconds:      float = 0.5
    area_change_threshold: float = 0.20
    pause_after_result:    float = 1.5
    num_challenges:        int   = 5
    smoothing_window:      int   = 7


@dataclass
class SequentialConfig:
    """Sequential challenge timing and depth detection."""
    hold_seconds:     float = 1.0
    pause_after_step: float = 1.0
    depth_threshold:  float = 0.20
    smoothing_window: int   = 7


@dataclass
class TouchTestConfig:
    """Touch Test session frame confirmation and pacing."""
    verify_frames:       int   = 10
    pause_after_success: float = 1.5


@dataclass
class ShapeTraceConfig:
    """Shape tracing DTW verification and UI timing."""
    draw_time:      float = 18.0
    dtw_threshold:  float = 0.25
    resample_n:     int   = 50
    min_hand_scale: float = 0.10
    pos_hold:       float = 1.50
    instruct_time:  float = 3.0


@dataclass
class ShapeEvalConfig:
    """Shape evaluation session settings."""
    dtw_threshold: float = 0.25
    auto_advance:  bool  = True


@dataclass
class LoggingConfig:
    """Session audit logger settings."""
    enabled: bool = True
    log_dir: str  = "logs"


@dataclass
class AppConfig:
    """Root configuration object.

    Access individual sections as attributes::

        cfg.tracker.detection_confidence
        cfg.touch.threshold
        cfg.liveness.time_limit
    """
    tracker:     TrackerConfig    = field(default_factory=TrackerConfig)
    gesture:     GestureConfig    = field(default_factory=GestureConfig)
    touch:       TouchConfig      = field(default_factory=TouchConfig)
    anti_spoof:  AntiSpoofConfig  = field(default_factory=AntiSpoofConfig)
    math:        MathConfig       = field(default_factory=MathConfig)
    liveness:    LivenessConfig   = field(default_factory=LivenessConfig)
    sequential:  SequentialConfig = field(default_factory=SequentialConfig)
    touch_test:  TouchTestConfig  = field(default_factory=TouchTestConfig)
    shape_trace: ShapeTraceConfig = field(default_factory=ShapeTraceConfig)
    shape_eval:  ShapeEvalConfig  = field(default_factory=ShapeEvalConfig)
    logging:     LoggingConfig    = field(default_factory=LoggingConfig)


# ── YAML loader ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Returns an empty dict (safe defaults) on any error:
    - pyyaml not installed
    - file not found
    - malformed YAML
    """
    try:
        import yaml  # pyyaml — optional dependency
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            logger.warning("config.yaml top level is not a mapping — using defaults.")
            return {}
        logger.debug("Loaded config from %s", path)
        return data
    except ImportError:
        warnings.warn(
            "pyyaml is not installed — all parameters use built-in defaults.\n"
            "Install with:  pip install pyyaml",
            RuntimeWarning,
            stacklevel=4,
        )
        return {}
    except FileNotFoundError:
        logger.debug("config.yaml not found at '%s' — using built-in defaults.", path)
        return {}
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to parse config.yaml ({exc}) — using built-in defaults.",
            RuntimeWarning,
            stacklevel=4,
        )
        return {}


def _merge_section(section_obj: Any, raw: dict) -> None:
    """Write YAML scalar values into a dataclass instance in-place.

    Rules
    -----
    - Only known field names are accepted; unknown keys emit a warning
      and are skipped.
    - Values are coerced to the declared field type (bool / int / float /
      str).  A coercion failure keeps the default and emits a warning.
    - Field type annotations are read as plain strings (``__annotations__``
      stores them as strings in dataclasses with ``from __future__ import
      annotations`` active).
    """
    known = {f.name: f for f in fields(section_obj)}
    for key, value in raw.items():
        if key not in known:
            logger.warning(
                "config.yaml: unknown key '%s' in section '%s' — ignored.",
                key, type(section_obj).__name__,
            )
            continue
        declared_type = known[key].type  # e.g. 'float', 'int', 'bool'
        try:
            if declared_type in ("bool", bool) or declared_type is bool:
                # YAML already parses true/false as bool; guard against
                # a user writing the string "true" instead.
                coerced: Any = value if isinstance(value, bool) else str(value).lower() == "true"
            elif declared_type in ("int", int) or declared_type is int:
                coerced = int(value)
            elif declared_type in ("float", float) or declared_type is float:
                coerced = float(value)
            else:
                coerced = value
            setattr(section_obj, key, coerced)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "config.yaml: invalid value for '%s.%s': %r (%s) — keeping default.",
                type(section_obj).__name__, key, value, exc,
            )


def load_config(path: Path | None = None) -> AppConfig:
    """Build and return a fully validated :class:`AppConfig`.

    Parameters
    ----------
    path:
        Override the default ``config.yaml`` location.  Pass ``None``
        (default) to use the file next to ``app_config.py``.
    """
    config = AppConfig()
    raw = _load_yaml(path or _CONFIG_PATH)

    # Map YAML section names to their corresponding dataclass instances.
    section_map: dict[str, Any] = {
        "tracker":     config.tracker,
        "gesture":     config.gesture,
        "touch":       config.touch,
        "anti_spoof":  config.anti_spoof,
        "math":        config.math,
        "liveness":    config.liveness,
        "sequential":  config.sequential,
        "touch_test":  config.touch_test,
        "shape_trace": config.shape_trace,
        "shape_eval":  config.shape_eval,
        "logging":     config.logging,
    }

    for section_name, section_obj in section_map.items():
        if section_name in raw:
            if isinstance(raw[section_name], dict):
                _merge_section(section_obj, raw[section_name])
            else:
                logger.warning(
                    "config.yaml: section '%s' is not a mapping — skipped.",
                    section_name,
                )

    return config


# ── Module-level singleton ────────────────────────────────────────────────────

#: Global configuration instance.  Import this in any module::
#:
#:     from app_config import cfg
#:     print(cfg.touch.threshold)
cfg: AppConfig = load_config()
