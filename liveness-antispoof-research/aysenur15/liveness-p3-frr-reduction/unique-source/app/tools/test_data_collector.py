"""Interactive tool to collect hybrid fusion test data.

Usage:
    python app/tools/test_data_collector.py

Workflow:
    1. Shows live camera feed
    2. Press SPACE to capture frame
    3. Select ground truth (1=LIVE, 2=SPOOF)
    4. Saves: frame image + metrics JSON
    5. Results in: data/test_frames/
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.container import get_face_detector, get_liveness_detector
from app.core.config import get_settings
from app.tools.live_liveness_preview import (
    AggregatedMetrics,
    FrameMetrics,
    LivenessPreviewFrameProcessor,
    TemporalLivenessAggregator,
    open_camera_capture,
)

logger = logging.getLogger(__name__)
_COLLECTOR_VERSION = "collector-preview-sync-v1"
_STATUS_MESSAGE_TTL_FRAMES = 45
_CAPTURE_KEYS = {32, 13, ord("c"), ord("C")}
_QUIT_KEYS = {ord("q"), ord("Q"), 27}
_LIVE_BUTTON = ((160, 10), (280, 50))
_SPOOF_BUTTON = ((290, 10), (430, 50))
_RPPG_FUSION_ENABLED = False


class TestDataCollector:
    """Collect labeled test frames for hybrid fusion validation."""

    BASE_DIR = Path("data")

    @staticmethod
    def _next_output_dir() -> Path:
        base = TestDataCollector.BASE_DIR
        index = 1
        while (base / f"test_frames{index}").exists():
            index += 1
        return base / f"test_frames{index}"

    def __init__(self) -> None:
        self.output_dir = self._next_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.SUMMARY_FILE = self.output_dir / "summary.jsonl"

        settings = get_settings()
        self.settings = settings
        self.detector = get_face_detector()
        self.liveness_detector = get_liveness_detector()
        self.frame_processor = LivenessPreviewFrameProcessor(
            face_detector=self.detector,
            liveness_detector=self.liveness_detector,
            settings=settings,
        )
        self.temporal_aggregator = TemporalLivenessAggregator(
            window_seconds=max(2.0, float(settings.DEV_LIVENESS_PREVIEW_WINDOW_SECONDS)),
            baseline_seconds=max(0.5, float(settings.DEV_LIVENESS_PREVIEW_BASELINE_SECONDS)),
        )

        self.frame_count = {"live": 0, "spoof": 0}

    def collect_from_camera(self) -> None:
        """Interactive camera capture loop (synchronous)."""
        cap = open_camera_capture(0)
        if not cap.isOpened():
            print("Camera not available")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 60)
        print("TEST DATA COLLECTOR - Hybrid Fusion Validation")
        print(f"VERSION: {_COLLECTOR_VERSION}")
        print(f"FILE: {Path(__file__).resolve()}")
        print("=" * 60)
        print("\nInstructions:")
        print("  SPACE  -> Capture frame")
        print("  1      -> Label as LIVE")
        print("  2      -> Label as SPOOF")
        print("  Q      -> Quit")
        print("\nTroubleshooting:")
        print("  - If stuck on 'NO FACE', move closer/farther from camera")
        print("  - Ensure good lighting")
        print("  - Center your face in the frame")
        print("\n" + "=" * 60 + "\n")

        captured_frame = None
        captured_metrics: FrameMetrics | None = None
        captured_aggregate: AggregatedMetrics | None = None
        frame_number = 0
        status_message = "SPACE: capture, 1: LIVE, 2: SPOOF, Q: quit"
        status_color = (200, 200, 200)
        status_ttl = _STATUS_MESSAGE_TTL_FRAMES
        last_key_code = -1
        mouse_action: dict[str, str | None] = {"action": None}
        latest_frame_metrics: FrameMetrics | None = None
        latest_aggregate: AggregatedMetrics | None = None

        window_name = f"Test Data Collector [{_COLLECTOR_VERSION}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            window_name,
            lambda event, x, y, flags, param: self._handle_mouse(
                event=event,
                x=x,
                y=y,
                mouse_action=mouse_action,
            ),
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            latest_frame_metrics, latest_aggregate = self._process_preview_frame(frame)
            detection_ok = latest_frame_metrics.face_detected if latest_frame_metrics is not None else False

            display_frame = frame.copy()
            self._apply_flash_debug_stimulus(display_frame)
            if captured_frame is not None:
                cv2.rectangle(display_frame, (10, 10), (140, 50), (0, 255, 0), 3)
                cv2.putText(
                    display_frame,
                    "CAPTURED",
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            self._draw_button(display_frame, _LIVE_BUTTON, "LIVE", (0, 170, 0))
            self._draw_button(display_frame, _SPOOF_BUTTON, "SPOOF", (0, 0, 170))

            face_status = "FACE DETECTED" if detection_ok else "NO FACE"
            face_color = (0, 255, 0) if detection_ok else (0, 0, 255)
            if latest_aggregate is not None:
                conf_text = f"{face_status} ({latest_aggregate.window_confidence:.2f})"
            else:
                conf_text = face_status

            cv2.putText(
                display_frame,
                conf_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                face_color,
                2,
            )

            status = (
                f"Frame: {frame_number} | Live: {self.frame_count['live']} | "
                f"Spoof: {self.frame_count['spoof']}"
            )
            cv2.putText(
                display_frame,
                status,
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                display_frame,
                f"Last key: {last_key_code}",
                (10, display_frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 160),
                1,
            )
            if status_ttl > 0 and status_message:
                cv2.putText(
                    display_frame,
                    status_message,
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    status_color,
                    2,
                )
                status_ttl -= 1
            if latest_aggregate is not None:
                self._draw_preview_metrics(display_frame, latest_frame_metrics, latest_aggregate)

            cv2.putText(
                display_frame,
                _COLLECTOR_VERSION,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.imshow(window_name, display_frame)
            raw_key = cv2.waitKey(10)
            key = raw_key & 0xFF if raw_key != -1 else -1
            if key != -1:
                last_key_code = key

            if mouse_action["action"] == "capture":
                key = ord("c")
                mouse_action["action"] = None
            elif mouse_action["action"] == "live":
                key = ord("1")
                mouse_action["action"] = None
            elif mouse_action["action"] == "spoof":
                key = ord("2")
                mouse_action["action"] = None

            if key in _CAPTURE_KEYS:
                captured_frame = frame.copy()
                captured_metrics = latest_frame_metrics
                captured_aggregate = latest_aggregate
                if detection_ok:
                    print(f"\nFrame captured (#{frame_number})")
                    print("  Press 1 (LIVE) or 2 (SPOOF) to label")
                    status_message = (
                        f"Captured frame #{frame_number}. Press 1=LIVE, 2=SPOOF, "
                        "or click buttons."
                    )
                    status_color = (0, 255, 0)
                else:
                    print(f"\nFrame captured without confirmed face (#{frame_number})")
                    print("  Detector did not confirm a face yet.")
                    print("  Press 1/2 to save anyway, click buttons, or recapture.")
                    status_message = (
                        f"Captured frame #{frame_number} without confirmed face. "
                        "Press 1/2, click buttons, or recapture."
                    )
                    status_color = (0, 215, 255)
                status_ttl = _STATUS_MESSAGE_TTL_FRAMES
            elif key in {ord("1"), 49} and captured_frame is not None:
                self._save_frame(
                    captured_frame,
                    label="live",
                    frame_metrics=captured_metrics,
                    aggregate=captured_aggregate,
                )
                captured_frame = None
                status_message = "Saved as LIVE. Press SPACE to capture next frame."
                status_color = (0, 255, 0)
                status_ttl = _STATUS_MESSAGE_TTL_FRAMES
            elif key in {ord("2"), 50} and captured_frame is not None:
                self._save_frame(
                    captured_frame,
                    label="spoof",
                    frame_metrics=captured_metrics,
                    aggregate=captured_aggregate,
                )
                captured_frame = None
                status_message = "Saved as SPOOF. Press SPACE to capture next frame."
                status_color = (0, 255, 0)
                status_ttl = _STATUS_MESSAGE_TTL_FRAMES
            elif key in _QUIT_KEYS:
                break

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    def _handle_mouse(
        self,
        *,
        event: int,
        x: int,
        y: int,
        mouse_action: dict[str, str | None],
    ) -> None:
        """Map mouse clicks to collector actions."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._point_in_rect(x, y, _LIVE_BUTTON):
            mouse_action["action"] = "live"
            return
        if self._point_in_rect(x, y, _SPOOF_BUTTON):
            mouse_action["action"] = "spoof"
            return
        mouse_action["action"] = "capture"

    @staticmethod
    def _draw_button(
        frame: np.ndarray,
        rect: tuple[tuple[int, int], tuple[int, int]],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a clickable action button."""
        (x1, y1), (x2, y2) = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1 + 12, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    @staticmethod
    def _point_in_rect(
        x: int,
        y: int,
        rect: tuple[tuple[int, int], tuple[int, int]],
    ) -> bool:
        """Return whether a point is inside a rectangle."""
        (x1, y1), (x2, y2) = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _process_preview_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[FrameMetrics | None, AggregatedMetrics | None]:
        """Run the same continuous temporal pipeline used by the debug preview."""
        try:
            frame_metrics = self.frame_processor.process_frame(frame)
            recent_entries = self.temporal_aggregator.get_recent_entries(now=frame_metrics.timestamp)
            frame_metrics = self.frame_processor.enrich_device_spoof_with_history(
                frame_metrics,
                recent_entries,
            )
            aggregate = self.temporal_aggregator.add(frame_metrics)
            return frame_metrics, aggregate
        except Exception as exc:
            logger.exception("Collector preview pipeline failed: %s", exc)
            return None, None

    def _save_frame(
        self,
        frame: np.ndarray,
        *,
        label: str,
        frame_metrics: FrameMetrics | None,
        aggregate: AggregatedMetrics | None,
    ) -> None:
        """Save frame and the synchronized preview metrics snapshot."""
        label = label.lower()
        self.frame_count[label] += 1
        timestamp = datetime.now().isoformat()
        filename = f"{label}_frame_{self.frame_count[label]:03d}"

        image_path = self.output_dir / f"{filename}.jpg"
        cv2.imwrite(str(image_path), frame)

        metrics = self._build_metrics_record(
            timestamp=timestamp,
            label=label,
            image_path=image_path,
            frame_metrics=frame_metrics,
            aggregate=aggregate,
        )

        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        with open(self.SUMMARY_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        raw_prediction = metrics["prediction"]
        binary_prediction = metrics["binary_prediction"]
        prediction = binary_prediction["label"]
        confidence = float(binary_prediction["confidence"] or 0.0)
        predicted_live = binary_prediction["is_live"]
        if predicted_live is None:
            correct = "UNDECIDED"
        else:
            correct = (
                "OK"
                if (predicted_live and label == "live") or (not predicted_live and label == "spoof")
                else "MISS"
            )
        print(
            f"\n{correct} Saved {filename}: Binary={prediction} "
            f"(conf={confidence:.2f}) Raw={raw_prediction['decision_state']}"
        )
        self._print_metric_summary(filename, metrics)

    def _build_metrics_record(
        self,
        *,
        timestamp: str,
        label: str,
        image_path: Path,
        frame_metrics: FrameMetrics | None,
        aggregate: AggregatedMetrics | None,
    ) -> dict[str, Any]:
        """Build a JSON-friendly record from the synchronized preview pipeline state."""
        details = dict(frame_metrics.details) if frame_metrics is not None else {}
        device_spoof = frame_metrics.device_spoof if frame_metrics is not None else None
        decision_state = aggregate.decision_state if aggregate is not None else "NO_DATA"
        prediction_is_live = decision_state == "LIVE"
        prediction_confidence = aggregate.window_confidence if aggregate is not None else 0.0
        rppg_frame_count = int(details.get("rppg_frame_count") or 0)
        rppg_score = float(details.get("rppg_score") or 0.5)
        rppg_signal_strength = float(details.get("rppg_signal_strength") or 0.0)
        rppg_bpm = float(details.get("rppg_bpm") or 0.0)
        rppg_live_signal = bool(
            rppg_frame_count >= 45
            and rppg_score >= 0.60
            and rppg_signal_strength >= 0.25
            and 40.0 <= rppg_bpm <= 180.0
        )
        binary_prediction = self._derive_binary_prediction(
            decision_state=decision_state,
            frame_metrics=frame_metrics,
            aggregate=aggregate,
            details=details,
            rppg_live_signal=rppg_live_signal,
        )

        record = {
            "timestamp": timestamp,
            "label": label,
            "frame_number": self.frame_count[label],
            "image_path": str(image_path),
            "collector_version": _COLLECTOR_VERSION,
            "collector_pipeline": "preview_temporal",
            "face_detected": frame_metrics.face_detected if frame_metrics is not None else False,
            "prediction": {
                "label": decision_state,
                "is_live": prediction_is_live,
                "confidence": prediction_confidence,
                "decision_state": decision_state,
            },
            "binary_prediction": binary_prediction,
            "liveness": {
                "is_live": prediction_is_live,
                "confidence": prediction_confidence,
                "method": "preview_temporal",
                "scores": {
                    "frame_raw_score": self._safe_float(frame_metrics.raw_score if frame_metrics is not None else None),
                    "frame_passive_score": self._safe_float(frame_metrics.passive_score if frame_metrics is not None else None),
                    "frame_active_score": self._safe_float(frame_metrics.active_score if frame_metrics is not None else None),
                    "supported_score": self._safe_float(aggregate.supported_score if aggregate is not None else None),
                    "smoothed_score": self._safe_float(aggregate.smoothed_score if aggregate is not None else None),
                    "temporal_consistency": self._safe_float(aggregate.temporal_consistency if aggregate is not None else None),
                    "moire_score": self._device_spoof_value(device_spoof, "moire_risk"),
                    "device_replay_score": self._device_spoof_value(device_spoof, "device_replay_risk"),
                    "flicker_score": self._device_spoof_value(device_spoof, "flicker_risk"),
                    "flash_response_score": self._device_spoof_value(device_spoof, "flash_response_score"),
                    "flash_replay_risk": self._device_spoof_value(device_spoof, "flash_replay_risk"),
                    "screen_frame_score": self._device_spoof_value(device_spoof, "screen_frame_risk"),
                    "reflection_score": self._device_spoof_value(device_spoof, "reflection_risk"),
                    "rppg_score": self._safe_float(rppg_score),
                    "rppg_signal_strength": self._safe_float(rppg_signal_strength),
                    "rppg_bpm": self._safe_float(rppg_bpm),
                },
                "checks": {
                    "rppg_available": rppg_frame_count > 0,
                    "rppg_fusion_enabled": _RPPG_FUSION_ENABLED,
                    "rppg_live_signal": rppg_live_signal,
                    "rppg_frame_count": rppg_frame_count,
                    "rppg_reason": details.get("rppg_reason"),
                    "decision_state": decision_state,
                    "face_detected": frame_metrics.face_detected if frame_metrics is not None else False,
                    "face_usable": bool(details.get("face_usable")) if "face_usable" in details else None,
                    "face_usability_reason": details.get("face_usability_reason"),
                    "background_active_detected": frame_metrics.background_active_detected if frame_metrics is not None else False,
                    "device_replay_triggered": self._device_spoof_value(device_spoof, "device_replay_risk", default=0.0) >= 0.70,
                    "moire_triggered": self._device_spoof_value(device_spoof, "moire_risk", default=0.0) >= 0.70,
                    "flash_triggered": self._device_spoof_value(device_spoof, "flash_replay_risk", default=0.0) >= 0.70,
                },
                "metadata": {
                    "preview_window_seconds": aggregate.window_seconds if aggregate is not None else None,
                    "preview_sample_count": aggregate.sample_count if aggregate is not None else 0,
                    "preview_security_profile": "standard",
                    "rppg_mode": "weighted" if _RPPG_FUSION_ENABLED else "observational_only",
                    "error": frame_metrics.error if frame_metrics is not None else "no_frame_metrics",
                },
            },
            "frame_metrics": self._jsonable(frame_metrics),
            "aggregate_metrics": self._jsonable(aggregate),
        }
        return self._jsonable(record)

    def _derive_binary_prediction(
        self,
        *,
        decision_state: str,
        frame_metrics: FrameMetrics | None,
        aggregate: AggregatedMetrics | None,
        details: dict[str, Any],
        rppg_live_signal: bool,
    ) -> dict[str, Any]:
        """Map preview decision states into a safer binary output for dataset labeling."""
        device_spoof = frame_metrics.device_spoof if frame_metrics is not None else None
        return self._derive_binary_prediction_static(
            decision_state=decision_state,
            face_detected=bool(frame_metrics.face_detected) if frame_metrics is not None else False,
            face_usable=bool(details.get("face_usable")) if "face_usable" in details else False,
            rppg_live_signal=rppg_live_signal,
            device_replay_risk=self._device_spoof_value(device_spoof, "device_replay_risk", default=0.0) or 0.0,
            moire_risk=self._device_spoof_value(device_spoof, "moire_risk", default=0.0) or 0.0,
            flash_replay_risk=self._device_spoof_value(device_spoof, "flash_replay_risk", default=0.0) or 0.0,
            screen_frame_risk=self._device_spoof_value(device_spoof, "screen_frame_risk", default=0.0) or 0.0,
            confidence=aggregate.window_confidence if aggregate is not None else 0.0,
        )

    @staticmethod
    def _derive_binary_prediction_static(
        *,
        decision_state: str,
        face_detected: bool,
        face_usable: bool,
        rppg_live_signal: bool,
        device_replay_risk: float,
        moire_risk: float,
        flash_replay_risk: float,
        screen_frame_risk: float,
        confidence: float,
    ) -> dict[str, Any]:
        """Pure helper shared by save-time and overlay binary labeling."""
        if decision_state == "LIVE":
            return {
                "label": "LIVE",
                "is_live": True,
                "confidence": confidence,
                "source": "decision_state",
            }
        if decision_state == "SPOOF":
            return {
                "label": "SPOOF",
                "is_live": False,
                "confidence": confidence,
                "source": "decision_state",
            }
        explicit_spoof_evidence = bool(
            device_replay_risk >= 0.75
            or moire_risk >= 0.72
            or flash_replay_risk >= 0.75
            or screen_frame_risk >= 0.72
        )
        explicit_live_evidence = bool(
            face_detected
            and face_usable
            and rppg_live_signal
            and device_replay_risk < 0.60
            and moire_risk < 0.65
            and flash_replay_risk < 0.70
        )

        if explicit_spoof_evidence:
            return {
                "label": "SPOOF",
                "is_live": False,
                "confidence": confidence,
                "source": "explicit_spoof_evidence",
            }
        if explicit_live_evidence:
            return {
                "label": "LIVE",
                "is_live": True,
                "confidence": confidence,
                "source": "explicit_live_evidence",
            }
        return {
            "label": "UNDECIDED",
            "is_live": None,
            "confidence": confidence,
            "source": "insufficient_evidence",
        }

    @staticmethod
    def _device_spoof_value(device_spoof: Any, field_name: str, *, default: float | None = None) -> float | None:
        """Read a numeric spoof metric from the assessment object."""
        if device_spoof is None:
            return default
        return TestDataCollector._safe_float(getattr(device_spoof, field_name, default))

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert values to JSON-safe floats."""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """Convert dataclasses, numpy scalars, and tuples into JSON-friendly values."""
        if is_dataclass(value):
            return cls._jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float):
            return cls._safe_float(value)
        return value

    @staticmethod
    def _draw_preview_metrics(
        frame: np.ndarray,
        frame_metrics: FrameMetrics | None,
        aggregate: AggregatedMetrics,
    ) -> None:
        """Show the synchronized temporal state directly on the collector window."""
        details = frame_metrics.details if frame_metrics is not None else {}
        device_spoof = frame_metrics.device_spoof if frame_metrics is not None else None
        binary_prediction = TestDataCollector._derive_binary_prediction_static(
            decision_state=aggregate.decision_state,
            face_detected=bool(frame_metrics.face_detected) if frame_metrics is not None else False,
            face_usable=bool(details.get("face_usable")) if "face_usable" in details else False,
            rppg_live_signal=bool(getattr(aggregate, "rppg_live_signal", False)),
            device_replay_risk=float(getattr(device_spoof, "device_replay_risk", 0.0) if device_spoof else 0.0),
            moire_risk=float(getattr(device_spoof, "moire_risk", 0.0) if device_spoof else 0.0),
            flash_replay_risk=float(getattr(device_spoof, "flash_replay_risk", 0.0) if device_spoof else 0.0),
            screen_frame_risk=float(getattr(device_spoof, "screen_frame_risk", 0.0) if device_spoof else 0.0),
            confidence=aggregate.window_confidence,
        )
        lines = [
            f"Binary: {binary_prediction['label']}",
            f"Raw: {aggregate.decision_state}",
            f"Samples: {aggregate.sample_count}",
            f"Smooth: {aggregate.smoothed_score:.2f}",
            f"Support: {aggregate.supported_score:.2f}",
            f"Active: {aggregate.debug_active_score:.1f}",
        ]
        if frame_metrics is not None:
            rppg_overlay = (
                f"rPPG n={int(details.get('rppg_frame_count') or 0)} "
                f"score={float(details.get('rppg_score') or 0.5):.2f}"
            )
            if not _RPPG_FUSION_ENABLED:
                rppg_overlay += " [obs-only]"
            lines.extend(
                [
                    rppg_overlay,
                    f"moire={float((getattr(frame_metrics.device_spoof, 'moire_risk', 0.0) if frame_metrics.device_spoof else 0.0)):.2f} replay={float((getattr(frame_metrics.device_spoof, 'device_replay_risk', 0.0) if frame_metrics.device_spoof else 0.0)):.2f}",
                    f"Guard: {aggregate.decision_guard_reason or '-'}",
                ]
            )
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, 125 + (index * 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

    def _apply_flash_debug_stimulus(self, overlay: np.ndarray) -> None:
        """Render the same flash stimulus used by the preview window."""
        flash_state = self.frame_processor.get_flash_visual_state()
        if not flash_state.get("enabled") or not flash_state.get("visible"):
            return
        color_name = str(flash_state.get("color") or "white")
        bgr = {
            "red": (40, 40, 255),
            "green": (60, 255, 60),
            "blue": (255, 90, 40),
            "yellow": (40, 255, 255),
            "white": (255, 255, 255),
        }.get(color_name, (255, 255, 255))
        stimulus = overlay.copy()
        cv2.rectangle(stimulus, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), bgr, -1)
        cv2.addWeighted(stimulus, 0.84, overlay, 0.16, 0, overlay)
        cv2.putText(
            overlay,
            f"FLASH {color_name.upper()}",
            (max(20, overlay.shape[1] // 2 - 90), 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0) if color_name == "white" else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _print_metric_summary(filename: str, metrics: dict[str, Any]) -> None:
        """Print the saved metric snapshot to the terminal for fast debugging."""
        prediction = metrics.get("prediction") or {}
        binary_prediction = metrics.get("binary_prediction") or {}
        liveness = metrics.get("liveness") or {}
        checks = liveness.get("checks") or {}
        scores = liveness.get("scores") or {}
        metadata = liveness.get("metadata") or {}
        aggregate = metrics.get("aggregate_metrics") or {}

        print("  Metrics:")
        print(
            "    decision="
            f"{prediction.get('decision_state')} | binary={binary_prediction.get('label')} "
            f"| conf={float(binary_prediction.get('confidence') or 0.0):.2f}"
        )
        print(
            "    guard="
            f"{aggregate.get('decision_guard_reason') or '-'} | "
            f"active:{float(aggregate.get('debug_active_score') or 0.0):.1f} | "
            f"sufficient:{aggregate.get('sufficient_evidence')} | "
            f"window_conf:{float(aggregate.get('window_confidence') or 0.0):.2f} | "
            f"reasons:{','.join((aggregate.get('suspicion_reasons') or [])) or '-'}"
        )
        print(
            "    face="
            f"detected:{int(bool(checks.get('face_detected')))} "
            f"usable:{checks.get('face_usable')} "
            f"reason:{checks.get('face_usability_reason') or '-'}"
        )
        print(
            "    rppg="
            f"mode:{'weighted' if _RPPG_FUSION_ENABLED else 'observational_only'} "
            f"available:{int(bool(checks.get('rppg_available')))} "
            f"live:{int(bool(checks.get('rppg_live_signal')))} "
            f"n:{int(checks.get('rppg_frame_count') or 0)} "
            f"score:{float(scores.get('rppg_score') or 0.0):.2f} "
            f"sig:{float(scores.get('rppg_signal_strength') or 0.0):.2f} "
            f"bpm:{float(scores.get('rppg_bpm') or 0.0):.1f} "
            f"reason:{checks.get('rppg_reason') or '-'}"
        )
        print(
            "    flash="
            f"score:{float(scores.get('flash_response_score') or 0.0):.2f} "
            f"risk:{float(scores.get('flash_replay_risk') or 0.0):.2f} "
            f"triggered:{int(bool(checks.get('flash_triggered')))}"
        )
        print(
            "    spoof="
            f"moire:{float(scores.get('moire_score') or 0.0):.2f} "
            f"replay:{float(scores.get('device_replay_score') or 0.0):.2f} "
            f"flicker:{float(scores.get('flicker_score') or 0.0):.2f} "
            f"screen:{float(scores.get('screen_frame_score') or 0.0):.2f}"
        )
        print(
            "    temporal="
            f"samples:{int(metadata.get('preview_sample_count') or 0)} "
            f"window:{float(metadata.get('preview_window_seconds') or 0.0):.1f}s "
            f"smoothed:{float(scores.get('smoothed_score') or 0.0):.2f} "
            f"supported:{float(scores.get('supported_score') or 0.0):.2f}"
        )

    def _print_summary(self) -> None:
        """Print collection summary."""
        total = self.frame_count["live"] + self.frame_count["spoof"]
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames: {total}")
        print(f"  LIVE:  {self.frame_count['live']}")
        print(f"  SPOOF: {self.frame_count['spoof']}")
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Summary JSONL: {self.SUMMARY_FILE.absolute()}")
        print("=" * 60 + "\n")


def main() -> None:
    """Run the collector."""
    collector = TestDataCollector()
    collector.collect_from_camera()


if __name__ == "__main__":
    main()
