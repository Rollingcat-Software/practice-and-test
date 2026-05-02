#!/usr/bin/env python3
"""
FIVUCSAS Spoof Detector — Real-Time Multi-Class Face Spoofing Analysis
======================================================================

A comprehensive spoof detection research tool that classifies faces
into 7 categories in real-time:
  - Real (genuine live person)
  - Static Image (printed photo or digital still)
  - Video Replay (pre-recorded video on screen)
  - 3D Mask (silicone/latex mask)
  - Heavy Makeup (contouring/prosthetics)
  - AR Filter (Snapchat, Instagram, OBS overlay)
  - Deepfake Injection (virtual webcam)

Part of the FIVUCSAS biometric authentication platform.
Research project for academic paper on AR-filter spoof detection.

Usage:
    python main.py                    # Default camera (index 0)
    python main.py --camera 1         # Specific camera
    python main.py --no-detail        # Hide probability panel
    python main.py --no-log           # Disable logging
    python main.py --config my.yaml   # Custom config

Controls:
    q / ESC  — Quit
    d        — Toggle detail panel
    l        — Toggle logging
    s        — Save frame + analysis
    p        — Toggle profiler
    h        — Help overlay
"""

import os
import sys
import argparse
import logging

# Suppress noisy library warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from src.presentation.camera import ThreadedCamera
from src.presentation.app import SpoofDetectorApp
from src.application.pipeline import SpoofDetectionPipeline
from src.application.face_tracker import FaceTracker
from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
from src.infrastructure.logging.structured_logger import StructuredLogger


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_pipeline(config: dict) -> tuple:
    """Build the full spoof detection pipeline from config.

    Returns:
        (pipeline, camera, struct_logger)
    """
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    ana_cfg = config.get("analyzers", {})
    fus_cfg = config.get("fusion", {})
    log_cfg = config.get("logging", {})

    # Camera
    camera = ThreadedCamera(
        src=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
    )

    # Face detector
    detector = MediaPipeFaceDetector(
        min_confidence=det_cfg.get("min_confidence", 0.5),
    )

    # Tracker
    tracker = FaceTracker(
        iou_threshold=trk_cfg.get("iou_threshold", 0.3),
        max_lost_frames=trk_cfg.get("max_lost_frames", 15),
    )

    # Per-face analyzers
    face_analyzers = []
    if ana_cfg.get("minifasnet", {}).get("enabled", True):
        face_analyzers.append(MiniFASNetAnalyzer())
    if ana_cfg.get("texture", {}).get("enabled", True):
        tex_cfg = ana_cfg.get("texture", {})
        face_analyzers.append(TextureAnalyzer(
            texture_threshold=tex_cfg.get("laplacian_threshold", 100.0),
            fft_downsample=tuple(tex_cfg.get("fft_downsample", [192, 108])),
        ))
    if ana_cfg.get("moire", {}).get("enabled", True):
        moire_cfg = ana_cfg.get("moire", {})
        face_analyzers.append(MoireAnalyzer(
            response_std_threshold=moire_cfg.get("response_std_threshold", 30.0),
        ))
    if ana_cfg.get("temporal", {}).get("enabled", True):
        tmp_cfg = ana_cfg.get("temporal", {})
        face_analyzers.append(TemporalAnalyzer(
            buffer_size=tmp_cfg.get("buffer_size", 30),
            min_motion_std=tmp_cfg.get("min_motion_std", 0.0003),
        ))

    if ana_cfg.get("device_boundary", {}).get("enabled", True):
        db_cfg = ana_cfg.get("device_boundary", {})
        face_analyzers.append(DeviceBoundaryAnalyzer(
            padding_ratio=db_cfg.get("padding_ratio", 0.55),
            spoof_threshold=db_cfg.get("spoof_threshold", 0.50),
        ))

    # Whole-frame analyzers
    frame_analyzers = []
    if ana_cfg.get("screen_replay", {}).get("enabled", True):
        frame_analyzers.append(ScreenReplayAnalyzer())

    # Fuser (uses calibrated defaults if no config override)
    fuser = MultiClassFuser(
        analyzer_weights=fus_cfg.get("weights") if fus_cfg.get("weights") else None,
    )

    # Pipeline
    pipeline = SpoofDetectionPipeline(
        detector=detector,
        tracker=tracker,
        face_analyzers=face_analyzers,
        frame_analyzers=frame_analyzers,
        fuser=fuser,
    )

    # Structured logger
    struct_logger = StructuredLogger(
        output_dir=log_cfg.get("output_dir", "logs"),
        log_every_n=log_cfg.get("log_every_n_frames", 30),
    )

    return pipeline, camera, struct_logger


def main():
    parser = argparse.ArgumentParser(
        description="FIVUCSAS Spoof Detector — Real-Time Multi-Class Face Spoofing Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--no-detail", action="store_true", help="Hide detail panel")
    parser.add_argument("--no-log", action="store_true", help="Disable structured logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)
    if args.camera is not None:
        config.setdefault("camera", {})["index"] = args.camera

    logging.getLogger("main").info("Building spoof detection pipeline...")
    pipeline, camera, struct_logger = build_pipeline(config)

    if not camera.is_opened:
        logging.getLogger("main").error("Could not open camera. Exiting.")
        sys.exit(1)

    # Create and run app
    app = SpoofDetectorApp(
        pipeline=pipeline,
        camera=camera,
        struct_logger=struct_logger,
        show_detail=not args.no_detail,
    )

    logging.getLogger("main").info("Starting spoof detector — press 'h' for help, 'q' to quit")
    app.run()


if __name__ == "__main__":
    main()
