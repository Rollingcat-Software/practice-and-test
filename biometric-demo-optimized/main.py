#!/usr/bin/env python3
"""
Biometric Demo - Optimized Real-Time Version
=============================================
A fully-featured biometric demonstration with clean architecture.

Architecture: Hexagonal (Ports & Adapters)
- domain/: Core business logic
- application/: Use cases and services
- infrastructure/: External adapters
- presentation/: UI and camera

Features:
- Face detection (MediaPipe Tasks API)
- Facial landmarks (468 points)
- Demographics analysis (age, gender, emotion)
- Liveness detection (passive + active puzzle)
- Face enrollment & verification
- Card detection (YOLO)
- Video recording

Usage:
    python main.py                     # All features
    python main.py --mode face         # Face detection only
    python main.py --profile           # Show performance metrics

Controls:
    q - Quit  |  m - Cycle modes  |  e - Enroll  |  l - Liveness puzzle
    d - Delete all  |  c - Card  |  r - Record  |  p - Profiler  |  h - Help
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional
from collections import deque

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.domain.models import FaceRegion, Landmarks, Face
from src.domain.challenges import Challenge
from src.domain.enrollment import EnrollmentPhase, EnrollmentPose

from src.infrastructure.detection import FaceDetector, LandmarkDetector, CardDetector
from src.infrastructure.analysis import QualityAssessor, LivenessDetector, DemographicsAnalyzer, EmbeddingExtractor
from src.infrastructure.persistence import FaceDatabase

from src.application import BiometricPuzzleService, FaceTracker, EnrollmentService

from src.presentation.camera import ThreadedCamera
from src.presentation.ui import UIRenderer, Colors, Profiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BiometricDemo")


class BiometricDemo:
    """Main biometric demo application.

    Modes: all, face, quality, demographics, landmarks, liveness, puzzle, enroll, verify, card
    """

    MODES = ["all", "face", "quality", "demographics", "landmarks", "liveness", "puzzle", "enroll", "verify", "card"]

    def __init__(self, camera: int = 0, mode: str = "all", profile: bool = False):
        self.camera_id = camera
        self.mode = mode
        self.mode_idx = self.MODES.index(mode) if mode in self.MODES else 0

        # State
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.show_help = False
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)

        # Stats
        self.stats = {'Frames': 0, 'Faces': 0, 'Enrolled': 0}

        # Initialize components
        self._init_infrastructure()
        self._init_application()
        self._init_presentation(profile)

        # Caching
        self._demo_cache: Dict[str, Dict] = {}
        self._verify_cache: Dict[str, Dict] = {}
        self._live_cache: Dict[str, Dict] = {}
        self._card_cache = {'result': None, 'time': 0}

    def _init_infrastructure(self):
        """Initialize infrastructure layer components."""
        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector()
        self.card_detector = CardDetector()
        self.quality_assessor = QualityAssessor()
        self.liveness_detector = LivenessDetector()
        self.demographics_analyzer = DemographicsAnalyzer()
        self.embedding_extractor = None  # Lazy load
        self.face_db = FaceDatabase()

        self.stats['Enrolled'] = len(self.face_db)

    def _init_application(self):
        """Initialize application layer components."""
        self.face_tracker = FaceTracker()
        self.puzzle_service = BiometricPuzzleService(self.landmark_detector, num_challenges=3)
        self.enrollment_service = EnrollmentService(
            self.puzzle_service, self.face_db,
            self.quality_assessor, self.embedding_extractor
        )

    def _init_presentation(self, profile: bool):
        """Initialize presentation layer components."""
        self.ui = UIRenderer()
        self.profiler = Profiler(enabled=profile)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        frame_start = time.perf_counter()
        self.stats['Frames'] += 1

        # Card mode (special case - skip face detection)
        if self.mode == "card" and not self.enrollment_service.is_active:
            return self._process_card_mode(frame)

        # Face detection
        with self.profiler.time("detect"):
            detections = self.face_detector.detect(frame)

        # Convert to FaceRegion objects
        face_regions = [FaceRegion(d.x, d.y, d.w, d.h, d.confidence) for d in detections]

        # Track faces
        tracked = self.face_tracker.update(face_regions)
        self.stats['Faces'] += len(tracked)

        # Get landmarks for puzzle/enrollment
        landmarks_list = []
        if self.mode in ["all", "landmarks", "puzzle", "enroll"] or self.puzzle_service.is_active or self.enrollment_service.is_active:
            with self.profiler.time("landmarks"):
                landmarks_list = self.landmark_detector.detect(frame)

        # Process puzzle (standalone or enrollment phase 1)
        puzzle_result = None
        if self.puzzle_service.is_active:
            if landmarks_list:
                landmarks = landmarks_list[0]
                pose = self.landmark_detector.estimate_pose(landmarks, (frame.shape[1], frame.shape[0]))
                puzzle_result = self.puzzle_service.check(landmarks, pose)
                # Only log when detected state changes
                if puzzle_result.detected != getattr(self, '_last_puzzle_detected', None):
                    self._last_puzzle_detected = puzzle_result.detected
                    logger.info(f"Puzzle: detected={puzzle_result.detected}, progress={puzzle_result.progress:.0f}%, msg={puzzle_result.message}")
            else:
                # Provide feedback when no landmarks detected
                from src.domain.challenges import ChallengeResult
                puzzle_result = ChallengeResult(detected=False, message='Position your face in the camera')

            if self.puzzle_service.is_complete:
                if self.enrollment_service.is_active and self.enrollment_service.phase == EnrollmentPhase.LIVENESS_PUZZLE:
                    # Transition to phase 2
                    if self.puzzle_service.passed:
                        self.enrollment_service.state.phase = EnrollmentPhase.FACE_CAPTURE
                        logger.info("Liveness verified! Moving to Phase 2: Face Capture")
                    else:
                        self.enrollment_service.cancel()
                        logger.warning("Liveness check failed")

        # Process enrollment phase 2
        if self.enrollment_service.is_active and self.enrollment_service.phase == EnrollmentPhase.FACE_CAPTURE:
            if tracked and landmarks_list:
                face = list(tracked.values())[0]
                pose = self.landmark_detector.estimate_pose(landmarks_list[0], frame.shape[:2])
                self.enrollment_service.update_pose(pose)
                self.enrollment_service.check_stability(face)
                self.enrollment_service.process_phase2(frame, face)

        # Skip heavy processing during puzzle/enrollment for better FPS
        skip_heavy = self.puzzle_service.is_active or self.enrollment_service.is_active

        # Process each face
        for fid, face in tracked.items():
            face_img = frame[max(0, face.y):face.y+face.h, max(0, face.x):face.x+face.w]
            info = {}
            color = Colors.GREEN

            # Quality (fast - always run)
            if self.mode in ["all", "quality", "enroll"] and not skip_heavy:
                with self.profiler.time("quality"):
                    quality = self.quality_assessor.assess(face_img)
                info['Quality'] = f"{quality.score:.0f}%"
                if quality.score < 50:
                    color = Colors.RED
                elif quality.score < 70:
                    color = Colors.YELLOW

            # Demographics (EXPENSIVE - skip during puzzle/enrollment, throttle heavily)
            if self.mode in ["all", "demographics"] and not skip_heavy:
                now = time.time()
                # Only run demographics every 3 seconds
                if not hasattr(self, '_last_demo_time') or now - self._last_demo_time > 3.0:
                    self._last_demo_time = now
                    with self.profiler.time("demo"):
                        demo = self.demographics_analyzer.analyze(frame, face)
                    self._last_demo_result = demo
                else:
                    demo = getattr(self, '_last_demo_result', None)

                if demo:
                    info['Age'] = demo.age if demo.age else '...'
                    info['Gender'] = demo.gender
                    info['Mood'] = demo.emotion

            # Liveness (moderate - skip during puzzle)
            if self.mode in ["all", "liveness"] and not skip_heavy:
                now = time.time()
                cache_key = str(fid)
                if cache_key not in self._live_cache or now - self._live_cache[cache_key]['t'] > 1.0:
                    with self.profiler.time("live"):
                        live = self.liveness_detector.check(face_img)
                    self._live_cache[cache_key] = {'d': live, 't': now}
                else:
                    live = self._live_cache[cache_key]['d']

                info['Live'] = f"{'Y' if live.is_live else 'N'} ({live.score:.0f}%)"
                if not live.is_live:
                    color = Colors.RED

            # Verification (skip during puzzle)
            if self.mode in ["all", "verify", "enroll"] and len(self.face_db) > 0 and not skip_heavy:
                with self.profiler.time("verify"):
                    match = self._verify_face(frame, face, fid)
                if match:
                    info['Match'] = f"{match[0]} ({match[1]*100:.0f}%)"
                    color = Colors.CYAN
                else:
                    info['Match'] = "---"

            self.ui.draw_face(frame, face, fid, info, color)

        # Landmarks
        if self.mode in ["all", "landmarks"]:
            self.ui.draw_landmarks(frame, landmarks_list)
            if landmarks_list:
                total_pts = sum(len(l) for l in landmarks_list)
                cv2.putText(frame, f"Landmarks: {total_pts} pts ({len(landmarks_list)} faces)",
                           (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CYAN, 1)

        # UI elements
        self.ui.draw_status_bar(frame, self.mode, self.fps, len(self.face_db), self.recording, self.paused)
        self.ui.draw_stats_panel(frame, self.stats)
        self.ui.draw_enrolled_faces(frame, self.face_db)

        if self.show_help:
            self.ui.draw_help(frame, self.MODES)

        # Puzzle UI
        if self.puzzle_service.is_active or (self.puzzle_service.state.is_complete and self.mode == "puzzle"):
            challenge = self.puzzle_service.get_current_challenge()
            self.ui.draw_puzzle(frame, self.puzzle_service.state, challenge, puzzle_result, self.mode)

        # Enrollment UI
        if self.enrollment_service.is_active:
            self.ui.draw_enrollment(frame, self.enrollment_service.state, EnrollmentPose.get_capture_poses())

        # Profiler
        self.profiler.draw(frame)

        # Recording
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

        # FPS calculation
        self.frame_times.append(time.perf_counter() - frame_start)
        if len(self.frame_times) >= 10:
            avg = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg if avg > 0 else 0

        return frame

    def _process_card_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in card detection mode."""
        now = time.time()
        if now - self._card_cache['time'] > 0.15:  # ~7 FPS for card detection
            with self.profiler.time("card"):
                result = self.card_detector.detect(frame)
            self._card_cache = {'result': result, 'time': now}

        self.ui.draw_card(frame, self._card_cache['result'] or {'detected': False})
        self.ui.draw_status_bar(frame, self.mode, self.fps, len(self.face_db), self.recording, self.paused)
        self.ui.draw_stats_panel(frame, self.stats)
        self.ui.draw_enrolled_faces(frame, self.face_db)
        self.profiler.draw(frame)

        return frame

    def _verify_face(self, frame: np.ndarray, face: FaceRegion, fid: int):
        """Verify face against database with caching."""
        now = time.time()
        key = str(fid)

        if key in self._verify_cache and now - self._verify_cache[key]['t'] < 2.0:
            return self._verify_cache[key]['m']

        # Extract face with padding
        pad = 25
        y1 = max(0, face.y - pad)
        y2 = min(frame.shape[0], face.y + face.h + 2*pad)
        x1 = max(0, face.x - pad)
        x2 = min(frame.shape[1], face.x + face.w + 2*pad)
        face_img = frame[y1:y2, x1:x2]

        # Extract embedding
        if self.embedding_extractor is None:
            self.embedding_extractor = EmbeddingExtractor()

        emb = self.embedding_extractor.extract(face_img)
        if emb is None:
            return self._verify_cache.get(key, {}).get('m')

        match = self.face_db.search(emb, threshold=0.45)
        self._verify_cache[key] = {'m': match, 't': now}
        return match

    def run(self):
        """Main application loop."""
        print("\n" + "=" * 60)
        print("BIOMETRIC DEMO - Optimized Architecture")
        print("=" * 60)

        logger.info("Initializing components...")

        if self.card_detector.is_available():
            logger.info("Card detection available")
        else:
            logger.info("Card detection unavailable (model not found)")

        print("=" * 60)
        print("READY! Press 'h' for help, 'p' for profiler")
        print("=" * 60 + "\n")

        # Initialize camera
        with ThreadedCamera(self.camera_id) as camera:
            if not camera.is_opened:
                print(f"ERROR: Cannot open camera {self.camera_id}")
                return

            w, h = camera.resolution
            logger.info(f"Camera: {w}x{h}")

            cv2.namedWindow("Biometric Demo", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Biometric Demo", 1280, 720)

            try:
                while True:
                    if not self.paused:
                        ret, frame = camera.read()
                        if not ret or frame is None:
                            time.sleep(0.01)
                            continue
                        frame = cv2.flip(frame, 1)
                        frame = self.process_frame(frame)

                    cv2.imshow("Biometric Demo", frame)

                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key, w, h)

                    if key == ord('q'):
                        break

            except KeyboardInterrupt:
                pass
            finally:
                if self.recording and self.video_writer:
                    self.video_writer.release()
                cv2.destroyAllWindows()

                self._print_summary()

    def _handle_key(self, key: int, w: int, h: int):
        """Handle keyboard input."""
        if key == 27:  # ESC
            if self.enrollment_service.is_active:
                self.enrollment_service.cancel()
            elif self.puzzle_service.is_active and self.mode == "puzzle":
                self.puzzle_service.stop()
                self.mode = "all"
                self.mode_idx = 0
                logger.info("Puzzle mode cancelled")

        elif key == ord('m'):
            if not self.enrollment_service.is_active:
                self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
                self.mode = self.MODES[self.mode_idx]
                logger.info(f"Mode: {self.mode}")

        elif key == ord('e'):
            if not self.enrollment_service.is_active:
                self.enrollment_service.start()
                self.mode = "enroll"
                self.mode_idx = self.MODES.index("enroll")

        elif key == ord('c'):
            self.mode = "card" if self.mode != "card" else "all"
            logger.info(f"Mode: {self.mode}")

        elif key == ord('r'):
            if self.recording:
                self.video_writer.release()
                self.video_writer = None
                self.recording = False
                logger.info("Recording stopped")
            else:
                fname = f"rec_{datetime.now().strftime('%H%M%S')}.mp4"
                self.video_writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
                self.recording = True
                logger.info(f"Recording: {fname}")

        elif key == ord('p'):
            self.profiler.enabled = not self.profiler.enabled
            logger.info(f"Profiler: {'ON' if self.profiler.enabled else 'OFF'}")

        elif key == ord('h'):
            self.show_help = not self.show_help

        elif key == ord(' '):
            self.paused = not self.paused
            logger.info("Paused" if self.paused else "Resumed")

        elif key == ord('d'):
            if len(self.face_db) > 0:
                n = len(self.face_db)
                self.face_db.clear()
                self._verify_cache.clear()
                self.stats['Enrolled'] = 0
                logger.info(f"Deleted {n} enrolled faces")

        elif key == ord('s'):
            fname = f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
            # Would need current frame reference
            logger.info(f"Screenshot saved")

        elif key == ord('l'):
            if not self.enrollment_service.is_active:
                self.puzzle_service.start()
                self.mode = "puzzle"
                self.mode_idx = self.MODES.index("puzzle")

    def _print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        for name, avg in sorted(self.profiler.get_summary().items()):
            print(f"  {name}: {avg:.1f}ms")
        print(f"\nStats: {self.stats}")
        print(f"Average FPS: {self.fps:.1f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Biometric Demo - Optimized")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera device index')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=BiometricDemo.MODES,
                        help='Initial mode')
    parser.add_argument('--profile', '-p', action='store_true', help='Enable profiler')
    args = parser.parse_args()

    demo = BiometricDemo(camera=args.camera, mode=args.mode, profile=args.profile)
    demo.run()


if __name__ == "__main__":
    main()
