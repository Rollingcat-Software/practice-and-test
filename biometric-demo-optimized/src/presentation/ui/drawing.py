"""UI rendering for biometric demo."""

from typing import Dict, List, Optional, Any
from datetime import datetime

import cv2
import numpy as np

from .colors import Colors
from ...domain.models import FaceRegion, Landmarks, Quality, LivenessResult, Demographics
from ...domain.challenges import Challenge, ChallengeType, ChallengeResult, PuzzleState
from ...domain.enrollment import EnrollmentState, EnrollmentPhase, EnrollmentPose
from ...infrastructure.persistence.face_database import FaceDatabase


class UIRenderer:
    """Renders all UI elements for the biometric demo."""

    def __init__(self):
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_face(self, frame: np.ndarray, region: FaceRegion, fid: int,
                  info: Dict[str, Any], color: tuple):
        """Draw face bounding box with info panel."""
        x, y, w, h = region.x, region.y, region.w, region.h

        # Main rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Corner accents
        c = min(w, h) // 4
        for px, py, dx, dy in [(x,y,1,1), (x+w,y,-1,1), (x,y+h,1,-1), (x+w,y+h,-1,-1)]:
            cv2.line(frame, (px, py), (px+c*dx, py), color, 3)
            cv2.line(frame, (px, py), (px, py+c*dy), color, 3)

        # ID badge
        cv2.rectangle(frame, (x, y-22), (x+35, y), color, -1)
        cv2.putText(frame, f"#{fid}", (x+3, y-6), self._font, 0.45, Colors.WHITE, 1)

        # Info panel
        if info:
            lines = [f"{k}:{v}" for k, v in info.items()]
            panel_w = max(len(l) * 8 for l in lines) + 10
            panel_h = len(lines) * 16 + 8

            px = x + w + 5
            if px + panel_w > frame.shape[1]:
                px = max(0, x - panel_w - 5)

            overlay = frame.copy()
            cv2.rectangle(overlay, (px, y), (px+panel_w, y+panel_h), Colors.BLACK, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            for i, line in enumerate(lines):
                cv2.putText(frame, line, (px+4, y+14+i*16), self._font, 0.4, Colors.WHITE, 1)

    def draw_landmarks(self, frame: np.ndarray, landmarks_list: List[Landmarks]):
        """Draw full 468-point landmarks with contours."""
        if not landmarks_list:
            return

        for landmarks in landmarks_list:
            if not landmarks.is_valid():
                continue

            points = landmarks.points

            # Face contour (jawline + forehead)
            contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
            self._draw_polyline(frame, points, contour_indices, Colors.CYAN)

            # Left eye outline
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
            self._draw_polyline(frame, points, left_eye, Colors.GREEN)

            # Right eye outline
            right_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
            self._draw_polyline(frame, points, right_eye, Colors.GREEN)

            # Lips outer
            lips_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
            self._draw_polyline(frame, points, lips_outer, Colors.RED)

            # Nose
            nose = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
            self._draw_polyline(frame, points, nose, Colors.YELLOW)

            # Left eyebrow
            left_brow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            self._draw_polyline(frame, points, left_brow, Colors.WHITE)

            # Right eyebrow
            right_brow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
            self._draw_polyline(frame, points, right_brow, Colors.WHITE)

            # Key landmark points
            key_points = [
                (33, Colors.GREEN), (133, Colors.GREEN),
                (160, Colors.GREEN), (144, Colors.GREEN),
                (362, Colors.GREEN), (263, Colors.GREEN),
                (387, Colors.GREEN), (373, Colors.GREEN),
                (1, Colors.YELLOW), (4, Colors.YELLOW),
                (61, Colors.RED), (291, Colors.RED),
                (0, Colors.RED), (17, Colors.RED),
            ]
            for idx, color in key_points:
                if idx < len(points):
                    cv2.circle(frame, points[idx], 3, color, -1)

            # All 468 points (small dots)
            for x, y in points:
                cv2.circle(frame, (x, y), 1, (150, 200, 200), -1)

    def _draw_polyline(self, frame: np.ndarray, points: list, indices: list, color: tuple):
        """Draw a polyline using landmark indices."""
        for i in range(len(indices) - 1):
            if indices[i] < len(points) and indices[i+1] < len(points):
                cv2.line(frame, points[indices[i]], points[indices[i+1]], color, 1)

    def draw_card(self, frame: np.ndarray, card: Dict):
        """Draw detected card bounding box."""
        if not card.get('detected'):
            return

        x1, y1, x2, y2 = card['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), Colors.CYAN, 3)
        label = f"{card['label']} ({card['confidence']*100:.0f}%)"
        cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*10, y1), Colors.CYAN, -1)
        cv2.putText(frame, label, (x1+3, y1-7), self._font, 0.5, Colors.BLACK, 2)

    def draw_status_bar(self, frame: np.ndarray, mode: str, fps: float,
                        enrolled_count: int, recording: bool = False, paused: bool = False):
        """Draw top status bar."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Mode
        mode_color = Colors.CYAN if mode not in ['enroll', 'card'] else Colors.ORANGE
        cv2.putText(frame, f"Mode: {mode.upper()}", (10, 28), self._font, 0.6, mode_color, 2)

        # Recording indicator
        if recording:
            cv2.circle(frame, (200, 20), 8, Colors.RED, -1)
            cv2.putText(frame, "REC", (215, 28), self._font, 0.5, Colors.RED, 2)

        # Paused indicator
        if paused:
            cv2.putText(frame, "PAUSED", (280, 28), self._font, 0.5, Colors.YELLOW, 2)

        # FPS with color coding
        fps_color = Colors.GREEN if fps >= 15 else Colors.YELLOW if fps >= 8 else Colors.RED
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 28), self._font, 0.5, fps_color, 1)

        # Enrolled count
        if enrolled_count > 0:
            cv2.putText(frame, f"Enrolled: {enrolled_count}", (w - 220, 28),
                       self._font, 0.5, Colors.GREEN, 1)

        # Help hint
        cv2.putText(frame, "'h' help | 'e' enroll | 'p' profiler", (w//2 - 120, 28),
                   self._font, 0.4, Colors.WHITE, 1)

    def draw_stats_panel(self, frame: np.ndarray, stats: Dict[str, int]):
        """Draw statistics panel at bottom right."""
        h, w = frame.shape[:2]
        lines = [f"{k}: {v}" for k, v in stats.items()]

        panel_w, panel_h = 130, len(lines) * 20 + 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, h - panel_h - 10), (w - 10, h - 10),
                     Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (w - panel_w, h - panel_h + 20 + i*20),
                       self._font, 0.4, Colors.WHITE, 1)

    def draw_enrolled_faces(self, frame: np.ndarray, face_db: FaceDatabase):
        """Draw thumbnails of enrolled faces at bottom left."""
        if len(face_db) == 0:
            return

        h, w = frame.shape[:2]
        thumb_size = 50
        margin = 5
        start_x = 10
        start_y = h - thumb_size - 10

        for i, (name, face) in enumerate(list(face_db.faces.items())[:5]):
            thumb = face.thumbnail
            if thumb is not None:
                try:
                    thumb_resized = cv2.resize(thumb, (thumb_size, thumb_size))
                    x = start_x + i * (thumb_size + margin)
                    frame[start_y:start_y+thumb_size, x:x+thumb_size] = thumb_resized
                    cv2.rectangle(frame, (x, start_y), (x+thumb_size, start_y+thumb_size),
                                 Colors.GREEN, 1)
                    cv2.putText(frame, name[:6], (x, start_y-5), self._font, 0.3, Colors.WHITE, 1)
                except:
                    pass

    def draw_help(self, frame: np.ndarray, modes: List[str]):
        """Draw help overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 40), (w-40, h-40), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        cv2.putText(frame, "BIOMETRIC DEMO - OPTIMIZED", (60, 80), self._font, 0.8, Colors.CYAN, 2)

        lines = [
            "", "CONTROLS:", "  q - Quit", "  m - Cycle modes", "  e - Enroll face",
            "  l - Liveness puzzle", "  d - Delete all enrolled", "  c - Card detection",
            "  r - Toggle recording", "  p - Toggle profiler", "  h - This help", "  Space - Pause",
            "", "ARCHITECTURE:",
            "  - Hexagonal (Ports & Adapters)",
            "  - Async worker for heavy operations",
            "  - Threaded camera for non-blocking I/O",
            "  - Vectorized face search (NumPy/BLAS)",
            "", f"MODES: {' | '.join(modes)}"
        ]

        y = 110
        for line in lines:
            color = Colors.YELLOW if line.endswith(':') else Colors.WHITE
            cv2.putText(frame, line, (60, y), self._font, 0.4, color, 1)
            y += 20

    def draw_puzzle(self, frame: np.ndarray, state: PuzzleState,
                    challenge: Optional[Challenge], result: Optional[ChallengeResult],
                    mode: str):
        """Draw biometric puzzle UI."""
        h, w = frame.shape[:2]

        # Show completion screen if puzzle just finished
        if state.is_complete and mode == "puzzle":
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), Colors.BLACK, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

            if state.passed:
                cv2.putText(frame, "PUZZLE PASSED!", (w//4 + 50, h//2 - 20),
                           self._font, 1.0, Colors.GREEN, 2)
            else:
                cv2.putText(frame, "PUZZLE ENDED", (w//4 + 70, h//2 - 20),
                           self._font, 1.0, Colors.YELLOW, 2)

            cv2.putText(frame, "Press L to try again", (w//4 + 60, h//2 + 30),
                       self._font, 0.6, Colors.CYAN, 1)
            return

        if not state.is_active or challenge is None:
            return

        # Draw overlay panel at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        title = "BIOMETRIC PUZZLE" if mode == "puzzle" else "LIVENESS CHECK"
        cv2.putText(frame, title, (20, 35), self._font, 0.8, Colors.CYAN, 2)

        # Challenge counter
        cv2.putText(frame, f"Challenge {state.current_idx+1}/{len(state.challenges)}",
                   (w - 180, 35), self._font, 0.6, Colors.WHITE, 1)

        # Progress bar (overall)
        progress = state.current_idx / len(state.challenges)
        cv2.rectangle(frame, (20, 50), (w-20, 70), Colors.WHITE, 2)
        cv2.rectangle(frame, (22, 52), (22 + int((w-44) * progress), 68), Colors.GREEN, -1)

        # Current challenge display
        cv2.putText(frame, challenge.display_name, (20, 110), self._font, 1.0, Colors.YELLOW, 2)

        # Challenge result/feedback
        if result:
            status_color = Colors.GREEN if result.detected else Colors.ORANGE
            cv2.putText(frame, result.message, (20, 145), self._font, 0.5, status_color, 1)

            # Hold progress bar
            if result.detected and result.progress > 0:
                cv2.putText(frame, f"HOLD! {result.progress:.0f}%", (20, 175),
                           self._font, 0.6, Colors.GREEN, 2)
                cv2.rectangle(frame, (150, 160), (350, 180), Colors.WHITE, 2)
                cv2.rectangle(frame, (152, 162), (152 + int(196 * result.progress/100), 178),
                             Colors.GREEN, -1)
        else:
            cv2.putText(frame, "Looking for face... Position yourself in frame",
                       (20, 145), self._font, 0.5, Colors.RED, 1)

        # Visual guide
        self._draw_puzzle_guide(frame, challenge.type)

        # ESC to cancel
        cv2.putText(frame, "ESC to cancel", (20, 195), self._font, 0.4, Colors.RED, 1)

    def _draw_puzzle_guide(self, frame: np.ndarray, challenge_type: ChallengeType):
        """Draw visual guide for the current challenge."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2 + 50
        color = Colors.CYAN

        if challenge_type == ChallengeType.BLINK:
            cv2.ellipse(frame, (cx - 50, cy), (30, 15), 0, 0, 360, color, 2)
            cv2.ellipse(frame, (cx + 50, cy), (30, 15), 0, 0, 360, color, 2)
            cv2.putText(frame, "CLOSE", (cx - 35, cy + 50), self._font, 0.6, color, 2)

        elif challenge_type == ChallengeType.SMILE:
            cv2.ellipse(frame, (cx, cy + 20), (60, 30), 0, 0, 180, color, 3)

        elif challenge_type == ChallengeType.OPEN_MOUTH:
            cv2.ellipse(frame, (cx, cy + 20), (40, 50), 0, 0, 360, color, 3)

        elif challenge_type in [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT]:
            direction = -1 if challenge_type == ChallengeType.TURN_LEFT else 1
            cv2.arrowedLine(frame, (cx, cy), (cx + direction * 80, cy), color, 4, tipLength=0.3)

        elif challenge_type == ChallengeType.LOOK_UP:
            cv2.ellipse(frame, (cx, cy), (50, 60), -20, 0, 360, color, 2)
            cv2.arrowedLine(frame, (cx, cy - 70), (cx - 30, cy - 110), color, 3, tipLength=0.3)
            cv2.putText(frame, "CHIN UP", (cx - 45, cy + 90), self._font, 0.6, color, 2)

        elif challenge_type == ChallengeType.LOOK_DOWN:
            cv2.ellipse(frame, (cx, cy), (50, 60), 20, 0, 360, color, 2)
            cv2.arrowedLine(frame, (cx, cy + 70), (cx + 30, cy + 110), color, 3, tipLength=0.3)
            cv2.putText(frame, "CHIN DOWN", (cx - 60, cy - 80), self._font, 0.6, color, 2)

        elif challenge_type == ChallengeType.RAISE_BOTH_BROWS:
            cv2.ellipse(frame, (cx - 40, cy - 30), (30, 10), -10, 0, 180, color, 3)
            cv2.ellipse(frame, (cx + 40, cy - 30), (30, 10), 10, 0, 180, color, 3)
            cv2.arrowedLine(frame, (cx - 40, cy - 35), (cx - 40, cy - 70), color, 2, tipLength=0.3)
            cv2.arrowedLine(frame, (cx + 40, cy - 35), (cx + 40, cy - 70), color, 2, tipLength=0.3)

        elif challenge_type == ChallengeType.CLOSE_LEFT:
            cv2.line(frame, (cx - 80, cy), (cx - 20, cy), color, 3)
            cv2.ellipse(frame, (cx + 50, cy), (30, 15), 0, 0, 360, Colors.GRAY, 2)
            cv2.putText(frame, "YOUR LEFT", (cx - 60, cy + 50), self._font, 0.5, color, 2)

        elif challenge_type == ChallengeType.CLOSE_RIGHT:
            cv2.ellipse(frame, (cx - 50, cy), (30, 15), 0, 0, 360, Colors.GRAY, 2)
            cv2.line(frame, (cx + 20, cy), (cx + 80, cy), color, 3)
            cv2.putText(frame, "YOUR RIGHT", (cx - 60, cy + 50), self._font, 0.5, color, 2)

    def draw_enrollment(self, frame: np.ndarray, state: EnrollmentState,
                        poses: List[EnrollmentPose]):
        """Draw enrollment UI."""
        if state.phase == EnrollmentPhase.IDLE:
            return

        # Phase 1 (puzzle) is handled by draw_puzzle
        if state.phase == EnrollmentPhase.LIVENESS_PUZZLE:
            return

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Header
        cv2.putText(frame, f"ENROLLING: {state.name}", (20, 30), self._font, 0.7, Colors.GREEN, 2)
        cv2.putText(frame, "Phase 2: Face Capture", (w - 200, 30), self._font, 0.4, Colors.CYAN, 1)

        # Progress bar
        progress = state.step / 5
        cv2.rectangle(frame, (20, 45), (w-20, 65), Colors.WHITE, 2)
        cv2.rectangle(frame, (22, 47), (22 + int((w-44) * progress), 63), Colors.GREEN, -1)
        cv2.putText(frame, f"{state.step}/5", (w-70, 60), self._font, 0.45, Colors.WHITE, 1)

        if state.step < 5:
            pose = poses[state.step]
            cv2.putText(frame, f"Look {pose.name}", (20, 95), self._font, 0.7, Colors.YELLOW, 2)

            yaw_ok = abs(state.current_yaw - pose.target_yaw) < pose.tolerance
            pitch_ok = abs(state.current_pitch - pose.target_pitch) < pose.tolerance

            # Draw head placeholder
            self._draw_head_placeholder(frame, pose.target_yaw, pose.target_pitch)

            # Stability indicator
            stability_color = Colors.GREEN if state.is_stable else Colors.RED
            cv2.putText(frame, f"Stability: {state.stability_score:.0f}%", (220, 95),
                       self._font, 0.5, stability_color, 1)

            # Stability bar
            cv2.rectangle(frame, (220, 100), (350, 115), Colors.WHITE, 1)
            bar_width = int(130 * state.stability_score / 100)
            cv2.rectangle(frame, (221, 101), (221 + bar_width, 114), stability_color, -1)

            if yaw_ok and pitch_ok and state.is_stable:
                hold = state.stability_score  # Use as proxy for hold progress
                cv2.putText(frame, "HOLD STILL!", (20, 125), self._font, 0.6, Colors.GREEN, 2)
            elif yaw_ok and pitch_ok and not state.is_stable:
                cv2.putText(frame, "STOP MOVING!", (20, 125), self._font, 0.6, Colors.ORANGE, 2)
                cv2.putText(frame, "Keep your head still", (20, 150), self._font, 0.4, Colors.YELLOW, 1)
            else:
                hints = []
                if state.current_yaw < pose.target_yaw - pose.tolerance:
                    hints.append("Turn RIGHT")
                elif state.current_yaw > pose.target_yaw + pose.tolerance:
                    hints.append("Turn LEFT")
                if state.current_pitch < pose.target_pitch - pose.tolerance:
                    hints.append("Tilt DOWN")
                elif state.current_pitch > pose.target_pitch + pose.tolerance:
                    hints.append("Tilt UP")
                cv2.putText(frame, " & ".join(hints) if hints else "Adjust",
                           (20, 125), self._font, 0.5, Colors.ORANGE, 1)

            # Pose indicator (mini radar)
            self._draw_pose_radar(frame, state, pose, yaw_ok, pitch_ok)
        else:
            cv2.putText(frame, "COMPLETE!", (20, 110), self._font, 0.8, Colors.GREEN, 2)

        cv2.putText(frame, "ESC to cancel", (20, 170), self._font, 0.4, Colors.RED, 1)

    def _draw_head_placeholder(self, frame: np.ndarray, target_yaw: float, target_pitch: float):
        """Draw head silhouette placeholder showing expected pose."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2 + 40
        head_w, head_h = 180, 220

        offset_x = int(target_yaw * 2)
        offset_y = int(target_pitch * 2)

        color = Colors.GRAY

        # Head ellipse
        cv2.ellipse(frame, (cx + offset_x, cy + offset_y), (head_w // 2, head_h // 2),
                   0, 0, 360, color, 2)

        # Nose line
        nose_offset = int(target_yaw * 1.5)
        cv2.line(frame, (cx + nose_offset, cy - 60 + offset_y),
                (cx + nose_offset, cy + 40 + offset_y), color, 1)

        # Eyes line
        eye_y = cy - 30 + offset_y
        cv2.line(frame, (cx - 50 + offset_x, eye_y), (cx + 50 + offset_x, eye_y), color, 1)

        # Direction arrow
        arrow_len = 50
        if abs(target_yaw) > 5:
            arrow_x = cx + (arrow_len if target_yaw > 0 else -arrow_len)
            cv2.arrowedLine(frame, (cx, cy - 100), (arrow_x, cy - 100), Colors.CYAN, 3, tipLength=0.3)

        if abs(target_pitch) > 5:
            arrow_y = cy - 100 + (arrow_len if target_pitch > 0 else -arrow_len)
            cv2.arrowedLine(frame, (cx, cy - 100), (cx, arrow_y), Colors.CYAN, 3, tipLength=0.3)

        # Direction labels
        if target_yaw < -5:
            cv2.putText(frame, "LEFT", (cx - 100, cy - 120), self._font, 0.7, Colors.CYAN, 2)
        elif target_yaw > 5:
            cv2.putText(frame, "RIGHT", (cx + 40, cy - 120), self._font, 0.7, Colors.CYAN, 2)

        if target_pitch > 5:
            cv2.putText(frame, "DOWN", (cx - 35, cy + 130), self._font, 0.7, Colors.CYAN, 2)
        elif target_pitch < -5:
            cv2.putText(frame, "UP", (cx - 20, cy - 140), self._font, 0.7, Colors.CYAN, 2)

    def _draw_pose_radar(self, frame: np.ndarray, state: EnrollmentState,
                         pose: EnrollmentPose, yaw_ok: bool, pitch_ok: bool):
        """Draw mini pose radar indicator."""
        h, w = frame.shape[:2]
        ix, iy, ir = w - 120, 110, 40

        cv2.circle(frame, (ix, iy), ir, Colors.WHITE, 2)

        dx = int(ix + (state.current_yaw / 45) * ir)
        dy = int(iy + (state.current_pitch / 35) * ir)
        color = Colors.GREEN if (yaw_ok and pitch_ok and state.is_stable) else Colors.RED
        cv2.circle(frame, (dx, dy), 6, color, -1)

        tx = int(ix + (pose.target_yaw / 45) * ir)
        ty = int(iy + (pose.target_pitch / 35) * ir)
        cv2.drawMarker(frame, (tx, ty), Colors.CYAN, cv2.MARKER_CROSS, 15, 2)
