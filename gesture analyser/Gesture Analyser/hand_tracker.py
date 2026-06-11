"""
Hand Tracking Core Module -- v6: Equal Left/Right Priority
===========================================================
Fixes left-hand detection bias with:

1. **Asymmetric confidence**: detection=0.7 (find hands reliably),
   tracking=0.5 (hold onto them once found -- critical for left hand).
2. **CLAHE contrast enhancement**: Equalises shadows before detection
   so the second hand (often in lower light) gets clearer landmarks.
3. **Wrist-position handedness**: Ignores MediaPipe's label entirely
   and assigns Left/Right purely by wrist screen position.
4. **Persistent hand state**: If a hand disappears for <= 2 frames,
   we keep its last-known state (landmarks + label) so downstream
   logic doesn't flicker or reset timers on brief tracking drops.
5. **Debug bounding boxes**: RED box = Left, BLUE box = Right.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from app_config import cfg

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions


@dataclass
class HandResult:
    """Lightweight container for one detected hand."""
    landmarks: list          # 21 NormalizedLandmark objects
    handedness: str          # "Left" or "Right" (user's perspective)
    frame_rgb: np.ndarray    # the RGB frame used for detection


# ── CLAHE contrast enhancer (created once, reused) ──────────────────
# Parameters sourced from config so operators can tune contrast enhancement
# for different lighting environments without touching source code.
_clahe = cv2.createCLAHE(
    clipLimit=cfg.tracker.clahe_clip_limit,
    tileGridSize=(cfg.tracker.clahe_tile_size, cfg.tracker.clahe_tile_size),
)


def _enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to make shadowed hands clearer.

    Converts to LAB colour space, enhances the L (lightness) channel,
    then converts back.  This brightens dark areas without blowing out
    bright ones -- helps the model detect a second hand in shadow.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Persistent hand slot (survives brief tracking drops) ────────────

class _SmoothedLandmark:
    """Lightweight EMA-smoothed landmark (x, y, z)."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def _smooth_landmarks(prev: Optional[list], curr: list, alpha: float = 0.7) -> list:
    """Weighted moving average: alpha * current + (1-alpha) * previous.

    Eliminates jitter while staying responsive (alpha=0.7 means 70%
    weight on the current frame).
    """
    if prev is None or len(prev) != len(curr):
        # First frame or size mismatch -- return current as-is but wrapped.
        return [_SmoothedLandmark(lm.x, lm.y, lm.z) for lm in curr]

    smoothed = []
    for p, c in zip(prev, curr):
        smoothed.append(_SmoothedLandmark(
            x=alpha * c.x + (1.0 - alpha) * p.x,
            y=alpha * c.y + (1.0 - alpha) * p.y,
            z=alpha * c.z + (1.0 - alpha) * p.z,
        ))
    return smoothed


class _HandSlot:
    """Holds last-known state for one hand label ("Left" or "Right").

    If the hand disappears for up to ``max_lost`` frames, the slot
    keeps returning its last landmarks.  After that it goes empty.

    Landmarks are EMA-smoothed (0.7 * current + 0.3 * previous) to
    reduce jitter without adding perceptible latency.
    """

    def __init__(self, label: str, max_lost: int = 2):
        self.label = label
        self.max_lost = max_lost
        self._landmarks: Optional[list] = None
        self._frame_rgb: Optional[np.ndarray] = None
        self._lost_count: int = 999
        self._no_hand_streak: int = 0  # consecutive frames with no hand

    def update(self, landmarks: Optional[list], frame_rgb: np.ndarray):
        if landmarks is not None:
            self._landmarks = _smooth_landmarks(self._landmarks, landmarks)
            self._frame_rgb = frame_rgb
            self._lost_count = 0
            self._no_hand_streak = 0
        else:
            self._lost_count += 1
            self._no_hand_streak += 1
            # After 10 frames with no hand, clear state to free memory.
            if self._no_hand_streak >= 10:
                self._landmarks = None

    @property
    def is_valid(self) -> bool:
        return self._landmarks is not None and self._lost_count <= self.max_lost

    def get(self) -> Optional[HandResult]:
        if not self.is_valid:
            return None
        return HandResult(
            landmarks=self._landmarks,
            handedness=self.label,
            frame_rgb=self._frame_rgb,
        )

    def reset(self):
        self._landmarks = None
        self._lost_count = 999


class HandTracker:
    """Dual-hand tracker with equal left/right priority.

    Parameters
    ----------
    max_num_hands : int
        Always 2.
    min_detection_confidence : float
        0.7 -- balanced: finds hands reliably without too many ghosts.
    min_tracking_confidence : float
        0.5 -- lower so the model holds onto a hand once found,
        preventing the left-hand "latency" issue.
    max_lost_frames : int
        How many frames a hand can disappear before we drop it.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_lost_frames: int = 2,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        # Persistent slots for each hand.
        self._left = _HandSlot("Left", max_lost_frames)
        self._right = _HandSlot("Right", max_lost_frames)

    def process(self, bgr_frame: np.ndarray) -> list[HandResult]:
        """Detect hands in a **pre-flipped** (mirrored) BGR frame.

        Pipeline:
        1. CLAHE contrast enhancement.
        2. BGR -> RGB conversion.
        3. MediaPipe detection (static_image_mode=False via VIDEO mode).
        4. Wrist-position handedness assignment (ignores model labels).
        5. Persistent slot update with lost-frame tolerance.
        """
        # -- Step 1: Contrast enhancement ---------------------------------
        enhanced = _enhance_frame(bgr_frame)

        # -- Step 2: Convert to RGB after flip (frame is already flipped) -
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # -- Step 3: Detection (VIDEO mode = static_image_mode=False) -----
        # Use the real wall-clock millisecond timestamp so MediaPipe's VIDEO
        # mode receives accurate inter-frame deltas regardless of the actual
        # camera frame rate.  The previous `+= 33` hardcoded 30 fps and drifted
        # immediately on 60 fps cameras, degrading optical-flow quality.
        result = self._landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        # -- Step 4: Assign handedness by wrist screen position -----------
        # We completely ignore MediaPipe's handedness classification.
        # Wrist.x < 0.5 = left side of mirrored frame = user's Left hand.
        left_lm = None
        right_lm = None

        if result.hand_landmarks:
            for lm_list in result.hand_landmarks:
                wrist_x = lm_list[0].x
                if wrist_x < 0.5:
                    # Left side -> user's Left hand.
                    # If we already have a left candidate, keep the one
                    # with wrist further to the left (more confident).
                    if left_lm is None or lm_list[0].x < left_lm[0].x:
                        left_lm = lm_list
                else:
                    if right_lm is None or lm_list[0].x > right_lm[0].x:
                        right_lm = lm_list

        # -- Step 5: Update persistent slots ------------------------------
        self._left.update(left_lm, rgb)
        self._right.update(right_lm, rgb)

        # -- Build output list from valid slots ---------------------------
        hands: list[HandResult] = []
        lh = self._left.get()
        if lh:
            hands.append(lh)
        rh = self._right.get()
        if rh:
            hands.append(rh)

        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Drawing helpers ─────────────────────────────────────────────────
_HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Landmark colours per hand.
_HAND_COLORS = {
    "Left":  {"line": (220, 180, 0), "dot": (255, 100, 0)},
    "Right": {"line": (180, 0, 180), "dot": (180, 105, 255)},
}
_DEFAULT_COLOR = {"line": (0, 220, 0), "dot": (0, 0, 255)}

# Debug bounding-box colours: RED = Left, BLUE = Right.
_BBOX_COLORS = {"Left": (0, 0, 255), "Right": (255, 0, 0)}


def draw_landmarks(bgr_frame: np.ndarray, hand: HandResult) -> None:
    """Draw landmarks, connections, label, and a debug bounding box."""
    h, w, _ = bgr_frame.shape
    landmarks = hand.landmarks
    colors = _HAND_COLORS.get(hand.handedness, _DEFAULT_COLOR)
    bbox_color = _BBOX_COLORS.get(hand.handedness, (255, 255, 255))

    # -- Connections + dots -----------------------------------------------
    for conn in _HAND_CONNECTIONS:
        s, e = landmarks[conn.start], landmarks[conn.end]
        cv2.line(
            bgr_frame,
            (int(s.x * w), int(s.y * h)),
            (int(e.x * w), int(e.y * h)),
            colors["line"], 2,
        )

    for lm in landmarks:
        cv2.circle(bgr_frame, (int(lm.x * w), int(lm.y * h)), 5, colors["dot"], -1)

    # -- Handedness label near wrist --------------------------------------
    wrist = landmarks[0]
    cv2.putText(
        bgr_frame, hand.handedness,
        (int(wrist.x * w) - 20, int(wrist.y * h) + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["line"], 2,
    )

    # -- Debug: finger-tip circles + tip-to-wrist ratio labels ---------------
    from gesture_validator import finger_debug_info, is_fist
    debug = finger_debug_info(landmarks, hand.handedness)
    fist_detected = is_fist(landmarks)
    for finger, (tip_id, is_open, ratio, tw_ratio) in debug.items():
        tip = landmarks[tip_id]
        cx, cy = int(tip.x * w), int(tip.y * h)
        tip_color = (0, 220, 0) if is_open else (0, 0, 220)  # green / red
        cv2.circle(bgr_frame, (cx, cy), 10, tip_color, 2)
        # Show tip-to-wrist ratio near each fingertip for threshold tuning.
        cv2.putText(bgr_frame, f"{tw_ratio:.2f}", (cx + 12, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, tip_color, 1)

    # Fist indicator near wrist.
    if fist_detected:
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(bgr_frame, "FIST", (wx - 20, wy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    # -- Debug bounding box (RED=Left, BLUE=Right) ------------------------
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 15
    x_min, x_max = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y_min, y_max = max(0, min(ys) - pad), min(h, max(ys) + pad)
    cv2.rectangle(bgr_frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)
    cv2.putText(
        bgr_frame, hand.handedness,
        (x_min, y_min - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2,
    )
