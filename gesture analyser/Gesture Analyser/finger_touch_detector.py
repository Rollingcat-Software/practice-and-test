"""
Finger Touch Detector -- Liveness Challenge Module
===================================================
Detects specific fingertip pinch gestures with four layers of protection:

  1. Fist gate (NEW)
       Immediately rejects the frame if the active hand is in a full fist.
       In a fist all fingertips converge near the palm, so raw distance
       checks between e.g. thumb-tip and index-tip can pass falsely.
       Uses the proven ``is_fist()`` geometry from gesture_validator.

  2. Bystander finger check (NEW)
       For single-hand commands, at least one finger that is NOT part of
       the touch pair must be sufficiently open.  This distinguishes a
       genuine pinch (bystanders extended) from a semi-clenched fist
       (all fingers curled together).

  3. Normalized Euclidean distance
       Distance between the two landmarks is divided by the hand bounding-
       box diagonal, making the threshold invariant to hand size and camera
       distance.

  4. Z-axis depth validation (tightened)
       Rejects cases where one finger passes *behind* the other in 3D space.
       This blocks the common 2D spoofing attack where fingers look adjacent
       in the image plane but are at very different depths.  The check is
       applied before the distance gate so it doubles as an early exit for
       fist-like poses where fingertip Z values diverge.

  5. 10-frame consecutive hold requirement
       A fleeting or accidental brush does not satisfy the challenge.
       The touch must be maintained for ``verify_frames`` (default 10)
       consecutive video frames before the gesture is marked 'Verified'.

Supported commands
------------------
  THUMB_TO_INDEX    -- Landmark 4 (thumb tip) & 8  (index tip), same hand
  THUMB_TO_MIDDLE   -- Landmark 4 (thumb tip) & 12 (middle tip), same hand
  THUMB_TO_RING     -- Landmark 4 (thumb tip) & 16 (ring tip),   same hand
  THUMB_TO_PINKY    -- Landmark 4 (thumb tip) & 20 (pinky tip),  same hand
  DOUBLE_THUMB_TOUCH-- Landmark 4 Left hand  & 4  Right hand
"""

import math
from enum import Enum, auto

from gesture_validator import _euclidean, _LM, finger_ratio, is_fist, Finger
from app_config import cfg


# ── Command catalogue ────────────────────────────────────────────────

class TouchCommand(Enum):
    THUMB_TO_INDEX     = auto()   # IDs 4 & 8  (same hand)
    THUMB_TO_MIDDLE    = auto()   # IDs 4 & 12 (same hand)
    THUMB_TO_RING      = auto()   # IDs 4 & 16 (same hand)
    THUMB_TO_PINKY     = auto()   # IDs 4 & 20 (same hand)
    DOUBLE_THUMB_TOUCH = auto()   # ID 4 Left hand & ID 4 Right hand


# Single-hand landmark pairs (tip_a_id, tip_b_id)
_SINGLE_HAND_PAIRS: dict[TouchCommand, tuple[int, int]] = {
    TouchCommand.THUMB_TO_INDEX:  (_LM["THUMB_TIP"], _LM["INDEX_TIP"]),   # 4, 8
    TouchCommand.THUMB_TO_MIDDLE: (_LM["THUMB_TIP"], _LM["MIDDLE_TIP"]),  # 4, 12
    TouchCommand.THUMB_TO_RING:   (_LM["THUMB_TIP"], _LM["RING_TIP"]),    # 4, 16
    TouchCommand.THUMB_TO_PINKY:  (_LM["THUMB_TIP"], _LM["PINKY_TIP"]),   # 4, 20
}

# Fingers that must NOT all be curled during a single-hand touch command.
# At least one bystander must be open to distinguish pinch from fist.
_BYSTANDER_FINGERS: dict[TouchCommand, list[Finger]] = {
    TouchCommand.THUMB_TO_INDEX:  [Finger.MIDDLE, Finger.RING, Finger.PINKY],
    TouchCommand.THUMB_TO_MIDDLE: [Finger.INDEX,  Finger.RING, Finger.PINKY],
    TouchCommand.THUMB_TO_RING:   [Finger.INDEX,  Finger.MIDDLE, Finger.PINKY],
    TouchCommand.THUMB_TO_PINKY:  [Finger.INDEX,  Finger.MIDDLE, Finger.RING],
}

# Default detector parameters — sourced from the central config so they
# update automatically when config.yaml is changed.
_DEFAULT_VERIFY_FRAMES   = cfg.touch.verify_frames    # frames of continuous touch required
_DEFAULT_TOUCH_THRESHOLD = cfg.touch.threshold        # fraction of bbox diagonal
_DEFAULT_Z_MAX_DIFF      = cfg.touch.z_max_diff       # maximum allowed Z-depth gap between tips

# Bystander openness gate — see config.yaml [touch] section.
_BYSTANDER_OPEN_THRESH = cfg.touch.bystander_open_thresh
_BYSTANDER_MIN_OPEN    = cfg.touch.bystander_min_open


# ── Geometry helpers ─────────────────────────────────────────────────

def _bbox_scale(landmarks) -> float:
    """Diagonal of the hand bounding box in normalised image coordinates.

    Grows when the hand is large (close to camera) and shrinks when small
    (far away), so dividing touch distances by this value makes the
    ``touch_threshold`` invariant to scale.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    diag = math.sqrt(w * w + h * h)
    return diag if diag > 1e-9 else 1e-9


def _z_valid(lm_a, lm_b, z_max_diff: float) -> bool:
    """Return True when both landmarks share a similar depth plane.

    MediaPipe Z values are positive further from the camera.  A large
    difference means one fingertip is behind the other — they only *look*
    close in the 2D image but are not actually touching.
    """
    return abs(lm_a.z - lm_b.z) <= z_max_diff


def _bystanders_clear(landmarks, command: TouchCommand) -> bool:
    """Return True when at least ``_BYSTANDER_MIN_OPEN`` non-touching fingers
    are sufficiently extended to rule out a fist or semi-fist posture.

    Logic
    -----
    In a genuine pinch the bystander fingers (those not involved in the
    touch pair) stay relatively open because the hand must separate the
    touching fingertips from the rest.  In a fist every finger curls, so
    all bystander ``finger_ratio`` values are at or below zero.

    We only require ``_BYSTANDER_MIN_OPEN`` (default 1) bystander to pass
    the threshold, which accommodates people who naturally curl one or two
    fingers when pinching.
    """
    bystanders = _BYSTANDER_FINGERS.get(command, [])
    open_count = sum(
        1 for f in bystanders
        if finger_ratio(landmarks, f) > _BYSTANDER_OPEN_THRESH
    )
    return open_count >= _BYSTANDER_MIN_OPEN


# ── Main detector class ──────────────────────────────────────────────

class FingerTouchDetector:
    """Per-frame detector for fingertip touch liveness challenges.

    Instantiate once per challenge, feed every video frame via ``update()``,
    and read ``is_verified`` (or use the bool return value) to know when the
    touch has been confirmed.

    Parameters
    ----------
    command : TouchCommand
        Which fingertip pair to watch.
    verify_frames : int
        Consecutive frames the touch must be held to become Verified.
        Default 10 (~0.33 s at 30 fps).
    touch_threshold : float
        Maximum normalised distance (as a fraction of the hand bounding-box
        diagonal) for the tips to count as touching.
    z_max_diff : float
        Maximum allowed Z-coordinate gap between the two landmark tips.
        Larger values tolerate more depth offset; smaller values are stricter
        about rejecting 2-D spoofing.
    """

    def __init__(
        self,
        command: TouchCommand,
        verify_frames: int   = _DEFAULT_VERIFY_FRAMES,
        touch_threshold: float = _DEFAULT_TOUCH_THRESHOLD,
        z_max_diff: float    = _DEFAULT_Z_MAX_DIFF,
    ):
        self.command         = command
        self.verify_frames   = verify_frames
        self.touch_threshold = touch_threshold
        self.z_max_diff      = z_max_diff
        self._consecutive    = 0
        self._verified       = False

    # -- Internal per-frame geometry checks ------------------------------

    def _check_single_hand(self, landmarks) -> bool:
        """Evaluate touch for same-hand landmark pairs.

        Validation pipeline
        -------------------
        Gate 1  — Fist detection
            Reject immediately if the hand geometry matches a closed fist.
            In a fist all finger_ratio scores are below their open threshold,
            so ``is_fist()`` returns True and we bail out before any distance
            arithmetic.  This is the primary fix for fist false-positives.

        Gate 2  — Z-axis depth
            The two touching tips must lie in the same depth plane.  A fist
            where fingertips pass over each other at different depths often
            survives the 2-D distance check but fails here.

        Gate 3  — Normalised Euclidean distance
            Classic proximity check, scale-invariant via bbox diagonal.

        Gate 4  — Bystander finger check
            At least one finger that is NOT in the touch pair must be
            measurably open.  This catches semi-fist postures that slip past
            Gate 1 (e.g. four fingers curled, thumb tucked slightly less).
        """
        # Gate 1: reject full fists outright
        if is_fist(landmarks):
            return False

        tip_a_id, tip_b_id = _SINGLE_HAND_PAIRS[self.command]
        lm_a = landmarks[tip_a_id]
        lm_b = landmarks[tip_b_id]

        # Gate 2: Z-axis anti-spoof (applied before distance to short-circuit)
        if not _z_valid(lm_a, lm_b, self.z_max_diff):
            return False

        # Gate 3: Normalised Euclidean distance
        scale = _bbox_scale(landmarks)
        dist  = _euclidean(lm_a, lm_b) / scale
        if dist >= self.touch_threshold:
            return False

        # Gate 4: at least one bystander finger must be open
        if not _bystanders_clear(landmarks, self.command):
            return False

        return True

    def _check_double_thumb(self, hands) -> bool:
        """Evaluate DOUBLE_THUMB_TOUCH across left and right hands.

        In addition to the existing distance + Z checks, both hands must
        individually pass ``is_fist()`` negation.  This rejects the case
        where a person holds two fists together with their thumbs tucked,
        which could otherwise pass the thumb-tip proximity test.
        """
        left_lm = right_lm = None
        for h in hands:
            if h.handedness == "Left":
                left_lm = h.landmarks
            elif h.handedness == "Right":
                right_lm = h.landmarks

        if left_lm is None or right_lm is None:
            return False  # both hands must be present

        # Gate 1: neither hand may be a fist
        if is_fist(left_lm) or is_fist(right_lm):
            return False

        lm_a = left_lm[_LM["THUMB_TIP"]]   # left  thumb tip (ID 4)
        lm_b = right_lm[_LM["THUMB_TIP"]]  # right thumb tip (ID 4)

        # Gate 2: Z-axis anti-spoof gate
        if not _z_valid(lm_a, lm_b, self.z_max_diff):
            return False

        # Gate 3: normalise against the *average* of both hand bounding boxes
        # so the threshold does not depend on inter-hand separation.
        scale = (_bbox_scale(left_lm) + _bbox_scale(right_lm)) / 2.0
        dist  = _euclidean(lm_a, lm_b) / scale
        return dist < self.touch_threshold

    def _is_touching_this_frame(self, hands) -> bool:
        """Raw, single-frame touch check (no temporal component)."""
        if not hands:
            return False
        if self.command == TouchCommand.DOUBLE_THUMB_TOUCH:
            return self._check_double_thumb(hands)
        return self._check_single_hand(hands[0].landmarks)

    # -- Public API ------------------------------------------------------

    def update(self, hands) -> bool:
        """Feed one video frame and check whether the touch is now Verified.

        Returns True on the frame that verification is first achieved, and
        on all subsequent frames (the verified state is sticky until
        ``reset()`` is called).

        Layer 5 -- consecutive-frame persistence:
          The touch must be held for ``verify_frames`` unbroken frames.
          Any frame without a valid touch resets the counter to zero.
        """
        if self._verified:
            return True

        if self._is_touching_this_frame(hands):
            self._consecutive += 1
            if self._consecutive >= self.verify_frames:
                self._verified = True
                return True
        else:
            self._consecutive = 0

        return False

    # -- Observable state ------------------------------------------------

    @property
    def consecutive_frames(self) -> int:
        """Current unbroken frame count toward verification (0 … verify_frames)."""
        return self._consecutive

    @property
    def progress(self) -> float:
        """Fraction of verify_frames reached (0.0 – 1.0), for progress bars."""
        return min(self._consecutive / self.verify_frames, 1.0)

    @property
    def is_verified(self) -> bool:
        """True once the touch has been held for the required number of frames."""
        return self._verified

    def reset(self) -> None:
        """Return the detector to its initial state."""
        self._consecutive = 0
        self._verified    = False
