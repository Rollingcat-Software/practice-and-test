"""
Fast-Response Liveness Challenge -- v5: Unified Shape Tracing
=============================================================
Command Types
-------------
1. GESTURE:       Finger-count match
2. MOVE_CLOSER:   Depth proxy increase >= 20%
3. MOVE_AWAY:     Depth proxy decrease >= 20%
4. WAVE:          Oscillation filter with frequency check
5. FINGER_TAP:    Touch specific fingers together (simple)
6. HAND_FLIP:     Show palm then back of hand (3D volume change)
7. PEEK_A_BOO:    Cover face with hands then reveal
8. FINGER_TOUCH:  Z-validated fingertip pinch with 10-frame hold
9. SHAPE_TRACE:   Dynamic shape tracing with DTW verification
                  (the ONLY drawing challenge — always shows a
                  visual template with Start/End points)

Legacy commands DRAW_CIRCLE and DRAW_SQUARE have been removed.
All drawing challenges now go through the ShapeTracerSession
module which renders a visual template on screen before the user
is asked to trace anything.

Verification Score: 0-100%.  Each successful challenge adds points.
Access granted only at 100%.  The system picks 5 random challenges
from the library; each is worth 20%.

Anti-spoofing: micro-tremor + brightness checks run every frame
EXCEPT during SHAPE_TRACE challenges, where deliberate stillness
at the Start Point would otherwise cause false positives.
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator, depth_proxy, is_finger_open, Finger, hand_scale, _euclidean, _LM
from motion_analyzer import WaveDetector
from anti_spoof import AntiSpoofAnalyzer, SpoofResult
from finger_touch_detector import FingerTouchDetector, TouchCommand
from shape_tracer import ShapeTracerSession, TracerState, DEFAULT_DRAW_TIME


# ── Command definitions ─────────────────────────────────────────────

class CmdType(Enum):
    GESTURE = auto()
    MOVE_CLOSER = auto()
    MOVE_AWAY = auto()
    WAVE = auto()
    FINGER_TAP = auto()
    HAND_FLIP = auto()
    PEEK_A_BOO = auto()
    FINGER_TOUCH = auto()  # validated touch with Z-axis check + 10-frame hold
    SHAPE_TRACE  = auto()  # dynamic shape tracing with DTW verification
                           # (sole drawing challenge — always renders a visual template)


@dataclass(frozen=True)
class Command:
    name: str
    cmd_type: CmdType
    gesture_reqs: dict[str, int] | None = None
    extra: dict | None = None  # extra params for special commands


# ── Challenge Library (modular, easy to expand) ─────────────────────

_GESTURE_CMDS = [
    Command("SHOW LEFT FIST!",      CmdType.GESTURE, {"Left": 0}),
    Command("SHOW RIGHT FIST!",     CmdType.GESTURE, {"Right": 0}),
    Command("SHOW 2 FINGERS!",      CmdType.GESTURE, {"Right": 2}),
    Command("SHOW 3 FINGERS!",      CmdType.GESTURE, {"Right": 3}),
    Command("SHOW 4 FINGERS!",      CmdType.GESTURE, {"Left": 4}),
    Command("OPEN BOTH HANDS!",     CmdType.GESTURE, {"Left": 5, "Right": 5}),
    Command("LEFT THUMB UP!",       CmdType.GESTURE, {"Left": 1}),
    Command("RIGHT OPEN HAND!",     CmdType.GESTURE, {"Right": 5}),
]

_SPATIAL_CMDS = [
    Command("MOVE HAND CLOSER!",    CmdType.MOVE_CLOSER),
    Command("MOVE HAND AWAY!",      CmdType.MOVE_AWAY),
]

_MOTION_CMDS = [
    Command("WAVE YOUR HAND!",      CmdType.WAVE),
]

_ADVANCED_CMDS = [
    # Finger tap: thumb tip must be close to pinky tip
    Command("TOUCH THUMB TO PINKY!", CmdType.FINGER_TAP,
            extra={"tip_a": _LM["THUMB_TIP"], "tip_b": _LM["PINKY_TIP"], "threshold": 0.3}),
    # Finger tap: thumb tip must be close to index tip
    Command("TOUCH THUMB TO INDEX!", CmdType.FINGER_TAP,
            extra={"tip_a": _LM["THUMB_TIP"], "tip_b": _LM["INDEX_TIP"], "threshold": 0.3}),
    # Hand flip: show palm then flip (detected by z-coordinate shift)
    Command("FLIP YOUR HAND!",      CmdType.HAND_FLIP),
    # Peek-a-boo: both hands must disappear then reappear
    Command("PEEK-A-BOO! (HIDE THEN SHOW)", CmdType.PEEK_A_BOO),
]

# ── Finger Touch commands (Z-validated, 10-frame hold) ───────────────
# These replace the simpler FINGER_TAP for challenges that require the
# full 3-layer protection (bbox normalisation + Z-axis + frame hold).

_TOUCH_CMDS = [
    Command("PINCH: THUMB + INDEX",   CmdType.FINGER_TOUCH,
            extra={"touch_cmd": TouchCommand.THUMB_TO_INDEX}),
    Command("PINCH: THUMB + MIDDLE",  CmdType.FINGER_TOUCH,
            extra={"touch_cmd": TouchCommand.THUMB_TO_MIDDLE}),
    Command("PINCH: THUMB + RING",    CmdType.FINGER_TOUCH,
            extra={"touch_cmd": TouchCommand.THUMB_TO_RING}),
    Command("PINCH: THUMB + PINKY",   CmdType.FINGER_TOUCH,
            extra={"touch_cmd": TouchCommand.THUMB_TO_PINKY}),
    Command("TOUCH BOTH THUMBS!",     CmdType.FINGER_TOUCH,
            extra={"touch_cmd": TouchCommand.DOUBLE_THUMB_TOUCH}),
]

# ── Shape Tracing commands (DTW-verified, depth-gated) ───────────────

_SHAPE_CMDS = [
    Command("TRACE THE SHAPE!",  CmdType.SHAPE_TRACE),
]

ALL_COMMANDS = _GESTURE_CMDS + _SPATIAL_CMDS + _MOTION_CMDS + _ADVANCED_CMDS + _TOUCH_CMDS + _SHAPE_CMDS

# Number of challenges to complete for 100% verification.
CHALLENGES_FOR_FULL_SCORE = 5


class LivenessState(Enum):
    ACTIVE = auto()
    DEBOUNCE = auto()
    SUCCESS = auto()
    FAILED = auto()
    VERIFIED_100 = auto()  # all challenges completed


@dataclass
class LivenessChallenge:
    """Production-grade liveness detector with verification score.

    Parameters
    ----------
    time_limit : float
        Base seconds per challenge.
    debounce_seconds : float
        Hold time for static gesture matches.
    area_change_threshold : float
        Required depth change for spatial commands.
    pause_after_result : float
        Display time for result before next challenge.
    smoothing_window : int
        Passed to GestureValidator.
    process_every_nth : int
        Skip motion math on alternate frames if needed.
    num_challenges : int
        Total challenges for 100% verification.
    """

    time_limit: float = 4.0
    debounce_seconds: float = 0.5
    area_change_threshold: float = 0.20
    pause_after_result: float = 1.5
    smoothing_window: int = 7
    process_every_nth: int = 1
    num_challenges: int = CHALLENGES_FOR_FULL_SCORE

    # -- internal state ---------------------------------------------------
    state: LivenessState = field(init=False, default=LivenessState.ACTIVE)
    current_cmd: Command = field(init=False, default=None)
    score: int = field(init=False, default=0)
    streak: int = field(init=False, default=0)

    # Verification progress: completed / num_challenges
    challenges_completed: int = field(init=False, default=0)
    _challenge_queue: list = field(init=False, default_factory=list)

    _challenge_start: Optional[float] = field(init=False, default=None)
    _debounce_start: Optional[float] = field(init=False, default=None)
    _result_at: Optional[float] = field(init=False, default=None)

    _baseline_area: Optional[float] = field(init=False, default=None)
    _current_area: Optional[float] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)
    _wave: WaveDetector = field(init=False)
    _anti_spoof: AntiSpoofAnalyzer = field(init=False)

    _frame_counter: int = field(init=False, default=0)

    # Hand flip state
    _flip_phase: int = field(init=False, default=0)  # 0=waiting palm, 1=saw palm, waiting back
    _flip_baseline_z: Optional[float] = field(init=False, default=None)

    # Peek-a-boo state
    _peekaboo_phase: int = field(init=False, default=0)  # 0=waiting hide, 1=hidden, waiting show
    _peekaboo_hidden_at: Optional[float] = field(init=False, default=None)

    # Finger touch state (FINGER_TOUCH commands)
    _touch_detector: Optional[FingerTouchDetector] = field(init=False, default=None)

    # Shape tracing state (SHAPE_TRACE commands)
    _shape_tracer: Optional[ShapeTracerSession] = field(init=False, default=None)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._wave = WaveDetector(buffer_size=40, min_swing=0.10,
                                  min_total_displacement=0.20, min_reversals=2)
        self._anti_spoof = AntiSpoofAnalyzer()
        self._build_queue()
        self._pick_next()

    def _build_queue(self):
        """Pick N random unique challenges for the verification session."""
        pool = list(ALL_COMMANDS)
        random.shuffle(pool)
        self._challenge_queue = pool[:self.num_challenges]
        self.challenges_completed = 0

    def _pick_next(self):
        """Pick the next challenge from the queue."""
        if self.challenges_completed >= len(self._challenge_queue):
            self.state = LivenessState.VERIFIED_100
            return

        self.current_cmd = self._challenge_queue[self.challenges_completed]
        self._challenge_start = None
        self._debounce_start = None
        self._result_at = None
        self._baseline_area = None
        self._current_area = None
        self._flip_phase = 0
        self._flip_baseline_z = None
        self._peekaboo_phase = 0
        self._peekaboo_hidden_at = None
        self._frame_counter = 0
        self.state = LivenessState.ACTIVE
        self._validator.clear_buffers()
        self._wave.reset()

        # Create a fresh touch detector for FINGER_TOUCH challenges.
        if self.current_cmd.cmd_type == CmdType.FINGER_TOUCH:
            touch_cmd = self.current_cmd.extra["touch_cmd"]
            self._touch_detector = FingerTouchDetector(
                command=touch_cmd,
                verify_frames=10,
            )
        else:
            self._touch_detector = None

        # Create a fresh shape tracer for SHAPE_TRACE challenges.
        if self.current_cmd.cmd_type == CmdType.SHAPE_TRACE:
            self._shape_tracer = ShapeTracerSession(
                time_limit=DEFAULT_DRAW_TIME,
                dtw_threshold=0.25,
            )
            # Flush the anti-spoof buffer so data from the previous
            # challenge (which may have involved a lot of motion) does
            # not bleed into the next non-tracing challenge.
            self._anti_spoof.reset()
        else:
            self._shape_tracer = None

    @property
    def _effective_time_limit(self) -> float:
        ct = self.current_cmd.cmd_type
        if ct in (CmdType.WAVE, CmdType.HAND_FLIP, CmdType.PEEK_A_BOO):
            return self.time_limit + 2.0
        if ct == CmdType.FINGER_TOUCH:
            return self.time_limit + 1.0
        if ct == CmdType.SHAPE_TRACE:
            # With the Point-to-Point Trigger the user spends an unknown
            # amount of time in IDLE / POSITIONING before TRACING begins.
            # Give the liveness wrapper a very wide safety-net (2 min) so
            # it never fires before the tracer's own TRACING timer runs out.
            return self.time_limit + DEFAULT_DRAW_TIME + 120.0
        return self.time_limit

    @property
    def verification_pct(self) -> float:
        """0-100 verification score."""
        return (self.challenges_completed / self.num_challenges) * 100.0

    @property
    def spoof_result(self) -> SpoofResult:
        return self._anti_spoof.analyze()

    # -- Matching helpers -------------------------------------------------

    def _gesture_matched(self, hands: list[HandResult]) -> bool:
        reqs = self.current_cmd.gesture_reqs
        if reqs is None:
            return False
        hand_map: dict[str, int] = {}
        for h in hands:
            hand_map[h.handedness] = self._validator.count_fingers(h.landmarks, h.handedness)
        return all(hand_map.get(label) == count for label, count in reqs.items())

    def _get_depth(self, hands: list[HandResult]) -> Optional[float]:
        return depth_proxy(hands[0].landmarks) if hands else None

    def _spatial_matched(self, hands: list[HandResult]) -> bool:
        area = self._get_depth(hands)
        if area is None or self._baseline_area is None:
            return False
        self._current_area = area
        ratio = area / self._baseline_area
        if self.current_cmd.cmd_type == CmdType.MOVE_CLOSER:
            return ratio >= (1.0 + self.area_change_threshold)
        elif self.current_cmd.cmd_type == CmdType.MOVE_AWAY:
            return ratio <= (1.0 - self.area_change_threshold)
        return False

    def _finger_tap_matched(self, hands: list[HandResult]) -> bool:
        """Check if two specific landmarks are close together."""
        if not hands or not self.current_cmd.extra:
            return False
        lm = hands[0].landmarks
        hs = hand_scale(lm)
        if hs < 1e-9:
            return False
        tip_a = lm[self.current_cmd.extra["tip_a"]]
        tip_b = lm[self.current_cmd.extra["tip_b"]]
        dist = _euclidean(tip_a, tip_b) / hs
        return dist < self.current_cmd.extra["threshold"]

    def _hand_flip_check(self, hands: list[HandResult]) -> bool:
        """Detect a hand flip by tracking the average Z of fingertips.

        Phase 0: Record baseline Z (palm facing camera = low avg Z).
        Phase 1: Wait for avg Z to shift significantly (back of hand).
        """
        if not hands:
            return False
        lm = hands[0].landmarks
        avg_z = sum(lm[i].z for i in [4, 8, 12, 16, 20]) / 5.0

        if self._flip_phase == 0:
            if self._flip_baseline_z is None:
                self._flip_baseline_z = avg_z
            # Wait for significant Z change (back of hand has different z)
            if self._flip_baseline_z is not None:
                delta = abs(avg_z - self._flip_baseline_z)
                if delta > 0.03:
                    self._flip_phase = 1
                    return True
        return False

    def _peekaboo_check(self, hands: list[HandResult]) -> bool:
        """Phase 0: Wait for hands to disappear (user covers face).
        Phase 1: Wait for hands to reappear."""
        now = time.monotonic()
        if self._peekaboo_phase == 0:
            if not hands:
                self._peekaboo_phase = 1
                self._peekaboo_hidden_at = now
        elif self._peekaboo_phase == 1:
            # Must stay hidden for at least 0.5s
            if not hands:
                return False
            if self._peekaboo_hidden_at and now - self._peekaboo_hidden_at > 0.5:
                return True  # Hands reappeared after hiding
        return False

    def _index_is_open(self, hands: list[HandResult]) -> bool:
        if not hands:
            return False
        return is_finger_open(hands[0].landmarks, hands[0].handedness, Finger.INDEX)

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> LivenessState:
        now = time.monotonic()
        self._frame_counter += 1

        # Anti-spoof: feed data every frame, EXCEPT during Shape Tracing.
        #
        # Shape Tracing requires the user to hold their finger perfectly still
        # on the Start Point for 0.5 s (POSITIONING phase) before recording
        # begins.  That deliberate stillness is indistinguishable from a
        # spoofed static image to the MicroTremorDetector, so feeding
        # anti-spoof data during this challenge produces guaranteed false
        # positives.  We skip the feed entirely for SHAPE_TRACE challenges
        # and reset the anti-spoof state so the next non-tracing challenge
        # starts with a clean buffer.
        _is_shape_trace = (self.current_cmd is not None and
                           self.current_cmd.cmd_type == CmdType.SHAPE_TRACE)
        if hands and not _is_shape_trace:
            wrist = hands[0].landmarks[0]
            # Use hand_scale as brightness proxy -- changes naturally with
            # hand movement/distance but stays constant for a static image.
            hs = hand_scale(hands[0].landmarks)
            self._anti_spoof.feed(wrist.x, wrist.y, hs)

        if self.state == LivenessState.VERIFIED_100:
            return self.state

        # After result, pause then advance.
        if self.state in (LivenessState.SUCCESS, LivenessState.FAILED):
            if self._result_at and now - self._result_at >= self.pause_after_result:
                if self.state == LivenessState.SUCCESS:
                    self.challenges_completed += 1
                self._pick_next()
            return self.state

        # Timer starts on first hand detection (or immediately for peek-a-boo).
        if self._challenge_start is None:
            if self.current_cmd.cmd_type == CmdType.PEEK_A_BOO:
                self._challenge_start = now  # start immediately
            elif not hands:
                return self.state
            else:
                self._challenge_start = now
                if self.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
                    self._baseline_area = self._get_depth(hands)

        # Timeout.
        if now - self._challenge_start >= self._effective_time_limit:
            self.state = LivenessState.FAILED
            self._result_at = now
            self.streak = 0
            return self.state

        should_process = (self._frame_counter % self.process_every_nth == 0)

        # -- DEBOUNCE ------------------------------------------------------
        if self.state == LivenessState.DEBOUNCE:
            if should_process and self._check_static_match(hands):
                if now - self._debounce_start >= self.debounce_seconds:
                    self._succeed(now)
            else:
                if should_process:
                    self.state = LivenessState.ACTIVE
                    self._debounce_start = None
            return self.state

        # -- ACTIVE --------------------------------------------------------
        ct = self.current_cmd.cmd_type

        # Spatial tracking.
        if ct in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            area = self._get_depth(hands)
            if area is not None:
                self._current_area = area
                if self._baseline_area is None:
                    self._baseline_area = area

        # Wave.
        if ct == CmdType.WAVE and hands:
            self._wave.push(hands[0].landmarks[0].x)
            if should_process and self._wave.is_waving():
                self._succeed(now)
                return self.state

        # Finger tap (simple, immediate).
        if ct == CmdType.FINGER_TAP:
            if should_process and self._finger_tap_matched(hands):
                self._succeed(now)
                return self.state

        # Finger touch (Z-validated, 10-frame hold).
        if ct == CmdType.FINGER_TOUCH and self._touch_detector is not None:
            if self._touch_detector.update(hands):
                self._succeed(now)
                return self.state

        # Shape tracing (DTW-verified, depth-gated).
        if ct == CmdType.SHAPE_TRACE and self._shape_tracer is not None:
            tracer_state = self._shape_tracer.update(hands)
            if tracer_state == TracerState.VERIFIED:
                self._succeed(now)
                return self.state
            if tracer_state == TracerState.FAILED:
                self.state       = LivenessState.FAILED
                self._result_at  = now
                self.streak      = 0
                return self.state
            return self.state   # tracer managing its own sub-states; skip other logic

        # Hand flip.
        if ct == CmdType.HAND_FLIP:
            if self._hand_flip_check(hands):
                self._succeed(now)
                return self.state

        # Peek-a-boo.
        if ct == CmdType.PEEK_A_BOO:
            if self._peekaboo_check(hands):
                self._succeed(now)
                return self.state

        # Static commands (gesture/spatial).
        if ct in (CmdType.GESTURE, CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            if should_process and self._check_static_match(hands):
                self.state = LivenessState.DEBOUNCE
                self._debounce_start = now

        return self.state

    def _check_static_match(self, hands):
        ct = self.current_cmd.cmd_type
        if ct == CmdType.GESTURE:
            return self._gesture_matched(hands)
        elif ct in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            return self._spatial_matched(hands)
        return False

    def _succeed(self, now: float):
        self.state = LivenessState.SUCCESS
        self._result_at = now
        self.score += 1
        self.streak += 1

    # -- Display helpers --------------------------------------------------

    @property
    def time_remaining(self) -> float:
        if self._challenge_start is None:
            return self._effective_time_limit
        return max(0.0, self._effective_time_limit - (time.monotonic() - self._challenge_start))

    @property
    def debounce_progress(self) -> float:
        if self._debounce_start is None:
            return 0.0
        return min((time.monotonic() - self._debounce_start) / self.debounce_seconds, 1.0)

    @property
    def area_change_pct(self) -> Optional[float]:
        if self._baseline_area is None or self._current_area is None:
            return None
        return ((self._current_area / self._baseline_area) - 1.0) * 100.0

    @property
    def wave_reversals(self) -> int:
        return self._wave.reversal_count

    @property
    def command_label(self) -> str:
        ct = self.current_cmd.cmd_type
        if ct == CmdType.GESTURE:
            return f"QUICK: {self.current_cmd.name}"
        return f"ACTION: {self.current_cmd.name}"

    @property
    def display_text(self) -> str:
        if self.state == LivenessState.VERIFIED_100:
            return "ACCESS GRANTED - 100% VERIFIED"
        if self.state == LivenessState.SUCCESS:
            return "VERIFIED!"
        if self.state == LivenessState.FAILED:
            return "VERIFICATION FAILED"
        return self.command_label

    @property
    def status_label(self) -> str:
        if self.state == LivenessState.VERIFIED_100:
            return "Status: All Challenges Complete"
        if self.state == LivenessState.SUCCESS:
            return "Status: Challenge Passed"
        if self.state == LivenessState.FAILED:
            return "Status: Time's Up!"
        if self.state == LivenessState.DEBOUNCE:
            return "Status: Confirming..."
        if self._challenge_start is None:
            return "Status: Show your hand to start"
        return "Status: Respond NOW!"

    @property
    def is_flash_red(self) -> bool:
        if self.state != LivenessState.FAILED or self._result_at is None:
            return False
        return (time.monotonic() - self._result_at) < 0.6

    @property
    def is_wave_cmd(self) -> bool:
        return self.current_cmd.cmd_type == CmdType.WAVE

    @property
    def is_touch_cmd(self) -> bool:
        return self.current_cmd.cmd_type == CmdType.FINGER_TOUCH

    @property
    def touch_frame_count(self) -> int:
        """Current consecutive-frame count toward the 10-frame touch hold."""
        if self._touch_detector is None:
            return 0
        return self._touch_detector.consecutive_frames

    @property
    def touch_frame_progress(self) -> float:
        """0.0-1.0 fraction of the 10-frame requirement met (for progress bars)."""
        if self._touch_detector is None:
            return 0.0
        return self._touch_detector.progress

    # -- Shape tracer display helpers ------------------------------------

    @property
    def is_shape_trace_cmd(self) -> bool:
        return self.current_cmd.cmd_type == CmdType.SHAPE_TRACE

    @property
    def shape_tracer(self) -> Optional[ShapeTracerSession]:
        """Direct access to the active ShapeTracerSession for the HUD."""
        return self._shape_tracer

    @property
    def shape_trace_label(self) -> str:
        """Readable command label including the shape name chosen for this round."""
        if self._shape_tracer is None:
            return "TRACE THE SHAPE!"
        return f"TRACE THE {self._shape_tracer.template.label}!"

    @property
    def challenge_progress_text(self) -> str:
        return f"Challenge {self.challenges_completed + 1}/{self.num_challenges}"

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    def reset(self) -> None:
        self.score = 0
        self.streak = 0
        self._anti_spoof.reset()
        self._build_queue()
        self._pick_next()
