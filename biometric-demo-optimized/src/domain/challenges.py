"""Biometric puzzle challenge domain models."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any
from collections import deque
import time


class ChallengeType(Enum):
    """All supported liveness challenge types."""
    BLINK = auto()
    CLOSE_LEFT = auto()
    CLOSE_RIGHT = auto()
    SMILE = auto()
    OPEN_MOUTH = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    LOOK_UP = auto()
    LOOK_DOWN = auto()
    RAISE_BOTH_BROWS = auto()
    RAISE_LEFT_BROW = auto()
    RAISE_RIGHT_BROW = auto()
    NOD = auto()
    SHAKE_HEAD = auto()


@dataclass
class Challenge:
    """Challenge definition."""
    type: ChallengeType
    display_name: str
    icon: str
    key: str

    @staticmethod
    def get_all() -> Dict[ChallengeType, 'Challenge']:
        return {
            ChallengeType.BLINK: Challenge(ChallengeType.BLINK, 'Close Both Eyes', '😌', 'blink'),
            ChallengeType.CLOSE_LEFT: Challenge(ChallengeType.CLOSE_LEFT, 'Close YOUR Left Eye', '😉', 'close_left'),
            ChallengeType.CLOSE_RIGHT: Challenge(ChallengeType.CLOSE_RIGHT, 'Close YOUR Right Eye', '😉', 'close_right'),
            ChallengeType.SMILE: Challenge(ChallengeType.SMILE, 'Smile Wide (Show Teeth)', '😁', 'smile'),
            ChallengeType.OPEN_MOUTH: Challenge(ChallengeType.OPEN_MOUTH, 'Open Mouth Wide', '😮', 'open_mouth'),
            ChallengeType.TURN_LEFT: Challenge(ChallengeType.TURN_LEFT, 'Turn Head Left', '👈', 'turn_left'),
            ChallengeType.TURN_RIGHT: Challenge(ChallengeType.TURN_RIGHT, 'Turn Head Right', '👉', 'turn_right'),
            ChallengeType.LOOK_UP: Challenge(ChallengeType.LOOK_UP, 'Look Up (Chin Up)', '👆', 'look_up'),
            ChallengeType.LOOK_DOWN: Challenge(ChallengeType.LOOK_DOWN, 'Look Down (Chin Down)', '👇', 'look_down'),
            ChallengeType.RAISE_BOTH_BROWS: Challenge(ChallengeType.RAISE_BOTH_BROWS, 'Raise Both Eyebrows', '🤨', 'raise_both'),
            ChallengeType.RAISE_LEFT_BROW: Challenge(ChallengeType.RAISE_LEFT_BROW, 'Raise YOUR Left Eyebrow', '🤔', 'raise_left'),
            ChallengeType.RAISE_RIGHT_BROW: Challenge(ChallengeType.RAISE_RIGHT_BROW, 'Raise YOUR Right Eyebrow', '🧐', 'raise_right'),
            ChallengeType.NOD: Challenge(ChallengeType.NOD, 'Nod Your Head', '↕️', 'nod'),
            ChallengeType.SHAKE_HEAD: Challenge(ChallengeType.SHAKE_HEAD, 'Shake Your Head', '↔️', 'shake_head'),
        }

    @staticmethod
    def get_simple_pool() -> List[ChallengeType]:
        """Get challenge types suitable for random selection."""
        return [
            ChallengeType.BLINK, ChallengeType.CLOSE_LEFT, ChallengeType.CLOSE_RIGHT,
            ChallengeType.SMILE, ChallengeType.OPEN_MOUTH,
            ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT,
            ChallengeType.LOOK_UP, ChallengeType.LOOK_DOWN,
            ChallengeType.RAISE_BOTH_BROWS, ChallengeType.RAISE_LEFT_BROW, ChallengeType.RAISE_RIGHT_BROW,
        ]


@dataclass
class ChallengeResult:
    """Result of checking a challenge."""
    detected: bool = False
    progress: float = 0.0
    message: str = ''
    completed: bool = False


@dataclass
class PuzzleState:
    """State of the biometric puzzle."""
    challenges: List[ChallengeType] = field(default_factory=list)
    current_idx: int = 0
    is_active: bool = False
    is_complete: bool = False
    passed: bool = False

    # Detection state
    hold_start: float = 0.0
    hold_duration: float = 0.6  # Seconds to hold action
    action_detected: bool = False

    # For dynamic challenges (nod, shake)
    motion_history: deque = field(default_factory=lambda: deque(maxlen=30))
    baseline_eyebrow_dist: Optional[Dict[str, float]] = None

    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_challenge(self) -> Optional[ChallengeType]:
        if 0 <= self.current_idx < len(self.challenges):
            return self.challenges[self.current_idx]
        return None

    def reset(self):
        self.current_idx = 0
        self.is_active = False
        self.is_complete = False
        self.passed = False
        self.hold_start = 0.0
        self.action_detected = False
        self.motion_history.clear()
        self.baseline_eyebrow_dist = None
        self.results = []

    def advance(self) -> bool:
        """Move to next challenge. Returns True if puzzle complete."""
        self.results.append({
            'challenge': self.challenges[self.current_idx].name,
            'passed': True,
            'time': time.time()
        })
        self.current_idx += 1
        self.action_detected = False
        self.hold_start = 0.0
        self.baseline_eyebrow_dist = None
        self.motion_history.clear()

        if self.current_idx >= len(self.challenges):
            self.is_complete = True
            self.is_active = False
            self.passed = True
            return True
        return False


# Thresholds for challenge detection
class ChallengeThresholds:
    """Detection thresholds for challenges."""
    EAR_OPEN = 0.20          # Above this = eye open (lowered for better detection)
    EAR_CLOSED = 0.18        # Must go BELOW this for closed eye (raised slightly)
    SMILE_CORNER = 0.05      # Lip corner raise ratio
    SMILE_WIDTH = 0.60       # Mouth width ratio
    MOUTH_OPEN = 0.12        # Mouth aspect ratio for open
    YAW = 20                 # Degrees for turn left/right
    PITCH = 12               # Degrees for look up/down
    EYEBROW_BOTH = 1.12      # Both eyebrows ratio (lowered from 1.20)
    EYEBROW_SINGLE = 1.15    # Single eyebrow raise (lowered from 1.25)
    NOD_RANGE = 25           # Degrees of pitch movement for nod
    SHAKE_RANGE = 35         # Degrees of yaw movement for shake
