"""
Dynamic Shape Tracing -- Liveness Challenge Module
===================================================
The user must trace a randomly generated geometric shape (Circle, Square,
Triangle, or S-Curve) using their index finger in real-time.

Architecture
------------

              ┌────────────┐
   start      │            │  shape guide shown on screen;
  ───────────►│    IDLE    │  waiting for finger to approach Start Point
              │            │
              └─────┬──────┘
                    │ finger enters Start Point radius
                    ▼
              ┌────────────┐
              │            │  hold position for 500 ms;
              │POSITIONING │  arc-fill shows countdown progress;
              │            │  if finger leaves → back to IDLE
              └─────┬──────┘
                    │ 500 ms hold complete → timer STARTS
                    ▼
              ┌────────────┐
              │            │  record Index Tip (ID 8) path each frame;
              │  TRACING   │  overlay trace on screen; timer counting down;
              │            │  liveness gate: hand_scale > min_hand_scale
              └──┬──────┬──┘
  finger at END  │      │ time expired  (or hand lost)
  (start_armed)  │      │
                 ▼      ▼
              ┌────────────┐
              │            │  single-frame DTW computation:
              │ COMPLETED  │    1. resample both paths to N points
              │            │    2. centroid-normalise (translation + scale invariant)
              └──┬──────┬──┘    3. DTW cost matrix (numpy distance mat + Python DP)
      cost ≤ th  │      │  cost > th / too few points
                 ▼      ▼
          ┌────────┐  ┌────────┐
          │VERIFIED│  │ FAILED │
          └────────┘  └────────┘

Shape templates (normalised screen coords, [0, 1]²)
----------------------------------------------------
  CIRCLE   -- 49-point parametric circle, centre (0.50, 0.50), r = 0.18
  SQUARE   -- 5-waypoint closed rectangle
  TRIANGLE -- 4-waypoint closed triangle
  S_CURVE  -- 18 control points (two opposing quarter-circle arcs)

Point-to-Point Trigger
-----------------------
  Each ShapeTemplate carries explicit start_point and end_point.
  • Recording (TRACING state) does NOT begin until the user holds their
    index fingertip within START_RADIUS (0.06) of start_point for
    position_hold_time seconds (default 0.50 s).
  • The countdown timer starts only after the TRACING state begins.
  • For closed shapes (start == end) a _start_armed flag is used:
    it becomes True once the finger moves away from start_point, so the
    end trigger cannot fire immediately on entry.
  • Recording ends when the fingertip is within END_RADIUS of end_point
    AND _start_armed is True, or when the time limit expires.

Verification algorithm (DTW)
-----------------------------
  1. Resample both paths to N = 50 evenly-spaced points (arc-length).
  2. Centroid-normalise: subtract centroid, divide by max radius.
     → invariant to translation AND scale; only shape matters.
  3. Build Euclidean distance matrix D[i,j] with numpy (vectorised).
  4. Fill DTW cost matrix with a Python DP loop (2 500 iterations ≈ 2 ms).
  5. Normalise total cost by N → per-point average distance.
  6. Accept if normalised DTW cost ≤ dtw_threshold (default 0.25).

Similarity score: max(0, 1 - cost / threshold)  → 0 … 1  → multiplied by
100 for display.  Green ≥ 70 %, Yellow ≥ 40 %, Red below that.
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from hand_tracker import HandResult
from gesture_validator import hand_scale as _hs, is_finger_open, Finger, _LM
from app_config import cfg


# ── Shape catalogue ──────────────────────────────────────────────────

class ShapeType(Enum):
    CIRCLE   = auto()
    SQUARE   = auto()
    TRIANGLE = auto()
    S_CURVE  = auto()


@dataclass(frozen=True)
class ShapeTemplate:
    """A target shape defined by ordered (x, y) waypoints in [0, 1]².

    Waypoints represent the expected tracing direction (clockwise for
    closed shapes, top-to-bottom for open curves).  They are drawn as a
    semi-transparent guide on the video frame.

    Parameters
    ----------
    start_point : (x, y)
        Normalised coordinate where the user must hold their finger to
        begin recording.  Highlighted with a pulsing cyan ring in IDLE state.
    end_point : (x, y)
        Normalised coordinate that, when reached, automatically ends the
        trace and triggers verification.
    min_trace_points : int
        Minimum number of recorded points required for a valid attempt.
        Attempts with fewer points always result in FAILED.
    """
    shape_type:       ShapeType
    label:            str
    waypoints:        tuple[tuple[float, float], ...]
    start_point:      tuple[float, float]
    end_point:        tuple[float, float]
    min_trace_points: int = 25


# ── Template builders ────────────────────────────────────────────────

def _circle_template(cx: float = 0.50, cy: float = 0.50,
                     r: float = 0.18, n: int = 48) -> ShapeTemplate:
    """Full circle starting at the 3-o'clock position, traced clockwise."""
    pts = tuple(
        (round(cx + r * math.cos(2 * math.pi * i / n), 6),
         round(cy + r * math.sin(2 * math.pi * i / n), 6))
        for i in range(n + 1)   # +1 closes the loop back to start
    )
    start = (round(cx + r, 6), round(cy, 6))   # 3-o'clock = (0.68, 0.50)
    return ShapeTemplate(
        ShapeType.CIRCLE, "CIRCLE", pts,
        start_point=start, end_point=start, min_trace_points=40,
    )


def _square_template(l: float = 0.32, t: float = 0.30,
                     r: float = 0.68, b: float = 0.70) -> ShapeTemplate:
    """Closed rectangle traced clockwise from the top-left corner."""
    start = (l, t)
    return ShapeTemplate(
        ShapeType.SQUARE, "SQUARE",
        ((l, t), (r, t), (r, b), (l, b), (l, t)),
        start_point=start, end_point=start, min_trace_points=30,
    )


def _triangle_template(cx: float = 0.50, ty: float = 0.25,
                        by: float = 0.72, hw: float = 0.22) -> ShapeTemplate:
    """Isosceles triangle: top → bottom-right → bottom-left → top."""
    start = (cx, ty)
    return ShapeTemplate(
        ShapeType.TRIANGLE, "TRIANGLE",
        ((cx, ty), (cx + hw, by), (cx - hw, by), (cx, ty)),
        start_point=start, end_point=start, min_trace_points=25,
    )


def _s_curve_template() -> ShapeTemplate:
    """S-shape from two opposing quarter-circle arcs, traced top-to-bottom."""
    pts: list[tuple[float, float]] = []
    # Upper arc: concave-right  (centre 0.62, 0.375, radius ≈ 0.125)
    for i in range(9):
        a = math.pi + (math.pi / 2) * (i / 8)      # π → 3π/2
        pts.append((round(0.62 + 0.14 * math.cos(a), 6),
                    round(0.375 + 0.125 * math.sin(a), 6)))
    # Lower arc: concave-left   (centre 0.38, 0.625, radius ≈ 0.125)
    for i in range(9):
        a = math.pi * 2 + (math.pi / 2) * (i / 8)  # 2π → 5π/2
        pts.append((round(0.38 + 0.14 * math.cos(a), 6),
                    round(0.625 + 0.125 * math.sin(a), 6)))

    start_pt = pts[0]   # (0.48, 0.375)  top of upper arc
    end_pt   = pts[-1]  # (0.38, 0.75)   bottom of lower arc
    return ShapeTemplate(
        ShapeType.S_CURVE, "S-CURVE", tuple(pts),
        start_point=start_pt, end_point=end_pt, min_trace_points=15,
    )


_TEMPLATES: list[ShapeTemplate] = [
    _circle_template(),
    _square_template(),
    _triangle_template(),
    _s_curve_template(),
]


def generate_random_shape() -> ShapeTemplate:
    """Return one of the four shape templates chosen at random."""
    return random.choice(_TEMPLATES)


# ── Path mathematics ─────────────────────────────────────────────────

def _resample(path: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Resample a polyline to exactly *n* evenly-spaced points by arc-length.

    This ensures both the template and the traced path have the same number
    of points before DTW, eliminating length bias from the alignment cost.
    """
    if len(path) == 0:
        return [(0.0, 0.0)] * n
    if len(path) == 1:
        return [path[0]] * n

    # Cumulative arc-lengths
    dists: list[float] = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    total = dists[-1]
    if total < 1e-9:
        return [path[0]] * n

    resampled: list[tuple[float, float]] = []
    j = 0
    for k in range(n):
        target = k * total / (n - 1)
        while j < len(dists) - 2 and dists[j + 1] < target:
            j += 1
        seg = dists[j + 1] - dists[j]
        t = ((target - dists[j]) / seg) if seg > 1e-9 else 0.0
        x = path[j][0] + t * (path[j + 1][0] - path[j][0])
        y = path[j][1] + t * (path[j + 1][1] - path[j][1])
        resampled.append((x, y))
    return resampled


def _centroid_normalise(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Translate to centroid; scale by max radius → points in ≈ [-1, 1]².

    This makes the DTW comparison invariant to both translation and scale:
    a small circle traced in the corner and a large circle traced in the
    centre produce the same normalised representation.
    """
    n = len(path)
    if n == 0:
        return []
    cx = sum(p[0] for p in path) / n
    cy = sum(p[1] for p in path) / n
    centred = [(p[0] - cx, p[1] - cy) for p in path]
    max_r = max(math.sqrt(x * x + y * y) for x, y in centred) or 1.0
    return [(x / max_r, y / max_r) for x, y in centred]


def dtw_normalised_cost(
    path_a: list[tuple[float, float]],
    path_b: list[tuple[float, float]],
) -> float:
    """Return the DTW alignment cost normalised by path length.

    Steps
    -----
    1. Build a Euclidean distance matrix D[i, j] with numpy (vectorised).
    2. Fill the DTW cost table with a Python DP loop.
    3. Divide the total cost by max(n, m) → per-point average distance.

    Both input paths should already be resampled to the same length N.
    For N = 50, the DP loop runs 2 500 iterations ≈ 1–3 ms.
    """
    n, m = len(path_a), len(path_b)
    if n == 0 or m == 0:
        return float("inf")

    a = np.asarray(path_a, dtype=np.float32)   # (n, 2)
    b = np.asarray(path_b, dtype=np.float32)   # (m, 2)

    # Vectorised Euclidean distance matrix
    diff = a[:, None, :] - b[None, :, :]        # (n, m, 2)
    D: np.ndarray = np.sqrt((diff ** 2).sum(axis=-1))  # (n, m)

    # Standard DTW DP with (n+1)×(m+1) table, INF borders
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(D[i - 1, j - 1])
            dtw[i, j] = cost + min(
                float(dtw[i - 1, j]),
                float(dtw[i, j - 1]),
                float(dtw[i - 1, j - 1]),
            )

    return float(dtw[n, m]) / max(n, m)


# ── Proximity helper ─────────────────────────────────────────────────

# Radius (in normalised [0,1] screen coords) for start/end trigger zones.
START_RADIUS = 0.06
END_RADIUS   = 0.07
# When finger is farther than this from start_point, _start_armed becomes True.
ARM_RADIUS   = 0.10


def _finger_near_point(fx: float, fy: float,
                       px: float, py: float,
                       radius: float = START_RADIUS) -> bool:
    """Return True if (fx, fy) is within *radius* of (px, py)."""
    return math.sqrt((fx - px) ** 2 + (fy - py) ** 2) <= radius


# ── Tracer state machine ─────────────────────────────────────────────

class TracerState(Enum):
    INSTRUCTING = auto()  # brief instruction overlay before showing shape
    IDLE        = auto()  # shape displayed; waiting for finger at Start Point
    POSITIONING = auto()  # finger on Start Point; 500ms countdown
    TRACING     = auto()  # recording; timer running
    COMPLETED   = auto()  # single-frame DTW computation (instant transition)
    VERIFIED    = auto()  # DTW cost ≤ threshold → accepted
    FAILED      = auto()  # timed out or DTW cost too high / too few points


# Defaults used by ShapeTracerSession and by liveness_session integration.
# Module-level constants — sourced from the central config so a single
# edit to config.yaml propagates here and to all importers automatically.
# (config.yaml: shape_trace.draw_time=12.0, shape_trace.min_hand_scale=0.05
#  reflect the spoof-fix tuning landed in fix/liveness-shape-trace-spoof-false-positive)
DEFAULT_DRAW_TIME     = cfg.shape_trace.draw_time      # seconds per trace attempt
DEFAULT_DTW_THRESH    = cfg.shape_trace.dtw_threshold  # normalised per-point DTW cost threshold
DEFAULT_RESAMPLE_N    = cfg.shape_trace.resample_n     # number of points for DTW comparison
DEFAULT_MIN_HS        = cfg.shape_trace.min_hand_scale # minimum hand_scale for depth liveness gate
DEFAULT_POS_HOLD      = cfg.shape_trace.pos_hold       # seconds to hold at start point before TRACING
DEFAULT_INSTRUCT_TIME = cfg.shape_trace.instruct_time  # seconds to show instruction overlay


@dataclass
class ShapeTracerSession:
    """State machine for one Dynamic Shape Tracing liveness challenge.

    Parameters
    ----------
    time_limit : float
        Seconds the user has to trace once the TRACING state begins.
        Does NOT count IDLE / POSITIONING time.
    min_hand_scale : float
        Minimum wrist-to-MiddleMCP distance (hand_scale) that counts as
        'hand in frame'.  Acts as a depth-based liveness gate.
    dtw_threshold : float
        Maximum normalised per-point DTW cost accepted as a valid trace.
    resample_n : int
        Number of resampled points used in the DTW comparison.
    position_hold_time : float
        Seconds the index finger must stay inside START_RADIUS before
        TRACING begins.
    pause_after_result : float
        Seconds to hold VERIFIED / FAILED before the session may be reset.
    """

    time_limit:         float = DEFAULT_DRAW_TIME
    min_hand_scale:     float = DEFAULT_MIN_HS
    dtw_threshold:      float = DEFAULT_DTW_THRESH
    resample_n:         int   = DEFAULT_RESAMPLE_N
    position_hold_time: float = DEFAULT_POS_HOLD
    instruction_duration: float = DEFAULT_INSTRUCT_TIME
    pause_after_result: float = 2.0

    # -- observable (read by HUD) -----------------------------------------
    state:      TracerState = field(init=False, default=TracerState.INSTRUCTING)
    template:   ShapeTemplate = field(init=False)
    similarity: float = field(init=False, default=0.0)   # 0 … 1
    dtw_cost:   float = field(init=False, default=0.0)

    # -- internal ---------------------------------------------------------
    _traced:          list[tuple[float, float]] = field(init=False, default_factory=list)
    _raw_buf:         list[tuple[float, float]] = field(init=False, default_factory=list)
    _draw_start:      Optional[float] = field(init=False, default=None)   # set on TRACING entry
    _position_start:  Optional[float] = field(init=False, default=None)   # set on POSITIONING entry
    _instruct_start:  Optional[float] = field(init=False, default=None)   # set on INSTRUCTING entry
    _result_at:       Optional[float] = field(init=False, default=None)
    _was_index_open:  bool = field(init=False, default=False)
    # True once finger moves far enough from start to enable end trigger
    _start_armed:     bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.template        = generate_random_shape()
        self._instruct_start = time.time()

    # -- Point recording (3-point moving-average smoothing) ---------------

    def _push_point(self, x: float, y: float) -> None:
        """Append smoothed point; mirrors ShapeRecognizer.push() logic."""
        self._raw_buf.append((x, y))
        if len(self._raw_buf) > 3:
            self._raw_buf.pop(0)
        n  = len(self._raw_buf)
        sx = sum(p[0] for p in self._raw_buf) / n
        sy = sum(p[1] for p in self._raw_buf) / n
        # Skip near-duplicates to keep the path compact.
        if self._traced:
            lx, ly = self._traced[-1]
            if abs(sx - lx) < 0.003 and abs(sy - ly) < 0.003:
                return
        self._traced.append((sx, sy))

    # -- Verification (runs in exactly one frame) -------------------------

    def _run_verification(self) -> None:
        """Compute DTW similarity and transition to VERIFIED or FAILED."""
        now = time.time()
        if len(self._traced) < self.template.min_trace_points:
            self.state      = TracerState.FAILED
            self._result_at = now
            return

        t_rs   = _resample(list(self.template.waypoints), self.resample_n)
        u_rs   = _resample(self._traced,                  self.resample_n)
        t_norm = _centroid_normalise(t_rs)
        u_norm = _centroid_normalise(u_rs)

        cost             = dtw_normalised_cost(t_norm, u_norm)
        self.dtw_cost    = cost
        self.similarity  = max(0.0, min(1.0, 1.0 - cost / self.dtw_threshold))
        self.state       = TracerState.VERIFIED if cost <= self.dtw_threshold else TracerState.FAILED
        self._result_at  = now

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> TracerState:
        """Feed one video frame.  Returns the current TracerState.

        Call this every frame; read ``state``, ``traced_path``, and
        ``template_waypoints`` to drive the HUD renderer.
        """
        now = time.time()

        # ── INSTRUCTING ───────────────────────────────────────────────
        # Show instructions for instruction_duration seconds, then go IDLE.
        # If the user is already impatient and holds finger at start → skip.
        if self.state == TracerState.INSTRUCTING:
            elapsed = now - (self._instruct_start or now)
            if elapsed >= self.instruction_duration:
                self.state = TracerState.IDLE
                return self.state
            # Fast-forward: finger already positioned at start point.
            if hands:
                hand0 = hands[0]
                lm0   = hand0.landmarks
                if _hs(lm0) >= self.min_hand_scale:
                    tip = lm0[_LM["INDEX_TIP"]]
                    if (is_finger_open(lm0, hand0.handedness, Finger.INDEX) and
                            _finger_near_point(tip.x, tip.y,
                                               *self.template.start_point,
                                               START_RADIUS)):
                        self._position_start = now
                        self.state = TracerState.POSITIONING
            return self.state

        # COMPLETED runs verification for exactly one frame then transitions.
        if self.state == TracerState.COMPLETED:
            self._run_verification()
            return self.state

        # Terminal states are sticky until reset() is called.
        if self.state in (TracerState.VERIFIED, TracerState.FAILED):
            return self.state

        # No hand detected → drop back from POSITIONING/TRACING to IDLE.
        if not hands:
            if self.state == TracerState.TRACING:
                # Complete with whatever was recorded (could be short).
                self.state = TracerState.COMPLETED
            elif self.state == TracerState.POSITIONING:
                self._position_start = None
                self.state = TracerState.IDLE
            return self.state

        hand = hands[0]
        lm   = hand.landmarks

        # ── Depth liveness gate ───────────────────────────────────────
        if _hs(lm) < self.min_hand_scale:
            return self.state

        index_tip  = lm[_LM["INDEX_TIP"]]
        index_open = is_finger_open(lm, hand.handedness, Finger.INDEX)
        fx, fy     = index_tip.x, index_tip.y

        # ── IDLE ──────────────────────────────────────────────────────
        if self.state == TracerState.IDLE:
            if index_open and _finger_near_point(fx, fy,
                                                 *self.template.start_point,
                                                 START_RADIUS):
                self._position_start = now
                self.state = TracerState.POSITIONING

        # ── POSITIONING ───────────────────────────────────────────────
        elif self.state == TracerState.POSITIONING:
            if not index_open or not _finger_near_point(fx, fy,
                                                        *self.template.start_point,
                                                        START_RADIUS):
                # Finger left the start zone → back to IDLE.
                self._position_start = None
                self.state = TracerState.IDLE
            elif now - self._position_start >= self.position_hold_time:
                # Held long enough → start recording.
                self._reset_drawing()
                self._draw_start = now
                # Closed shapes: arm flag only after finger moves away.
                sp = self.template.start_point
                ep = self.template.end_point
                self._start_armed = (sp != ep)   # True for open shapes
                self.state = TracerState.TRACING
                self._push_point(fx, fy)
                print("Tracing Started, Jitter Check: Disabled")

        # ── TRACING ───────────────────────────────────────────────────
        elif self.state == TracerState.TRACING:
            # Hard time-limit.
            if now - self._draw_start >= self.time_limit:
                self.state = TracerState.COMPLETED
                return self.state

            if index_open:
                self._push_point(fx, fy)
                self._was_index_open = True

                # Arm trigger once finger moves away from the start point.
                if not self._start_armed:
                    if not _finger_near_point(fx, fy,
                                              *self.template.start_point,
                                              ARM_RADIUS):
                        self._start_armed = True

                # End trigger: finger at end_point AND armed.
                if self._start_armed and _finger_near_point(fx, fy,
                                                            *self.template.end_point,
                                                            END_RADIUS):
                    self.state = TracerState.COMPLETED
            else:
                # Fist after a valid stroke → trigger completion.
                if self._was_index_open and len(self._traced) >= self.template.min_trace_points:
                    self.state = TracerState.COMPLETED

        return self.state

    # -- Internals --------------------------------------------------------

    def _reset_drawing(self) -> None:
        self._traced.clear()
        self._raw_buf.clear()
        self._was_index_open = False
        self._draw_start     = None

    # -- Observable helpers for HUD ---------------------------------------

    @property
    def traced_path(self) -> list[tuple[float, float]]:
        """Current recorded trace in normalised screen coordinates."""
        return list(self._traced)

    @property
    def template_waypoints(self) -> list[tuple[float, float]]:
        return list(self.template.waypoints)

    @property
    def time_remaining(self) -> float:
        """Seconds remaining in the TRACING window (0 when not tracing)."""
        if self._draw_start is None:
            return self.time_limit
        return max(0.0, self.time_limit - (time.time() - self._draw_start))

    @property
    def draw_progress(self) -> float:
        """0.0 → 1.0 fraction of the TRACING time window used."""
        if self._draw_start is None:
            return 0.0
        return min((time.time() - self._draw_start) / self.time_limit, 1.0)

    @property
    def position_progress(self) -> float:
        """0.0 → 1.0 arc-fill for the POSITIONING countdown."""
        if self._position_start is None:
            return 0.0
        return min((time.time() - self._position_start) / self.position_hold_time, 1.0)

    @property
    def instruct_progress(self) -> float:
        """0.0 → 1.0 fraction of the instruction overlay elapsed."""
        if self._instruct_start is None:
            return 1.0
        return min((time.time() - self._instruct_start) / max(self.instruction_duration, 0.001), 1.0)

    @property
    def instruct_remaining(self) -> float:
        """Seconds remaining in the instruction phase."""
        if self._instruct_start is None:
            return 0.0
        return max(0.0, self.instruction_duration - (time.time() - self._instruct_start))

    @property
    def similarity_pct(self) -> float:
        """0 – 100 similarity percentage (only meaningful after COMPLETED)."""
        return self.similarity * 100.0

    @property
    def point_count(self) -> int:
        return len(self._traced)

    def reset(self) -> None:
        """Pick a new random shape and restart from the INSTRUCTING phase."""
        self.template        = generate_random_shape()
        self.state           = TracerState.INSTRUCTING
        self.similarity      = 0.0
        self.dtw_cost        = 0.0
        self._instruct_start = time.time()
        self._result_at      = None
        self._position_start = None
        self._start_armed    = False
        self._reset_drawing()
