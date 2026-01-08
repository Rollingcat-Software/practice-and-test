"""Signal processing filters for biometric data smoothing."""

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OneEuroFilterParams:
    """Parameters for One Euro Filter."""
    min_cutoff: float = 1.0    # Minimum cutoff frequency (Hz) - lower = more smoothing
    beta: float = 0.0          # Speed coefficient - higher = less lag during fast motion
    d_cutoff: float = 1.0      # Cutoff for derivative filtering


class OneEuroFilter:
    """One Euro Filter for real-time signal smoothing.

    The One Euro Filter is an adaptive low-pass filter that provides:
    - High smoothing when signal is stable (reduces jitter)
    - Low smoothing when signal changes rapidly (reduces lag)

    Based on: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems" (CHI 2012)

    Usage:
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        smoothed = filter(raw_value)
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """Initialize the filter.

        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing when slow)
            beta: Speed coefficient (higher = less lag when fast)
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def reset(self):
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """Filter a value.

        Args:
            x: Raw input value
            t: Timestamp (optional, uses current time if not provided)

        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()

        if self._t_prev is None:
            # First sample
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        # Compute time delta
        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero

        # Estimate derivative
        dx = (x - self._x_prev) / dt

        # Filter the derivative
        alpha_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_filtered = alpha_d * dx + (1 - alpha_d) * self._dx_prev

        # Adaptive cutoff based on speed
        speed = abs(dx_filtered)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal
        alpha = self._smoothing_factor(cutoff, dt)
        x_filtered = alpha * x + (1 - alpha) * self._x_prev

        # Update state
        self._x_prev = x_filtered
        self._dx_prev = dx_filtered
        self._t_prev = t

        return x_filtered

    @staticmethod
    def _smoothing_factor(cutoff: float, dt: float) -> float:
        """Compute exponential smoothing factor from cutoff frequency."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    """2D One Euro Filter for coordinate smoothing (e.g., landmarks).

    Applies independent One Euro Filters to x and y coordinates.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self._filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self):
        """Reset both filters."""
        self._filter_x.reset()
        self._filter_y.reset()

    def __call__(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        """Filter a 2D coordinate.

        Args:
            x: Raw x coordinate
            y: Raw y coordinate
            t: Timestamp (optional)

        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        return (self._filter_x(x, t), self._filter_y(y, t))


class LandmarkSmoother:
    """Smoother for facial landmarks using One Euro Filters.

    Maintains independent filters for each landmark point,
    providing smooth but responsive tracking.
    """

    def __init__(self, num_landmarks: int = 478, min_cutoff: float = 0.5, beta: float = 0.01):
        """Initialize landmark smoother.

        Args:
            num_landmarks: Number of landmarks to track (MediaPipe uses 478)
            min_cutoff: Base cutoff (lower = smoother, default 0.5 for landmarks)
            beta: Speed coefficient (higher = more responsive)
        """
        self._filters = [
            OneEuroFilter2D(min_cutoff, beta)
            for _ in range(num_landmarks)
        ]
        self._initialized = False

    def reset(self):
        """Reset all filters."""
        for f in self._filters:
            f.reset()
        self._initialized = False

    def smooth(self, landmarks: list, t: Optional[float] = None) -> list:
        """Smooth a list of landmark coordinates.

        Args:
            landmarks: List of (x, y, z) or (x, y) tuples
            t: Timestamp (optional)

        Returns:
            List of smoothed landmarks
        """
        if t is None:
            t = time.time()

        result = []
        for i, lm in enumerate(landmarks):
            if i >= len(self._filters):
                result.append(lm)
                continue

            if len(lm) >= 2:
                sx, sy = self._filters[i](lm[0], lm[1], t)
                if len(lm) >= 3:
                    result.append((sx, sy, lm[2]))  # Keep z unfiltered
                else:
                    result.append((sx, sy))
            else:
                result.append(lm)

        self._initialized = True
        return result

    @property
    def is_initialized(self) -> bool:
        """Check if smoother has processed at least one frame."""
        return self._initialized
