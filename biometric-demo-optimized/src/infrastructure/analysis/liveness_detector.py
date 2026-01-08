"""Passive liveness detection with 5-factor analysis."""

import cv2
import numpy as np

from ...domain.models import LivenessResult


# Pre-computed Gabor kernels (class-level constant, smaller size for speed)
_GABOR_KERNELS = tuple(
    cv2.getGaborKernel((15, 15), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]
)


class LivenessDetector:
    """Fast passive liveness detection.

    Uses 5 factors to detect presentation attacks:
    1. Texture score - Real faces have natural texture variation
    2. Color naturalness - Real skin has moderate saturation
    3. Skin tone check - Hue should be in skin range
    4. Moire detection - Print attacks show periodic patterns
    5. Local contrast variation - Real faces have varying texture across regions
    """

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
        # Use pre-computed class-level kernels
        self._gabor_kernels = _GABOR_KERNELS

    def check(self, face_img: np.ndarray) -> LivenessResult:
        """Check if face is live (not a presentation attack).

        Args:
            face_img: BGR face image

        Returns:
            LivenessResult with scores for each factor
        """
        if face_img is None or face_img.size == 0:
            return LivenessResult(is_live=False, score=0)

        try:
            # Single color conversion to HSV, extract V channel as grayscale
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            gray = hsv[:, :, 2]  # V channel is grayscale equivalent

            # 1. Texture score - Laplacian variance measures edge sharpness
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture = min(100, max(0, (lap_var - 20) / 3))

            # 2. Color naturalness - Real skin has moderate saturation (40-100)
            sat_mean = np.mean(hsv[:, :, 1])
            if 30 <= sat_mean <= 120:
                color = 100
            elif sat_mean < 30:
                color = max(0, sat_mean * 2)  # Penalty for too gray
            else:
                color = max(0, 100 - (sat_mean - 120) * 0.8)  # Penalty for oversaturated

            # 3. Skin tone check - Hue should be in skin range (0-25 in OpenCV)
            hue_mean = np.mean(hsv[:, :, 0])
            if hue_mean < 25 or hue_mean > 165:
                skin_tone = 100
            else:
                skin_tone = max(0, 100 - abs(hue_mean - 15) * 3)

            # 4. Moire/pattern detection - Print artifacts show periodic patterns
            moire = 100
            gray_f32 = gray.astype(np.float32)  # Convert once for all kernels
            for kernel in self._gabor_kernels:
                gabor_std = np.std(cv2.filter2D(gray_f32, -1, kernel))
                if gabor_std > 40:
                    moire -= 20

            # 5. Local contrast variation - Real faces have varying texture
            h, w = gray.shape
            if h >= 20 and w >= 20:
                regions = [
                    gray[:h//2, :w//2], gray[:h//2, w//2:],
                    gray[h//2:, :w//2], gray[h//2:, w//2:]
                ]
                variances = [np.var(r) for r in regions]
                var_range = max(variances) - min(variances)
                local_var = min(100, var_range / 10)
            else:
                local_var = 50

            # Combined score with weights
            score = (texture * 0.25 + color * 0.25 + skin_tone * 0.15 +
                    moire * 0.20 + local_var * 0.15)

            return LivenessResult(
                is_live=score >= self.threshold,
                score=score,
                texture_score=texture,
                color_score=color,
                skin_tone_score=skin_tone,
                moire_score=moire,
                local_var_score=local_var
            )
        except Exception:
            return LivenessResult(is_live=False, score=0)
