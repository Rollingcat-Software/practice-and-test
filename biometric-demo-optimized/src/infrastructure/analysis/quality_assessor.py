"""Face image quality assessment."""

import cv2
import numpy as np

from ...domain.models import Quality


class QualityAssessor:
    """Fast quality assessment for face images.

    Assesses:
    - Blur (Laplacian variance)
    - Size (minimum dimension)
    - Brightness
    """

    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    def assess(self, face_img: np.ndarray) -> Quality:
        """Assess quality of face image.

        Args:
            face_img: BGR face image

        Returns:
            Quality object with scores and issues
        """
        if face_img is None or face_img.size == 0:
            return Quality(score=0, issues=['No image'])

        h, w = face_img.shape[:2]

        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Blur score (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100, (lap_var / self.blur_threshold) * 100)

        # Size score
        size_score = min(100, min(h, w) / 80 * 50)

        # Brightness
        brightness = np.mean(gray)
        brightness_ok = 50 < brightness < 200

        # Identify issues
        issues = []
        if blur_score < 50:
            issues.append('Blurry')
        if size_score < 50:
            issues.append('Small')
        if not brightness_ok:
            issues.append('Dark' if brightness < 50 else 'Bright')

        # Combined score
        score = (blur_score + size_score + (100 if brightness_ok else 50)) / 3

        return Quality(
            score=score,
            blur_score=blur_score,
            size_score=size_score,
            brightness_ok=brightness_ok,
            issues=issues
        )
