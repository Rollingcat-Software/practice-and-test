"""Face embedding extraction using DeepFace."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """DeepFace embedding extractor.

    Extracts 512-dimensional face embeddings using Facenet512 model.
    Used for enrollment and verification.
    """

    _deepface = None

    def __init__(self, model: str = "Facenet512"):
        self.model = model
        if EmbeddingExtractor._deepface is None:
            from deepface import DeepFace
            EmbeddingExtractor._deepface = DeepFace
            logger.info(f"DeepFace loaded for embeddings (model={model})")

    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from face image.

        Args:
            face_img: BGR face image (already cropped)

        Returns:
            512-dimensional embedding vector or None if failed
        """
        if face_img is None or face_img.size == 0 or min(face_img.shape[:2]) < 48:
            return None

        try:
            results = self._deepface.represent(
                img_path=face_img,
                model_name=self.model,
                enforce_detection=False,
                detector_backend="skip",  # Skip detection, we already have face
            )
            if results:
                return np.array(results[0]['embedding'])
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")

        return None
