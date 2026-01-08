"""Face database with vectorized search for fast verification."""

import os
import pickle
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np

from ...domain.models import EnrolledFace

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Face embedding database with vectorized similarity search.

    Features:
    - Pickle-based persistence
    - Vectorized cosine similarity using NumPy (BLAS/SIMD optimized)
    - Automatic matrix rebuild on changes
    - Multiple embeddings per face for robustness
    """

    def __init__(self, path: str = "face_db.pkl", threshold: float = 0.45):
        self.threshold = threshold
        self._faces: Dict[str, EnrolledFace] = {}
        self._matrix: Optional[np.ndarray] = None
        self._names: List[str] = []
        self._dirty = True

        # Find existing database or use default path
        self.path = self._find_db_path(path)
        self._load()

    def _find_db_path(self, default_path: str) -> str:
        """Find existing face database or return default path."""
        if os.path.exists(default_path):
            return default_path

        # Search in biometric-processor directory
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "biometric-processor", "face_db.pkl"),
            os.path.normpath(os.path.join(os.getcwd(), "..", "biometric-processor", "face_db.pkl")),
        ]

        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found existing face database at: {path}")
                return os.path.abspath(path)

        # Return default path (will create new database)
        return default_path

    def _load(self):
        """Load database from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle both old and new formats
                    if isinstance(data, dict):
                        for name, face_data in data.items():
                            if isinstance(face_data, EnrolledFace):
                                self._faces[name] = face_data
                            else:
                                # Convert old format
                                self._faces[name] = EnrolledFace(
                                    name=name,
                                    embeddings=face_data.get('embeddings', []),
                                    thumbnail=face_data.get('thumbnail'),
                                    enrolled_at=face_data.get('enrolled_at', datetime.now().isoformat())
                                )
                self._dirty = True
                logger.info(f"Loaded {len(self._faces)} enrolled faces")
            except Exception as e:
                logger.warning(f"Failed to load face database: {e}")
                self._faces = {}

    def save(self):
        """Save database to disk."""
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self._faces, f)
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")

    def _rebuild_matrix(self):
        """Rebuild the NumPy matrix for vectorized search."""
        if not self._faces:
            self._matrix = None
            self._names = []
            self._dirty = False
            return

        embeddings = []
        names = []

        for name, face in self._faces.items():
            for emb in face.embeddings:
                emb_flat = np.array(emb).flatten()
                norm = np.linalg.norm(emb_flat)
                if norm > 0:
                    embeddings.append(emb_flat / norm)  # Normalize for cosine similarity
                    names.append(name)

        if embeddings:
            self._matrix = np.array(embeddings)  # Shape: (N_samples, embedding_dim)
            self._names = names
        else:
            self._matrix = None
            self._names = []

        self._dirty = False
        logger.info(f"Face database matrix rebuilt: {len(names)} vectors")

    @property
    def faces(self) -> Dict[str, EnrolledFace]:
        """Get all enrolled faces."""
        return self._faces

    def __len__(self) -> int:
        return len(self._faces)

    def enroll(self, name: str, embedding: np.ndarray, thumbnail: np.ndarray):
        """Enroll a new face.

        Args:
            name: Identity name
            embedding: Face embedding vector
            thumbnail: Thumbnail image for display
        """
        self._faces[name] = EnrolledFace(
            name=name,
            embeddings=[embedding],
            thumbnail=thumbnail,
            enrolled_at=datetime.now().isoformat()
        )
        self._dirty = True
        self.save()
        logger.info(f"Enrolled new face: {name}")

    def add_embedding(self, name: str, embedding: np.ndarray, max_embeddings: int = 5):
        """Add additional embedding to existing face.

        Args:
            name: Identity name
            embedding: Additional embedding vector
            max_embeddings: Maximum embeddings to keep per face
        """
        if name in self._faces:
            self._faces[name].add_embedding(embedding, max_embeddings)
            self._dirty = True
            self.save()

    def search(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """Search for matching face using vectorized cosine similarity.

        Uses BLAS/SIMD via NumPy for fast matrix-vector multiplication.
        Much faster than looping for large databases.

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold (default: instance threshold)

        Returns:
            Tuple of (name, similarity) if match found, None otherwise
        """
        if self._dirty or self._matrix is None:
            self._rebuild_matrix()

        if self._matrix is None:
            return None

        threshold = threshold or self.threshold

        # Normalize query vector
        query = np.array(embedding).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None
        query = query / query_norm

        # Vectorized cosine similarity: matrix @ vector
        # (N, embedding_dim) @ (embedding_dim,) -> (N,)
        scores = np.dot(self._matrix, query)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score >= threshold:
            return (self._names[best_idx], float(best_score))

        return None

    def delete(self, name: str) -> bool:
        """Delete an enrolled face.

        Args:
            name: Identity name to delete

        Returns:
            True if deleted, False if not found
        """
        if name in self._faces:
            del self._faces[name]
            self._dirty = True
            self.save()
            logger.info(f"Deleted face: {name}")
            return True
        return False

    def clear(self):
        """Delete all enrolled faces."""
        count = len(self._faces)
        self._faces.clear()
        self._matrix = None
        self._names = []
        self._dirty = False
        self.save()
        logger.info(f"Cleared {count} enrolled faces")
