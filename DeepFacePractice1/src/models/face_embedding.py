"""
Face Embedding Model
Represents a face as a mathematical vector (embedding).
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


@dataclass
class FaceEmbedding:
    """
    Encapsulates face embedding (vector representation).

    Educational Notes:
    - Embeddings are numerical representations of faces
    - Similar faces have similar embeddings
    - Dimension varies by model (128, 512, 2622, 4096, etc.)
    - Used for face recognition and similarity comparison
    """
    embedding: List[float]
    model: str
    img_path: str
    face_region: Dict[str, int]
    raw_result: Dict[str, Any]

    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embedding vector."""
        return len(self.embedding)

    @property
    def as_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array for calculations."""
        return np.array(self.embedding)

    def statistics(self) -> Dict[str, float]:
        """Calculate statistical properties of the embedding."""
        arr = self.as_numpy
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr))
        }

    def cosine_similarity(self, other: 'FaceEmbedding') -> float:
        """
        Calculate cosine similarity with another embedding.
        Returns value between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite).
        """
        a = self.as_numpy
        b = other.as_numpy

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        return float(dot_product / (norm_a * norm_b))

    def euclidean_distance(self, other: 'FaceEmbedding') -> float:
        """
        Calculate Euclidean distance with another embedding.
        Lower values indicate more similar faces.
        """
        return float(np.linalg.norm(self.as_numpy - other.as_numpy))

    def __str__(self) -> str:
        """Human-readable representation."""
        stats = self.statistics()
        return (
            f"Face Embedding ({self.model})\n"
            f"  Image: {self.img_path}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Statistics: min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
