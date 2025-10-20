"""
Face Recognition Service
Handles face embeddings and database-based recognition.

Educational Notes:
- Recognition = 1:N comparison (searching for a face in a database)
- Uses embeddings (vector representations) for efficient searching
- DeepFace can build and search face databases automatically
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from deepface import DeepFace

from ..models.face_embedding import FaceEmbedding


class FaceRecognitionService:
    """
    Service for face recognition and embedding operations.

    Demonstrates:
    - SRP: Focused only on recognition/embedding
    - DRY: Reusable methods for common operations
    """

    AVAILABLE_MODELS = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ]

    def __init__(
            self,
            model_name: str = "Facenet512",
            detector_backend: str = "opencv"
    ):
        """
        Initialize the recognition service.

        Args:
            model_name: Which face recognition model to use
            detector_backend: Which face detection algorithm to use
        """
        self.model_name = model_name
        self.detector_backend = detector_backend

        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Choose from: {', '.join(self.AVAILABLE_MODELS)}"
            )

    def extract_embedding(
            self,
            img_path: str,
            enforce_detection: bool = True
    ) -> FaceEmbedding:
        """
        Extract face embedding (vector representation) from an image.

        Educational Note:
        Embeddings are the foundation of face recognition:
        - They convert faces into numerical vectors
        - Similar faces have similar vectors
        - Can be compared using distance metrics

        Args:
            img_path: Path to the image
            enforce_detection: If True, raises error when no face detected

        Returns:
            FaceEmbedding object containing the vector and metadata

        Raises:
            ValueError: If face detection fails
        """
        try:
            # DeepFace.represent returns a list of embeddings (one per face)
            raw_results = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=enforce_detection
            )

            # Take first face
            raw_result = raw_results[0] if isinstance(raw_results, list) else raw_results

            return FaceEmbedding(
                embedding=raw_result["embedding"],
                model=self.model_name,
                img_path=img_path,
                face_region=raw_result.get("facial_area", {}),
                raw_result=raw_result
            )

        except Exception as e:
            raise ValueError(f"Embedding extraction failed: {str(e)}") from e

    def extract_multiple_embeddings(
            self,
            img_paths: List[str],
            enforce_detection: bool = True
    ) -> Dict[str, FaceEmbedding]:
        """
        Extract embeddings from multiple images.

        Args:
            img_paths: List of image paths
            enforce_detection: If True, raises error when no face detected

        Returns:
            Dictionary mapping image paths to their embeddings
        """
        embeddings = {}

        for img_path in img_paths:
            try:
                embedding = self.extract_embedding(img_path, enforce_detection)
                embeddings[img_path] = embedding
            except Exception as e:
                print(f"Warning: Failed to extract embedding from {img_path}: {e}")
                continue

        return embeddings

    def find_faces_in_database(
            self,
            img_path: str,
            db_path: str,
            distance_metric: str = "cosine",
            model_name: Optional[str] = None,
            detector_backend: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find matching faces in a database directory.

        Educational Note:
        This is true "face recognition" - searching for a person
        in a database of known faces.

        How it works:
        1. Scans db_path directory for all images
        2. Extracts embeddings from all faces
        3. Compares target image against all database faces
        4. Returns matches sorted by similarity

        Args:
            img_path: Path to the query image (face to search for)
            db_path: Path to database directory containing face images
            distance_metric: How to calculate similarity
            model_name: Override default model
            detector_backend: Override default detector

        Returns:
            List of dictionaries containing match information

        Example database structure:
            database/
                person1/
                    img1.jpg
                    img2.jpg
                person2/
                    img1.jpg
        """
        try:
            # Use provided parameters or fall back to instance defaults
            model = model_name or self.model_name
            detector = detector_backend or self.detector_backend

            # DeepFace.find searches the database
            results = DeepFace.find(
                img_path=img_path,
                db_path=db_path,
                model_name=model,
                detector_backend=detector,
                distance_metric=distance_metric,
                enforce_detection=True,
                silent=True
            )

            # Convert DataFrame results to list of dicts
            if len(results) > 0 and not results[0].empty:
                return results[0].to_dict('records')
            else:
                return []

        except Exception as e:
            raise ValueError(f"Database search failed: {str(e)}") from e

    def compare_embeddings(
            self,
            embedding1: FaceEmbedding,
            embedding2: FaceEmbedding,
            metric: str = "cosine"
    ) -> float:
        """
        Compare two embeddings using specified metric.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: 'cosine' or 'euclidean'

        Returns:
            Similarity score (interpretation depends on metric)
        """
        if metric == "cosine":
            return embedding1.cosine_similarity(embedding2)
        elif metric == "euclidean":
            return embedding1.euclidean_distance(embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def build_face_database(
            self,
            source_dir: str,
            output_file: str = "face_database.pkl"
    ) -> Dict[str, FaceEmbedding]:
        """
        Build a face database from a directory of images.

        This creates a searchable database of face embeddings.

        Args:
            source_dir: Directory containing face images
            output_file: Where to save the database

        Returns:
            Dictionary of image paths to embeddings
        """
        import pickle

        # Get all image files
        source_path = Path(source_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            str(f) for f in source_path.rglob('*')
            if f.suffix.lower() in image_extensions
        ]

        print(f"Found {len(image_files)} images in {source_dir}")

        # Extract embeddings
        embeddings = self.extract_multiple_embeddings(image_files, enforce_detection=False)

        # Save to file
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"Database saved to {output_file}")

        return embeddings
