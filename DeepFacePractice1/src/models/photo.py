"""
Photo Model
Comprehensive photo entity with quality metrics and metadata.

Implements:
- Entity Pattern: Core domain model
- Value Object Pattern: Immutable quality metrics
- Rich Domain Model: Business logic embedded
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List


class PhotoQualityLevel(Enum):
    """Photo quality classification."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # 30-49%
    VERY_POOR = "very_poor"  # 0-29%


class PhotoStatus(Enum):
    """Photo processing status."""
    PENDING = "pending"  # Not yet processed
    PROCESSING = "processing"  # Currently being processed
    VALIDATED = "validated"  # Passed quality checks
    PROBLEMATIC = "problematic"  # Quality issues detected
    FAILED = "failed"  # Processing failed
    ARCHIVED = "archived"  # Moved to archive


@dataclass
class QualityMetrics:
    """
    Immutable quality metrics for a photo.

    Value Object Pattern: Represents quality characteristics.
    """
    overall_score: float  # 0-100
    sharpness_score: float  # 0-100
    brightness_score: float  # 0-100
    contrast_score: float  # 0-100
    face_detection_confidence: float  # 0-100
    face_size_score: float  # 0-100 (based on face size in image)
    pose_quality_score: float  # 0-100 (frontal = 100, profile = lower)
    occlusion_score: float  # 0-100 (no occlusion = 100)

    # Derived properties
    quality_level: PhotoQualityLevel = field(init=False)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate quality level based on overall score."""
        if self.overall_score >= 90:
            object.__setattr__(self, 'quality_level', PhotoQualityLevel.EXCELLENT)
        elif self.overall_score >= 70:
            object.__setattr__(self, 'quality_level', PhotoQualityLevel.GOOD)
        elif self.overall_score >= 50:
            object.__setattr__(self, 'quality_level', PhotoQualityLevel.FAIR)
        elif self.overall_score >= 30:
            object.__setattr__(self, 'quality_level', PhotoQualityLevel.POOR)
        else:
            object.__setattr__(self, 'quality_level', PhotoQualityLevel.VERY_POOR)

    def is_acceptable(self, min_threshold: float = 50.0) -> bool:
        """Check if quality meets minimum threshold."""
        return self.overall_score >= min_threshold

    def get_problematic_metrics(self, threshold: float = 60.0) -> Dict[str, float]:
        """Get metrics that are below threshold."""
        problematic = {}

        metrics = {
            'sharpness': self.sharpness_score,
            'brightness': self.brightness_score,
            'contrast': self.contrast_score,
            'face_detection': self.face_detection_confidence,
            'face_size': self.face_size_score,
            'pose_quality': self.pose_quality_score,
            'occlusion': self.occlusion_score,
        }

        for name, value in metrics.items():
            if value < threshold:
                problematic[name] = value

        return problematic

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'overall_score': round(self.overall_score, 2),
            'sharpness_score': round(self.sharpness_score, 2),
            'brightness_score': round(self.brightness_score, 2),
            'contrast_score': round(self.contrast_score, 2),
            'face_detection_confidence': round(self.face_detection_confidence, 2),
            'face_size_score': round(self.face_size_score, 2),
            'pose_quality_score': round(self.pose_quality_score, 2),
            'occlusion_score': round(self.occlusion_score, 2),
            'quality_level': self.quality_level.value,
            'issues': self.issues,
            'warnings': self.warnings,
        }


@dataclass
class Photo:
    """
    Comprehensive photo entity with rich metadata.

    Entity Pattern: Has unique identity (photo_id) and mutable state.
    Aggregate Root: Manages its quality metrics and validation state.
    """

    # Identity
    photo_id: str  # Unique identifier (e.g., "person_0001_img_003")
    file_path: Path

    # Basic Metadata
    filename: str
    file_size_bytes: int
    file_extension: str
    created_at: datetime
    modified_at: datetime
    added_to_system_at: datetime

    # Image Properties
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    color_mode: str  # RGB, RGBA, L (grayscale), etc.

    # Ownership & Organization
    person_id: Optional[int] = None
    person_folder: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Quality & Validation
    quality_metrics: Optional[QualityMetrics] = None
    status: PhotoStatus = PhotoStatus.PENDING
    validation_date: Optional[datetime] = None

    # Technical Details
    file_hash: Optional[str] = None  # MD5 hash for duplicate detection
    exif_data: Dict = field(default_factory=dict)

    # Processing History
    processing_notes: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    # Custom Metadata
    custom_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.file_path, str):
            object.__setattr__(self, 'file_path', Path(self.file_path))

    @classmethod
    def from_file_path(
            cls,
            file_path: Path,
            photo_id: Optional[str] = None,
            person_id: Optional[int] = None,
            person_folder: Optional[str] = None
    ) -> 'Photo':
        """
        Factory method to create Photo from file path.

        Args:
            file_path: Path to image file
            photo_id: Optional custom photo ID
            person_id: Optional person ID
            person_folder: Optional person folder name

        Returns:
            Photo instance
        """
        from PIL import Image

        file_path = Path(file_path)

        # Generate photo ID if not provided
        if photo_id is None:
            if person_folder and file_path.name:
                photo_id = f"{person_folder}_{file_path.stem}"
            else:
                photo_id = file_path.stem

        # Get file stats
        stat = file_path.stat()

        # Load image to get dimensions
        with Image.open(file_path) as img:
            width, height = img.size
            color_mode = img.mode

        aspect_ratio = width / height if height > 0 else 0
        megapixels = (width * height) / 1_000_000

        # Calculate file hash
        file_hash = cls._calculate_file_hash(file_path)

        return cls(
            photo_id=photo_id,
            file_path=file_path,
            filename=file_path.name,
            file_size_bytes=stat.st_size,
            file_extension=file_path.suffix.lower(),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            added_to_system_at=datetime.now(),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            color_mode=color_mode,
            person_id=person_id,
            person_folder=person_folder,
            file_hash=file_hash,
        )

    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def mark_as_validated(self, quality_metrics: QualityMetrics) -> None:
        """Mark photo as validated with quality metrics."""
        self.quality_metrics = quality_metrics
        self.validation_date = datetime.now()

        if quality_metrics.is_acceptable():
            self.status = PhotoStatus.VALIDATED
        else:
            self.status = PhotoStatus.PROBLEMATIC
            self.add_processing_note(
                f"Quality below threshold: {quality_metrics.overall_score:.1f}%"
            )

    def mark_as_problematic(self, reason: str) -> None:
        """Mark photo as problematic."""
        self.status = PhotoStatus.PROBLEMATIC
        self.add_processing_note(f"Problematic: {reason}")

    def mark_as_failed(self, error: str) -> None:
        """Mark photo as failed."""
        self.status = PhotoStatus.FAILED
        self.error_messages.append(error)
        self.add_processing_note(f"Failed: {error}")

    def add_processing_note(self, note: str) -> None:
        """Add a processing note with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.processing_notes.append(f"[{timestamp}] {note}")

    def add_tag(self, tag: str) -> None:
        """Add a tag to the photo."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the photo."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_quality_issues(self) -> bool:
        """Check if photo has quality issues."""
        if self.quality_metrics is None:
            return False
        return not self.quality_metrics.is_acceptable()

    def get_quality_summary(self) -> str:
        """Get human-readable quality summary."""
        if self.quality_metrics is None:
            return "Not validated"

        qm = self.quality_metrics
        summary = f"{qm.quality_level.value.upper()} ({qm.overall_score:.1f}%)"

        problematic = qm.get_problematic_metrics()
        if problematic:
            issues = ", ".join([f"{k}: {v:.1f}%" for k, v in problematic.items()])
            summary += f" | Issues: {issues}"

        return summary

    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    def is_high_resolution(self, min_megapixels: float = 2.0) -> bool:
        """Check if photo is high resolution."""
        return self.megapixels >= min_megapixels

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'photo_id': self.photo_id,
            'file_path': str(self.file_path),
            'filename': self.filename,
            'file_size_mb': round(self.get_file_size_mb(), 2),
            'file_extension': self.file_extension,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'added_to_system_at': self.added_to_system_at.isoformat(),
            'dimensions': f"{self.width}x{self.height}",
            'aspect_ratio': round(self.aspect_ratio, 2),
            'megapixels': round(self.megapixels, 2),
            'color_mode': self.color_mode,
            'person_id': self.person_id,
            'person_folder': self.person_folder,
            'tags': self.tags,
            'quality_metrics': self.quality_metrics.to_dict() if self.quality_metrics else None,
            'status': self.status.value,
            'validation_date': self.validation_date.isoformat() if self.validation_date else None,
            'file_hash': self.file_hash,
            'processing_notes': self.processing_notes,
            'error_messages': self.error_messages,
            'custom_metadata': self.custom_metadata,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        quality = self.get_quality_summary()
        return (
            f"Photo(id={self.photo_id}, "
            f"{self.width}x{self.height}, "
            f"{self.get_file_size_mb():.1f}MB, "
            f"quality={quality})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Photo(photo_id='{self.photo_id}', "
            f"file_path='{self.file_path}', "
            f"status={self.status.value})"
        )
