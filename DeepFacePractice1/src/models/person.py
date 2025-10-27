"""
Enhanced Person Model
Professional person entity with photo management and relationships.

Implements:
- Aggregate Root Pattern: Manages photo collection
- Repository Pattern Support: Designed for data persistence
- Rich Domain Model: Business logic and validation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Set

from .photo import Photo, PhotoStatus, PhotoQualityLevel


class PersonStatus(Enum):
    """Person status in the system."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_VALIDATION = "pending_validation"
    ARCHIVED = "archived"


@dataclass
class PersonStatistics:
    """Statistical information about a person's photos."""
    total_photos: int = 0
    validated_photos: int = 0
    problematic_photos: int = 0
    failed_photos: int = 0
    average_quality_score: float = 0.0
    highest_quality_score: float = 0.0
    lowest_quality_score: float = 0.0
    total_storage_mb: float = 0.0

    excellent_quality_count: int = 0
    good_quality_count: int = 0
    fair_quality_count: int = 0
    poor_quality_count: int = 0
    very_poor_quality_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_photos': self.total_photos,
            'validated_photos': self.validated_photos,
            'problematic_photos': self.problematic_photos,
            'failed_photos': self.failed_photos,
            'average_quality_score': round(self.average_quality_score, 2),
            'highest_quality_score': round(self.highest_quality_score, 2),
            'lowest_quality_score': round(self.lowest_quality_score, 2),
            'total_storage_mb': round(self.total_storage_mb, 2),
            'quality_distribution': {
                'excellent': self.excellent_quality_count,
                'good': self.good_quality_count,
                'fair': self.fair_quality_count,
                'poor': self.poor_quality_count,
                'very_poor': self.very_poor_quality_count,
            }
        }


@dataclass
class Person:
    """
    Enhanced Person entity with comprehensive photo management.

    Aggregate Root: Manages collection of Photos.
    Follows Hexagonal Architecture principles.
    """

    # Core Identity
    person_id: int  # Unique numeric ID
    folder_name: str  # Standardized folder name (e.g., "person_0001")

    # Personal Information
    name: Optional[str] = None
    alternative_names: List[str] = field(default_factory=list)
    notes: str = ""

    # System Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None
    status: PersonStatus = PersonStatus.ACTIVE

    # Photo Management
    photos: List[Photo] = field(default_factory=list)
    _photos_by_id: Dict[str, Photo] = field(default_factory=dict, repr=False)

    # Tags & Categorization
    tags: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)

    # Statistics (cached)
    statistics: Optional[PersonStatistics] = None

    # Custom Metadata
    custom_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields."""
        self._rebuild_photo_index()
        if self.statistics is None:
            self.refresh_statistics()

    # ========================================
    # Photo Management
    # ========================================

    def add_photo(self, photo: Photo) -> None:
        """
        Add a photo to this person.

        Args:
            photo: Photo instance to add
        """
        if photo.photo_id in self._photos_by_id:
            raise ValueError(f"Photo {photo.photo_id} already exists for this person")

        # Set ownership
        photo.person_id = self.person_id
        photo.person_folder = self.folder_name

        self.photos.append(photo)
        self._photos_by_id[photo.photo_id] = photo
        self.updated_at = datetime.now()
        self.refresh_statistics()

    def remove_photo(self, photo_id: str) -> bool:
        """
        Remove a photo from this person.

        Args:
            photo_id: Photo ID to remove

        Returns:
            True if removed, False if not found
        """
        if photo_id not in self._photos_by_id:
            return False

        photo = self._photos_by_id.pop(photo_id)
        self.photos.remove(photo)
        self.updated_at = datetime.now()
        self.refresh_statistics()
        return True

    def get_photo(self, photo_id: str) -> Optional[Photo]:
        """
        Get a specific photo by ID.

        Args:
            photo_id: Photo ID

        Returns:
            Photo instance or None
        """
        return self._photos_by_id.get(photo_id)

    def get_all_photos(self) -> List[Photo]:
        """Get all photos."""
        return self.photos.copy()

    def get_photos_by_status(self, status: PhotoStatus) -> List[Photo]:
        """Get photos filtered by status."""
        return [p for p in self.photos if p.status == status]

    def get_validated_photos(self) -> List[Photo]:
        """Get only validated photos."""
        return self.get_photos_by_status(PhotoStatus.VALIDATED)

    def get_problematic_photos(self) -> List[Photo]:
        """Get problematic photos."""
        return self.get_photos_by_status(PhotoStatus.PROBLEMATIC)

    def get_high_quality_photos(self, min_score: float = 70.0) -> List[Photo]:
        """Get photos above quality threshold."""
        return [
            p for p in self.photos
            if p.quality_metrics and p.quality_metrics.overall_score >= min_score
        ]

    def get_best_quality_photo(self) -> Optional[Photo]:
        """Get the highest quality photo."""
        validated = [p for p in self.photos if p.quality_metrics]
        if not validated:
            return None
        return max(validated, key=lambda p: p.quality_metrics.overall_score)

    def get_photo_paths(self) -> List[str]:
        """Get all photo file paths as strings."""
        return [str(p.file_path) for p in self.photos]

    def _rebuild_photo_index(self) -> None:
        """Rebuild internal photo index."""
        self._photos_by_id = {p.photo_id: p for p in self.photos}

    # ========================================
    # Statistics & Metrics
    # ========================================

    def refresh_statistics(self) -> PersonStatistics:
        """
        Recalculate statistics from photos.

        Returns:
            Updated PersonStatistics
        """
        stats = PersonStatistics()
        stats.total_photos = len(self.photos)

        if not self.photos:
            self.statistics = stats
            return stats

        quality_scores = []

        for photo in self.photos:
            # Count by status
            if photo.status == PhotoStatus.VALIDATED:
                stats.validated_photos += 1
            elif photo.status == PhotoStatus.PROBLEMATIC:
                stats.problematic_photos += 1
            elif photo.status == PhotoStatus.FAILED:
                stats.failed_photos += 1

            # Storage
            stats.total_storage_mb += photo.get_file_size_mb()

            # Quality metrics
            if photo.quality_metrics:
                score = photo.quality_metrics.overall_score
                quality_scores.append(score)

                # Count by quality level
                level = photo.quality_metrics.quality_level
                if level == PhotoQualityLevel.EXCELLENT:
                    stats.excellent_quality_count += 1
                elif level == PhotoQualityLevel.GOOD:
                    stats.good_quality_count += 1
                elif level == PhotoQualityLevel.FAIR:
                    stats.fair_quality_count += 1
                elif level == PhotoQualityLevel.POOR:
                    stats.poor_quality_count += 1
                elif level == PhotoQualityLevel.VERY_POOR:
                    stats.very_poor_quality_count += 1

        if quality_scores:
            stats.average_quality_score = sum(quality_scores) / len(quality_scores)
            stats.highest_quality_score = max(quality_scores)
            stats.lowest_quality_score = min(quality_scores)

        self.statistics = stats
        return stats

    def get_quality_summary(self) -> str:
        """Get human-readable quality summary."""
        if not self.statistics or self.statistics.total_photos == 0:
            return "No photos"

        s = self.statistics
        return (
            f"{s.validated_photos}/{s.total_photos} validated, "
            f"avg quality: {s.average_quality_score:.1f}%"
        )

    # ========================================
    # Tag & Category Management
    # ========================================

    def add_tag(self, tag: str) -> None:
        """Add a tag."""
        self.tags.add(tag)
        self.updated_at = datetime.now()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag."""
        self.tags.discard(tag)
        self.updated_at = datetime.now()

    def has_tag(self, tag: str) -> bool:
        """Check if person has a tag."""
        return tag in self.tags

    def add_category(self, category: str) -> None:
        """Add a category."""
        self.categories.add(category)
        self.updated_at = datetime.now()

    def remove_category(self, category: str) -> None:
        """Remove a category."""
        self.categories.discard(category)
        self.updated_at = datetime.now()

    # ========================================
    # Validation & Status
    # ========================================

    def mark_as_validated(self) -> None:
        """Mark person as validated."""
        self.status = PersonStatus.ACTIVE
        self.last_validated_at = datetime.now()
        self.updated_at = datetime.now()

    def needs_validation(self) -> bool:
        """Check if person needs validation."""
        if self.status == PersonStatus.PENDING_VALIDATION:
            return True

        # Check if has enough high-quality photos
        high_quality = self.get_high_quality_photos(min_score=70.0)
        return len(high_quality) < 3

    def get_validation_status(self) -> Dict[str, any]:
        """
        Get detailed validation status.

        Returns:
            Dictionary with validation information
        """
        high_quality = self.get_high_quality_photos(min_score=70.0)
        problematic = self.get_problematic_photos()

        return {
            'status': self.status.value,
            'total_photos': len(self.photos),
            'high_quality_photos': len(high_quality),
            'problematic_photos': len(problematic),
            'needs_validation': self.needs_validation(),
            'recommended_actions': self._get_recommended_actions(),
        }

    def _get_recommended_actions(self) -> List[str]:
        """Get recommended actions for this person."""
        actions = []

        if len(self.photos) == 0:
            actions.append("Add photos for this person")
        elif len(self.photos) < 3:
            actions.append("Add more photos (recommended: 5+ photos)")

        high_quality = self.get_high_quality_photos(min_score=70.0)
        if len(high_quality) < 3:
            actions.append("Add more high-quality photos")

        problematic = self.get_problematic_photos()
        if len(problematic) > 0:
            actions.append(f"Review and fix {len(problematic)} problematic photos")

        return actions

    # ========================================
    # Serialization
    # ========================================

    def to_dict(self, include_photos: bool = False) -> Dict:
        """
        Convert to dictionary for serialization.

        Args:
            include_photos: Whether to include full photo details

        Returns:
            Dictionary representation
        """
        data = {
            'person_id': self.person_id,
            'folder_name': self.folder_name,
            'name': self.name,
            'alternative_names': self.alternative_names,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_validated_at': self.last_validated_at.isoformat() if self.last_validated_at else None,
            'status': self.status.value,
            'tags': list(self.tags),
            'categories': list(self.categories),
            'statistics': self.statistics.to_dict() if self.statistics else None,
            'photo_count': len(self.photos),
            'quality_summary': self.get_quality_summary(),
            'validation_status': self.get_validation_status(),
            'custom_metadata': self.custom_metadata,
        }

        if include_photos:
            data['photos'] = [p.to_dict() for p in self.photos]
        else:
            data['photo_ids'] = [p.photo_id for p in self.photos]

        return data

    def to_summary(self) -> str:
        """Get a one-line summary."""
        name_str = self.name if self.name else f"Person {self.person_id}"
        return (
            f"{name_str} ({self.folder_name}): "
            f"{len(self.photos)} photos, "
            f"{self.get_quality_summary()}"
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        name = self.name or f"Person {self.person_id}"
        return f"Person(id={self.person_id}, name='{name}', photos={len(self.photos)})"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Person(person_id={self.person_id}, folder_name='{self.folder_name}', photo_count={len(self.photos)})"
