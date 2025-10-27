"""
Image Inspection Tool
Visualize and analyze problematic images with quality overlays.

Implements:
- Service Layer Pattern: Visualization service
- Strategy Pattern: Multiple visualization strategies
- Single Responsibility: Only image inspection/visualization
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..models.person import Person
from ..models.photo import Photo, QualityMetrics, PhotoQualityLevel
from ..utils.logger import setup_logger

logger = setup_logger("ImageInspectionTool")


@dataclass
class InspectionReport:
    """Report from image inspection."""
    total_images: int
    problematic_images: int
    quality_distribution: Dict[str, int]
    issues_summary: Dict[str, int]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_images': self.total_images,
            'problematic_images': self.problematic_images,
            'quality_distribution': self.quality_distribution,
            'issues_summary': self.issues_summary,
            'recommendations': self.recommendations,
        }


class ImageInspectionTool:
    """
    Tool for visualizing and analyzing image quality issues.

    Provides:
    - Visual quality overlays
    - Comparison visualizations
    - Quality reports
    - Problem detection
    """

    def __init__(self, output_dir: str = "output/inspection"):
        """
        Initialize inspection tool.

        Args:
            output_dir: Directory for output visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def inspect_photo(
            self,
            photo: Photo,
            show: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create detailed inspection visualization for a photo.

        Args:
            photo: Photo to inspect
            show: Whether to display the visualization
            save_path: Optional save path (auto-generated if None)
        """
        if photo.quality_metrics is None:
            logger.warning(f"Photo {photo.photo_id} has no quality metrics")
            return

        # Load image
        img = cv2.imread(str(photo.file_path))
        if img is None:
            logger.error(f"Could not load image: {photo.file_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"Image Inspection: {photo.filename}\n"
            f"Quality: {photo.quality_metrics.quality_level.value.upper()} "
            f"({photo.quality_metrics.overall_score:.1f}%)",
            fontsize=16,
            fontweight='bold'
        )

        # 1. Original image with quality overlay
        ax1 = axes[0, 0]
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Add quality badge
        self._add_quality_badge(ax1, photo.quality_metrics)

        # 2. Quality metrics radar chart
        ax2 = axes[0, 1]
        self._plot_quality_radar(ax2, photo.quality_metrics)

        # 3. Quality metrics bar chart
        ax3 = axes[1, 0]
        self._plot_quality_bars(ax3, photo.quality_metrics)

        # 4. Issues and metadata
        ax4 = axes[1, 1]
        self._plot_metadata_and_issues(ax4, photo)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"inspection_{photo.photo_id}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved inspection to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def inspect_person(
            self,
            person: Person,
            show: bool = False,
            save_path: Optional[str] = None
    ) -> InspectionReport:
        """
        Create inspection report for all photos of a person.

        Args:
            person: Person to inspect
            show: Whether to display the visualization
            save_path: Optional save path

        Returns:
            InspectionReport with findings
        """
        if not person.photos:
            logger.warning(f"Person {person.person_id} has no photos")
            return self._empty_report()

        # Filter photos with quality metrics
        photos_with_metrics = [p for p in person.photos if p.quality_metrics]

        if not photos_with_metrics:
            logger.warning(f"No photos with quality metrics for person {person.person_id}")
            return self._empty_report()

        # Create grid visualization
        n_photos = len(photos_with_metrics)
        cols = min(4, n_photos)
        rows = (n_photos + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"Person Inspection: {person.name or f'Person {person.person_id}'}\n"
            f"{person.get_quality_summary()}",
            fontsize=16,
            fontweight='bold'
        )

        # Plot each photo
        for idx, photo in enumerate(photos_with_metrics):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # Load and display image
            img = cv2.imread(str(photo.file_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)

                # Add title with quality info
                qm = photo.quality_metrics
                color = self._get_quality_color(qm.quality_level)
                ax.set_title(
                    f"{photo.filename}\n{qm.quality_level.value.upper()} ({qm.overall_score:.0f}%)",
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )

                # Add quality border
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
            else:
                ax.text(0.5, 0.5, f"Failed to load\n{photo.filename}",
                        ha='center', va='center')

            ax.axis('off')

        # Hide unused subplots
        for idx in range(n_photos, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"person_inspection_{person.person_id}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved person inspection to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # Generate report
        return self._generate_person_report(person)

    def compare_photos(
            self,
            photos: List[Photo],
            title: str = "Photo Comparison",
            show: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Compare multiple photos side-by-side with quality metrics.

        Args:
            photos: List of photos to compare
            title: Comparison title
            show: Whether to display
            save_path: Optional save path
        """
        n_photos = len(photos)
        if n_photos == 0:
            logger.warning("No photos to compare")
            return

        fig, axes = plt.subplots(2, n_photos, figsize=(n_photos * 4, 8))
        if n_photos == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(title, fontsize=16, fontweight='bold')

        for idx, photo in enumerate(photos):
            # Top row: images
            ax_img = axes[0, idx]
            img = cv2.imread(str(photo.file_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_img.imshow(img_rgb)

                if photo.quality_metrics:
                    qm = photo.quality_metrics
                    color = self._get_quality_color(qm.quality_level)
                    ax_img.set_title(
                        f"{photo.filename}\n{qm.overall_score:.1f}%",
                        color=color,
                        fontweight='bold'
                    )
                else:
                    ax_img.set_title(photo.filename)

            ax_img.axis('off')

            # Bottom row: quality bars
            ax_bar = axes[1, idx]
            if photo.quality_metrics:
                self._plot_quality_bars(ax_bar, photo.quality_metrics, compact=True)
            else:
                ax_bar.text(0.5, 0.5, "No metrics", ha='center', va='center')
                ax_bar.axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "photo_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _add_quality_badge(self, ax, quality_metrics: QualityMetrics) -> None:
        """Add quality badge to image."""
        qm = quality_metrics
        color = self._get_quality_color(qm.quality_level)

        # Add text box
        text = f"{qm.quality_level.value.upper()}\n{qm.overall_score:.0f}%"
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, edgecolor='white', linewidth=2),
            color='white'
        )

    def _plot_quality_radar(self, ax, quality_metrics: QualityMetrics) -> None:
        """Plot quality metrics as radar chart."""
        qm = quality_metrics

        categories = ['Sharpness', 'Brightness', 'Contrast', 'Face\nDetection',
                      'Face\nSize', 'Pose', 'Occlusion']
        values = [
            qm.sharpness_score,
            qm.brightness_score,
            qm.contrast_score,
            qm.face_detection_confidence,
            qm.face_size_score,
            qm.pose_quality_score,
            qm.occlusion_score,
        ]

        # Number of variables
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        ax = plt.subplot(2, 2, 2, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], size=8)
        ax.set_title("Quality Metrics Radar", fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    def _plot_quality_bars(
            self,
            ax,
            quality_metrics: QualityMetrics,
            compact: bool = False
    ) -> None:
        """Plot quality metrics as bar chart."""
        qm = quality_metrics

        if compact:
            categories = ['Sharp', 'Bright', 'Contr', 'Face', 'Size', 'Pose', 'Occl']
        else:
            categories = ['Sharpness', 'Brightness', 'Contrast', 'Face Det.',
                          'Face Size', 'Pose', 'Occlusion']

        values = [
            qm.sharpness_score,
            qm.brightness_score,
            qm.contrast_score,
            qm.face_detection_confidence,
            qm.face_size_score,
            qm.pose_quality_score,
            qm.occlusion_score,
        ]

        colors = [self._get_score_color(v) for v in values]

        bars = ax.barh(categories, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (%)', fontsize=10)

        if not compact:
            ax.set_title("Quality Scores", fontsize=12, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                    f'{value:.0f}%',
                    ha='left', va='center', fontsize=8, fontweight='bold')

        ax.grid(axis='x', alpha=0.3)

    def _plot_metadata_and_issues(self, ax, photo: Photo) -> None:
        """Plot metadata and issues."""
        ax.axis('off')

        text_content = []

        # Metadata
        text_content.append("=== METADATA ===")
        text_content.append(f"Dimensions: {photo.width} x {photo.height}")
        text_content.append(f"Size: {photo.get_file_size_mb():.2f} MB")
        text_content.append(f"Megapixels: {photo.megapixels:.1f} MP")
        text_content.append(f"Status: {photo.status.value}")
        text_content.append("")

        # Quality info
        if photo.quality_metrics:
            qm = photo.quality_metrics
            text_content.append("=== QUALITY ===")
            text_content.append(f"Overall: {qm.overall_score:.1f}%")
            text_content.append(f"Level: {qm.quality_level.value.upper()}")
            text_content.append("")

            # Issues
            if qm.issues:
                text_content.append("=== ISSUES ===")
                for issue in qm.issues:
                    text_content.append(f"• {issue}")
                text_content.append("")

            # Warnings
            if qm.warnings:
                text_content.append("=== WARNINGS ===")
                for warning in qm.warnings:
                    text_content.append(f"• {warning}")

        # Display text
        text = "\n".join(text_content)
        ax.text(0.05, 0.95, text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    def _get_quality_color(self, level: PhotoQualityLevel) -> str:
        """Get color for quality level."""
        colors = {
            PhotoQualityLevel.EXCELLENT: '#28a745',  # Green
            PhotoQualityLevel.GOOD: '#5cb85c',  # Light green
            PhotoQualityLevel.FAIR: '#ffc107',  # Yellow
            PhotoQualityLevel.POOR: '#fd7e14',  # Orange
            PhotoQualityLevel.VERY_POOR: '#dc3545',  # Red
        }
        return colors.get(level, '#6c757d')

    def _get_score_color(self, score: float) -> str:
        """Get color for quality score."""
        if score >= 90:
            return '#28a745'
        elif score >= 70:
            return '#5cb85c'
        elif score >= 50:
            return '#ffc107'
        elif score >= 30:
            return '#fd7e14'
        else:
            return '#dc3545'

    def _generate_person_report(self, person: Person) -> InspectionReport:
        """Generate inspection report for person."""
        photos_with_metrics = [p for p in person.photos if p.quality_metrics]

        if not photos_with_metrics:
            return self._empty_report()

        # Quality distribution
        quality_dist = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0,
            'very_poor': 0,
        }

        # Issues summary
        issues_summary = {}
        problematic_count = 0

        for photo in photos_with_metrics:
            qm = photo.quality_metrics

            # Count quality levels
            quality_dist[qm.quality_level.value] += 1

            # Count problematic
            if not qm.is_acceptable():
                problematic_count += 1

            # Count issues
            for issue in qm.issues:
                # Extract issue type
                issue_type = issue.split(':')[0]
                issues_summary[issue_type] = issues_summary.get(issue_type, 0) + 1

        # Generate recommendations
        recommendations = []
        if problematic_count > 0:
            recommendations.append(f"Review {problematic_count} problematic photos")

        if quality_dist['poor'] + quality_dist['very_poor'] > 0:
            recommendations.append("Replace low-quality photos with better ones")

        if len(photos_with_metrics) < 5:
            recommendations.append("Add more photos (recommended: 5-10 per person)")

        return InspectionReport(
            total_images=len(photos_with_metrics),
            problematic_images=problematic_count,
            quality_distribution=quality_dist,
            issues_summary=issues_summary,
            recommendations=recommendations,
        )

    def _empty_report(self) -> InspectionReport:
        """Create empty report."""
        return InspectionReport(
            total_images=0,
            problematic_images=0,
            quality_distribution={},
            issues_summary={},
            recommendations=["Add photos for inspection"],
        )
