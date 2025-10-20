"""
Visualization Utility
Creates visual outputs for face analysis results.

Educational Notes:
- Makes results easier to understand
- Good for presentations and reports
- Demonstrates using matplotlib for data visualization
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..models.face_analysis_result import FaceAnalysisResult
from ..models.face_embedding import FaceEmbedding
from ..models.verification_result import VerificationResult


class FaceVisualizer:
    """
    Handles visualization of face analysis results.

    Demonstrates:
    - SRP: Only responsible for visualization
    - Encapsulation: Hides matplotlib complexity
    """

    def __init__(
            self,
            figure_size: Tuple[int, int] = (12, 8),
            dpi: int = 100
    ):
        """
        Initialize visualizer.

        Args:
            figure_size: Default figure size (width, height) in inches
            dpi: Dots per inch for output images
        """
        self.figure_size = figure_size
        self.dpi = dpi

    def visualize_verification(
            self,
            result: VerificationResult,
            save_path: Optional[str] = None,
            show: bool = True
    ) -> None:
        """
        Visualize face verification result (comparing two faces).

        Args:
            result: VerificationResult object
            save_path: Where to save the visualization (None = don't save)
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)

        # Load images
        try:
            img1 = Image.open(result.img1_path)
            img2 = Image.open(result.img2_path)
        except Exception as e:
            print(f"Error loading images: {e}")
            return

        # Display images
        axes[0].imshow(img1)
        axes[0].set_title(f"Image 1\n{Path(result.img1_path).name}", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img2)
        axes[1].set_title(f"Image 2\n{Path(result.img2_path).name}", fontsize=10)
        axes[1].axis('off')

        # Add result text
        status = "✓ MATCH" if result.verified else "✗ NO MATCH"
        status_color = "green" if result.verified else "red"

        result_text = (
            f"{status}\n\n"
            f"Distance: {result.distance:.4f}\n"
            f"Threshold: {result.threshold:.4f}\n"
            f"Confidence: {result.confidence_percentage:.2f}%\n\n"
            f"Model: {result.model}\n"
            f"Detector: {result.detector_backend}\n"
            f"Metric: {result.similarity_metric}"
        )

        fig.suptitle(result_text, fontsize=12, color=status_color, weight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_analysis(
            self,
            result: FaceAnalysisResult,
            save_path: Optional[str] = None,
            show: bool = True
    ) -> None:
        """
        Visualize comprehensive face analysis result.

        Args:
            result: FaceAnalysisResult object
            save_path: Where to save the visualization
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=self.figure_size)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Load image
        try:
            img = Image.open(result.img_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # 1. Main image with face box
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(img)

        if result.region:
            rect = patches.Rectangle(
                (result.region['x'], result.region['y']),
                result.region['w'],
                result.region['h'],
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax_img.add_patch(rect)

        ax_img.set_title(f"Detected Face\n{Path(result.img_path).name}", fontsize=10)
        ax_img.axis('off')

        # 2. Demographics info
        ax_demo = fig.add_subplot(gs[0, 1])
        ax_demo.axis('off')

        demo_text = "Demographics\n" + "=" * 30 + "\n\n"
        if result.age is not None:
            demo_text += f"Age: ~{result.age:.0f} years\n\n"
        if result.gender:
            demo_text += f"Gender: {result.gender}\n"
            if result.gender_confidence:
                demo_text += f"Confidence: {result.gender_confidence:.1f}%\n\n"
        if result.dominant_race:
            demo_text += f"Ethnicity: {result.dominant_race}\n"

        ax_demo.text(0.1, 0.5, demo_text, fontsize=11, verticalalignment='center',
                     family='monospace')

        # 3. Emotion chart
        if result.emotion_scores:
            ax_emotion = fig.add_subplot(gs[1, 1])
            emotions = list(result.emotion_scores.keys())
            scores = list(result.emotion_scores.values())

            # Sort by score
            sorted_pairs = sorted(zip(emotions, scores), key=lambda x: x[1], reverse=True)
            emotions, scores = zip(*sorted_pairs)

            colors = ['green' if e == result.dominant_emotion else 'skyblue' for e in emotions]

            ax_emotion.barh(emotions, scores, color=colors)
            ax_emotion.set_xlabel('Confidence (%)')
            ax_emotion.set_title('Emotion Analysis', fontsize=10, weight='bold')
            ax_emotion.set_xlim(0, 100)

            # Add value labels
            for i, (emotion, score) in enumerate(zip(emotions, scores)):
                ax_emotion.text(score + 1, i, f'{score:.1f}%', va='center', fontsize=8)

        plt.suptitle(f"Face Analysis Results", fontsize=14, weight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_embedding_comparison(
            self,
            embeddings: Dict[str, FaceEmbedding],
            save_path: Optional[str] = None,
            show: bool = True
    ) -> None:
        """
        Visualize and compare multiple face embeddings.

        Args:
            embeddings: Dictionary of name -> FaceEmbedding
            save_path: Where to save the visualization
            show: Whether to display the plot
        """
        n_embeddings = len(embeddings)
        if n_embeddings < 2:
            print("Need at least 2 embeddings to compare")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 1. Embedding vectors visualization (first 100 dimensions)
        ax_vectors = axes[0]
        names = list(embeddings.keys())

        for i, (name, emb) in enumerate(embeddings.items()):
            vector = emb.as_numpy[:100]  # Plot first 100 dimensions
            ax_vectors.plot(vector, label=name, alpha=0.7, linewidth=2)

        ax_vectors.set_xlabel('Dimension')
        ax_vectors.set_ylabel('Value')
        ax_vectors.set_title('Embedding Vectors (first 100 dimensions)', fontsize=12, weight='bold')
        ax_vectors.legend()
        ax_vectors.grid(True, alpha=0.3)

        # 2. Similarity matrix
        ax_similarity = axes[1]

        # Calculate pairwise similarities
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i, emb1 in enumerate(embeddings.values()):
            for j, emb2 in enumerate(embeddings.values()):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = emb1.cosine_similarity(emb2)

        # Plot heatmap
        im = ax_similarity.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)

        # Labels
        ax_similarity.set_xticks(range(n))
        ax_similarity.set_yticks(range(n))
        ax_similarity.set_xticklabels(names, rotation=45, ha='right')
        ax_similarity.set_yticklabels(names)

        # Add values to cells
        for i in range(n):
            for j in range(n):
                text = ax_similarity.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                                          ha="center", va="center", color="black", fontsize=9)

        ax_similarity.set_title('Cosine Similarity Matrix\n(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)',
                                fontsize=12, weight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_similarity)
        cbar.set_label('Similarity', rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_emotion_timeline(
            self,
            results: List[Tuple[str, FaceAnalysisResult]],
            save_path: Optional[str] = None,
            show: bool = True
    ) -> None:
        """
        Plot emotion changes across multiple images (e.g., video frames).

        Args:
            results: List of (label, FaceAnalysisResult) tuples
            save_path: Where to save the visualization
            show: Whether to display the plot
        """
        if not results:
            print("No results to plot")
            return

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Extract data
        labels = [r[0] for r in results]
        emotions_data = {}

        # Collect all emotions
        for label, result in results:
            if result.emotion_scores:
                for emotion, score in result.emotion_scores.items():
                    if emotion not in emotions_data:
                        emotions_data[emotion] = []
                    emotions_data[emotion].append(score)

        # Plot each emotion
        x = range(len(labels))
        for emotion, scores in emotions_data.items():
            ax.plot(x, scores, marker='o', label=emotion, linewidth=2)

        ax.set_xlabel('Frame/Image')
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Emotion Analysis Timeline', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def display_image_grid(
            image_paths: List[str],
            titles: Optional[List[str]] = None,
            cols: int = 3,
            figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Display multiple images in a grid.

        Args:
            image_paths: List of image file paths
            titles: Optional titles for each image
            cols: Number of columns in the grid
            figsize: Figure size
        """
        n_images = len(image_paths)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)

                if titles and i < len(titles):
                    axes[i].set_title(titles[i], fontsize=10)
                else:
                    axes[i].set_title(Path(img_path).name, fontsize=10)

                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{Path(img_path).name}',
                             ha='center', va='center')
                axes[i].axis('off')

        # Hide extra subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
