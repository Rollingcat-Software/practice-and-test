"""
Advanced Visualization System
Generates comprehensive visual reports for face verification results.

Features:
- Individual comparison images
- Verification matrix heatmaps
- Complete HTML reports
- Grid views of all comparisons
"""

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class AdvancedVisualizer:
    """Advanced visualization for face verification results."""

    def __init__(self, output_dir: str = "output"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_verification_grid(
            self,
            results: List,
            persons,
            output_file: str = "verification_grid.png",
            max_per_row: int = 4
    ):
        """
        Create a grid showing all verification comparisons.

        Args:
            results: List of (person1, person2, verification_result) tuples
            persons: List of Person objects
            output_file: Output filename
            max_per_row: Maximum comparisons per row
        """
        if not results:
            print("[!] No results to visualize")
            return

        n_results = len(results)
        n_rows = (n_results + max_per_row - 1) // max_per_row
        n_cols = min(n_results, max_per_row)

        fig = plt.figure(figsize=(5 * n_cols, 6 * n_rows))
        fig.suptitle('Face Verification Results - All Comparisons',
                     fontsize=16, fontweight='bold')

        for idx, (person1, person2, result) in enumerate(results):
            ax = plt.subplot(n_rows, n_cols, idx + 1)

            # Load and display both images side by side
            try:
                img1 = Image.open(result.img1_path)
                img2 = Image.open(result.img2_path)

                # Resize for uniform display
                target_size = (200, 200)
                img1 = img1.resize(target_size)
                img2 = img2.resize(target_size)

                # Create combined image
                combined = Image.new('RGB', (400, 200))
                combined.paste(img1, (0, 0))
                combined.paste(img2, (200, 0))

                ax.imshow(combined)
                ax.axis('off')

                # Add verification info
                verified_text = "✓ MATCH" if result.verified else "✗ DIFFERENT"
                color = 'green' if result.verified else 'red'

                title = f"{person1.folder_name} vs {person2.folder_name}\n"
                title += f"{verified_text}\n"
                title += f"Distance: {result.distance:.4f}"

                ax.set_title(title, fontsize=10, fontweight='bold', color=color)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading images\n{e}",
                        ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved grid visualization: {output_path}")

    def create_verification_matrix_heatmap(
            self,
            persons,
            verification_service,
            output_file: str = "verification_heatmap.png"
    ):
        """
        Create a heatmap showing verification distances between all persons.

        Args:
            persons: List of Person objects
            verification_service: FaceVerificationService instance
            output_file: Output filename
        """
        n = len(persons)

        if n == 0:
            print("[!] No persons to visualize")
            return

        # Create distance matrix
        distances = np.zeros((n, n))

        for i, person1 in enumerate(persons):
            for j, person2 in enumerate(persons):
                if not person1.image_paths or not person2.image_paths:
                    distances[i, j] = np.nan
                    continue

                try:
                    result = verification_service.verify(
                        person1.image_paths[0],
                        person2.image_paths[0],
                        enforce_detection=False
                    )
                    distances[i, j] = result.distance
                except:
                    distances[i, j] = np.nan

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(distances, cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=1.0)

        # Set ticks
        person_labels = [p.folder_name for p in persons]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(person_labels, rotation=45, ha='right')
        ax.set_yticklabels(person_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Distance (Lower = More Similar)', rotation=270, labelpad=20)

        # Add distance values
        for i in range(n):
            for j in range(n):
                if not np.isnan(distances[i, j]):
                    text_color = 'white' if distances[i, j] > 0.5 else 'black'
                    text = ax.text(j, i, f'{distances[i, j]:.3f}',
                                   ha="center", va="center", color=text_color,
                                   fontsize=8)

        ax.set_title('Face Verification Distance Matrix\n' +
                     '(Diagonal = Same Person, Off-Diagonal = Different People)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved heatmap: {output_path}")

    def create_person_gallery(
            self,
            person,
            output_file: Optional[str] = None
    ):
        """
        Create a gallery showing all images for a specific person.

        Args:
            person: Person object
            output_file: Optional custom output filename
        """
        if not person.image_paths:
            print(f"[!] No images for {person.folder_name}")
            return

        n_images = len(person.image_paths)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        fig.suptitle(f'{person.folder_name} - All Images ({n_images} total)',
                     fontsize=16, fontweight='bold')

        # Handle single row/col case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, img_path in enumerate(person.image_paths):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(Path(img_path).name, fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center')

            ax.axis('off')

        # Hide empty subplots
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if output_file is None:
            output_file = f"gallery_{person.folder_name}.png"

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved gallery: {output_path}")

    def create_html_report(
            self,
            persons,
            verification_results,
            output_file: str = "verification_report.html"
    ):
        """
        Create a comprehensive HTML report with all verification results.

        Args:
            persons: List of Person objects
            verification_results: List of verification results
            output_file: Output HTML filename
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Face Verification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .person-section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .person-section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .image-item {{
            text-align: center;
        }}
        .image-item img {{
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .image-item .label {{
            margin-top: 5px;
            font-size: 0.9em;
            color: #666;
        }}
        .verification-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .verification-table th, .verification-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .verification-table th {{
            background-color: #667eea;
            color: white;
        }}
        .match {{
            color: green;
            font-weight: bold;
        }}
        .no-match {{
            color: red;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Face Verification Report</h1>
        <p>Comprehensive analysis of face verification results</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <h3>Total Persons</h3>
            <div class="value">{len(persons)}</div>
        </div>
        <div class="stat-card">
            <h3>Total Images</h3>
            <div class="value">{sum(p.image_count for p in persons)}</div>
        </div>
        <div class="stat-card">
            <h3>Verifications</h3>
            <div class="value">{len(verification_results)}</div>
        </div>
        <div class="stat-card">
            <h3>Avg Images/Person</h3>
            <div class="value">{sum(p.image_count for p in persons) / len(persons) if persons else 0:.1f}</div>
        </div>
    </div>
"""

        # Add person galleries
        for person in persons:
            html_content += f"""
    <div class="person-section">
        <h2>{person.folder_name} ({person.image_count} images)</h2>
        <div class="image-grid">
"""
            for img_path in person.image_paths:
                # Convert to relative path for HTML
                rel_path = os.path.relpath(img_path, self.output_dir.parent).replace('\\', '/')
                img_name = Path(img_path).name
                html_content += f"""
            <div class="image-item">
                <img src="../{rel_path}" alt="{img_name}">
                <div class="label">{img_name}</div>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

        # Add verification results table
        if verification_results:
            html_content += """
    <div class="person-section">
        <h2>Verification Results</h2>
        <table class="verification-table">
            <tr>
                <th>Person 1</th>
                <th>Person 2</th>
                <th>Result</th>
                <th>Distance</th>
                <th>Threshold</th>
                <th>Confidence</th>
            </tr>
"""
            for person1, person2, result in verification_results:
                match_class = "match" if result.verified else "no-match"
                match_text = "MATCH" if result.verified else "DIFFERENT"

                html_content += f"""
            <tr>
                <td>{person1.folder_name}</td>
                <td>{person2.folder_name}</td>
                <td class="{match_class}">{match_text}</td>
                <td>{result.distance:.4f}</td>
                <td>{result.threshold:.4f}</td>
                <td>{result.confidence_percentage:.1f}%</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""

        html_content += """
</body>
</html>
"""

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[OK] Saved HTML report: {output_path}")
        print(f"    Open in browser: file://{output_path.absolute()}")
