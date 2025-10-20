"""
Interactive Carousel Viewer
Creates an interactive HTML viewer for browsing face verification results.

Features:
- Carousel navigation (Next/Previous/First/Last)
- Clickable list to jump to any comparison
- Full comparison details (distance, threshold, confidence)
- Image metadata (filename, size, dimensions, person info)
- Statistics and progress tracking
"""

import json
import os
from pathlib import Path
from typing import List

from PIL import Image


class InteractiveViewer:
    """Interactive HTML viewer for face verification results."""

    def __init__(self, output_dir: str = "output"):
        """Initialize viewer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_image_details(self, img_path: str) -> dict:
        """Extract detailed information about an image."""
        try:
            img = Image.open(img_path)
            file_size = os.path.getsize(img_path) / 1024  # KB

            return {
                "filename": Path(img_path).name,
                "person_folder": Path(img_path).parent.name,
                "full_path": str(Path(img_path).absolute()),
                "dimensions": f"{img.width} x {img.height}",
                "size_kb": f"{file_size:.1f} KB",
                "format": img.format or "Unknown",
                "mode": img.mode
            }
        except Exception as e:
            return {
                "filename": Path(img_path).name,
                "person_folder": Path(img_path).parent.name,
                "error": str(e)
            }

    def create_interactive_viewer(
            self,
            verification_results: List,
            output_file: str = "interactive_viewer.html"
    ):
        """
        Create interactive carousel viewer for all verification results.

        Args:
            verification_results: List of (person1, person2, result) tuples
            output_file: Output HTML filename
        """
        if not verification_results:
            print("[!] No verification results to display")
            return

        # Prepare data for each comparison
        comparisons_data = []

        for idx, (person1, person2, result) in enumerate(verification_results):
            img1_details = self.get_image_details(result.img1_path)
            img2_details = self.get_image_details(result.img2_path)

            # Convert paths to relative for HTML
            rel_img1 = os.path.relpath(result.img1_path, self.output_dir.parent)
            rel_img2 = os.path.relpath(result.img2_path, self.output_dir.parent)

            comparison = {
                "id": idx,
                "person1": {
                    "id": person1.person_id,
                    "folder": person1.folder_name,
                    "image_path": rel_img1,
                    "details": img1_details
                },
                "person2": {
                    "id": person2.person_id,
                    "folder": person2.folder_name,
                    "image_path": rel_img2,
                    "details": img2_details
                },
                "result": {
                    "verified": result.verified,
                    "distance": result.distance,
                    "threshold": result.threshold,
                    "confidence": result.confidence_percentage,
                    "model": result.model,
                    "detector": result.detector_backend,
                    "metric": result.similarity_metric
                },
                "comparison_type": "Same Person" if person1.person_id == person2.person_id else "Different People"
            }
            comparisons_data.append(comparison)

        # Generate HTML
        html_content = self._generate_html(comparisons_data)

        # Save HTML file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[OK] Created interactive viewer: {output_path}")
        print(f"    Open in browser: file://{output_path.absolute()}")

    def _generate_html(self, comparisons_data: List[dict]) -> str:
        """Generate complete HTML content."""

        # Convert data to JSON for JavaScript
        comparisons_json = json.dumps(comparisons_data, indent=2)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Verification Interactive Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .stats {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 0;
            min-height: 700px;
        }}

        /* Sidebar - List */
        .sidebar {{
            background: #f8f9fa;
            border-right: 2px solid #e0e0e0;
            overflow-y: auto;
            max-height: 700px;
        }}

        .sidebar-header {{
            padding: 20px;
            background: #667eea;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .comparison-list {{
            list-style: none;
        }}

        .comparison-item {{
            padding: 15px 20px;
            border-bottom: 1px solid #e0e0e0;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .comparison-item:hover {{
            background: #e3f2fd;
        }}

        .comparison-item.active {{
            background: #667eea;
            color: white;
        }}

        .comparison-item .item-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .comparison-item .item-subtitle {{
            font-size: 0.85em;
            opacity: 0.8;
        }}

        .comparison-item .match-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-top: 5px;
        }}

        .match-badge.match {{
            background: #4caf50;
            color: white;
        }}

        .match-badge.different {{
            background: #f44336;
            color: white;
        }}

        /* Viewer Area */
        .viewer-area {{
            padding: 30px;
        }}

        .controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}

        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}

        .nav-buttons button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}

        .nav-buttons button:hover:not(:disabled) {{
            background: #5568d3;
            transform: translateY(-2px);
        }}

        .nav-buttons button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}

        .progress-info {{
            font-size: 1.1em;
            font-weight: bold;
            color: #667eea;
        }}

        .comparison-viewer {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .image-panel {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .image-panel h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .image-panel img {{
            width: 100%;
            height: 400px;
            object-fit: contain;
            border-radius: 10px;
            background: white;
            padding: 10px;
        }}

        .image-details {{
            margin-top: 15px;
            background: white;
            padding: 15px;
            border-radius: 8px;
        }}

        .detail-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}

        .detail-row:last-child {{
            border-bottom: none;
        }}

        .detail-label {{
            font-weight: bold;
            color: #666;
        }}

        .detail-value {{
            color: #333;
        }}

        .verification-results {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}

        .verification-results h2 {{
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        .result-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .result-card {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}

        .result-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}

        .result-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}

        .result-status {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            margin-top: 20px;
        }}

        .result-status .status-icon {{
            font-size: 4em;
            margin-bottom: 10px;
        }}

        .result-status .status-text {{
            font-size: 1.5em;
            font-weight: bold;
        }}

        .technical-details {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}

        .tech-detail {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}

        .tech-detail .tech-label {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}

        .tech-detail .tech-value {{
            font-size: 1.1em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Face Verification Interactive Viewer</h1>
            <div class="stats">
                <span id="total-comparisons">{len(comparisons_data)}</span> Total Comparisons
            </div>
        </div>

        <div class="main-content">
            <!-- Sidebar: List of comparisons -->
            <div class="sidebar">
                <div class="sidebar-header">
                    📋 Comparison List
                </div>
                <ul class="comparison-list" id="comparison-list">
                    <!-- Populated by JavaScript -->
                </ul>
            </div>

            <!-- Main Viewer Area -->
            <div class="viewer-area">
                <!-- Navigation Controls -->
                <div class="controls">
                    <div class="nav-buttons">
                        <button id="btn-first" onclick="navigateFirst()">⏮ First</button>
                        <button id="btn-prev" onclick="navigatePrev()">◀ Previous</button>
                        <button id="btn-next" onclick="navigateNext()">Next ▶</button>
                        <button id="btn-last" onclick="navigateLast()">Last ⏭</button>
                    </div>
                    <div class="progress-info">
                        <span id="current-index">1</span> / <span id="total-count">{len(comparisons_data)}</span>
                    </div>
                </div>

                <!-- Verification Results Summary -->
                <div class="verification-results">
                    <h2>Verification Results</h2>
                    <div class="result-grid">
                        <div class="result-card">
                            <div class="label">Comparison Type</div>
                            <div class="value" id="comparison-type">-</div>
                        </div>
                        <div class="result-card">
                            <div class="label">Distance</div>
                            <div class="value" id="distance">-</div>
                        </div>
                        <div class="result-card">
                            <div class="label">Threshold</div>
                            <div class="value" id="threshold">-</div>
                        </div>
                        <div class="result-card">
                            <div class="label">Confidence</div>
                            <div class="value" id="confidence">-</div>
                        </div>
                    </div>

                    <div class="result-status">
                        <div class="status-icon" id="status-icon">-</div>
                        <div class="status-text" id="status-text">-</div>
                    </div>

                    <div class="technical-details">
                        <div class="tech-detail">
                            <div class="tech-label">Model</div>
                            <div class="tech-value" id="model">-</div>
                        </div>
                        <div class="tech-detail">
                            <div class="tech-label">Detector</div>
                            <div class="tech-value" id="detector">-</div>
                        </div>
                        <div class="tech-detail">
                            <div class="tech-label">Metric</div>
                            <div class="tech-value" id="metric">-</div>
                        </div>
                    </div>
                </div>

                <!-- Image Comparison -->
                <div class="comparison-viewer">
                    <!-- Image 1 -->
                    <div class="image-panel">
                        <h3 id="person1-title">Person 1</h3>
                        <img id="img1" src="" alt="Image 1">
                        <div class="image-details" id="img1-details">
                            <!-- Populated by JavaScript -->
                        </div>
                    </div>

                    <!-- Image 2 -->
                    <div class="image-panel">
                        <h3 id="person2-title">Person 2</h3>
                        <img id="img2" src="" alt="Image 2">
                        <div class="image-details" id="img2-details">
                            <!-- Populated by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Comparisons data
        const comparisons = {comparisons_json};
        let currentIndex = 0;

        // Initialize
        function init() {{
            populateList();
            displayComparison(0);
        }}

        // Populate sidebar list
        function populateList() {{
            const list = document.getElementById('comparison-list');
            list.innerHTML = '';

            comparisons.forEach((comp, index) => {{
                const li = document.createElement('li');
                li.className = 'comparison-item';
                if (index === 0) li.classList.add('active');
                li.onclick = () => displayComparison(index);

                const matchClass = comp.result.verified ? 'match' : 'different';
                const matchText = comp.result.verified ? 'MATCH' : 'DIFFERENT';

                li.innerHTML = `
                    <div class="item-title">#${{index + 1}}</div>
                    <div class="item-subtitle">
                        ${{comp.person1.folder}} vs ${{comp.person2.folder}}
                    </div>
                    <span class="match-badge ${{matchClass}}">${{matchText}}</span>
                `;

                list.appendChild(li);
            }});
        }}

        // Display specific comparison
        function displayComparison(index) {{
            if (index < 0 || index >= comparisons.length) return;

            currentIndex = index;
            const comp = comparisons[index];

            // Update active item in list
            document.querySelectorAll('.comparison-item').forEach((item, i) => {{
                item.classList.toggle('active', i === index);
            }});

            // Scroll active item into view
            const activeItem = document.querySelectorAll('.comparison-item')[index];
            if (activeItem) {{
                activeItem.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}

            // Update progress
            document.getElementById('current-index').textContent = index + 1;

            // Update navigation buttons
            document.getElementById('btn-first').disabled = index === 0;
            document.getElementById('btn-prev').disabled = index === 0;
            document.getElementById('btn-next').disabled = index === comparisons.length - 1;
            document.getElementById('btn-last').disabled = index === comparisons.length - 1;

            // Update result summary
            document.getElementById('comparison-type').textContent = comp.comparison_type;
            document.getElementById('distance').textContent = comp.result.distance.toFixed(4);
            document.getElementById('threshold').textContent = comp.result.threshold.toFixed(4);
            document.getElementById('confidence').textContent = comp.result.confidence.toFixed(1) + '%';

            // Update status
            const statusIcon = comp.result.verified ? '✅' : '❌';
            const statusText = comp.result.verified ? 'MATCH - Same Person' : 'DIFFERENT - Not Same Person';
            document.getElementById('status-icon').textContent = statusIcon;
            document.getElementById('status-text').textContent = statusText;

            // Update technical details
            document.getElementById('model').textContent = comp.result.model;
            document.getElementById('detector').textContent = comp.result.detector;
            document.getElementById('metric').textContent = comp.result.metric;

            // Update images
            document.getElementById('person1-title').textContent = comp.person1.folder;
            document.getElementById('person2-title').textContent = comp.person2.folder;
            document.getElementById('img1').src = '../' + comp.person1.image_path;
            document.getElementById('img2').src = '../' + comp.person2.image_path;

            // Update image details
            updateImageDetails('img1-details', comp.person1.details);
            updateImageDetails('img2-details', comp.person2.details);
        }}

        // Update image details panel
        function updateImageDetails(elementId, details) {{
            const container = document.getElementById(elementId);
            container.innerHTML = `
                <div class="detail-row">
                    <span class="detail-label">Filename:</span>
                    <span class="detail-value">${{details.filename}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Person:</span>
                    <span class="detail-value">${{details.person_folder}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Dimensions:</span>
                    <span class="detail-value">${{details.dimensions || 'N/A'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Size:</span>
                    <span class="detail-value">${{details.size_kb || 'N/A'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Format:</span>
                    <span class="detail-value">${{details.format || 'N/A'}}</span>
                </div>
            `;
        }}

        // Navigation functions
        function navigateFirst() {{ displayComparison(0); }}
        function navigatePrev() {{ displayComparison(currentIndex - 1); }}
        function navigateNext() {{ displayComparison(currentIndex + 1); }}
        function navigateLast() {{ displayComparison(comparisons.length - 1); }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') navigatePrev();
            if (e.key === 'ArrowRight') navigateNext();
            if (e.key === 'Home') navigateFirst();
            if (e.key === 'End') navigateLast();
        }});

        // Initialize on load
        window.onload = init;
    </script>
</body>
</html>
"""
        return html
