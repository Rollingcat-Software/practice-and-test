"""
Optimized Interactive Viewer
Professional, compact, user-friendly interface for face verification.

Design principles:
- Images are the focus (large, center)
- Navigation always visible (bottom, sticky)
- Key stats prominent but compact
- Details available but not overwhelming
- Everything visible on one screen
"""

import json
import os
from pathlib import Path
from typing import List

from PIL import Image


class OptimizedViewer:
    """Optimized interactive viewer with professional UI."""

    def __init__(self, output_dir: str = "output"):
        """Initialize viewer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_image_details(self, img_path: str) -> dict:
        """Extract image information."""
        try:
            img = Image.open(img_path)
            file_size = os.path.getsize(img_path) / 1024

            return {
                "filename": Path(img_path).name,
                "person_folder": Path(img_path).parent.name,
                "dimensions": f"{img.width}×{img.height}",
                "size_kb": f"{file_size:.0f}KB",
                "format": img.format or "Unknown"
            }
        except Exception as e:
            return {
                "filename": Path(img_path).name,
                "person_folder": Path(img_path).parent.name,
                "error": str(e)
            }

    def create_optimized_viewer(
            self,
            verification_results: List,
            output_file: str = "viewer.html"
    ):
        """Create optimized interactive viewer."""
        if not verification_results:
            print("[!] No verification results to display")
            return

        # Prepare comparison data
        comparisons_data = []
        for idx, (person1, person2, result) in enumerate(verification_results):
            img1_details = self.get_image_details(result.img1_path)
            img2_details = self.get_image_details(result.img2_path)

            rel_img1 = os.path.relpath(result.img1_path, self.output_dir.parent).replace('\\', '/')
            rel_img2 = os.path.relpath(result.img2_path, self.output_dir.parent).replace('\\', '/')

            comparisons_data.append({
                "id": idx,
                "person1": {
                    "folder": person1.folder_name,
                    "image_path": rel_img1,
                    "details": img1_details
                },
                "person2": {
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
                "type": "Same Person" if person1.person_id == person2.person_id else "Different People"
            })

        html_content = self._generate_html(comparisons_data)

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[OK] Created optimized viewer: {output_path}")
        print(f"    Open in browser: file://{output_path.absolute()}")

    def _generate_html(self, comparisons_data: List[dict]) -> str:
        """Generate optimized HTML."""
        comparisons_json = json.dumps(comparisons_data, indent=2)

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Verification Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1e;
            color: #fff;
            overflow: hidden;
            height: 100vh;
        }}

        .viewer-container {{
            display: grid;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
        }}

        /* TOP BAR */
        .top-bar {{
            background: rgba(15, 15, 30, 0.95);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            border-bottom: 2px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }}

        .title {{
            font-size: 1.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .stats-bar {{
            display: flex;
            gap: 30px;
            font-size: 0.95em;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .stat-label {{
            opacity: 0.7;
        }}

        .stat-value {{
            font-weight: bold;
            color: #667eea;
        }}

        /* MAIN CONTENT */
        .main-content {{
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 0;
            overflow: hidden;
        }}

        /* SIDEBAR */
        .sidebar {{
            background: rgba(20, 20, 35, 0.8);
            border-right: 1px solid rgba(102, 126, 234, 0.3);
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #667eea #0f0f1e;
        }}

        .sidebar::-webkit-scrollbar {{ width: 6px; }}
        .sidebar::-webkit-scrollbar-track {{ background: #0f0f1e; }}
        .sidebar::-webkit-scrollbar-thumb {{ background: #667eea; border-radius: 3px; }}

        .list-item {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            cursor: pointer;
            transition: all 0.2s;
        }}

        .list-item:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}

        .list-item.active {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-left: 4px solid #fff;
        }}

        .list-item-num {{
            font-size: 0.85em;
            opacity: 0.7;
            margin-bottom: 4px;
        }}

        .list-item-persons {{
            font-size: 0.9em;
            font-weight: 500;
        }}

        .list-item-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            margin-top: 4px;
            font-weight: bold;
        }}

        .badge-match {{ background: #10b981; }}
        .badge-diff {{ background: #ef4444; }}

        /* HAMBURGER MENU */
        .hamburger {{
            display: none;
            flex-direction: column;
            gap: 4px;
            cursor: pointer;
            padding: 8px;
            background: rgba(102, 126, 234, 0.2);
            border-radius: 6px;
            transition: all 0.3s;
        }}

        .hamburger:hover {{
            background: rgba(102, 126, 234, 0.4);
        }}

        .hamburger span {{
            width: 24px;
            height: 3px;
            background: #667eea;
            border-radius: 2px;
            transition: all 0.3s;
        }}

        .hamburger.active span:nth-child(1) {{
            transform: rotate(45deg) translate(5px, 5px);
        }}

        .hamburger.active span:nth-child(2) {{
            opacity: 0;
        }}

        .hamburger.active span:nth-child(3) {{
            transform: rotate(-45deg) translate(7px, -6px);
        }}

        /* COMPARISON VIEW */
        .comparison-view {{
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 20px;
            overflow-y: auto;
        }}

        /* STATUS BANNER */
        .status-banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 25px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        }}

        .status-main {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .status-icon {{
            font-size: 2.5em;
        }}

        .status-text {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .status-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.95em;
        }}

        .status-stats div {{
            text-align: center;
        }}

        .status-stats .label {{
            opacity: 0.8;
            font-size: 0.85em;
            margin-bottom: 2px;
        }}

        .status-stats .value {{
            font-weight: bold;
            font-size: 1.2em;
        }}

        /* IMAGES COMPARISON */
        .images-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .image-card {{
            background: rgba(20, 20, 35, 0.6);
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(102, 126, 234, 0.3);
            transition: all 0.3s;
        }}

        .image-card:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}

        .image-header {{
            padding: 12px 15px;
            background: rgba(102, 126, 234, 0.2);
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .image-header .person-name {{
            font-size: 1.1em;
        }}

        .image-header .image-name {{
            font-size: 0.85em;
            opacity: 0.7;
        }}

        .image-container {{
            position: relative;
            padding: 15px;
            background: #000;
        }}

        .image-container img {{
            width: 100%;
            height: 350px;
            object-fit: contain;
            border-radius: 8px;
        }}

        .image-details {{
            padding: 10px 15px;
            display: flex;
            gap: 15px;
            font-size: 0.85em;
            flex-wrap: wrap;
        }}

        .detail-chip {{
            background: rgba(102, 126, 234, 0.2);
            padding: 4px 10px;
            border-radius: 12px;
        }}

        /* TECHNICAL INFO */
        .tech-info {{
            background: rgba(20, 20, 35, 0.6);
            border-radius: 12px;
            padding: 15px 20px;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }}

        .tech-info-title {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 10px;
        }}

        .tech-info-content {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .tech-item {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .tech-label {{
            opacity: 0.7;
            font-size: 0.9em;
        }}

        .tech-value {{
            font-weight: 600;
            color: #667eea;
        }}

        /* BOTTOM NAV */
        .bottom-nav {{
            background: rgba(15, 15, 30, 0.95);
            backdrop-filter: blur(10px);
            border-top: 2px solid #667eea;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.3);
        }}

        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}

        .nav-btn {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .nav-btn:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        }}

        .nav-btn:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}

        .progress {{
            font-size: 1.1em;
            font-weight: bold;
        }}

        .progress .current {{
            color: #667eea;
            font-size: 1.3em;
        }}

        /* RESPONSIVE DESIGN */

        /* Tablet (portrait) - 1024px and below */
        @media (max-width: 1024px) {{
            .top-bar {{
                padding: 12px 20px;
            }}

            .title {{
                font-size: 1.2em;
            }}

            .stats-bar {{
                gap: 15px;
                font-size: 0.85em;
            }}

            .main-content {{
                grid-template-columns: 220px 1fr;
            }}

            .image-container img {{
                height: 280px;
            }}

            .status-icon {{
                font-size: 2em;
            }}

            .status-text {{
                font-size: 1.1em;
            }}
        }}

        /* Tablet (portrait) and small laptops - 768px and below */
        @media (max-width: 768px) {{
            .top-bar {{
                padding: 10px 15px;
            }}

            .title {{
                font-size: 1em;
            }}

            .hamburger {{
                display: flex;
            }}

            .stats-bar {{
                gap: 10px;
                font-size: 0.8em;
            }}

            .stat-item {{
                gap: 4px;
            }}

            /* Collapsible sidebar */
            .main-content {{
                grid-template-columns: 1fr;
            }}

            .sidebar {{
                position: fixed;
                left: -280px;
                top: 0;
                bottom: 0;
                width: 280px;
                z-index: 1000;
                transition: left 0.3s ease;
                box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
            }}

            .sidebar.open {{
                left: 0;
            }}

            .sidebar-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.7);
                z-index: 999;
            }}

            .sidebar-overlay.show {{
                display: block;
            }}

            .comparison-view {{
                padding: 15px;
                gap: 15px;
            }}

            .status-banner {{
                flex-direction: column;
                gap: 15px;
                padding: 15px;
            }}

            .status-stats {{
                width: 100%;
                justify-content: space-around;
            }}

            .images-comparison {{
                grid-template-columns: 1fr;
                gap: 15px;
            }}

            .image-container img {{
                height: 250px;
            }}

            .bottom-nav {{
                padding: 10px 15px;
                flex-direction: column;
                gap: 10px;
            }}

            .nav-buttons {{
                width: 100%;
                justify-content: space-between;
            }}

            .nav-btn {{
                padding: 8px 12px;
                font-size: 0.85em;
                flex: 1;
            }}

            .progress {{
                font-size: 1em;
            }}

            .progress .current {{
                font-size: 1.1em;
            }}
        }}

        /* Mobile - 480px and below */
        @media (max-width: 480px) {{
            .top-bar {{
                flex-direction: column;
                gap: 8px;
                align-items: flex-start;
            }}

            .title {{
                font-size: 0.95em;
                width: 100%;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .stats-bar {{
                width: 100%;
                justify-content: space-between;
                font-size: 0.75em;
            }}

            .comparison-view {{
                padding: 10px;
                gap: 12px;
            }}

            .status-banner {{
                padding: 12px;
            }}

            .status-icon {{
                font-size: 1.8em;
            }}

            .status-text {{
                font-size: 0.95em;
            }}

            .status-stats {{
                gap: 10px;
                font-size: 0.8em;
            }}

            .status-stats .label {{
                font-size: 0.75em;
            }}

            .status-stats .value {{
                font-size: 1em;
            }}

            .image-header .person-name {{
                font-size: 0.95em;
            }}

            .image-header .image-name {{
                font-size: 0.75em;
            }}

            .image-container img {{
                height: 200px;
            }}

            .image-details {{
                font-size: 0.75em;
                gap: 8px;
            }}

            .tech-info {{
                padding: 12px;
            }}

            .tech-info-content {{
                gap: 12px;
                font-size: 0.85em;
            }}

            .nav-btn {{
                padding: 10px 8px;
                font-size: 0.75em;
            }}

            .nav-btn span {{
                display: none;
            }}

            .bottom-nav {{
                padding: 8px 10px;
            }}
        }}

        /* Very small mobile - 360px and below */
        @media (max-width: 360px) {{
            .image-container img {{
                height: 150px;
            }}

            .status-stats {{
                flex-direction: column;
                gap: 8px;
                align-items: stretch;
            }}

            .status-stats div {{
                display: flex;
                justify-content: space-between;
                padding: 5px 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
            }}

            .nav-buttons {{
                gap: 5px;
            }}

            .nav-btn {{
                padding: 8px 5px;
                min-width: 0;
            }}
        }}

        /* Touch device optimizations */
        @media (hover: none) and (pointer: coarse) {{
            .nav-btn {{
                min-height: 44px;
                min-width: 44px;
            }}

            .list-item {{
                padding: 15px;
            }}

            .hamburger {{
                min-height: 44px;
                min-width: 44px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
        }}

        /* Landscape mode on mobile */
        @media (max-width: 768px) and (orientation: landscape) {{
            .viewer-container {{
                grid-template-rows: auto 1fr auto;
            }}

            .comparison-view {{
                overflow-y: auto;
            }}

            .images-comparison {{
                grid-template-columns: 1fr 1fr;
            }}

            .image-container img {{
                height: 180px;
            }}

            .status-banner {{
                flex-direction: row;
            }}
        }}
    </style>
</head>
<body>
    <div class="viewer-container">
        <!-- TOP BAR -->
        <div class="top-bar">
            <div class="title">
                <span>Face Verification Viewer</span>
                <div class="hamburger" id="hamburger" onclick="toggleSidebar()">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-label">Total:</span>
                    <span class="stat-value">{len(comparisons_data)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Viewing:</span>
                    <span class="stat-value" id="viewing-type">-</span>
                </div>
            </div>
        </div>

        <!-- SIDEBAR OVERLAY (for mobile) -->
        <div class="sidebar-overlay" id="sidebar-overlay" onclick="closeSidebar()"></div>

        <!-- MAIN CONTENT -->
        <div class="main-content">
            <!-- SIDEBAR -->
            <div class="sidebar" id="sidebar">
                <!-- Populated by JS -->
            </div>

            <!-- COMPARISON VIEW -->
            <div class="comparison-view">
                <!-- STATUS BANNER -->
                <div class="status-banner">
                    <div class="status-main">
                        <div class="status-icon" id="status-icon">-</div>
                        <div class="status-text" id="status-text">-</div>
                    </div>
                    <div class="status-stats">
                        <div>
                            <div class="label">Distance</div>
                            <div class="value" id="distance-val">-</div>
                        </div>
                        <div>
                            <div class="label">Threshold</div>
                            <div class="value" id="threshold-val">-</div>
                        </div>
                        <div>
                            <div class="label">Confidence</div>
                            <div class="value" id="confidence-val">-</div>
                        </div>
                    </div>
                </div>

                <!-- IMAGES COMPARISON -->
                <div class="images-comparison">
                    <div class="image-card">
                        <div class="image-header">
                            <span class="person-name" id="person1-name">-</span>
                            <span class="image-name" id="img1-name">-</span>
                        </div>
                        <div class="image-container">
                            <img id="img1" src="" alt="Image 1">
                        </div>
                        <div class="image-details" id="img1-details"></div>
                    </div>

                    <div class="image-card">
                        <div class="image-header">
                            <span class="person-name" id="person2-name">-</span>
                            <span class="image-name" id="img2-name">-</span>
                        </div>
                        <div class="image-container">
                            <img id="img2" src="" alt="Image 2">
                        </div>
                        <div class="image-details" id="img2-details"></div>
                    </div>
                </div>

                <!-- TECHNICAL INFO -->
                <div class="tech-info">
                    <div class="tech-info-title">Technical Details</div>
                    <div class="tech-info-content">
                        <div class="tech-item">
                            <span class="tech-label">Model:</span>
                            <span class="tech-value" id="model">-</span>
                        </div>
                        <div class="tech-item">
                            <span class="tech-label">Detector:</span>
                            <span class="tech-value" id="detector">-</span>
                        </div>
                        <div class="tech-item">
                            <span class="tech-label">Metric:</span>
                            <span class="tech-value" id="metric">-</span>
                        </div>
                        <div class="tech-item">
                            <span class="tech-label">Type:</span>
                            <span class="tech-value" id="comp-type">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- BOTTOM NAV -->
        <div class="bottom-nav">
            <div class="nav-buttons">
                <button class="nav-btn" id="btn-first" onclick="navigateFirst()">
                    ⏮ First
                </button>
                <button class="nav-btn" id="btn-prev" onclick="navigatePrev()">
                    ◀ Prev
                </button>
                <button class="nav-btn" id="btn-next" onclick="navigateNext()">
                    Next ▶
                </button>
                <button class="nav-btn" id="btn-last" onclick="navigateLast()">
                    Last ⏭
                </button>
            </div>
            <div class="progress">
                <span class="current" id="current-num">1</span>
                <span>/ {len(comparisons_data)}</span>
            </div>
        </div>
    </div>

    <script>
        const comparisons = {comparisons_json};
        let currentIndex = 0;

        function init() {{
            populateList();
            displayComparison(0);
        }}

        function displayComparison(index) {{
            if (index < 0 || index >= comparisons.length) return;
            currentIndex = index;
            const comp = comparisons[index];

            // Update list
            document.querySelectorAll('.list-item').forEach((item, i) => {{
                item.classList.toggle('active', i === index);
            }});

            // Update navigation
            document.getElementById('btn-first').disabled = index === 0;
            document.getElementById('btn-prev').disabled = index === 0;
            document.getElementById('btn-next').disabled = index === comparisons.length - 1;
            document.getElementById('btn-last').disabled = index === comparisons.length - 1;
            document.getElementById('current-num').textContent = index + 1;

            // Update status
            const icon = comp.result.verified ? '✅' : '❌';
            const text = comp.result.verified ? 'MATCH' : 'DIFFERENT';
            document.getElementById('status-icon').textContent = icon;
            document.getElementById('status-text').textContent = text;
            document.getElementById('viewing-type').textContent = comp.type;

            // Update stats
            document.getElementById('distance-val').textContent = comp.result.distance.toFixed(4);
            document.getElementById('threshold-val').textContent = comp.result.threshold.toFixed(4);
            document.getElementById('confidence-val').textContent = comp.result.confidence.toFixed(1) + '%';

            // Update images
            document.getElementById('person1-name').textContent = comp.person1.folder;
            document.getElementById('person2-name').textContent = comp.person2.folder;
            document.getElementById('img1-name').textContent = comp.person1.details.filename;
            document.getElementById('img2-name').textContent = comp.person2.details.filename;
            document.getElementById('img1').src = '../' + comp.person1.image_path;
            document.getElementById('img2').src = '../' + comp.person2.image_path;

            // Update details
            document.getElementById('img1-details').innerHTML = `
                <div class="detail-chip">${{comp.person1.details.dimensions}}</div>
                <div class="detail-chip">${{comp.person1.details.size_kb}}</div>
                <div class="detail-chip">${{comp.person1.details.format}}</div>
            `;
            document.getElementById('img2-details').innerHTML = `
                <div class="detail-chip">${{comp.person2.details.dimensions}}</div>
                <div class="detail-chip">${{comp.person2.details.size_kb}}</div>
                <div class="detail-chip">${{comp.person2.details.format}}</div>
            `;

            // Update technical info
            document.getElementById('model').textContent = comp.result.model;
            document.getElementById('detector').textContent = comp.result.detector;
            document.getElementById('metric').textContent = comp.result.metric;
            document.getElementById('comp-type').textContent = comp.type;

            // Scroll active item into view
            document.querySelectorAll('.list-item')[index].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }}

        function navigateFirst() {{ displayComparison(0); }}
        function navigatePrev() {{ displayComparison(currentIndex - 1); }}
        function navigateNext() {{ displayComparison(currentIndex + 1); }}
        function navigateLast() {{ displayComparison(comparisons.length - 1); }}

        // Sidebar toggle functions
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            const hamburger = document.getElementById('hamburger');

            sidebar.classList.toggle('open');
            overlay.classList.toggle('show');
            hamburger.classList.toggle('active');
        }}

        function closeSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            const hamburger = document.getElementById('hamburger');

            sidebar.classList.remove('open');
            overlay.classList.remove('show');
            hamburger.classList.remove('active');
        }}

        // Close sidebar when item is clicked on mobile
        function handleListItemClick(index) {{
            displayComparison(index);
            if (window.innerWidth <= 768) {{
                closeSidebar();
            }}
        }}

        // Touch swipe gesture support
        let touchStartX = 0;
        let touchEndX = 0;
        let touchStartY = 0;
        let touchEndY = 0;

        function handleSwipeGesture() {{
            const swipeThreshold = 50;
            const verticalSwipeThreshold = 100;

            const horizontalDiff = touchEndX - touchStartX;
            const verticalDiff = Math.abs(touchEndY - touchStartY);

            // Only process horizontal swipes (ignore vertical scrolling)
            if (verticalDiff < verticalSwipeThreshold) {{
                if (horizontalDiff > swipeThreshold) {{
                    // Swipe right - go to previous
                    navigatePrev();
                }} else if (horizontalDiff < -swipeThreshold) {{
                    // Swipe left - go to next
                    navigateNext();
                }}
            }}
        }}

        // Add touch event listeners
        const comparisonView = document.querySelector('.comparison-view');
        comparisonView.addEventListener('touchstart', (e) => {{
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        }}, {{ passive: true }});

        comparisonView.addEventListener('touchend', (e) => {{
            touchEndX = e.changedTouches[0].screenX;
            touchEndY = e.changedTouches[0].screenY;
            handleSwipeGesture();
        }}, {{ passive: true }});

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') navigatePrev();
            if (e.key === 'ArrowRight') navigateNext();
            if (e.key === 'Home') navigateFirst();
            if (e.key === 'End') navigateLast();
        }});

        // Update list item click handler
        function populateList() {{
            const sidebar = document.getElementById('sidebar');
            comparisons.forEach((comp, idx) => {{
                const div = document.createElement('div');
                div.className = 'list-item';
                if (idx === 0) div.classList.add('active');
                div.onclick = () => handleListItemClick(idx);

                const badgeClass = comp.result.verified ? 'badge-match' : 'badge-diff';
                const badgeText = comp.result.verified ? 'MATCH' : 'DIFF';

                div.innerHTML = `
                    <div class="list-item-num">#${{idx + 1}}</div>
                    <div class="list-item-persons">${{comp.person1.folder}} vs ${{comp.person2.folder}}</div>
                    <span class="list-item-badge ${{badgeClass}}">${{badgeText}}</span>
                `;
                sidebar.appendChild(div);
            }});
        }}

        window.onload = init;
    </script>
</body>
</html>
"""
