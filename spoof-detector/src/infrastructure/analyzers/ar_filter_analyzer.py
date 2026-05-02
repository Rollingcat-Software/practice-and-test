"""AR filter detection analyzer.

Detects live augmented reality overlays (Snapchat, Instagram, TikTok,
FaceApp, OBS plugins) applied to real faces.

This is the NOVEL academic contribution — no existing FAS benchmark
isolates AR-filter attacks. AR filters preserve pulse (rPPG), motion
(temporal), and show no screen artifacts (moire), so existing anti-spoof
models classify them as "real."

Phase 5 implementation plan:
1. Collect AR-filter dataset via amispoof.com (500+ clips per filter type)
2. Train MobileNetV3-Small on face crops (224x224, binary AR/not-AR)
3. Export to ONNX (~5MB) for CPU deployment
4. Integrate into this analyzer

Until the model is trained, this analyzer uses heuristic signals:
- Spatial-frequency discontinuity at filter boundaries
- Unusual color saturation patterns (AR filters boost colors)
- Landmark stability anomalies (filters smooth micro-movements)
"""

import time
import logging

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)


class ARFilterAnalyzer:
    """AR filter detection using heuristics + optional ONNX model.

    Current implementation (heuristic): analyzes color saturation
    distribution and edge coherence at face boundary. When a trained
    ONNX model is available, loads it for deep classification.

    Score:
    - 0-30: Strong AR filter indicators
    - 30-60: Some anomalies (possible filter)
    - 60-100: Natural appearance (no filter detected)
    """

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._model_path = model_path
        if model_path:
            self._load_model(model_path)

    @property
    def name(self) -> str:
        return "ar_filter"

    def _load_model(self, path: str):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            logger.info(f"AR filter ONNX model loaded from {path}")
        except Exception as e:
            logger.warning(f"AR filter model load failed: {e}")

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        # If trained model is available, use it
        if self._model is not None:
            return self._analyze_with_model(face_crop, face_roi, start)

        # Otherwise use heuristic analysis
        return self._analyze_heuristic(face_crop, face_roi, start)

    def _analyze_with_model(self, crop: np.ndarray, roi: FaceROI, start: float) -> AnalyzerResult:
        """Deep classification using trained ONNX model."""
        try:
            # Resize to model input size
            input_img = cv2.resize(crop, (224, 224))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
            input_img = np.expand_dims(input_img, 0)  # Add batch dim

            input_name = self._model.get_inputs()[0].name
            output = self._model.run(None, {input_name: input_img})
            prob_real = float(output[0][0][0])  # Sigmoid output

            score = prob_real * 100.0  # Higher = more real (no filter)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                name=self.name,
                score=max(0.0, min(100.0, score)),
                details={"method": "model", "prob_real": round(prob_real, 4)},
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            return self._analyze_heuristic(crop, roi, start)

    def _analyze_heuristic(self, crop: np.ndarray, roi: FaceROI, start: float) -> AnalyzerResult:
        """Heuristic AR filter detection via color and edge analysis."""
        h, w = crop.shape[:2]
        if h < 30 or w < 30:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"error": "crop_too_small"},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)

        # Signal 1: Saturation uniformity
        # AR filters often boost saturation uniformly (skin smoothing)
        sat_std = float(np.std(saturation))
        sat_mean = float(np.mean(saturation))

        # Very high saturation + low std = artificial smoothing
        sat_anomaly = 0.0
        if sat_mean > 100 and sat_std < 25:
            sat_anomaly = min(1.0, (100 - sat_std) / 75.0)

        # Signal 2: Edge coherence at face boundary
        # AR filters create sharp edges at overlay boundaries
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check edge density in face border region (outer 20%)
        border_mask = np.zeros_like(edges)
        bw = max(1, w // 5)
        bh = max(1, h // 5)
        border_mask[:bh, :] = 1
        border_mask[-bh:, :] = 1
        border_mask[:, :bw] = 1
        border_mask[:, -bw:] = 1
        center_mask = 1 - border_mask

        border_edge_density = float(np.mean(edges[border_mask == 1])) / 255.0
        center_edge_density = float(np.mean(edges[center_mask == 1])) / 255.0

        # AR filters: high border edges (overlay boundary), smooth center
        edge_anomaly = 0.0
        if border_edge_density > 0.15 and center_edge_density < 0.08:
            edge_anomaly = min(1.0, (border_edge_density - center_edge_density) / 0.15)

        # Signal 3: Color temperature uniformity
        # Real skin has natural variation; AR filters homogenize
        b, g, r = cv2.split(crop)
        color_var = float(np.std([np.mean(b), np.mean(g), np.mean(r)]))
        color_anomaly = 0.0
        if color_var < 5.0:
            color_anomaly = min(1.0, (5.0 - color_var) / 5.0)

        # Combined anomaly score
        anomaly = 0.40 * sat_anomaly + 0.35 * edge_anomaly + 0.25 * color_anomaly

        # Convert to liveness score (higher = more natural)
        score = 100.0 * (1.0 - anomaly)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "method": "heuristic",
                "sat_anomaly": round(sat_anomaly, 3),
                "edge_anomaly": round(edge_anomaly, 3),
                "color_anomaly": round(color_anomaly, 3),
                "anomaly": round(anomaly, 3),
                "sat_mean": round(sat_mean, 1),
                "sat_std": round(sat_std, 1),
            },
            elapsed_ms=elapsed_ms,
        )
