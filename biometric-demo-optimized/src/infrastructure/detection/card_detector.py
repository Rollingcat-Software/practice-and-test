"""YOLO-based ID card detection with temporal smoothing."""

import logging
from typing import Dict, Optional
from collections import deque

import cv2
import numpy as np

from .face_detector import get_resource_path

logger = logging.getLogger(__name__)


class CardDetector:
    """YOLO card detector with lazy loading and temporal smoothing."""

    CARD_LABELS = {
        'tc_kimlik': 'Turkish ID',
        'ehliyet': 'License',
        'pasaport': 'Passport',
        'ogrenci_karti': 'Student',
        'akademisyen_karti': 'Academic',
    }

    def __init__(self):
        self._model = None
        self._available: Optional[bool] = None
        self._detection_history: deque = deque(maxlen=5)
        self._last_stable_result: Dict = {'detected': False}

    def _load(self):
        """Lazy load YOLO model."""
        if self._model is not None:
            return self._model

        import os

        # Search paths for the card model
        search_paths = [
            get_resource_path("app/core/card_type_model/best.pt"),
            "app/core/card_type_model/best.pt",
            # Look in biometric-processor sibling directory
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "biometric-processor", "app", "core", "card_type_model", "best.pt"),
            # Absolute fallback to common project location
            os.path.normpath(os.path.join(os.getcwd(), "..", "biometric-processor", "app", "core", "card_type_model", "best.pt")),
        ]

        model_path = None
        for path in search_paths:
            if path and os.path.exists(path):
                model_path = os.path.abspath(path)
                break

        if not model_path:
            self._available = False
            logger.info("Card model not found in: " + ", ".join([p for p in search_paths if p]))
            return None

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._available = True
            logger.info(f"YOLO card model loaded from {model_path}")
            return self._model
        except Exception as e:
            logger.warning(f"Card detection unavailable: {e}")
            self._available = False
            return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better detection in varying lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _get_stable_result(self, current_result: Dict) -> Dict:
        """Apply temporal smoothing to reduce flickering."""
        self._detection_history.append(current_result)

        if len(self._detection_history) < 3:
            return current_result

        recent_detections = [r for r in self._detection_history if r.get('detected')]

        if len(recent_detections) >= 2:
            class_counts = {}
            for r in recent_detections:
                cls = r.get('class', '')
                if cls not in class_counts:
                    class_counts[cls] = {'count': 0, 'total_conf': 0, 'last_result': r}
                class_counts[cls]['count'] += 1
                class_counts[cls]['total_conf'] += r.get('confidence', 0)
                class_counts[cls]['last_result'] = r

            best_class = max(class_counts.items(), key=lambda x: (x[1]['count'], x[1]['total_conf']))
            self._last_stable_result = best_class[1]['last_result']
            return self._last_stable_result
        elif len(recent_detections) == 0:
            self._last_stable_result = {'detected': False}
            return self._last_stable_result
        else:
            return self._last_stable_result

    def detect(self, frame: np.ndarray, use_smoothing: bool = True) -> Dict:
        """Detect card in frame.

        Args:
            frame: BGR image
            use_smoothing: Apply temporal smoothing

        Returns:
            Dict with keys: detected, class, label, confidence, box
        """
        model = self._load()
        if model is None:
            return {'detected': False}

        try:
            processed = self._preprocess(frame)
            results = model(processed, conf=0.35, verbose=False, imgsz=640)

            if len(results[0].boxes) == 0:
                current = {'detected': False}
            else:
                best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                cls_id = int(best.cls[0])
                conf = float(best.conf[0])
                name = model.names[cls_id]
                box = best.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                current = {
                    'detected': True,
                    'class': name,
                    'label': self.CARD_LABELS.get(name, name),
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                }

            if use_smoothing:
                return self._get_stable_result(current)
            return current

        except Exception as e:
            logger.debug(f"Card detection error: {e}")
            return {'detected': False}

    def is_available(self) -> bool:
        """Check if card detection is available."""
        if self._available is None:
            self._load()
        return self._available or False
