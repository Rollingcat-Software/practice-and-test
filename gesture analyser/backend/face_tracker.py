"""
Yüz Takip Çekirdeği — MediaPipe FaceLandmarker
================================================
PythonProject1'de yüz takibi YOKTUR; bu modül web köprüsüne özgü, sıfırdan
yazılmış bir yüz takip katmanıdır (el tarafındaki hand_tracker.py'nin yüz
karşılığı). PythonProject1 kaynağına dokunmaz.

Her karede şunları üretir:
  • 478 yüz landmark'ı (normalize [0,1] koordinat)
  • 52 ARKit-tarzı blendshape skoru (eyeBlinkLeft, jawOpen, mouthSmileLeft, ...)
  • Landmark-tabanlı baş pozu sinyalleri: yaw_norm (sağa/sola çevirme),
    pitch_norm (yukarı/aşağı bakma) — ölçek/öteleme bağımsız oranlar.

Baş pozu için transformation-matrix yerine landmark oranları kullanıyoruz:
konvansiyon belirsizliği yok, işaretler sezgisel ve kalibrasyonu kolay.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

BaseOptions          = mp.tasks.BaseOptions
FaceLandmarker       = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode


# ── MediaPipe FaceMesh kanonik landmark indeksleri ───────────────────
_NOSE_TIP    = 1
_CHIN        = 152
_LEFT_EYE_O  = 33     # görüntüde sol göz dış köşesi
_RIGHT_EYE_O = 263    # görüntüde sağ göz dış köşesi
_LEFT_CHEEK  = 234
_RIGHT_CHEEK = 454
_FOREHEAD    = 10


@dataclass
class FaceResult:
    """Tek bir algılanan yüzün hafif konteyneri."""
    landmarks: list                      # 478 NormalizedLandmark
    blendshapes: dict[str, float]        # {category_name: score}
    yaw_norm: float = 0.0                # baş çevirme (sağ/sol) — Y ekseni, ~[-1,1]
    pitch_norm: float = 0.0             # baş eğme (yukarı/aşağı) — X ekseni, oran
    roll_deg: float = 0.0               # baş yana yatma — Z ekseni, derece
    face_size: float = 0.0              # yüz yüksekliği (derinlik proxy)
    present: bool = True


def _bs_to_dict(categories) -> dict[str, float]:
    return {c.category_name: float(c.score) for c in categories}


def _head_pose(lm) -> tuple[float, float, float, float]:
    """Landmark'lardan (yaw_norm, pitch_norm, roll_deg, face_size) hesaplar."""
    nose = lm[_NOSE_TIP]
    chin = lm[_CHIN]
    le   = lm[_LEFT_EYE_O]
    re   = lm[_RIGHT_EYE_O]
    lc   = lm[_LEFT_CHEEK]
    rc   = lm[_RIGHT_CHEEK]
    top  = lm[_FOREHEAD]

    half_w = abs(rc.x - lc.x) / 2.0
    cx     = (lc.x + rc.x) / 2.0
    # Yaw (Y ekseni): burun ucunun yatay merkez sapması
    yaw_norm = (nose.x - cx) / (half_w + 1e-6)

    # Pitch (X ekseni): burun ucunun göz-çene ekseni üzerindeki dikey konumu
    eye_mid_y = (le.y + re.y) / 2.0
    denom = (chin.y - eye_mid_y)
    pitch_norm = (nose.y - eye_mid_y) / (denom + 1e-6)

    # Roll (Z ekseni): göz hattının yatayla açısı (baş yana yatma)
    roll_deg = math.degrees(math.atan2(re.y - le.y, re.x - le.x))

    face_size = abs(chin.y - top.y)
    return yaw_norm, pitch_norm, roll_deg, face_size


class FaceTracker:
    """Tek yüz için MediaPipe FaceLandmarker sarmalayıcısı (VIDEO modu)."""

    def __init__(self, min_face_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def process(self, bgr_frame: np.ndarray) -> Optional[FaceResult]:
        """Önceden flip'lenmiş (aynalı) BGR kareyi işler. Yüz yoksa None."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        blends: dict[str, float] = {}
        if result.face_blendshapes:
            blends = _bs_to_dict(result.face_blendshapes[0])

        yaw, pitch, roll, fsize = _head_pose(lm)
        return FaceResult(
            landmarks=lm,
            blendshapes=blends,
            yaw_norm=yaw,
            pitch_norm=pitch,
            roll_deg=roll,
            face_size=fsize,
            present=True,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Çizim için landmark alt kümesi (HUD overlay) ─────────────────────
# 478 noktanın hepsini çizmek ağır; karakteristik kontur/özellik noktaları:
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS_OUTER = [61, 39, 0, 269, 291, 405, 17, 181]
LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
