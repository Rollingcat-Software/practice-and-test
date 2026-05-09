import numpy as np

from app.infrastructure.ml.liveness.face_quality_illumination_gate import (
    FaceQualityIlluminationGate,
)


def _face_frame(*, dark_face: bool = False, shadow_left: bool = False) -> np.ndarray:
    frame = np.full((160, 160, 3), (92, 126, 168), dtype=np.uint8)
    frame[20:140, 20:140] = (122, 162, 201)
    frame[30:130:5, 30:130:6] = (138, 176, 214)
    frame[56:74, 50:66] = (35, 50, 70)
    frame[56:74, 94:110] = (35, 50, 70)
    frame[72:100, 76:84] = (80, 105, 130)
    frame[104:118, 56:104] = (44, 58, 84)
    frame[86:134:6, 34:126:7] = (150, 187, 223)
    if dark_face:
        frame[20:140, 20:140] = np.clip(frame[20:140, 20:140] * 0.28, 0, 255).astype(np.uint8)
    if shadow_left:
        frame[20:140, 20:80] = np.clip(frame[20:140, 20:80] * 0.38, 0, 255).astype(np.uint8)
    return frame


def test_illumination_gate_accepts_well_lit_face():
    gate = FaceQualityIlluminationGate()

    result = gate.evaluate(frame_bgr=_face_frame(), face_bounding_box=(20, 20, 120, 120))

    assert result.quality_ok is True
    assert result.quality_status == "OK"


def test_illumination_gate_marks_dark_face_as_low_quality():
    gate = FaceQualityIlluminationGate()

    result = gate.evaluate(frame_bgr=_face_frame(dark_face=True), face_bounding_box=(20, 20, 120, 120))

    assert result.quality_ok is False
    assert result.quality_status == "LOW_QUALITY"
    assert result.quality_reason == "poor_face_illumination"
    assert result.global_face_brightness < 62.0


def test_illumination_gate_marks_shadow_as_low_quality():
    gate = FaceQualityIlluminationGate()

    result = gate.evaluate(frame_bgr=_face_frame(shadow_left=True), face_bounding_box=(20, 20, 120, 120))

    assert result.quality_ok is False
    assert result.quality_status == "LOW_QUALITY"
    assert result.quality_reason in {"uneven_face_lighting", "poor_face_illumination"}
    assert result.shadow_asymmetry > 0.0
