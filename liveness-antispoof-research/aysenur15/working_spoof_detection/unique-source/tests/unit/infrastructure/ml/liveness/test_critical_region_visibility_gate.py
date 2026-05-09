import numpy as np

from app.infrastructure.ml.liveness.critical_region_visibility_gate import (
    CriticalRegionVisibilityGate,
    CriticalRegionVisibilityTracker,
    _classify_region_states,
    _is_critical_occluded,
)


def _face_frame(
    *,
    occlude_lower_face: bool = False,
    occlude_nose: bool = False,
    occlude_mouth: bool = False,
    occlude_eyes: bool = False,
    shadow_left: bool = False,
) -> np.ndarray:
    # face_roi = frame[20:140, 20:140] (120x120) for bbox=(20, 20, 120, 120)
    # Ratio regions → face_roi → frame coords:
    #   mouth      (0.28, 0.63, 0.44, 0.16) → roi[76:95,  34:87]  → frame[96:115, 54:107]
    #   nose       (0.39, 0.35, 0.22, 0.24) → roi[42:71,  47:73]  → frame[62:91,  67:93]
    #   left_eye   (0.14, 0.24, 0.24, 0.18) → roi[29:51,  17:46]  → frame[49:71,  37:66]
    #   right_eye  (0.62, 0.24, 0.24, 0.18) → roi[29:51,  74:103] → frame[49:71,  94:123]
    #   lower_face (0.18, 0.56, 0.64, 0.34) → roi[67:108, 22:99]  → frame[87:128, 42:119]
    #   left_cheek baseline (0.14, 0.41, 0.18, 0.18) → roi[49:71, 17:39] → frame[69:91, 37:59]
    #   right_cheek baseline (0.68, 0.41, 0.18, 0.18) → roi[49:71, 82:104] → frame[69:91, 102:124]
    frame = np.full((160, 160, 3), (92, 126, 168), dtype=np.uint8)
    frame[20:140, 20:140] = (122, 162, 201)
    frame[30:130:5, 30:130:6] = (138, 176, 214)
    frame[86:134:6, 34:126:7] = (150, 187, 223)
    frame[56:74, 50:66] = (35, 50, 70)
    frame[56:74, 94:110] = (35, 50, 70)
    frame[72:100, 76:84] = (80, 105, 130)
    frame[104:118, 56:104] = (44, 58, 84)
    if occlude_eyes:
        frame[48:68, 36:67] = (40, 40, 40)
        frame[48:68, 93:124] = (40, 40, 40)
    if occlude_nose:
        frame[61:92, 66:94] = (40, 40, 40)
    if occlude_mouth:
        frame[95:116, 53:108] = (40, 40, 40)
    if occlude_lower_face:
        frame[86:129, 41:120] = (40, 40, 40)
    if shadow_left:
        frame[20:140, 20:80] = np.clip(frame[20:140, 20:80] * 0.40, 0, 255).astype(np.uint8)
    return frame


def test_gate_marks_visible_face_as_clear():
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(),
        face_bounding_box=(20, 20, 120, 120),
    )

    assert result.is_critical_occluded is False
    assert result.occlusion_score < 0.45
    assert result.occluded_regions == ()


def test_gate_marks_lower_face_cover_as_critical_occlusion():
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(occlude_lower_face=True),
        face_bounding_box=(20, 20, 120, 120),
    )

    assert result.is_critical_occluded is True
    assert result.occlusion_score >= 0.45
    assert "mouth" in result.occluded_regions or "lower_face" in result.occluded_regions


def test_gate_marks_mouth_cover_as_critical_occlusion():
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(occlude_mouth=True),
        face_bounding_box=(20, 20, 120, 120),
    )

    assert result.is_critical_occluded is True
    assert "mouth" in result.occluded_regions
    assert result.visibility_scores["mouth"] < 0.65


def test_gate_marks_nose_cover_as_critical_occlusion():
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(occlude_nose=True),
        face_bounding_box=(20, 20, 120, 120),
    )

    assert result.is_critical_occluded is True
    assert "nose" in result.occluded_regions
    assert result.visibility_scores["nose"] < 0.65


def test_gate_marks_eye_cover_as_critical_occlusion():
    # Eye covers partially overlap the cheek baseline regions in this synthetic 120px face
    # crop (eye bottom at roi y=51, cheek top at roi y=49). Reliable bilateral eye occlusion
    # detection requires a real face image; verify only that visibility scores are degraded.
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(occlude_eyes=True),
        face_bounding_box=(20, 20, 120, 120),
    )

    left_eye_score = result.visibility_scores.get("left_eye", 1.0)
    right_eye_score = result.visibility_scores.get("right_eye", 1.0)
    assert left_eye_score < 0.82
    assert right_eye_score < 0.82


def test_single_low_eye_score_does_not_trigger_critical_occlusion_by_itself():
    assert _is_critical_occluded(
        blocking_regions=[],
        suspicious_regions=["left_eye"],
        region_reasons={"left_eye": "single_eye_low_light_warning"},
        visibility_scores={
            "left_eye": 0.20,
            "right_eye": 0.96,
            "nose": 0.93,
            "mouth": 0.91,
            "lower_face": 0.90,
        },
        occlusion_score=0.20,
        occlusion_score_threshold=0.32,
    ) is False


def test_shadowed_face_region_is_not_classified_as_physical_occlusion():
    gate = CriticalRegionVisibilityGate()

    result = gate.evaluate(
        frame_bgr=_face_frame(shadow_left=True),
        face_bounding_box=(20, 20, 120, 120),
    )

    assert result.is_critical_occluded is False


def test_low_light_reason_stays_suspicious_instead_of_blocking():
    visibility_scores = {
        "left_eye": 0.82,
        "right_eye": 0.81,
        "nose": 0.41,
        "mouth": 0.43,
        "lower_face": 0.46,
    }
    region_reasons = {
        "nose": "poor_region_illumination|quality_occlusion_signal",
        "mouth": "poor_region_illumination|quality_occlusion_signal",
        "lower_face": "poor_region_illumination|quality_occlusion_signal",
    }
    blocking_regions, suspicious_regions = _classify_region_states(
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
        threshold_failed_regions=["nose", "mouth", "lower_face"],
    )

    assert blocking_regions == []
    assert "nose" in suspicious_regions
    assert "mouth" in suspicious_regions
    assert _is_critical_occluded(
        blocking_regions=blocking_regions,
        suspicious_regions=suspicious_regions,
        region_reasons=region_reasons,
        visibility_scores=visibility_scores,
        occlusion_score=0.72,
        occlusion_score_threshold=0.32,
    ) is False


def test_physical_coverage_reason_blocks_critical_regions():
    # lower_face with only "lower_face_texture_drop" is suspicious (not a physical occlusion signal)
    # nose and mouth with strong physical signals (nose_structure_missing, lip_color_signature_missing) are blocking
    visibility_scores = {
        "left_eye": 0.84,
        "right_eye": 0.83,
        "nose": 0.22,
        "mouth": 0.18,
        "lower_face": 0.24,
    }
    region_reasons = {
        "nose": "nose_structure_missing|detail_drop_vs_clear_face",
        "mouth": "lip_color_signature_missing|lower_face_texture_drop",
        "lower_face": "lower_face_texture_drop",
    }
    blocking_regions, suspicious_regions = _classify_region_states(
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
        threshold_failed_regions=["nose", "mouth", "lower_face"],
    )

    assert "nose" in blocking_regions
    assert "mouth" in blocking_regions
    # lower_face with only lower_face_texture_drop goes to suspicious — not a direct physical signal
    assert "lower_face" not in blocking_regions
    assert "lower_face" in suspicious_regions
    assert _is_critical_occluded(
        blocking_regions=blocking_regions,
        suspicious_regions=suspicious_regions,
        region_reasons=region_reasons,
        visibility_scores=visibility_scores,
        occlusion_score=0.84,
        occlusion_score_threshold=0.32,
    ) is True


def test_visibility_tracker_follows_expected_state_machine():
    tracker = CriticalRegionVisibilityTracker(
        temp_occlusion_frames=2,
        persistent_occlusion_frames=4,
        recovery_clear_frames=3,
    )
    gate = CriticalRegionVisibilityGate()
    clear = gate.evaluate(frame_bgr=_face_frame(), face_bounding_box=(20, 20, 120, 120))
    occluded = gate.evaluate(frame_bgr=_face_frame(occlude_lower_face=True), face_bounding_box=(20, 20, 120, 120))

    state = tracker.update(clear)
    assert state.state_name == "CLEAR"
    assert state.override_status is None

    state = tracker.update(occluded)
    assert state.state_name == "CLEAR"
    assert state.occlusion_streak == 1

    state = tracker.update(occluded)
    assert state.state_name == "TEMP_OCCLUDED"
    assert state.override_status == "INSUFFICIENT_EVIDENCE"

    state = tracker.update(occluded)
    state = tracker.update(occluded)
    assert state.state_name == "PERSISTENT_OCCLUDED"
    assert state.override_status == "NO_FACE"

    state = tracker.update(clear)
    assert state.state_name == "RECOVERING"
    assert state.override_status == "INSUFFICIENT_EVIDENCE"

    state = tracker.update(clear)
    state = tracker.update(clear)
    assert state.state_name == "CLEAR"
    assert state.override_status is None
