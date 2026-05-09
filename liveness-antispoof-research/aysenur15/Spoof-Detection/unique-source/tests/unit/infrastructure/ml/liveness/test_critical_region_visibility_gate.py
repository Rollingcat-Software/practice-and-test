import numpy as np

from app.infrastructure.ml.liveness.critical_region_visibility_gate import (
    CriticalRegionVisibilityGate,
    CriticalRegionVisibilityTracker,
)


def _face_frame(*, occlude_lower_face: bool = False) -> np.ndarray:
    frame = np.full((160, 160, 3), (92, 126, 168), dtype=np.uint8)
    frame[20:140, 20:140] = (122, 162, 201)
    frame[30:130:5, 30:130:6] = (138, 176, 214)
    frame[86:134:6, 34:126:7] = (150, 187, 223)
    frame[56:74, 50:66] = (35, 50, 70)
    frame[56:74, 94:110] = (35, 50, 70)
    frame[72:100, 76:84] = (80, 105, 130)
    frame[104:118, 56:104] = (44, 58, 84)
    if occlude_lower_face:
        frame[88:140, 34:126] = (40, 40, 40)
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
