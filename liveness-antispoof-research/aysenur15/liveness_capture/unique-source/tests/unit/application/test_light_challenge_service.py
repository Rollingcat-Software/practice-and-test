import numpy as np

from app.api.schemas.active_liveness import ActiveLivenessConfig, ChallengeStatus, ChallengeType
from app.application.services.active_liveness_manager import ActiveLivenessManager
from app.application.services.active_liveness_token_service import ActiveLivenessTokenService
from app.application.services.light_challenge_service import LightChallengeService


def test_generate_light_challenge_contains_client_metadata():
    service = LightChallengeService()

    challenge = service.generate_challenge()

    assert challenge["color"] in LightChallengeService.COLORS
    assert challenge["duration_ms"] == 150
    assert challenge["expected_response_window_ms"] == 500
    assert challenge["expires_at"] > challenge["issued_at"]
    assert challenge["baseline_required"] is True
    assert challenge["ready_for_flash"] is False
    assert "issued_at" in challenge


def test_verify_response_accepts_expected_color_shift():
    service = LightChallengeService()
    baseline = [40.0, 40.0, 40.0]
    red_frame = np.full((20, 20, 3), (40, 40, 80), dtype=np.uint8)

    result = service.verify_response(
        frame=red_frame,
        expected_color="red",
        flash_timestamp=100.0,
        frame_timestamp=100.12,
        baseline_bgr=baseline,
    )

    assert result["passed"] is True
    assert result["color_shift"] > 0.05


def test_verify_response_rejects_timing_mismatch():
    service = LightChallengeService()
    frame = np.full((20, 20, 3), 120, dtype=np.uint8)

    result = service.verify_response(
        frame=frame,
        expected_color="white",
        flash_timestamp=100.0,
        frame_timestamp=100.8,
    )

    assert result["passed"] is False
    assert result["reason"] == "timing_mismatch"


def test_active_liveness_manager_prepares_light_challenge_on_start():
    manager = ActiveLivenessManager()

    session = manager.create_session(
        ActiveLivenessConfig(
            num_challenges=1,
            required_challenges=[ChallengeType.LIGHT],
            randomize=False,
        )
    )

    challenge = session.challenges[0]
    assert challenge.type == ChallengeType.LIGHT
    assert challenge.status.value == "in_progress"
    assert challenge.metadata["color"] in LightChallengeService.COLORS
    assert challenge.metadata["duration_ms"] == 150
    assert challenge.metadata["ready_for_flash"] is False


def test_active_liveness_manager_generates_verification_token_on_pass():
    manager = ActiveLivenessManager(token_service=ActiveLivenessTokenService(ttl_seconds=60))

    session = manager.create_session(
        ActiveLivenessConfig(
            num_challenges=1,
            required_challenges=[ChallengeType.BLINK],
            randomize=False,
        )
    )
    session.is_complete = True
    session.passed = True
    session.overall_score = 100.0
    session.challenges[0].status = ChallengeStatus.COMPLETED

    response = manager.build_response(session)

    assert response.verification_token is not None
    assert response.verification_token_expires_at is not None


def test_active_liveness_challenge_type_does_not_expose_nod():
    assert not hasattr(ChallengeType, "NOD")
