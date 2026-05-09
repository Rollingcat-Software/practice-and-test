import json
import logging

from app.application.use_cases.verify_puzzle import VerifyPuzzleUseCase
from app.core.config import Settings
from app.core.container import clear_cache, get_verify_puzzle_use_case
from app.core.logging_config import StructuredFormatter
from app.infrastructure.ml.liveness.uniface_liveness_detector import UniFaceLivenessDetector


def test_combined_mode_defaults_to_uniface_backend():
    settings = Settings(_env_file=None, JWT_ENABLED=False)

    assert settings.LIVENESS_MODE == "combined"
    assert settings.LIVENESS_UNIFACE_DEFAULT_ENABLED is True
    assert settings.get_liveness_backend() == "uniface"


def test_verify_puzzle_use_case_pins_spot_check_to_uniface():
    clear_cache()
    use_case = get_verify_puzzle_use_case()

    assert isinstance(use_case, VerifyPuzzleUseCase)
    assert isinstance(use_case._spot_check_detector, UniFaceLivenessDetector)

    clear_cache()


def test_structured_formatter_includes_calibration_payload():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="liveness_calibration",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="liveness_calibration",
        args=(),
        exc_info=None,
    )
    record.event_type = "liveness_calibration"
    record.payload = {
        "score": 91.2,
        "confidence": 0.93,
        "backend": "uniface",
        "sub_scores": {"texture": 0.4},
    }

    output = json.loads(formatter.format(record))

    assert output["event_type"] == "liveness_calibration"
    assert output["score"] == 91.2
    assert output["confidence"] == 0.93
    assert output["backend"] == "uniface"
    assert output["sub_scores"]["texture"] == 0.4
