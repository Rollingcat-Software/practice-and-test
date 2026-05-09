"""Integration tests for API routes."""

import pytest
import io
import sys
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
import numpy as np
import cv2

# Mock DeepFace before any imports that depend on it
sys.modules['deepface'] = Mock()
sys.modules['deepface.DeepFace'] = Mock()

from app.main import app
from app.core.container import (
    get_enroll_face_use_case,
    get_verify_face_use_case,
    get_check_liveness_use_case,
    get_file_storage,
    get_process_active_liveness_frame_use_case,
    get_start_active_liveness_use_case,
)
from app.domain.entities.face_embedding import FaceEmbedding
from app.domain.entities.verification_result import VerificationResult
from app.domain.entities.liveness_result import LivenessResult
from app.domain.exceptions.face_errors import (
    FaceNotDetectedError,
    PoorImageQualityError,
)
from app.domain.exceptions.verification_errors import EmbeddingNotFoundError
from app.api.schemas.active_liveness import (
    ActiveLivenessResponse,
    Challenge,
    ChallengeResult,
    ChallengeStatus,
    ChallengeType,
)


@pytest.fixture
def client():
    """Create test client."""
    # Clear any existing overrides
    app.dependency_overrides.clear()

    yield TestClient(app)

    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture
def mock_enroll_use_case():
    """Create mock enrollment use case."""
    use_case = Mock()

    # Create mock result
    embedding = FaceEmbedding.create_new(
        user_id="test_user",
        vector=np.random.randn(128).astype(np.float32),
        quality_score=85.0,
    )

    use_case.execute = AsyncMock(return_value=embedding)
    return use_case


@pytest.fixture
def mock_verify_use_case():
    """Create mock verification use case."""
    use_case = Mock()

    # Create mock result (successful match)
    result = VerificationResult(
        verified=True,
        confidence=0.87,
        distance=0.13,
        threshold=0.6,
    )

    use_case.execute = AsyncMock(return_value=result)
    return use_case


@pytest.fixture
def mock_liveness_use_case():
    """Create mock liveness check use case."""
    use_case = Mock()

    # Create mock result (liveness passed)
    result = LivenessResult(
        is_live=True,
        score=92.0,
        challenge="none",
        challenge_completed=True,
        confidence=0.91,
    )

    use_case.execute = AsyncMock(return_value=result)
    return use_case


@pytest.fixture
def mock_file_storage():
    """Create mock file storage."""
    storage = Mock()
    storage.save_temp = AsyncMock(return_value="/tmp/test_image.jpg")
    storage.cleanup = AsyncMock()
    return storage


@pytest.fixture
def mock_start_active_liveness_use_case():
    """Create mock active liveness start use case."""
    use_case = Mock()
    challenge = Challenge(
        type=ChallengeType.BLINK,
        instruction="Please blink your eyes",
        status=ChallengeStatus.PENDING,
    )
    use_case.execute = AsyncMock(
        return_value=ActiveLivenessResponse(
            session_id="session-123",
            current_challenge=challenge,
            challenge=challenge,
            challenge_progress=0.0,
            time_remaining=5.0,
            challenges_completed=0,
            challenges_total=3,
            session_complete=False,
            session_passed=False,
            overall_score=0.0,
            instruction=challenge.instruction,
            feedback="",
        )
    )
    return use_case


@pytest.fixture
def mock_process_active_liveness_frame_use_case():
    """Create mock active liveness frame use case."""
    use_case = Mock()
    next_challenge = Challenge(
        type=ChallengeType.SMILE,
        instruction="Please smile",
        status=ChallengeStatus.IN_PROGRESS,
    )
    use_case.execute = AsyncMock(
        return_value=ActiveLivenessResponse(
            session_id="session-123",
            current_challenge=next_challenge,
            challenge=next_challenge,
            challenge_progress=1 / 3,
            time_remaining=4.0,
            detection=ChallengeResult(
                challenge_type=ChallengeType.BLINK,
                detected=True,
                confidence=0.95,
                details={"method": "test"},
            ),
            challenges_completed=1,
            challenges_total=3,
            session_complete=False,
            session_passed=False,
            overall_score=0.0,
            instruction=next_challenge.instruction,
            feedback="Great job!",
        )
    )
    return use_case


@pytest.fixture
def test_image_file():
    """Create test image file."""
    # Create a small valid image (100x100 black image)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', img)
    assert success, "Failed to encode test image"

    image_bytes = buffer.tobytes()
    return ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")


# ============================================================================
# Health Check Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns 200."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "version" in data
        assert "model" in data
        assert "detector" in data

    def test_health_check_response_structure(self, client):
        """Test health check response has correct structure."""
        response = client.get("/api/v1/health")

        data = response.json()

        # Verify all required fields
        assert "status" in data
        assert "version" in data
        assert "model" in data
        assert "detector" in data


# ============================================================================
# Enrollment Endpoint Tests
# ============================================================================


class TestEnrollmentEndpoint:
    """Test enrollment endpoint."""

    def test_enroll_success(
        self, client, mock_enroll_use_case, mock_file_storage, test_image_file
    ):
        """Test successful face enrollment."""
        # Override dependencies
        app.dependency_overrides[get_enroll_face_use_case] = lambda: mock_enroll_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["user_id"] == "test_user"
        assert data["quality_score"] == 85.0
        assert "message" in data
        assert data["embedding_dimension"] == 128

    def test_enroll_with_tenant_id(
        self, client, mock_enroll_use_case, mock_file_storage, test_image_file
    ):
        """Test enrollment with tenant ID."""
        # Override dependencies
        app.dependency_overrides[get_enroll_face_use_case] = lambda: mock_enroll_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123", "tenant_id": "tenant_xyz"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200

        # Verify use case was called with tenant_id
        mock_enroll_use_case.execute.assert_called_once()
        call_kwargs = mock_enroll_use_case.execute.call_args.kwargs
        assert call_kwargs["tenant_id"] == "tenant_xyz"

    def test_enroll_invalid_file_type(self, client):
        """Test enrollment rejects non-image files."""
        text_file = ("test.txt", io.BytesIO(b"not an image"), "text/plain")

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123"},
            files={"file": text_file},
        )

        assert response.status_code == 400
        data = response.json()
        # HTTPException uses "detail" field
        assert "image" in data["detail"].lower()

    def test_enroll_no_face_detected(
        self, client, mock_enroll_use_case, mock_file_storage, test_image_file
    ):
        """Test enrollment fails when no face detected."""
        # Override dependencies
        app.dependency_overrides[get_enroll_face_use_case] = lambda: mock_enroll_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to raise FaceNotDetectedError
        mock_enroll_use_case.execute = AsyncMock(
            side_effect=FaceNotDetectedError()
        )

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 400
        data = response.json()
        assert "face" in data["message"].lower()

    def test_enroll_poor_quality(
        self, client, mock_enroll_use_case, mock_file_storage, test_image_file
    ):
        """Test enrollment fails with poor image quality."""
        # Override dependencies
        app.dependency_overrides[get_enroll_face_use_case] = lambda: mock_enroll_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to raise PoorImageQualityError
        mock_enroll_use_case.execute = AsyncMock(
            side_effect=PoorImageQualityError(
                quality_score=35.0, min_threshold=70.0
            )
        )

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 400
        data = response.json()
        assert "quality" in data["message"].lower()

    def test_enroll_cleanup_called(
        self, client, mock_enroll_use_case, mock_file_storage, test_image_file
    ):
        """Test that file cleanup is always called."""
        # Override dependencies
        app.dependency_overrides[get_enroll_face_use_case] = lambda: mock_enroll_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/enroll",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200

        # Verify cleanup was called
        mock_file_storage.cleanup.assert_called_once()
        cleanup_path = mock_file_storage.cleanup.call_args.args[0]
        assert cleanup_path == "/tmp/test_image.jpg"


# ============================================================================
# Verification Endpoint Tests
# ============================================================================


class TestVerificationEndpoint:
    """Test verification endpoint."""

    def test_verify_success_match(
        self, client, mock_verify_use_case, mock_file_storage, test_image_file
    ):
        """Test successful verification with match."""
        # Override dependencies
        app.dependency_overrides[get_verify_face_use_case] = lambda: mock_verify_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["verified"] is True
        assert data["confidence"] == 0.87
        assert data["distance"] == 0.13
        assert data["threshold"] == 0.6
        assert "verified" in data["message"].lower() or "match" in data["message"].lower()

    def test_verify_success_no_match(
        self, client, mock_verify_use_case, mock_file_storage, test_image_file
    ):
        """Test successful verification with no match."""
        # Override dependencies
        app.dependency_overrides[get_verify_face_use_case] = lambda: mock_verify_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to return no match
        result = VerificationResult(
            verified=False,
            confidence=0.32,
            distance=0.68,
            threshold=0.6,
        )
        mock_verify_use_case.execute = AsyncMock(return_value=result)

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["verified"] is False
        assert data["confidence"] == 0.32

    def test_verify_user_not_enrolled(
        self, client, mock_verify_use_case, mock_file_storage, test_image_file
    ):
        """Test verification fails when user not enrolled."""
        # Override dependencies
        app.dependency_overrides[get_verify_face_use_case] = lambda: mock_verify_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to raise EmbeddingNotFoundError
        mock_verify_use_case.execute = AsyncMock(
            side_effect=EmbeddingNotFoundError(user_id="unknown_user")
        )

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "unknown_user"},
            files={"file": test_image_file},
        )

        assert response.status_code == 404
        data = response.json()
        assert "enroll" in data["message"].lower()

    def test_verify_invalid_file_type(self, client):
        """Test verification rejects non-image files."""
        text_file = ("test.txt", io.BytesIO(b"not an image"), "text/plain")

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "test_user_123"},
            files={"file": text_file},
        )

        assert response.status_code == 400
        data = response.json()
        # HTTPException uses "detail" field
        assert "image" in data["detail"].lower()

    def test_verify_with_tenant_id(
        self, client, mock_verify_use_case, mock_file_storage, test_image_file
    ):
        """Test verification with tenant ID."""
        # Override dependencies
        app.dependency_overrides[get_verify_face_use_case] = lambda: mock_verify_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "test_user_123", "tenant_id": "tenant_xyz"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200

        # Verify use case was called with tenant_id
        mock_verify_use_case.execute.assert_called_once()
        call_kwargs = mock_verify_use_case.execute.call_args.kwargs
        assert call_kwargs["tenant_id"] == "tenant_xyz"

    def test_verify_cleanup_called(
        self, client, mock_verify_use_case, mock_file_storage, test_image_file
    ):
        """Test that file cleanup is always called."""
        # Override dependencies
        app.dependency_overrides[get_verify_face_use_case] = lambda: mock_verify_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/verify",
            data={"user_id": "test_user_123"},
            files={"file": test_image_file},
        )

        assert response.status_code == 200

        # Verify cleanup was called
        mock_file_storage.cleanup.assert_called_once()
        cleanup_path = mock_file_storage.cleanup.call_args.args[0]
        assert cleanup_path == "/tmp/test_image.jpg"


# ============================================================================
# Liveness Check Endpoint Tests
# ============================================================================


class TestLivenessEndpoint:
    """Test liveness check endpoint."""

    def test_liveness_success_pass(
        self, client, mock_liveness_use_case, mock_file_storage, test_image_file
    ):
        """Test successful liveness check (passed)."""
        # Override dependencies
        app.dependency_overrides[get_check_liveness_use_case] = lambda: mock_liveness_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness",
            files={"file": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["is_live"] is True
        assert data["score"] == 92.0
        assert data["confidence"] == 0.91
        assert data["challenge"] == "none"
        assert data["challenge_completed"] is True

    def test_liveness_success_fail(
        self, client, mock_liveness_use_case, mock_file_storage, test_image_file
    ):
        """Test successful liveness check (failed)."""
        # Override dependencies
        app.dependency_overrides[get_check_liveness_use_case] = lambda: mock_liveness_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to return liveness failed
        result = LivenessResult(
            is_live=False,
            score=35.0,
            challenge="none",
            challenge_completed=True,
            confidence=0.42,
        )
        mock_liveness_use_case.execute = AsyncMock(return_value=result)

        response = client.post(
            "/api/v1/liveness",
            files={"file": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["is_live"] is False
        assert data["score"] == 35.0
        assert data["confidence"] == 0.42

    def test_liveness_invalid_file_type(self, client):
        """Test liveness check rejects non-image files."""
        text_file = ("test.txt", io.BytesIO(b"not an image"), "text/plain")

        response = client.post(
            "/api/v1/liveness",
            files={"file": text_file},
        )

        assert response.status_code == 400
        data = response.json()
        # HTTPException uses "detail" field
        assert "image" in data["detail"].lower()

    def test_liveness_no_face_detected(
        self, client, mock_liveness_use_case, mock_file_storage, test_image_file
    ):
        """Test liveness check fails when no face detected."""
        # Override dependencies
        app.dependency_overrides[get_check_liveness_use_case] = lambda: mock_liveness_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        # Configure use case to raise FaceNotDetectedError
        mock_liveness_use_case.execute = AsyncMock(
            side_effect=FaceNotDetectedError()
        )

        response = client.post(
            "/api/v1/liveness",
            files={"file": test_image_file},
        )

        assert response.status_code == 400
        data = response.json()
        assert "face" in data["message"].lower()

    def test_liveness_cleanup_called(
        self, client, mock_liveness_use_case, mock_file_storage, test_image_file
    ):
        """Test that file cleanup is always called."""
        # Override dependencies
        app.dependency_overrides[get_check_liveness_use_case] = lambda: mock_liveness_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness",
            files={"file": test_image_file},
        )

        assert response.status_code == 200

        # Verify cleanup was called
        mock_file_storage.cleanup.assert_called_once()
        cleanup_path = mock_file_storage.cleanup.call_args.args[0]
        assert cleanup_path == "/tmp/test_image.jpg"


class TestActiveLivenessEndpoint:
    """Test active liveness endpoints."""

    def test_start_active_liveness_success(
        self, client, mock_start_active_liveness_use_case
    ):
        """Test starting an active liveness session."""
        app.dependency_overrides[get_start_active_liveness_use_case] = lambda: mock_start_active_liveness_use_case

        response = client.post("/api/v1/liveness/active/start", json={})

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "session-123"
        assert data["current_challenge"]["type"] == "blink"
        assert data["instruction"] == "Please blink your eyes"
        assert data["session_complete"] is False

    def test_process_active_liveness_frame_success(
        self,
        client,
        mock_process_active_liveness_frame_use_case,
        mock_file_storage,
        test_image_file,
    ):
        """Test submitting a valid active liveness frame."""
        app.dependency_overrides[get_process_active_liveness_frame_use_case] = lambda: mock_process_active_liveness_frame_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "session-123", "frame_timestamp": "1710000000.125"},
            files={"image": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "session-123"
        assert data["detection"]["detected"] is True
        assert data["challenges_completed"] == 1
        assert data["current_challenge"]["type"] == "smile"

    def test_process_active_liveness_frame_forwards_frame_timestamp(
        self,
        client,
        mock_process_active_liveness_frame_use_case,
        mock_file_storage,
        test_image_file,
    ):
        """Test active liveness frame forwards client timestamp."""
        app.dependency_overrides[get_process_active_liveness_frame_use_case] = lambda: mock_process_active_liveness_frame_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "session-123", "frame_timestamp": "1710000000.125"},
            files={"image": test_image_file},
        )

        assert response.status_code == 200
        call_kwargs = mock_process_active_liveness_frame_use_case.execute.call_args.kwargs
        assert call_kwargs["frame_timestamp"] == 1710000000.125

    def test_process_active_liveness_frame_invalid_session(
        self,
        client,
        mock_process_active_liveness_frame_use_case,
        mock_file_storage,
        test_image_file,
    ):
        """Test submitting a frame with an invalid session ID."""
        from app.application.use_cases.process_active_liveness_frame import ActiveLivenessSessionNotFoundError

        mock_process_active_liveness_frame_use_case.execute = AsyncMock(
            side_effect=ActiveLivenessSessionNotFoundError("Active liveness session not found")
        )
        app.dependency_overrides[get_process_active_liveness_frame_use_case] = lambda: mock_process_active_liveness_frame_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "missing-session", "frame_timestamp": "1710000000.125"},
            files={"image": test_image_file},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_process_active_liveness_frame_invalid_upload(self, client):
        """Test active liveness frame rejects non-image uploads."""
        text_file = ("test.txt", io.BytesIO(b"not an image"), "text/plain")

        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "session-123", "frame_timestamp": "1710000000.125"},
            files={"image": text_file},
        )

        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_process_active_liveness_frame_requires_frame_timestamp(self, client, test_image_file):
        """Test active liveness frame requires frame timestamp."""
        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "session-123"},
            files={"image": test_image_file},
        )

        assert response.status_code == 422

    def test_process_active_liveness_frame_completed_session(
        self,
        client,
        mock_process_active_liveness_frame_use_case,
        mock_file_storage,
        test_image_file,
    ):
        """Test completed sessions return a terminal response."""
        mock_process_active_liveness_frame_use_case.execute = AsyncMock(
            return_value=ActiveLivenessResponse(
                session_id="session-123",
                current_challenge=None,
                challenge=None,
                challenge_progress=1.0,
                time_remaining=0.0,
                challenges_completed=3,
                challenges_total=3,
                session_complete=True,
                session_passed=True,
                overall_score=100.0,
                instruction="All challenges completed! Liveness verified.",
                feedback="Passed 3/3 challenges",
            )
        )
        app.dependency_overrides[get_process_active_liveness_frame_use_case] = lambda: mock_process_active_liveness_frame_use_case
        app.dependency_overrides[get_file_storage] = lambda: mock_file_storage

        response = client.post(
            "/api/v1/liveness/active/frame",
            data={"session_id": "session-123", "frame_timestamp": "1710000000.125"},
            files={"image": test_image_file},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_complete"] is True
        assert data["session_passed"] is True
