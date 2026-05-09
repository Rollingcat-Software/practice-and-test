"""Short-lived verification token generation for active liveness."""

from __future__ import annotations

import time
import uuid

import jwt

from app.core.config import get_settings


class ActiveLivenessTokenService:
    """Creates signed verification tokens for successful sessions."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self._settings = get_settings()

    def create_token(self, session_id: str) -> tuple[str, float]:
        issued_at = int(time.time())
        expires_at = issued_at + self._ttl_seconds
        payload = {
            "sub": session_id,
            "scope": "active_liveness_verification",
            "jti": str(uuid.uuid4()),
            "iat": issued_at,
            "exp": expires_at,
        }
        token = jwt.encode(
            payload,
            self._settings.JWT_SECRET,
            algorithm=self._settings.JWT_ALGORITHM,
        )
        return token, float(expires_at)
