"""Active liveness challenge manager."""

import logging
import random
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.api.schemas.active_liveness import (
    ActiveLivenessConfig,
    ActiveLivenessResponse,
    ActiveLivenessSession,
    Challenge,
    ChallengeResult,
    ChallengeStatus,
    ChallengeType,
    get_challenge_instruction,
)
from app.application.services.active_liveness_token_service import ActiveLivenessTokenService
from app.application.services.light_challenge_service import LightChallengeService

logger = logging.getLogger(__name__)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
NOSE_TIP = 1
LEFT_EAR = 234
RIGHT_EAR = 454


class ActiveLivenessManager:
    """Manages active liveness challenge sessions."""

    def __init__(
        self,
        blink_threshold: float = 0.21,
        smile_threshold: float = 0.4,
        head_turn_threshold: float = 0.15,
        mouth_open_threshold: float = 0.5,
        eyebrow_threshold: float = 0.08,
        light_challenge_service: Optional[LightChallengeService] = None,
        token_service: Optional[ActiveLivenessTokenService] = None,
    ) -> None:
        self._blink_threshold = blink_threshold
        self._smile_threshold = smile_threshold
        self._head_turn_threshold = head_turn_threshold
        self._mouth_open_threshold = mouth_open_threshold
        self._eyebrow_threshold = eyebrow_threshold
        self._light_challenge_service = light_challenge_service or LightChallengeService()
        self._token_service = token_service or ActiveLivenessTokenService()
        self._face_landmarker = None
        logger.info("ActiveLivenessManager initialized")

    def _get_face_landmarker(self):
        if self._face_landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = Path(__file__).parent.parent.parent.parent / "models" / "face_landmarker.task"
                if not model_path.exists():
                    alt_paths = [
                        Path("models/face_landmarker.task"),
                        Path("C:/Users/hp/Documents/GitHub/Rollingcat-Software/biometric-processor/models/face_landmarker.task"),
                    ]
                    for alt in alt_paths:
                        if alt.exists():
                            model_path = alt
                            break
                    else:
                        raise FileNotFoundError(f"Face landmarker model not found. Expected at: {model_path}")

                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                )
                self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe Face Landmarker initialized for active liveness")
            except Exception:
                logger.exception("Failed to initialize Face Landmarker")
                raise
        return self._face_landmarker

    def create_session(self, config: Optional[ActiveLivenessConfig] = None) -> ActiveLivenessSession:
        if config is None:
            config = ActiveLivenessConfig()

        created_at = time.time()
        challenges = self._generate_challenges(config)
        session = ActiveLivenessSession(
            session_id=str(uuid.uuid4()),
            challenges=challenges,
            current_challenge_index=0,
            started_at=created_at,
            expires_at=created_at + config.session_timeout_seconds,
            last_activity_at=created_at,
            current_challenge_started_at=created_at,
        )
        first_challenge = self.get_current_challenge(session)
        if first_challenge is not None:
            self._prepare_challenge(session, first_challenge, created_at=created_at)
        logger.info("Created active liveness session %s with %s challenges", session.session_id, len(challenges))
        return session

    def _generate_challenges(self, config: ActiveLivenessConfig) -> List[Challenge]:
        if config.required_challenges:
            challenge_types = config.required_challenges[: config.num_challenges]
        else:
            available = [
                ChallengeType.LIGHT,
                ChallengeType.BLINK,
                ChallengeType.SMILE,
                ChallengeType.TURN_LEFT,
                ChallengeType.TURN_RIGHT,
            ]
            if config.randomize:
                random.shuffle(available)
            challenge_types = available[: config.num_challenges]

        return [
            Challenge(
                type=challenge_type,
                instruction=get_challenge_instruction(challenge_type),
                timeout_seconds=config.challenge_timeout,
            )
            for challenge_type in challenge_types
        ]

    def get_current_challenge(self, session: ActiveLivenessSession) -> Optional[Challenge]:
        if session.is_complete:
            return None
        if session.current_challenge_index >= len(session.challenges):
            return None
        return session.challenges[session.current_challenge_index]

    def is_expired(self, session: ActiveLivenessSession, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= session.expires_at

    async def process_frame(
        self,
        session: ActiveLivenessSession,
        image: np.ndarray,
        frame_timestamp: Optional[float] = None,
    ) -> ActiveLivenessResponse:
        current_challenge = self.get_current_challenge(session)
        if current_challenge is None:
            return self.build_response(session=session)

        if current_challenge.status == ChallengeStatus.PENDING:
            self._prepare_challenge(session, current_challenge)

        elapsed = time.time() - session.current_challenge_started_at
        time_remaining = max(0.0, current_challenge.timeout_seconds - elapsed)

        if time_remaining <= 0:
            current_challenge.attempts += 1
            feedback = "Challenge failed."
            if current_challenge.attempts >= current_challenge.max_attempts:
                current_challenge.status = ChallengeStatus.FAILED
                self._advance_to_next_challenge(session)
            else:
                self._prepare_challenge(session, current_challenge)
                time_remaining = current_challenge.timeout_seconds
                feedback = "Time's up! Try again."
            return self.build_response(session=session, feedback=feedback)

        detection = await self._detect_challenge(
            session,
            image,
            current_challenge.type,
            challenge=current_challenge,
            frame_timestamp=frame_timestamp,
        )
        feedback = self._get_guidance(current_challenge.type, detection)

        if detection.detected:
            current_challenge.status = ChallengeStatus.COMPLETED
            current_challenge.confidence = detection.confidence
            self._advance_to_next_challenge(session)
            if session.is_complete:
                return self.build_response(session=session, detection=detection, feedback="Challenge sequence completed.")
            feedback = "Great job!"

        return self.build_response(session=session, detection=detection, feedback=feedback)

    def _advance_to_next_challenge(self, session: ActiveLivenessSession) -> None:
        session.current_challenge_index += 1
        session.blink_detected = False
        session.baseline_ear = None
        session.baseline_mar = None

        next_challenge = self.get_current_challenge(session)
        if next_challenge is None:
            self._complete_session(session)
            return
        self._prepare_challenge(session, next_challenge)

    def _prepare_challenge(
        self,
        session: ActiveLivenessSession,
        challenge: Challenge,
        created_at: Optional[float] = None,
    ) -> None:
        session.current_challenge_started_at = created_at or time.time()
        challenge.status = ChallengeStatus.IN_PROGRESS
        challenge.metadata = (
            self._light_challenge_service.generate_challenge()
            if challenge.type == ChallengeType.LIGHT
            else {}
        )
        if challenge.type == ChallengeType.LIGHT:
            session.light_baseline_captured = False

    def _complete_session(self, session: ActiveLivenessSession) -> None:
        completed = sum(1 for challenge in session.challenges if challenge.status == ChallengeStatus.COMPLETED)
        total = len(session.challenges)
        session.completed_at = time.time()
        session.is_complete = True
        session.overall_score = (completed / total) * 100 if total > 0 else 0.0
        session.passed = completed >= (total * 0.6)

    async def _detect_challenge(
        self,
        session: ActiveLivenessSession,
        image: np.ndarray,
        challenge_type: ChallengeType,
        challenge: Optional[Challenge] = None,
        frame_timestamp: Optional[float] = None,
    ) -> ChallengeResult:
        import mediapipe as mp

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self._get_face_landmarker().detect(mp_image)

        if not results.face_landmarks:
            return ChallengeResult(
                challenge_type=challenge_type,
                detected=False,
                confidence=0.0,
                details={"error": "No face detected"},
            )

        face_landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        if challenge_type != ChallengeType.LIGHT:
            session.last_face_mean_bgr = image.mean(axis=(0, 1)).astype(float).tolist()
        blendshapes = {}
        if results.face_blendshapes:
            for bs in results.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

        if challenge_type == ChallengeType.LIGHT:
            return self._detect_light_challenge(session, image, challenge, frame_timestamp)
        if challenge_type == ChallengeType.BLINK:
            return self._detect_blink(session, points, face_landmarks, blendshapes)
        if challenge_type == ChallengeType.SMILE:
            return self._detect_smile(session, points, blendshapes)
        if challenge_type == ChallengeType.TURN_LEFT:
            return self._detect_head_turn(face_landmarks, "left")
        if challenge_type == ChallengeType.TURN_RIGHT:
            return self._detect_head_turn(face_landmarks, "right")
        if challenge_type == ChallengeType.OPEN_MOUTH:
            return self._detect_mouth_open(points, blendshapes)
        if challenge_type == ChallengeType.RAISE_EYEBROWS:
            return self._detect_eyebrow_raise(points, face_landmarks, blendshapes)
        return ChallengeResult(
            challenge_type=challenge_type,
            detected=False,
            confidence=0.0,
            details={"error": f"Unknown challenge: {challenge_type}"},
        )

    def _detect_light_challenge(
        self,
        session: ActiveLivenessSession,
        image: np.ndarray,
        challenge: Optional[Challenge],
        frame_timestamp: Optional[float],
    ) -> ChallengeResult:
        metadata = challenge.metadata if challenge is not None else {}
        if not session.light_baseline_captured:
            session.last_face_mean_bgr = image.mean(axis=(0, 1)).astype(float).tolist()
            session.light_baseline_captured = True
            refreshed_metadata = self._light_challenge_service.generate_challenge()
            refreshed_metadata["ready_for_flash"] = True
            metadata.update(refreshed_metadata)
            if challenge is not None:
                challenge.metadata = metadata
            return ChallengeResult(
                challenge_type=ChallengeType.LIGHT,
                detected=False,
                confidence=0.0,
                details={
                    "baseline_captured": True,
                    "expected_color": metadata.get("color"),
                    "issued_at": metadata.get("issued_at"),
                    "expires_at": metadata.get("expires_at"),
                },
            )

        verification = self._light_challenge_service.verify_response(
            frame=image,
            expected_color=metadata.get("color", "white"),
            flash_timestamp=metadata.get("issued_at", session.current_challenge_started_at),
            frame_timestamp=frame_timestamp,
            baseline_bgr=session.last_face_mean_bgr,
        )

        face_mean_bgr = verification.get("face_mean_bgr")
        if isinstance(face_mean_bgr, list) and len(face_mean_bgr) == 3:
            session.last_face_mean_bgr = [float(value) for value in face_mean_bgr]

        if verification["passed"]:
            return ChallengeResult(
                challenge_type=ChallengeType.LIGHT,
                detected=True,
                confidence=min(1.0, float(verification.get("color_shift", 0.0)) * 8.0),
                details=verification,
            )

        return ChallengeResult(
            challenge_type=ChallengeType.LIGHT,
            detected=False,
            confidence=0.0,
            details=verification,
        )

    def _calculate_ear(self, points: List[Tuple[int, int]], eye_indices: List[int]) -> float:
        try:
            p1 = np.array(points[eye_indices[0]])
            p2 = np.array(points[eye_indices[1]])
            p3 = np.array(points[eye_indices[2]])
            p4 = np.array(points[eye_indices[3]])
            p5 = np.array(points[eye_indices[4]])
            p6 = np.array(points[eye_indices[5]])
            vertical_1 = np.linalg.norm(p2 - p6)
            vertical_2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)
            if horizontal == 0:
                return 0.0
            return float((vertical_1 + vertical_2) / (2.0 * horizontal))
        except (IndexError, ValueError):
            return 0.0

    def _calculate_mar(self, points: List[Tuple[int, int]]) -> float:
        try:
            left_corner = np.array(points[MOUTH_CORNER_LEFT])
            right_corner = np.array(points[MOUTH_CORNER_RIGHT])
            upper_lip = np.array(points[UPPER_LIP_CENTER])
            lower_lip = np.array(points[LOWER_LIP_CENTER])
            horizontal = np.linalg.norm(right_corner - left_corner)
            vertical = np.linalg.norm(lower_lip - upper_lip)
            if horizontal == 0:
                return 0.0
            return float(vertical / horizontal)
        except (IndexError, ValueError):
            return 0.0

    def _detect_blink(self, session: ActiveLivenessSession, points: List[Tuple[int, int]], landmarks, blendshapes: dict) -> ChallengeResult:
        if blendshapes:
            left_blink = blendshapes.get("eyeBlinkLeft", 0)
            right_blink = blendshapes.get("eyeBlinkRight", 0)
            avg_blink = (left_blink + right_blink) / 2.0
            eyes_closed = avg_blink > 0.5
            if eyes_closed and not session.blink_detected:
                session.blink_detected = True
            elif not eyes_closed and session.blink_detected:
                session.blink_detected = False
                return ChallengeResult(
                    challenge_type=ChallengeType.BLINK,
                    detected=True,
                    confidence=min(1.0, avg_blink + 0.3),
                    details={"blink_score": avg_blink, "method": "blendshapes"},
                )
            return ChallengeResult(
                challenge_type=ChallengeType.BLINK,
                detected=False,
                confidence=0.0,
                details={"blink_score": avg_blink, "eyes_closed": eyes_closed, "blink_started": session.blink_detected},
            )

        left_ear = self._calculate_ear(points, LEFT_EYE_INDICES)
        right_ear = self._calculate_ear(points, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        if session.baseline_ear is None and avg_ear > 0.2:
            session.baseline_ear = avg_ear

        eyes_closed = avg_ear < self._blink_threshold
        if eyes_closed and not session.blink_detected:
            session.blink_detected = True
        elif not eyes_closed and session.blink_detected:
            confidence = min(1.0, (session.baseline_ear or 0.3) / avg_ear) if avg_ear > 0 else 0.5
            session.blink_detected = False
            return ChallengeResult(
                challenge_type=ChallengeType.BLINK,
                detected=True,
                confidence=confidence,
                details={"ear": avg_ear, "baseline": session.baseline_ear, "method": "ear"},
            )

        session.last_ear = avg_ear
        return ChallengeResult(
            challenge_type=ChallengeType.BLINK,
            detected=False,
            confidence=0.0,
            details={"ear": avg_ear, "eyes_closed": eyes_closed, "blink_started": session.blink_detected},
        )

    def _detect_smile(self, session: ActiveLivenessSession, points: List[Tuple[int, int]], blendshapes: dict) -> ChallengeResult:
        if blendshapes:
            left_smile = blendshapes.get("mouthSmileLeft", 0)
            right_smile = blendshapes.get("mouthSmileRight", 0)
            avg_smile = (left_smile + right_smile) / 2.0
            detected = avg_smile > 0.4
            confidence = min(1.0, avg_smile) if detected else 0.0
            return ChallengeResult(
                challenge_type=ChallengeType.SMILE,
                detected=detected,
                confidence=confidence,
                details={"smile_score": avg_smile, "method": "blendshapes"},
            )

        mar = self._calculate_mar(points)
        if session.baseline_mar is None:
            session.baseline_mar = mar
        if session.baseline_mar and session.baseline_mar > 0:
            smile_ratio = mar / session.baseline_mar
        else:
            smile_ratio = mar / 0.3
        detected = mar > self._smile_threshold and smile_ratio > 1.3
        confidence = min(1.0, smile_ratio - 1.0) if detected else 0.0
        return ChallengeResult(
            challenge_type=ChallengeType.SMILE,
            detected=detected,
            confidence=confidence,
            details={"mar": mar, "baseline": session.baseline_mar, "ratio": smile_ratio, "method": "mar"},
        )

    def _detect_head_turn(self, landmarks, direction: str) -> ChallengeResult:
        nose = landmarks[NOSE_TIP]
        left_ear = landmarks[LEFT_EAR]
        right_ear = landmarks[RIGHT_EAR]
        center_x = (left_ear.x + right_ear.x) / 2
        deviation = nose.x - center_x

        if direction == "left":
            detected = deviation > self._head_turn_threshold
            confidence = min(1.0, deviation / 0.2) if detected else 0.0
            challenge_type = ChallengeType.TURN_LEFT
        else:
            detected = deviation < -self._head_turn_threshold
            confidence = min(1.0, abs(deviation) / 0.2) if detected else 0.0
            challenge_type = ChallengeType.TURN_RIGHT

        return ChallengeResult(
            challenge_type=challenge_type,
            detected=detected,
            confidence=confidence,
            details={"deviation": deviation, "direction": direction},
        )

    def _detect_mouth_open(self, points: List[Tuple[int, int]], blendshapes: dict) -> ChallengeResult:
        if blendshapes:
            jaw_open = blendshapes.get("jawOpen", 0)
            detected = jaw_open > 0.4
            confidence = min(1.0, jaw_open) if detected else 0.0
            return ChallengeResult(
                challenge_type=ChallengeType.OPEN_MOUTH,
                detected=detected,
                confidence=confidence,
                details={"jaw_open": jaw_open, "method": "blendshapes"},
            )

        mar = self._calculate_mar(points)
        detected = mar > self._mouth_open_threshold
        confidence = min(1.0, mar / 0.7) if detected else 0.0
        return ChallengeResult(
            challenge_type=ChallengeType.OPEN_MOUTH,
            detected=detected,
            confidence=confidence,
            details={"mar": mar, "method": "mar"},
        )

    def _detect_eyebrow_raise(self, points: List[Tuple[int, int]], landmarks, blendshapes: dict) -> ChallengeResult:
        if blendshapes:
            inner_up = blendshapes.get("browInnerUp", 0)
            outer_up_left = blendshapes.get("browOuterUpLeft", 0)
            outer_up_right = blendshapes.get("browOuterUpRight", 0)
            avg_brow = (inner_up + outer_up_left + outer_up_right) / 3.0
            detected = avg_brow > 0.3
            confidence = min(1.0, avg_brow) if detected else 0.0
            return ChallengeResult(
                challenge_type=ChallengeType.RAISE_EYEBROWS,
                detected=detected,
                confidence=confidence,
                details={"brow_raise_score": avg_brow, "method": "blendshapes"},
            )

        try:
            left_brow_y = np.mean([landmarks[i].y for i in LEFT_EYEBROW])
            right_brow_y = np.mean([landmarks[i].y for i in RIGHT_EYEBROW])
            avg_brow_y = (left_brow_y + right_brow_y) / 2
            left_eye_y = landmarks[LEFT_EYE_INDICES[0]].y
            right_eye_y = landmarks[RIGHT_EYE_INDICES[0]].y
            avg_eye_y = (left_eye_y + right_eye_y) / 2
            distance = avg_eye_y - avg_brow_y
            detected = distance > self._eyebrow_threshold
            confidence = min(1.0, distance / 0.12) if detected else 0.0
            return ChallengeResult(
                challenge_type=ChallengeType.RAISE_EYEBROWS,
                detected=detected,
                confidence=confidence,
                details={"brow_eye_distance": distance, "method": "landmarks"},
            )
        except (IndexError, AttributeError):
            return ChallengeResult(
                challenge_type=ChallengeType.RAISE_EYEBROWS,
                detected=False,
                confidence=0.0,
                details={"error": "Could not calculate eyebrow position"},
            )

    def _get_guidance(self, challenge_type: ChallengeType, detection: ChallengeResult) -> str:
        details = detection.details
        if challenge_type == ChallengeType.BLINK:
            return "Good, now open your eyes!" if details.get("blink_started") else "Blink your eyes slowly"
        if challenge_type == ChallengeType.SMILE:
            return "Almost there, smile wider!" if details.get("ratio", 0) > 1.1 else "Give a natural smile"
        if challenge_type == ChallengeType.LIGHT:
            if details.get("reason") == "timing_mismatch":
                return "Capture the frame right after the flash appears"
            if details.get("baseline_captured"):
                return "Baseline captured. Show the flash now and send the next frame immediately"
            return "Keep looking at the camera while the screen flashes"
        if challenge_type in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT):
            direction = "left" if challenge_type == ChallengeType.TURN_LEFT else "right"
            return f"Keep turning {direction}..." if abs(details.get("deviation", 0)) > 0.05 else f"Turn your head to the {direction}"
        if challenge_type == ChallengeType.OPEN_MOUTH:
            return "Open wider!" if details.get("mar", 0) > 0.3 else "Open your mouth wide"
        if challenge_type == ChallengeType.RAISE_EYEBROWS:
            return "Raise your eyebrows high"
        return ""

    def build_response(
        self,
        session: ActiveLivenessSession,
        detection: Optional[ChallengeResult] = None,
        feedback: str = "",
    ) -> ActiveLivenessResponse:
        completed = sum(1 for challenge in session.challenges if challenge.status == ChallengeStatus.COMPLETED)
        total = len(session.challenges)
        current_challenge = self.get_current_challenge(session)
        time_remaining = 0.0
        instruction = "Session complete" if session.is_complete else ""

        if current_challenge is not None:
            elapsed = max(0.0, time.time() - session.current_challenge_started_at)
            time_remaining = max(0.0, current_challenge.timeout_seconds - elapsed)
            instruction = current_challenge.instruction
        elif session.is_complete:
            feedback = feedback or (
                f"Passed {completed}/{total} challenges"
                if session.passed
                else f"Only {completed}/{total} challenges passed"
            )
            instruction = "All challenges completed! Liveness verified." if session.passed else "Session complete. Please try again."

        progress = completed / total if total > 0 else 0.0
        if session.is_complete:
            progress = 1.0

        if session.is_complete and session.passed and not session.verification_token:
            token, token_expires_at = self._token_service.create_token(session.session_id)
            session.verification_token = token
            session.verification_token_expires_at = token_expires_at

        return ActiveLivenessResponse(
            session_id=session.session_id,
            current_challenge=current_challenge,
            challenge=current_challenge,
            challenge_progress=progress,
            time_remaining=time_remaining,
            detection=detection,
            challenges_completed=completed,
            challenges_total=total,
            session_complete=session.is_complete,
            session_passed=session.passed,
            overall_score=session.overall_score,
            instruction=instruction,
            feedback=feedback,
            verification_token=session.verification_token,
            verification_token_expires_at=session.verification_token_expires_at,
        )
