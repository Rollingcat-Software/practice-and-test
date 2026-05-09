"""WebSocket endpoint for live camera analysis.

This module provides real-time frame-by-frame analysis for:
- Face detection
- Quality assessment
- Demographics analysis
- Liveness detection
- Enrollment readiness checks

Clients send camera frames via WebSocket and receive instant feedback.
"""

import base64
import io
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from PIL import Image

from app.api.schemas.live_analysis import (
    AnalysisMode,
    LiveAnalysisRequest,
    LiveAnalysisResponse,
    SessionStats,
)
from app.application.use_cases.live_camera_analysis import LiveCameraAnalysisUseCase
from app.core.container import (
    get_face_detector,
    get_quality_assessor,
    get_liveness_detector,
    get_landmark_detector,
    get_embedding_extractor,
    get_embedding_repository,
    get_similarity_calculator,
)
from app.domain.interfaces.face_detector import IFaceDetector
from app.domain.interfaces.quality_assessor import IQualityAssessor
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.domain.interfaces.landmark_detector import ILandmarkDetector
from app.domain.interfaces.embedding_extractor import IEmbeddingExtractor
from app.domain.interfaces.embedding_repository import IEmbeddingRepository
from app.domain.interfaces.similarity_calculator import ISimilarityCalculator
from app.infrastructure.ml.liveness.rppg_analyzer import RPPGAnalyzer
from app.infrastructure.ml.liveness.temporal_consistency_analyzer import TemporalConsistencyAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Live Analysis"])


class LiveAnalysisSession:
    """Manages state for a live analysis WebSocket session."""

    def __init__(
        self,
        websocket: WebSocket,
        detector: IFaceDetector,
        quality_assessor: IQualityAssessor,
        liveness_detector: ILivenessDetector,
        landmark_detector: ILandmarkDetector,
        embedding_extractor: Optional[IEmbeddingExtractor] = None,
        embedding_repository: Optional[IEmbeddingRepository] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        self.websocket = websocket
        self.detector = detector
        self.quality_assessor = quality_assessor
        self.liveness_detector = liveness_detector
        self.landmark_detector = landmark_detector
        self.embedding_extractor = embedding_extractor
        self.embedding_repository = embedding_repository
        self.similarity_calculator = similarity_calculator

        # Session state
        self.mode: AnalysisMode = AnalysisMode.QUALITY_ONLY
        self.frame_skip: int = 0
        self.quality_threshold: float = 70.0
        self.user_id: Optional[str] = None
        self.tenant_id: Optional[str] = None

        # Statistics
        self.stats = SessionStats()
        self.frame_number = 0
        self.processing_times: list[float] = []

        # Use case
        self.use_case: Optional[LiveCameraAnalysisUseCase] = None

    async def initialize(self, config: LiveAnalysisRequest):
        """Initialize session with configuration."""
        self.mode = config.mode
        self.frame_skip = config.frame_skip
        self.quality_threshold = config.quality_threshold
        self.user_id = config.user_id
        self.tenant_id = config.tenant_id

        # Create use case with all dependencies
        self.use_case = LiveCameraAnalysisUseCase(
            detector=self.detector,
            quality_assessor=self.quality_assessor,
            liveness_detector=self.liveness_detector,
            landmark_detector=self.landmark_detector,
            temporal_consistency_analyzer=TemporalConsistencyAnalyzer(window_size=10),
            rppg_analyzer=RPPGAnalyzer(fps=30.0, window_seconds=5.0),
            embedding_extractor=self.embedding_extractor,
            embedding_repository=self.embedding_repository,
            similarity_calculator=self.similarity_calculator,
        )

        logger.info(
            f"Live analysis session initialized: mode={self.mode}, "
            f"frame_skip={self.frame_skip}, user_id={self.user_id}"
        )

    async def process_frame(self, frame_data: str) -> LiveAnalysisResponse:
        """Process a single frame.

        Args:
            frame_data: Base64-encoded image

        Returns:
            Analysis results
        """
        start_time = time.time()
        self.frame_number += 1
        self.stats.frames_received += 1

        # Check if we should skip this frame
        if self.frame_skip > 0 and self.frame_number % (self.frame_skip + 1) != 0:
            self.stats.frames_skipped += 1
            return LiveAnalysisResponse(
                frame_number=self.frame_number,
                timestamp=start_time,
                processing_time_ms=0,
                skipped=True,
            )

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Process based on mode
            result = await self.use_case.analyze_frame(
                image=image_np,
                mode=self.mode,
                quality_threshold=self.quality_threshold,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
            )

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.stats.frames_processed += 1
            self.stats.average_processing_time_ms = sum(self.processing_times) / len(
                self.processing_times
            )

            # Track best quality
            if result.quality and result.quality.score > self.stats.best_quality_score:
                self.stats.best_quality_score = result.quality.score

            # Track enrollment ready frames
            if result.enrollment_ready and result.enrollment_ready.ready:
                self.stats.enrollment_ready_count += 1

            # Add metadata
            result.frame_number = self.frame_number
            result.timestamp = start_time
            result.processing_time_ms = processing_time

            return result

        except Exception as e:
            logger.error(f"Error processing frame {self.frame_number}: {str(e)}")
            return LiveAnalysisResponse(
                frame_number=self.frame_number,
                timestamp=start_time,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def get_stats(self) -> SessionStats:
        """Get current session statistics."""
        return self.stats


@router.websocket("/ws/live-analysis")
async def live_analysis_websocket(
    websocket: WebSocket,
    detector: IFaceDetector = Depends(get_face_detector),
    quality_assessor: IQualityAssessor = Depends(get_quality_assessor),
    liveness_detector: ILivenessDetector = Depends(get_liveness_detector),
    landmark_detector: ILandmarkDetector = Depends(get_landmark_detector),
    embedding_extractor: IEmbeddingExtractor = Depends(get_embedding_extractor),
    embedding_repository: IEmbeddingRepository = Depends(get_embedding_repository),
    similarity_calculator: ISimilarityCalculator = Depends(get_similarity_calculator),
):
    """WebSocket endpoint for live camera analysis.

    Protocol:
        1. Client connects
        2. Client sends config: {"type": "config", "data": LiveAnalysisRequest}
        3. Client sends frames: {"type": "frame", "data": "base64_encoded_image"}
        4. Server responds: {"type": "result", "data": LiveAnalysisResponse}
        5. Client can request stats: {"type": "stats"}
        6. Server responds: {"type": "stats", "data": SessionStats}

    Example Client (JavaScript):
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/live-analysis');

        ws.onopen = () => {
            // Send configuration
            ws.send(JSON.stringify({
                type: 'config',
                data: {
                    mode: 'enrollment_ready',
                    frame_skip: 1,  // Process every other frame
                    quality_threshold: 75.0
                }
            }));
        };

        // Send camera frame
        canvas.toBlob((blob) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                ws.send(JSON.stringify({
                    type: 'frame',
                    data: base64
                }));
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.8);

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'result') {
                // Update UI with analysis results
                console.log('Quality:', msg.data.quality?.score);
                console.log('Enrollment ready:', msg.data.enrollment_ready?.ready);
            }
        };
        ```
    """
    await websocket.accept()
    logger.info(f"Live analysis WebSocket connected: {websocket.client}")

    session = LiveAnalysisSession(
        websocket=websocket,
        detector=detector,
        quality_assessor=quality_assessor,
        liveness_detector=liveness_detector,
        landmark_detector=landmark_detector,
        embedding_extractor=embedding_extractor,
        embedding_repository=embedding_repository,
        similarity_calculator=similarity_calculator,
    )

    try:
        while True:
            # Receive message from client (handle both text and JSON)
            try:
                # Try to receive as text first to handle plain "ping" messages
                raw_message = await websocket.receive_text()

                # Check if it's a plain ping heartbeat
                if raw_message == "ping":
                    await websocket.send_text("pong")
                    continue

                # Otherwise, parse as JSON
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON: {raw_message[:100]}")
                continue

            msg_type = message.get("type")
            msg_data = message.get("data")

            if msg_type == "config":
                # Initialize session with config
                config = LiveAnalysisRequest(**msg_data)
                await session.initialize(config)

                await websocket.send_json({
                    "type": "config_ack",
                    "data": {"status": "configured", "mode": config.mode}
                })

            elif msg_type == "frame":
                # Process frame
                result = await session.process_frame(msg_data)

                await websocket.send_json({
                    "type": "result",
                    "data": result.model_dump()
                })

            elif msg_type == "stats":
                # Send statistics
                stats = session.get_stats()

                await websocket.send_json({
                    "type": "stats",
                    "data": stats.model_dump()
                })

            elif msg_type == "ping":
                # Keepalive (JSON format)
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"}
                })

    except WebSocketDisconnect:
        logger.info(
            f"Live analysis WebSocket disconnected: {websocket.client} "
            f"(processed {session.stats.frames_processed} frames)"
        )
    except Exception as e:
        logger.error(f"Live analysis WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except Exception:
            pass
