"""Face landmarks domain entities."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel


@dataclass
class Landmark:
    """Single facial landmark point.

    Attributes:
        id: Landmark index
        x: X coordinate
        y: Y coordinate
        z: Z coordinate (optional, for 3D landmarks)
        visibility: Per-landmark visibility score 0–1 from the detector (optional)
    """

    id: int
    x: int
    y: int
    z: Optional[float] = None
    visibility: Optional[float] = None


@dataclass
class HeadPose:
    """Head pose estimation.

    Attributes:
        pitch: Rotation around X-axis (up/down)
        yaw: Rotation around Y-axis (left/right)
        roll: Rotation around Z-axis (tilt)
    """

    pitch: float
    yaw: float
    roll: float


@dataclass
class LandmarkResult:
    """Face landmark detection result.

    Attributes:
        model: Model used for detection
        landmark_count: Number of landmarks detected
        landmarks: List of landmark points
        regions: Mapping of facial regions to landmark indices
        head_pose: Estimated head pose
    """

    model: str
    landmark_count: int
    landmarks: List[Landmark] = field(default_factory=list)
    regions: Dict[str, List[int]] = field(default_factory=dict)
    head_pose: Optional[HeadPose] = None


# Pydantic models for API responses


class LandmarkResponse(BaseModel):
    """API response model for single landmark."""

    id: int
    x: int
    y: int
    z: Optional[float] = None


class HeadPoseResponse(BaseModel):
    """API response model for head pose."""

    pitch: float
    yaw: float
    roll: float


class LandmarkResultResponse(BaseModel):
    """API response model for landmark detection result."""

    model: str
    landmark_count: int
    landmarks: List[LandmarkResponse]
    regions: Dict[str, List[int]]
    head_pose: Optional[HeadPoseResponse] = None

    @classmethod
    def from_result(cls, result: LandmarkResult) -> "LandmarkResultResponse":
        """Create response from domain result."""
        head_pose = None
        if result.head_pose:
            head_pose = HeadPoseResponse(
                pitch=result.head_pose.pitch,
                yaw=result.head_pose.yaw,
                roll=result.head_pose.roll,
            )

        return cls(
            model=result.model,
            landmark_count=result.landmark_count,
            landmarks=[
                LandmarkResponse(id=lm.id, x=lm.x, y=lm.y, z=lm.z)
                for lm in result.landmarks
            ],
            regions=result.regions,
            head_pose=head_pose,
        )
