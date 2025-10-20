"""
Configuration Management
Centralized configuration for the application.

Educational Notes:
- Keeps all settings in one place (DRY principle)
- Easy to modify behavior without changing code
- Demonstrates separation of configuration from logic
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Configuration for face recognition models."""
    default_model: str = "Facenet512"
    default_detector: str = "opencv"
    default_distance_metric: str = "cosine"


@dataclass
class AnalysisConfig:
    """Configuration for face analysis."""
    default_detector: str = "opencv"
    enforce_detection: bool = True
    default_actions: List[str] = None

    def __post_init__(self):
        if self.default_actions is None:
            self.default_actions = ["age", "gender", "emotion", "race"]


@dataclass
class PathConfig:
    """Configuration for file paths."""
    images_dir: str = "images"
    output_dir: str = "output"
    database_dir: str = "database"


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_size: tuple = (12, 8)
    dpi: int = 100
    save_plots: bool = True
    show_plots: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    analysis: AnalysisConfig
    paths: PathConfig
    visualization: VisualizationConfig

    @classmethod
    def default(cls) -> 'AppConfig':
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            analysis=AnalysisConfig(),
            paths=PathConfig(),
            visualization=VisualizationConfig()
        )

    @classmethod
    def fast_mode(cls) -> 'AppConfig':
        """Configuration optimized for speed."""
        return cls(
            model=ModelConfig(
                default_model="OpenFace",
                default_detector="opencv"
            ),
            analysis=AnalysisConfig(
                default_detector="opencv",
                enforce_detection=False
            ),
            paths=PathConfig(),
            visualization=VisualizationConfig(show_plots=False)
        )

    @classmethod
    def accurate_mode(cls) -> 'AppConfig':
        """Configuration optimized for accuracy."""
        return cls(
            model=ModelConfig(
                default_model="ArcFace",
                default_detector="retinaface"
            ),
            analysis=AnalysisConfig(
                default_detector="retinaface",
                enforce_detection=True
            ),
            paths=PathConfig(),
            visualization=VisualizationConfig()
        )
