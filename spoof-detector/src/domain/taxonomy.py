"""Spoof taxonomy mapping rules.

Maps analyzer signals to spoof category probabilities.
Each analyzer contributes evidence toward or against each category.
"""

from __future__ import annotations

from .models import SpoofCategory

# Signal-to-category contribution matrix.
# Each entry is (category, weight) where weight is how much a LOW analyzer score
# increases the probability of that category.
# Conversely, a HIGH score reduces the probability.
#
# Key insight: different analyzers detect different attack types.
# - MiniFASNet: general binary (catches static, replay, mask)
# - Texture: catches static images and printed photos
# - Moire: catches screen-based attacks (replay, static on screen)
# - Screen Replay: catches display-based attacks
# - Temporal: catches static images (no motion)
# - rPPG: catches static, replay, mask (no pulse)
# - AR Filter: specifically detects AR overlays
# - Makeup: detects heavy contouring

# When an analyzer reports LOW score (spoof-like), these are the category weights
SPOOF_SIGNAL_MAP: dict[str, dict[SpoofCategory, float]] = {
    "minifasnet": {
        SpoofCategory.STATIC_IMAGE: 0.35,
        SpoofCategory.VIDEO_REPLAY: 0.25,
        SpoofCategory.MASK_3D: 0.20,
        SpoofCategory.DEEPFAKE_INJECT: 0.10,
        SpoofCategory.AR_FILTER: 0.05,
        SpoofCategory.HEAVY_MAKEUP: 0.05,
    },
    "texture": {
        SpoofCategory.STATIC_IMAGE: 0.50,
        SpoofCategory.VIDEO_REPLAY: 0.30,
        SpoofCategory.MASK_3D: 0.10,
        SpoofCategory.DEEPFAKE_INJECT: 0.10,
    },
    "moire": {
        SpoofCategory.VIDEO_REPLAY: 0.50,
        SpoofCategory.STATIC_IMAGE: 0.40,
        SpoofCategory.DEEPFAKE_INJECT: 0.10,
    },
    "screen_replay": {
        SpoofCategory.VIDEO_REPLAY: 0.45,
        SpoofCategory.STATIC_IMAGE: 0.35,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "temporal": {
        SpoofCategory.STATIC_IMAGE: 0.70,
        SpoofCategory.MASK_3D: 0.15,
        SpoofCategory.VIDEO_REPLAY: 0.15,
    },
    "rppg": {
        SpoofCategory.STATIC_IMAGE: 0.30,
        SpoofCategory.VIDEO_REPLAY: 0.25,
        SpoofCategory.MASK_3D: 0.25,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "ar_filter": {
        SpoofCategory.AR_FILTER: 0.85,
        SpoofCategory.HEAVY_MAKEUP: 0.10,
        SpoofCategory.DEEPFAKE_INJECT: 0.05,
    },
    "makeup": {
        SpoofCategory.HEAVY_MAKEUP: 0.70,
        SpoofCategory.AR_FILTER: 0.20,
        SpoofCategory.MASK_3D: 0.10,
    },
    "device_boundary": {
        SpoofCategory.STATIC_IMAGE: 0.40,
        SpoofCategory.VIDEO_REPLAY: 0.40,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "blink": {
        SpoofCategory.STATIC_IMAGE: 0.40,
        SpoofCategory.VIDEO_REPLAY: 0.20,
        SpoofCategory.MASK_3D: 0.15,
        SpoofCategory.DEEPFAKE_INJECT: 0.15,
        SpoofCategory.AR_FILTER: 0.10,
    },
    "rppg": {
        SpoofCategory.STATIC_IMAGE: 0.30,
        SpoofCategory.VIDEO_REPLAY: 0.25,
        SpoofCategory.MASK_3D: 0.25,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "landmark_variance": {
        SpoofCategory.STATIC_IMAGE: 0.50,
        SpoofCategory.VIDEO_REPLAY: 0.15,
        SpoofCategory.MASK_3D: 0.15,
        SpoofCategory.DEEPFAKE_INJECT: 0.10,
        SpoofCategory.AR_FILTER: 0.10,
    },
    "screen_flicker": {
        SpoofCategory.VIDEO_REPLAY: 0.50,
        SpoofCategory.STATIC_IMAGE: 0.30,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "micro_tremor": {
        SpoofCategory.STATIC_IMAGE: 0.30,
        SpoofCategory.VIDEO_REPLAY: 0.30,
        SpoofCategory.MASK_3D: 0.20,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
    "blink_rhythm": {
        SpoofCategory.VIDEO_REPLAY: 0.70,
        SpoofCategory.DEEPFAKE_INJECT: 0.15,
        SpoofCategory.STATIC_IMAGE: 0.05,
        SpoofCategory.MASK_3D: 0.05,
        SpoofCategory.AR_FILTER: 0.05,
    },
    "background_grid": {
        SpoofCategory.STATIC_IMAGE: 0.40,
        SpoofCategory.VIDEO_REPLAY: 0.40,
        SpoofCategory.DEEPFAKE_INJECT: 0.20,
    },
}
