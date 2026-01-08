# Biometric Demo - Optimized Architecture

A fully-featured biometric demonstration application with clean Hexagonal Architecture.

## Architecture

```
biometric-demo-optimized/
├── main.py                     # Application entry point
├── requirements.txt            # Dependencies
└── src/
    ├── domain/                 # Core business logic
    │   ├── models.py           # Face, Quality, Liveness entities
    │   ├── challenges.py       # Challenge types and thresholds
    │   └── enrollment.py       # Enrollment state machine
    │
    ├── application/            # Use cases and services
    │   ├── puzzle_service.py   # BiometricPuzzle with 14 challenges
    │   ├── face_tracker.py     # Centroid-based face tracking
    │   └── enrollment_service.py # Two-phase enrollment
    │
    ├── infrastructure/         # External adapters
    │   ├── detection/
    │   │   ├── face_detector.py     # MediaPipe Tasks API
    │   │   ├── landmark_detector.py # 468-point landmarks + pose
    │   │   └── card_detector.py     # YOLO card detection
    │   ├── analysis/
    │   │   ├── quality_assessor.py      # Blur, size, brightness
    │   │   ├── liveness_detector.py     # 5-factor passive liveness
    │   │   ├── demographics_analyzer.py # Age, gender, emotion
    │   │   └── embedding_extractor.py   # Facenet512 embeddings
    │   └── persistence/
    │       └── face_database.py    # Vectorized cosine similarity
    │
    └── presentation/           # UI layer
        ├── camera/
        │   └── threaded_camera.py  # Non-blocking I/O
        └── ui/
            ├── drawing.py      # All UI rendering
            ├── colors.py       # Color constants
            └── profiler.py     # Performance metrics
```

## Features

### All Features from Original `demo_local_fast.py`:
- **Face Detection**: MediaPipe Tasks API with Haar fallback
- **468-Point Landmarks**: Full face mesh with contour drawing
- **Pose Estimation**: Yaw/pitch calculation from landmarks
- **Demographics**: Age, gender, emotion with CLAHE preprocessing
- **Passive Liveness**: 5-factor analysis (texture, color, skin tone, moire, local contrast)
- **Active Liveness Puzzle**: 14 challenge types with proper detection
- **Two-Phase Enrollment**: Liveness verification + 5-angle capture
- **Face Verification**: Vectorized cosine similarity search
- **Card Detection**: YOLO with temporal smoothing
- **Video Recording**: MP4 output
- **Performance Profiler**: Real-time timing metrics
- **Complete UI**: Status bar, stats panel, enrolled thumbnails, help overlay

### Optimizations:
- **Threaded Camera**: Non-blocking I/O for frame capture
- **Vectorized Search**: NumPy BLAS/SIMD for face matching
- **Smart Caching**: Spatial and temporal caching for heavy operations
- **Lazy Loading**: DeepFace and YOLO loaded on demand

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run with all features
python main.py

# Face detection only (fastest)
python main.py --mode face

# With profiler enabled
python main.py --profile
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Cycle modes |
| `e` | Start enrollment |
| `l` | Start liveness puzzle |
| `c` | Toggle card detection |
| `r` | Toggle recording |
| `p` | Toggle profiler |
| `h` | Toggle help |
| `d` | Delete all enrolled faces |
| `s` | Screenshot |
| `Space` | Pause/Resume |
| `ESC` | Cancel current operation |

## Modes

- `all`: All features enabled
- `face`: Face detection only
- `quality`: Face quality assessment
- `demographics`: Age, gender, emotion
- `landmarks`: 468-point visualization
- `liveness`: Passive liveness check
- `puzzle`: Active liveness challenges
- `enroll`: Face enrollment
- `verify`: Face verification
- `card`: ID card detection

## Challenge Types (Biometric Puzzle)

1. **Eye Challenges**: Blink, Close Left Eye, Close Right Eye
2. **Mouth Challenges**: Smile, Open Mouth
3. **Head Pose**: Turn Left/Right, Look Up/Down
4. **Eyebrow Challenges**: Raise Both, Raise Left, Raise Right
5. **Dynamic**: Nod, Shake Head

## Architecture Principles

- **Hexagonal Architecture**: Clean separation of concerns
- **SOLID Principles**: Single responsibility, dependency injection
- **Domain-Driven Design**: Rich domain models
- **Clean Code**: Self-documenting, minimal comments
