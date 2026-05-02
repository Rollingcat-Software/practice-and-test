# FIVUCSAS Spoof Detector

Session-based multi-method face presentation attack detection system. Part of the [FIVUCSAS](https://fivucsas.com) biometric authentication platform.

## What It Does

Real-time face spoof detection that runs for **5 seconds to 3 hours**, accumulating evidence to produce a session-level verdict (LIVE or SPOOF). Classifies attacks into 7 categories:

| Category | Detection Method | Status |
|----------|-----------------|--------|
| Real (live person) | MiniFASNet + temporal + motion | Working |
| Static Image (photo) | MiniFASNet + device boundary | Working |
| Video Replay (screen) | MiniFASNet + moire + screen replay | Working |
| 3D Mask | MiniFASNet (partial) | Partial |
| Heavy Makeup | Planned | Phase 5 |
| AR Filter | MobileNetV3-Small (planned) | Phase 5 |
| Deepfake Injection | Active illumination (planned) | Phase 5 |

## Architecture

```
Session Engine (5s - 3hr)
  |-- accumulates per-frame evidence
  |-- detects incidents (spoof bursts, frozen face, missing face)
  |-- peak-sensitive verdict (worst-window prevents dilution)
  |-- session report on exit
  |
Pipeline (per-frame, ~20ms)
  |-- MediaPipe face detection (~2ms)
  |-- IoU multi-face tracking
  |-- 6 analyzers (calibrated weights from ground-truth testing):
  |     MiniFASNet ONNX      (3.0x) +94.7 gap  PROVEN
  |     Device Boundary      (2.5x) +19.2 gap  GOOD
  |     Screen Replay        (0.5x) +9.6 gap   WEAK
  |     Temporal Consistency  (0.3x) motion-based
  |     Texture              (0.1x) anti-correlated
  |     Moire                (0.1x) anti-correlated
  |-- MultiClassFuser -> 7-category probability distribution
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the detector
python main.py

# Controls: q=quit, d=detail panel, s=save frame, h=help
```

## Testing

```bash
# Unit tests (60 tests)
python -m pytest tests/ -v

# Benchmark (per-analyzer timing + accuracy)
python tools/benchmark.py

# Live diagnostic dashboard
python tools/diagnose.py

# Guided test protocol (walks through attack scenarios)
python tools/test_protocol.py

# Analyze saved captures with ground truth
python tools/analyze_captures.py

# Diagnose a single image
python tools/diagnose.py --image path/to/image.jpg
```

## Project Structure

```
spoof-detector/
  main.py                           Entry point
  config.yaml                       Configuration
  src/
    domain/
      models.py                     BBox, FaceROI, SpoofCategory (7), SpoofClassification
      session.py                    SessionState, SessionVerdict, TemporalSignals, Incident
      interfaces.py                 IFaceDetector, IFaceAnalyzer, IFrameAnalyzer
      taxonomy.py                   Signal-to-category mapping rules
    application/
      session_engine.py             Session-based verdict engine (core)
      pipeline.py                   Per-frame pipeline orchestrator
      face_tracker.py               IoU multi-face tracking
      data_collector.py             Research data capture
    infrastructure/
      detection/
        mediapipe_detector.py       MediaPipe Tasks face detection
      analyzers/
        minifasnet_analyzer.py      UniFace MiniFASNet ONNX (binary real/spoof)
        device_boundary_analyzer.py Phone/tablet bezel detection (Canny+Hough)
        screen_replay_analyzer.py   FFT + skin color + specular analysis
        moire_analyzer.py           Gabor filter bank + FFT periodicity
        texture_analyzer.py         Laplacian + color + frequency analysis
        temporal_analyzer.py        Micro-motion naturalness per face ID
      fusion/
        multi_class_fuser.py        Calibrated 7-class probability fusion
      logging/
        structured_logger.py        JSONL session logging
    presentation/
      app.py                        Main GUI loop + session rendering
      overlay.py                    OpenCV HUD (stats, boxes, probability bars)
      camera.py                     Threaded double-buffer camera
  tests/                            60 unit tests
  tools/                            Benchmark, diagnostic, test protocol
  paper/                            Academic paper outline
```

## Ground-Truth Results (2026-05-02)

Calibration from labeled captures:
- **Per-capture accuracy**: 8/8 (100%) with calibrated weights
- **Session accuracy**: Real-only -> LIVE 95%, Spoof-only -> SPOOF 63%
- **MiniFASNet**: Only analyzer with proven discrimination (+94.7 gap)
- **Texture/Moire**: Anti-correlated on screen attacks (suppressed to 0.1 weight)

## Academic Paper

Target: BIOSIG 2026 / IJCB 2026. See `paper/outline.md` for full structure.

Title: "AR-Spoofing: Session-Based Multi-Method Face Presentation Attack Detection"

Novel contributions:
1. Session-based verdict engine (vs per-frame classification)
2. Calibrated fusion weights from ground-truth testing
3. Peak-sensitive verdict (worst-window prevents spoof dilution)
4. AR-filter detection dataset (Phase 5, via amispoof.com)

## Roadmap

- [x] Phase 1: Foundation (detection, tracking, overlay)
- [x] Phase 2: Analyzer integration (MiniFASNet, texture, moire, screen replay, temporal)
- [x] Phase 2.5: Device boundary detection, calibrated fusion, session engine
- [ ] Phase 3: Temporal analyzers (blink detection, rPPG pulse)
- [ ] Phase 4: Data collection (AR-filter dataset via amispoof.com)
- [ ] Phase 5: AR filter detector training (MobileNetV3-Small)
- [ ] Phase 6: Paper + evaluation (APCER/BPCER/ACER benchmarks)
- [ ] Phase 7: amispoof.com demo
- [ ] Phase 8: Production integration into FIVUCSAS

## Requirements

- Python 3.11+
- OpenCV (with GUI support, not headless)
- MediaPipe 0.10.9+
- UniFace 3.0+ (MiniFASNet ONNX)
- Webcam

## License

Part of FIVUCSAS project. Marmara University - Computer Engineering Department.
