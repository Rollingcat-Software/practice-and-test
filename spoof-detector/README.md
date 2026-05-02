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
  |-- "guilty until proven innocent" liveness prover
  |-- accumulates per-frame evidence + liveness proof score (0-100)
  |-- detects incidents (spoof bursts, frozen face, MiniFASNet instability)
  |-- peak-sensitive verdict (worst-window prevents dilution)
  |-- session report with liveness breakdown on exit
  |
Pipeline (per-frame)
  |-- MediaPipe face detection (~2ms)
  |-- MediaPipe FaceLandmarker (478 points, shared across analyzers)
  |-- IoU multi-face tracking
  |-- 13 analyzers in 3 layers:
  |
  |  Layer 1 — Pixel Forensics:
  |     MiniFASNet ONNX       (5.0x) +94.7 gap  PROVEN
  |     Screen Flicker        (3.0x) 50/60Hz temporal aliasing
  |     Device Boundary       (2.5x) phone bezel detection
  |     Screen Replay         (0.5x) FFT + skin color
  |
  |  Layer 2 — Behavioral Signals:
  |     Micro-Tremor          (2.5x) 8-12Hz involuntary oscillation
  |     Landmark Variance     (2.0x) 478-point motion tracking
  |     Blink (EAR)           (0.5x) V-shape validated blinks
  |     Temporal              (0.3x) micro-motion naturalness
  |     rPPG                  (0.0x) DISABLED — false pulse on screens
  |
  |  Layer 3 — Environment:
  |     Background Grid       (1.5x) 6x4 cell stability monitoring
  |     AR Filter             (0.3x) heuristic (ONNX model pending)
  |     Texture               (0.1x) anti-correlated, suppressed
  |     Moire                 (0.1x) anti-correlated, suppressed
  |
  |-- MultiClassFuser (calibrated weights) -> 7-category probabilities
  |-- LivenessProver: blinks(25) + motion(20) + rotation(15) + expression(15) = 75 max
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

ISO 30107-3 metrics: BPCER 0.00% | APCER 30% | ACER 15% | **Grade C**

Session-level accuracy (4 test scenarios):
- **Real face**: LIVE 78%, liveness 63/100 PROVEN, 5 blinks, 0 incidents
- **Phone screen photo**: SPOOF 43%, liveness 23/100, 0 blinks, 7 incidents
- **Printed photo**: SPOOF 58%, liveness 50/100, 3 incidents
- **Video replay**: LIVE 60% — remaining challenge (video shows real blinks/motion)

Key calibration findings:
- **MiniFASNet**: Only reliable per-frame discriminator (+94.7 gap)
- **rPPG**: Anti-correlated — detects screen flicker as false pulse (disabled)
- **Texture/Moire**: Anti-correlated on screen attacks (suppressed to 0.1 weight)
- **Blink EAR 0.20**: Works for real faces, some false blinks on video playback

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
- [x] Phase 2.5: Device boundary, calibrated fusion, session engine
- [x] Phase 3: Temporal analyzers (blink EAR, rPPG, landmark variance)
- [x] Phase 3.5: Guilty-until-proven liveness architecture
- [x] Phase 3.6: Three-layer detection (screen flicker, micro-tremor, background grid)
- [ ] Phase 3.7: Connect fusion ↔ liveness prover, fix video replay
- [ ] Phase 4: Data collection (AR-filter dataset via amispoof.com)
- [ ] Phase 5: AR filter detector training (MobileNetV3-Small)
- [ ] Phase 6: Paper + evaluation (APCER/BPCER/ACER → Grade B target)
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
