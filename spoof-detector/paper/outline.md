# AR-Spoofing: Session-Based Multi-Method Face Presentation Attack Detection

## Paper Outline (v2 — Session Architecture)

### Target Venues
1. **BIOSIG 2026** (September, Darmstadt) — biometrics focused
2. **IJCB 2026** — International Joint Conference on Biometrics
3. **IEEE FG 2027** — Face and Gesture Recognition

### Abstract (~250 words)
Face anti-spoofing (FAS) systems typically classify individual frames as real or fake, ignoring the temporal dimension of real-world attacks. We present a **session-based spoof detection engine** that accumulates evidence across 5 seconds to 3 hours, producing verdicts that strengthen over time. Our system combines six per-frame analyzers (MiniFASNet ONNX, device boundary detection, screen replay heuristics, moire pattern analysis, texture analysis, and temporal consistency) with a calibrated multi-class fusion engine that maps signals into a 7-category spoof taxonomy: real, static image, video replay, 3D mask, heavy makeup, AR filter, and deepfake injection. Calibration weights are derived from ground-truth testing, revealing that MiniFASNet alone achieves +94.7 discrimination gap while texture and moire analyzers are anti-correlated for screen attacks. The session engine uses peak-sensitive verdict computation — a single sustained spoof burst permanently affects the session verdict, preventing dilution by real-face frames (critical for exam proctoring). On our labeled test set, the system achieves 100% accuracy on per-capture classification and correctly identifies spoof sessions with 13 incidents and SPOOF verdict at 63% confidence. We additionally present a labeled dataset of AR-filter attacks collected via amispoof.com and a lightweight MobileNetV3-Small detector targeting the AR-filter gap in existing FAS benchmarks.

### 1. Introduction
- Biometric face authentication is ubiquitous
- FAS literature focuses on per-frame classification
- Real attacks happen over sessions (5 seconds to hours)
- Session-based aggregation reveals signals invisible to single frames
- AR filters remain a blind spot in existing benchmarks

### 2. Related Work
- Classical FAS: LBP, frequency analysis, moire detection
- Deep FAS: CDCN, FAS-SGTD, Silent-Face, MiniFASNet
- Benchmarks: OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof (all per-frame)
- Session-based systems: proctoring literature, limited FAS integration
- AR filter detection: largely unaddressed

### 3. Spoof Taxonomy (7 Categories)
- Real (genuine live person)
- Static Image (printed photo, digital still on screen)
- Video Replay (pre-recorded video on display)
- 3D Mask (silicone/latex mask)
- Heavy Makeup (contouring, prosthetics)
- AR Filter (Snapchat, Instagram, FaceApp, OBS)
- Deepfake Injection (virtual webcam, DeepFaceLive)

### 4. System Architecture

#### 4.1 Per-Frame Analysis Pipeline
- Face detection (MediaPipe Tasks API, ~2ms)
- Multi-face tracking (IoU-based persistent IDs)
- 6 parallel analyzers per face:
  - MiniFASNet ONNX (3.0 weight, +94.7 gap)
  - Device Boundary (2.5 weight, phone bezel detection via Canny+Hough)
  - Screen Replay (0.5 weight, FFT+skin color+specular)
  - Temporal (0.3 weight, micro-motion naturalness)
  - Texture (0.1 weight, Laplacian+color+FFT — anti-correlated)
  - Moire (0.1 weight, Gabor bank+FFT — anti-correlated)
- Multi-class fusion engine with calibrated weights

#### 4.2 Session Engine (Novel Contribution)
- Ingests per-frame classifications into rolling session state
- Multi-timescale signal accumulation:
  - Per-frame (0-33ms): MiniFASNet, device boundary
  - Short-term (1-5s): blink detection, micro-motion baseline
  - Medium-term (5-30s): blink rate, movement naturalness
  - Long-term (30s-3hr): identity consistency, behavior patterns
- Incident detection: P(real) drops, frozen face, face missing, spoof bursts
- **Peak-sensitive verdict**: blends average with worst-window (50/50)
  - Prevents spoof dilution in mixed sessions
  - Critical for proctoring: 10% cheating = still cheating
- Session lifecycle: WARMING_UP -> ANALYZING -> CONCLUDED

#### 4.3 Fusion Weight Calibration
- Ground-truth testing protocol
- Discrimination gap analysis per analyzer
- Evidence-based weight assignment
- Anti-correlated analyzers suppressed (0.1 weight)

### 5. AR Filter Detector (Phase 5)
- Architecture: MobileNetV3-Small backbone
- Key insight: spatial-frequency artifacts at filter boundaries
- Training: PyTorch, mixed precision, GTX 1650
- ONNX export for CPU deployment (~15ms)
- Dataset collection via amispoof.com

### 6. Dataset Collection
- amispoof.com web platform
- Consent framework (KVKK/GDPR)
- 5-second sessions, 720p, 30fps
- Filter sources: Snapchat, Instagram, TikTok, FaceApp, OBS
- Labeling: tool-assisted + manual verification

### 7. Experiments

#### 7.1 Per-Analyzer Discrimination Analysis
- MiniFASNet: real=99.9, spoof=5.1, gap=+94.7
- Device boundary: real=34.2, spoof=15.0, gap=+19.2
- Screen replay: real=46.7, spoof=37.1, gap=+9.6
- Texture: real=72.1, spoof=78.4, gap=-6.3 (anti-correlated)
- Moire: real=39.1, spoof=44.1, gap=-5.0 (anti-correlated)

#### 7.2 Session-Level Accuracy
- Real-only session: LIVE 95% confidence, 0 incidents
- Spoof-only session: SPOOF 63% confidence, 13 incidents
- Mixed sessions: peak-sensitive verdict drops confidence

#### 7.3 Ablation Study
- Impact of calibrated vs equal weights
- Impact of peak-sensitive vs average-only verdict
- Individual analyzer contribution

#### 7.4 Cross-Filter Generalization (Phase 5)
- Train on Snapchat, test on Instagram/TikTok/OBS
- Per-filter-type breakdown

### 8. Results
- Confusion matrix per attack type
- ROC curves, APCER/BPCER/ACER
- Session verdict accuracy across durations (5s, 30s, 5min)
- Latency benchmarks

### 9. Discussion
- Why per-frame FAS misses session-level attacks
- Calibration methodology transfers to new analyzers
- Limitations: makeup false positives, no depth sensor
- Ethical considerations: consent, KVKK/GDPR compliance

### 10. Conclusion
- Session-based approach > per-frame classification
- Calibrated fusion > equal weighting
- AR-filter detection remains a gap (addressed by Phase 5 model)
- Open-source dataset and model weights
