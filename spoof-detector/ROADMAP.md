# Spoof Detector Roadmap

**Project**: FIVUCSAS Session-Based Face Presentation Attack Detection
**Paper Target**: BIOSIG 2026 / IJCB 2026
**Demo**: amispoof.com
**Last Updated**: 2026-05-02

## Current State (v1)

- 9 analyzers (MiniFASNet, Device Boundary, Blink, rPPG, Screen Replay, Temporal, Texture, Moire, AR Filter)
- Session engine with incident detection and peak-sensitive verdict
- 68 tests, 23 source modules, 8 tools, ~7000 lines
- Tested accuracy: LIVE session 98% confidence, SPOOF session 63% + 13 incidents
- Blink detection working (20 blinks detected in 31s)
- rPPG pulse detection implemented (needs validation)

## Priority Levels
- P0: Must fix before next test (broken/blocking)
- P1: Must complete for paper submission
- P2: Should complete for enterprise quality
- P3: Nice to have / future work

---

## P0 — Critical Fixes

- [x] Fullscreen content scaling (frame resize to screen resolution)
- [x] rPPG FPS measurement (measure actual FPS instead of assuming 30)
- [ ] Blink analyzer caching — runs FaceLandmarker on full frame per face, should cache per frame
- [ ] Session verdict threshold tuning — phone-photo-only session still borders LIVE/SPOOF at 55%
- [ ] EAR threshold calibration — 20 blinks in 31s = 38/min is too high, likely false blinks

## P1 — Paper Requirements

### Detection Quality
- [ ] Validate rPPG on real vs screen (measure actual BPM on live face, confirm 0 on screen)
- [ ] Cross-validate blink on screen spoofs (verify 0 blinks detected on photo/video)
- [ ] Identity consistency checker — face embedding comparison across session
- [ ] Active challenge mode — ask user to turn head / blink on command / respond to color flash
- [ ] Collect OULU-NPU benchmark results for comparison table

### AR Filter Detection (Novel Contribution)
- [ ] Collect 500+ AR filter samples (50 per filter type: Snapchat, Instagram, TikTok, FaceApp, OBS)
- [ ] Collect 500+ real face baseline samples
- [ ] Train MobileNetV3-Small classifier
- [ ] Validate cross-filter generalization
- [ ] ONNX export and integration

### Evaluation
- [ ] Run test_protocol.py with all 5 scenarios, 100+ samples total
- [ ] Compute APCER/BPCER/ACER via evaluate.py
- [ ] ROC curves per attack type
- [ ] Ablation study: contribution of each analyzer
- [ ] Session duration vs accuracy curve (5s, 10s, 30s, 1min, 5min)

### Paper Writing
- [ ] Abstract (250 words)
- [ ] Introduction + related work
- [ ] Method: session architecture + calibrated fusion + peak-sensitive verdict
- [ ] Experiments section with all metrics
- [ ] Results tables and figures
- [ ] Discussion + limitations
- [ ] Submit to BIOSIG 2026 or IJCB 2026

## P2 — Enterprise Quality

### Performance
- [ ] GPU acceleration — install onnxruntime-gpu for MiniFASNet + future AR model
- [ ] Blink analyzer frame caching (avoid running FaceLandmarker N times for N faces)
- [ ] Disable anti-correlated analyzers (moire, texture) — save ~10ms/frame
- [ ] Profile and optimize to sustain 30+ FPS with all analyzers

### Robustness
- [ ] Low-light testing and compensation
- [ ] Multiple face policy (proctoring: only 1 face allowed)
- [ ] Face occlusion handling (glasses, mask worn for health)
- [ ] Camera resolution auto-detection and adaptation

### Session Engine
- [ ] Identity verification via face embedding comparison across session
- [ ] Attention/gaze tracking (is the person looking at the screen?)
- [ ] Anomaly detection for behavioral patterns (long absence, repeated identity switches)
- [ ] Session pause/resume support
- [ ] Configurable session duration limits

### Integration
- [ ] REST API wrapper for amispoof.com backend
- [ ] WebSocket streaming mode for proctoring
- [ ] FIVUCSAS biometric-processor integration (feature-flagged ARFilterLivenessDetector)
- [ ] Docker container packaging
- [ ] CI/CD pipeline

## P3 — Future Work / Research Extensions

### New Attack Detection
- [ ] Deepfake injection via virtual webcam (OBS Virtual Camera, DeepFaceLive)
- [ ] Active illumination challenge (random color flash, measure skin response)
- [ ] 3D mask detection via rPPG (mask blocks pulse)
- [ ] Voice-face consistency check (speaker verification during session)
- [ ] Audio environment analysis (detect pre-recorded audio playback)

### amispoof.com
- [ ] Static frontend (Vite + vanilla JS)
- [ ] 5-second quick test mode
- [ ] Consent framework (KVKK/GDPR)
- [ ] Ephemeral storage (30-min auto-delete)
- [ ] "Donate to research" opt-in
- [ ] Public API for third-party integration
- [ ] Domain registration and DNS setup

### Dataset & Benchmarks
- [ ] Public labeled dataset release (with consent)
- [ ] Cross-dataset evaluation (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof)
- [ ] Demographic fairness analysis
- [ ] Adversarial robustness testing

### Proctoring Features
- [ ] Browser lockdown integration (detect tab switches, screen sharing)
- [ ] Multi-camera support (front + side)
- [ ] Exam session management (start/pause/end with timestamps)
- [ ] Incident review interface for proctors
- [ ] Webhook/callback on incident detection
- [ ] LTI integration for LMS platforms (Moodle, Canvas)

---

## Self-Critique & Known Weaknesses

### Architecture
- Blink analyzer downloads 15MB model on first run — should bundle or lazy-load with user prompt
- Session engine has no face identity verification (different person could swap in)
- No active challenges — all detection is passive
- Pipeline runs ALL analyzers every frame even if some are warming up and returning 50

### Accuracy
- MiniFASNet is the ONLY proven per-frame discriminator (+94.7 gap)
- Texture and moire are ANTI-CORRELATED (score spoofs higher than real) — suppressed but still waste compute
- Phone-screen close-up (no visible bezel) fools all analyzers except MiniFASNet
- 60% accuracy on STATIC_SCREEN scenario in test protocol — needs temporal signals (blink/rPPG) to improve
- Blink rate of 38/min is double normal — EAR threshold likely too sensitive

### Paper Gaps
- No public benchmark comparison yet
- Small test set (43 labeled samples from one person, one webcam, one environment)
- AR filter model not yet trained (need data collection)
- No cross-dataset evaluation
- Single-person testing — need multi-demographic validation

### Performance
- No GPU utilization (GTX 1650 available but unused)
- FaceLandmarker runs once per face per frame (~15ms each)
- Total pipeline ~25-35ms (30 FPS) but could be 15ms with optimization

---

## Proposed Paper Title

**"Beyond Single Frames: Session-Based Face Anti-Spoofing with Calibrated Multi-Analyzer Fusion"**

Alternative:
**"Session-Level Face Presentation Attack Detection: A Multi-Signal Temporal Approach for Proctoring and Identity Verification"**
