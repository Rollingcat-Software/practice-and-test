# Liveness & Anti-Spoofing — Comprehensive Investigation 2026-05-09

> Read-only audit of every liveness / anti-spoof artifact across `biometric-processor`
> branches and `practice-and-test` submodules. Cross-referenced against the prod
> binary `b670f218` (api `b670f218` + bio `a0a763b5`, 2026-05-07).

## Executive summary

- **There is *substantially more* anti-spoof code in side branches than in prod.**
  `origin/working_spoof_detection` (the most advanced of Aysenur's lines) adds
  ~9.5k LoC across 74 app files vs `main`, including Gabor/FFT moire analysis,
  flash-color challenges, focal-blur/cutout anomaly detection, face-usability
  gates and a 4 803-line `live_liveness_preview.py` desktop tuner. Prod ships
  only the UniFace MiniFASNet ONNX backend gated by DeepFace's anti-spoof flag.
- **The strongest standalone work is the user's own `practice-and-test/spoof-detector/`**
  — a session-engine architecture, 14 analyzers, calibrated 7-class fusion, 60+
  unit tests, and a near-publishable paper outline (BIOSIG/IJCB 2026 target).
  It is the only artifact in the inventory with measured ISO 30107-3 numbers
  (`BPCER 0.00% / APCER 30% / ACER 15%`, Grade C, 4 scenarios).
- **Aysenur's branches are not mergeable as-is.** Every branch reverts the
  Dependabot security pins in `requirements.txt`, deletes shipped tests
  (`test_embedding_cipher`, `test_request_size_limit`, gesture liveness),
  drops `embedding_cipher.py`, drops migrations work, and bundles a 6.5 MB
  `yolov8n.pt` binary. The signal is real but the deliverable is unreviewably
  large and security-regressing.
- **Prod gap is narrow but real.** UniFace MiniFASNet alone covers print and
  some screen replay; it has no replay-burst aggregation, no flash challenge,
  no rPPG, no moire/Gabor screen check, no device-bezel detection, and no
  session-level verdict. `ANTI_SPOOFING_ENABLED=true` only toggles
  *DeepFace's* per-frame veto (`app/application/use_cases/check_liveness.py:155-186`),
  not Aysenur's pipeline.
- **Recommendation.** Ship the user's `spoof-detector` as the basis: extract
  it as a sidecar microservice (or library), integrate the MiniFASNet+device-boundary
  layer into `biometric-processor` first, write the paper around the
  session-engine + calibrated fusion novelty. Treat Aysenur's `working_spoof_detection`
  as a *donor branch* — cherry-pick `screen_replay_anti_spoof.py`,
  `moire_pattern_analysis.py`, `flash_spoof_analyzer.py`,
  `device_spoof_risk_evaluator.py`, `cutout_anomaly_detector.py`,
  `face_usability_gate.py`, `critical_region_visibility_gate.py`, and
  `light_challenge_service.py` into focused PRs against `main` (each with
  its own tests and zero `requirements.txt` regressions).

---

## Inventory

### Aysenur's branches (biometric-processor)

| Branch | Tip | Unique commits vs main | Authors | Key techniques | Wired into prod? | Tests? |
|---|---|---:|---|---|---|---|
| `origin/liveness_capture` | `504067e` Color Shaded Screen | 6 | Ayşe Gülsüm EREN | enhanced+UniFace baseline, face bbox, "color-shaded screen" | No | New tests added but several existing tests deleted |
| `origin/liveness_capture2` | `504067e` | 6 | Ayşe Gülsüm EREN | identical commit set to `liveness_capture` (no divergence found in tip log) | No | same |
| `origin/working_spoof_detection` | `cbdbe0b` Spoof Detection | 27 | Ayşe Gülsüm EREN + Aysenur15 | superset of `liveness_capture` + Gabor/FFT moire + flash challenge + cutout anomaly + face-usability gate + critical-region visibility gate + reaction baseline + sklearn `train_spoof_classifier.py` + `test_data_collector.py` + 4 803-line tuner | No | adds `test_hybrid_fusion_evaluator.py`, `test_critical_region_visibility_gate.py`, `test_face_quality_illumination_gate.py`, `test_face_usability_gate.py`, `test_live_liveness_preview.py` (2 314 lines); deletes ~10 prod tests |
| `origin/Spoof-Detection` | `0685f05` No Face Update | 9 | Aysenur15 | subset of `working_spoof_detection` (no-face handling fixes, color-shaded screen, face bbox, liveness score) | No | same |
| `origin/fix/liveness-cascade-frr-reduction` | `b730a6d` | 27 | Ahmet + Aysenur | hijab/head-turn FRR fix (nose-alone occlusion no longer critical), built on `working_spoof_detection` | No | shares branch tests |
| `origin/fix/liveness-p0-frr-reduction` | `00bf4d7` | 28 | Ahmet | EMA freeze on skipped frames + decision-guard re-enable + multiple revert/iter cycles (P1, P2, cascade-guard) | No | shares branch tests |
| `origin/fix/liveness-p3-frr-reduction` | `1229f48` | 24 | Aysenur15 | "P3 phase" + previous P0 work | No | shares branch tests |
| `feat/anti-spoof-pipeline` (local-only) | `9ca51a2` | 6 | Aysenur15 + Ahmet | cleaned-up squash of the anti-spoof integration: moire + device-spoof + reaction + baseline + flash-spoof + cutout + screen-replay veto + hybrid backend + strict-profile config | No (local branch, never pushed to PR) | yes (subset) |

#### Per-branch notes

**`origin/liveness_capture` / `liveness_capture2`** — Identical history. Promotes
`enhanced` to default backend (`set enhanced as default liveness baseline`)
and adjusts `EnhancedLivenessDetector` confidence/passive scoring. Adds
`color-shaded-screen` heuristic (likely a macro flicker check on display
characteristic colors). Limited scope. Memory's claim that this branch already
has rPPG + screen-replay + MRZ does **not** match the actual file diff —
those features land in `working_spoof_detection` and never include MRZ in
`biometric-processor` at all (MRZ work is in `practice-and-test/`).

**`origin/working_spoof_detection`** — The flagship branch. Net change:
`+45 249 / −23 327` across **624 files**. Substantively adds:

- `app/infrastructure/ml/liveness/critical_region_visibility_gate.py` (901 lines):
  per-region (left/right eye, nose, mouth, lower-face) pixel-based occlusion gate
  with hijab-aware token exclusions (`mouth_roi_color_invalid`,
  `mouth_chrominance_anomaly` excluded — see comments at lines 41-50 of that
  file). Returns `CriticalRegionVisibilityResult` with blocking_regions,
  suspicious_regions, occlusion_score. Threshold-based, no learned model.
- `app/infrastructure/ml/liveness/face_quality_illumination_gate.py` (241 lines):
  brightness uniformity + shadow-asymmetry + over/underexposed-region detection.
- `app/infrastructure/ml/liveness/face_usability_gate.py` (341 lines):
  composes the above two gates with frame-confirmation streaks
  (LOW_QUALITY_CONFIRM_FRAMES=2, OCCLUSION_CONFIRM_FRAMES=2,
  NO_FACE_CONFIRM_FRAMES=6). Outputs a `FaceUsabilityResult` that can `block`
  liveness scoring entirely when the face is unusable.
- `app/infrastructure/ml/liveness/moire_pattern_analysis.py`: Gabor bank
  (4 orientations, ksize 21, sigma 5, lambda 10) + FFT periodicity, returns
  `moire_risk` ∈ [0,1] and `moire_score` ∈ [0,100] (also lives in main today).
- `app/infrastructure/ml/liveness/screen_replay_anti_spoof.py`: 5-signal cheap
  layered fusion (FFT, Gabor, Laplacian, skin-coverage, specular). Hard veto
  triggers when ≥ 2 sub-signals fall below 30. Already shipped in main and
  consumed by `EnhancedLivenessDetector` (`app/infrastructure/ml/liveness/enhanced_liveness_detector.py:31,152,192-201`).
- `app/application/services/flash_spoof_analyzer.py`: per-region (forehead,
  cheeks, nose) BGR delta analysis between baseline and flash frames; classifies
  diffuse 3D skin response vs planar replay-media response.
- `app/application/services/device_spoof_risk_evaluator.py`: aggregates
  moire_risk, reflection_risk, flicker_risk, flash_response_score,
  hole_cutout_risk, focal_blur_anomaly_risk, screen_frame_risk into a
  `device_replay_risk` with hard-coded weights (moire 0.28, reflection 0.20,
  flicker 0.14, flash 0.28, screen-frame 0.10). Already in main but not wired
  to `/verify`.
- `app/application/services/cutout_anomaly_detector.py`: per-region
  (eyes, mouth) hole-detection + boundary-edge + sharpness-ratio + focus-jump
  heuristics. Already in main.
- `app/application/services/light_challenge_service.py`: random color flash
  challenge (red/green/blue/white/yellow), verifies BGR shift in expected
  channel within `[minimum_delay_ms=50, expected_response_window_ms=500]` ms.
  Already in main but no public route exposes it.
- `app/application/services/hybrid_fusion_evaluator.py` (190 lines): fuses
  pretrained MiniFASNet score with flash/moire/device signals. Weights:
  pretrained 0.30, flash 0.30, moire 0.20, device 0.20. Hard-veto if
  flicker > 0.85 or (flicker > 0.75 AND device_replay > 0.55). **Not in main.**
- `app/tools/live_liveness_preview.py` (4 803 lines): standalone OpenCV
  desktop tuner with frame metrics, temporal aggregator, baseline calibrator,
  background reaction evaluator. Effectively a research workbench; replaces
  the deleted `live_liveness_preview.py` paths.
- `app/tools/test_data_collector.py` (789 lines): interactive CV2 capture
  tool that saves `(frame, metrics_json, label)` triples into
  `data/test_frames/`. Used to generate ground truth for the sklearn
  classifier below.
- `app/tools/train_spoof_classifier.py` (268 lines): sklearn
  `GradientBoostingClassifier` / `RandomForest` / `LogisticRegression` /
  `SVC` 5-fold CV trainer that pickles `models/spoof_classifier.pkl`.

  Authorship: 6 commits by **Ayşe Gülsüm EREN**, 21 by **Aysenur15**
  (aysenurarici@hotmail.com). Latest commit 2026-05-06 20:57. The branch
  reverts `requirements.txt` from `tensorflow-cpu==2.21.0`+pinned-transitives
  back to `tensorflow-cpu==2.15.0` and removes Dependabot security pins —
  this alone makes it un-mergeable without significant cleanup.

**`origin/Spoof-Detection`** — A subset of `working_spoof_detection`. Same
authorship pattern (Aysenur15) but stops earlier in the iteration. Effectively
superseded.

**`origin/fix/liveness-cascade-frr-reduction` / `p0-frr-reduction` / `p3-frr-reduction`** —
Three sibling FRR-tuning branches built on top of `working_spoof_detection`'s
post-Spoof-Detection commits. P0 branch is the most disciplined: each fix
(`fix(liveness): P0 FRR reduction — freeze EMA on skipped frames + re-enable decision guards`)
is followed by a revert if the user pushed back, suggesting Ayhmet was
shepherding the FRR knob. Cascade branch adds the hijab fix
(`fix(liveness): nose-alone physical block is not critical occlusion (head-turn FRR)`).
None of these can land without first landing the parent `working_spoof_detection`.

**`feat/anti-spoof-pipeline` (local)** — 6 commits, presents as a clean
"squashed-history" view of the same work but **never pushed**. Net diff
vs main: `+5 248 / −7 102` across 80 files — much more contained than the
huge upstream branches because it's restricted to just the anti-spoof
modules and drops the test-deletion noise. **This is the most
review-friendly version of Aysenur's contribution and is the right
starting point if we want to upstream her work.**

### `practice-and-test/` work (the user's R&D)

| Directory | Owner | Models / techniques | Standalone? | Quality |
|---|---|---|---|---|
| `spoof-detector/` | Ahmet (user) | MiniFASNet ONNX, MediaPipe FaceLandmarker (478pt), IoU tracking, 14 analyzers in 3 layers, calibrated 7-class fusion, session engine with peak-sensitive verdict, blink (EAR), rPPG, screen-flicker, micro-tremor, landmark-variance, background-grid, AR-filter (heuristic), texture, moire, device-boundary, screen-replay, temporal | Yes (own `requirements.txt`, own `main.py`, own `tests/`) | High — best-organized artifact in the inventory; has paper outline, ROADMAP, ISO 30107-3 metrics, structured logging |
| `biometric-demo-optimized/` | Ahmet | MediaPipe Tasks + 468pt landmarks + Facenet512 + Hexagonal Architecture demo; threaded camera; vectorized cosine search; YOLO card detector | Yes | Moderate — used as the visual reference for the web `FacePuzzle` overlay (per `CLAUDE.md`) |
| `DeepFace_InsightFace_Pipeline/` | Ahmet | Side-by-side DeepFace vs InsightFace comparison scripts, FaceNet baseline, plain ID-pipeline | Yes | Low — pure exploration scripts, no paper-grade artifacts |
| `GestureAnalysis/` | Ahmet | MediaPipe HandLandmarker (`hand_landmarker.task` shipped), `anti_spoof.py`, motion analyzer, `liveness_session.py`, math/shape/sequential challenge sessions, finger-touch detector | Yes | Moderate — the gesture/active-liveness counterpart; some overlap with `biometric-processor`'s active liveness flow |
| `archive/`, `optimization-experiments/` | Ahmet | mostly stale, archived | Yes | n/a |

#### Per-directory notes

**`practice-and-test/spoof-detector/`** — The only artifact in the entire
audit with explicit ISO 30107-3 measurements
(`README.md:134`: `BPCER 0.00% | APCER 30% | ACER 15% | Grade C`).
3 274 LoC of actual source code, 60 unit tests, structured taxonomy
(`src/domain/taxonomy.py`), incident detection, peak-sensitive verdicts.
The novelty claim is sound: per-frame FAS is well-trodden; *session-based*
FAS with multi-timescale signal accumulation (per-frame → 1-5s → 5-30s →
30s-3hr) is genuinely under-explored. Paper outline targets BIOSIG 2026 /
IJCB 2026 — both legitimate venues. Phase 4-7 (AR-filter dataset
collection via amispoof.com) is unfinished. `data/captures/` and
`data/annotations/` are present but **empty** at HEAD (`ls` returns 0
entries) — the calibration numbers in the README came from sessions that
weren't committed. This is the largest publishability gap.

**`practice-and-test/biometric-demo-optimized/`** — Production-style hex-arch
demo with `presentation/ui/drawing.py` that the web-app `FacePuzzle`
component mirrors (`CLAUDE.md` references it explicitly). Useful as a
reference, not as a source of anti-spoof signals.

**`practice-and-test/GestureAnalysis/`** — Active-liveness companion. Has its
own `anti_spoof.py` and `liveness_session.py`. Not yet examined in depth,
but worth keeping on the radar as the integration layer if the project
adds back challenge-response active liveness (which `biometric-processor`'s
`light_challenge_service.py` would also benefit from).

### Current production state (`main` branch)

What's *actually* compiled into the running prod image (`a0a763b5`,
2026-05-07 07:28 UTC):

- **Backend selection** (`.env.prod`): `LIVENESS_BACKEND=hybrid`,
  `LIVENESS_UNIFACE_DEFAULT_ENABLED=True`, `ANTI_SPOOFING_ENABLED=true`,
  `ANTI_SPOOFING_THRESHOLD=0.5`. (Note: `CLAUDE.md` claim of
  `LIVENESS_MODE=passive` + `LIVENESS_BACKEND=uniface` is now stale.)
- **Resolved detector**: `HybridLivenessDetector`
  (`app/infrastructure/ml/liveness/hybrid_liveness_detector.py`) =
  `EnhancedLivenessDetector` first → UniFace as second opinion → fall
  back to enhanced verdict if UniFace returns indeterminate.
- **EnhancedLivenessDetector** (`enhanced_liveness_detector.py:31-201`)
  *does* invoke `ScreenReplayAntiSpoof` (5-signal Gabor/FFT/Laplacian/skin/specular)
  on every frame and keeps a 3-frame veto streak. So screen-replay
  hard-veto **is shipping in prod today** — this contradicts the audit
  memory saying screen-replay landed only in Aysenur's branches.
- **Anti-spoof gate**: DeepFace's built-in anti-spoof model runs in
  `extract_face_with_detection` (called from `check_liveness.py:153`),
  vetoes via `LIVENESS_VERDICT_POLICY=conservative`
  (`app/application/use_cases/check_liveness.py:175-186`).
- **rPPG**: `RPPGAnalyzer` is *in main* and is wired *only* into
  `LiveCameraAnalysisUseCase` (`app/application/use_cases/live_camera_analysis.py:32,134-147,515`)
  which is exposed via `app/api/routes/live_analysis.py:50,119`. rPPG
  is **not** invoked from `/verify` or `/enroll` paths.
- **Device spoof / cutout / flash / moire / hybrid-fusion**: present in
  main as standalone modules but **not invoked** by `check_liveness` or
  `live_camera_analysis`. They're dead code outside `live_liveness_preview.py`
  (developer tuner). This is the largest "shipped but unused" surface.
- **Active liveness / gesture liveness / biometric-puzzle**: still in
  main, used by `/verify_puzzle` and biometric-puzzle web flow.

#### Gap list (prod vs. Aysenur's branches vs. spoof-detector)

| Capability | In prod `main`? | In Aysenur's branches? | In `spoof-detector/`? |
|---|:-:|:-:|:-:|
| MiniFASNet ONNX | yes (UniFace) | yes | yes |
| Texture (LBP) | yes | yes | yes |
| Screen-replay (Gabor+FFT+skin+specular) | **yes (active veto)** | yes | yes |
| Moire (Gabor bank) | yes (module exists) | yes (wired into device-spoof) | yes |
| Device boundary (phone bezel) | no | yes | yes |
| Flash/color challenge | no (module unused) | yes (light_challenge_service wired) | no |
| Cutout anomaly | no (module unused) | yes (wired) | no |
| Face usability gate (occlusion+illumination) | no | yes | no |
| Critical region visibility | no | yes (901 LoC, hijab-aware) | no |
| Hybrid fusion evaluator (weighted multi-signal) | no | yes | yes (multi-class) |
| rPPG | yes (only in `/live_analysis`) | no (deferred) | yes (disabled — false pulse on screens) |
| Blink (EAR) | yes (in EnhancedLivenessDetector) | yes | yes |
| Smile / mouth movement | yes | yes | no |
| Micro-tremor (8-12Hz oscillation) | no | no | yes |
| Landmark variance (478pt) | no | no | yes |
| Screen flicker (50/60Hz aliasing) | no | yes (flicker_risk in device-spoof) | yes |
| Background-grid stability | no | no | yes |
| AR filter detector | no | no | yes (heuristic; ONNX planned) |
| Session engine (peak-sensitive verdict) | no | no | yes |
| Liveness prover (active challenges) | active-liveness/puzzle exists | flash challenge added | yes (blinks/motion/rotation/expression) |
| Calibrated fusion weights from ground truth | no | no | yes |
| ISO 30107-3 metrics measured | no | no | yes (Grade C) |
| 7-category spoof taxonomy | no | no | yes |
| Sklearn meta-classifier | no | yes (Aysenur, GradientBoosting/RF/LR/SVC) | no |

---

## Technical deep-dive

### 1. UniFace MiniFASNet (prod baseline)

**Algorithm.** Binary real/spoof CNN distilled to ONNX via UniFace 3.0+.
Single forward pass on a face crop; no temporal accumulation. Per
`spoof-detector/README.md:142-147`, the user measured a **+94.7
discrimination gap** on real vs spoof in their own ground-truth set —
this is the strongest single signal available.

**Files.** `app/infrastructure/ml/liveness/uniface_liveness_detector.py`,
gated by `LIVENESS_UNIFACE_DEFAULT_ENABLED`, with cache pinned to
`/app/uniface-cache` (named volume, uid 100, see
`/opt/projects/fivucsas/CLAUDE.md`).

**Limitation.** Per-frame only. No replay-burst aggregation. Confused
by high-quality screen replays where MiniFASNet alone votes "live" because
the screen renders convincing skin texture (the user's `data/test_protocol`
phase notes: video-replay session reads as LIVE 60% — see
`spoof-detector/ROADMAP.md`).

### 2. Screen-replay anti-spoof (shipping in prod, not headlined)

**Algorithm.** Five cheap signals fused into a [0, 100] live-likeness
score with hard-veto policy:
- FFT periodicity (radial energy ratio in a band centered on
  `fft_ratio_center=0.85`, width `0.20`)
- Gabor bank `analyze_moire_pattern()` (4 orientations, sigma 5, lambda 10)
- Laplacian variance (blur-vs-sharp screen artifacts)
- Skin coverage (HSV mask, expected `[0.20, 0.95]`)
- Specular coverage (luminance percentile, warn=0.020, fail=0.060)

Hard veto trips when `low_signal_count >= 2` and signals < 30, capped at
`veto_score_cap=35`. Veto streak of 3 needed inside
`EnhancedLivenessDetector` before it propagates to the verdict
(`enhanced_liveness_detector.py:153,154,198`).

**Files.** `app/infrastructure/ml/liveness/screen_replay_anti_spoof.py:78-128`,
`app/infrastructure/ml/liveness/moire_pattern_analysis.py:1-130`.

**Limitation.** Tuned by hand. No per-attack-class weighting. False
positives on macro-prints; false negatives on high-DPI screens at
distance. The `_blur_floor=25.0` short-circuit
(`screen_replay_anti_spoof.py:90-105`) silently abstains on out-of-focus
frames, which an attacker can exploit by deliberately defocusing.

### 3. Aysenur's hybrid-fusion / device-spoof pipeline (working_spoof_detection only)

**Algorithm.** Linear weighted sum
`w_pre · pretrained + w_flash · flash + w_moire · moire + w_dev · device`
with weights `(0.30, 0.30, 0.20, 0.20)` and threshold 0.45
(`app/application/services/hybrid_fusion_evaluator.py:14-32,52-95`).
Hard veto if `flicker > 0.85` or (`flicker > 0.75` AND
`device_replay > 0.55`) — bypasses the linear sum entirely.

**Limitation.** Weights are hand-coded, not learned from data. The
sklearn `train_spoof_classifier.py` exists but its data dir is empty at
HEAD (`data/training_data.csv` referenced but never committed). Treat
the weights as priors, not calibration.

### 4. User's session engine (spoof-detector)

**Algorithm.** Per-frame
`pipeline.process(frame)` → `engine.ingest(analysis, frame)` accumulates
into `SessionState`. Multi-timescale signals collected at frame, 1-5s,
5-30s, 30s-3hr horizons. Verdict =
`0.5 * average_p_real + 0.5 * worst_window_p_real` ("peak-sensitive" —
single sustained spoof burst permanently degrades the session even if
mostly real). Liveness prover (separate from category fusion) accumulates
gold proofs: blinks 25, motion 20, rotation 15, expression 15 → max 75
(`spoof-detector/src/application/session_engine.py:1-60`,
`spoof-detector/src/application/liveness_prover.py:1-338`).

**Limitation.** Single-author, untested at scale. Calibration data
(`data/captures/`, `data/annotations/`) not in git. Phase 3.7 ("connect
fusion ↔ liveness prover, fix video replay") and Phase 4-7 (AR-filter
dataset, MobileNetV3 training, paper) are unfinished
(`spoof-detector/README.md:160-173`).

### 5. rPPG (prod, but only in `/live_analysis`)

**Algorithm.** Sliding 5-second window of mean green-channel intensity,
detrend, Butterworth bandpass [0.83, 2.5] Hz (50-150 BPM), FFT, dominant
frequency → BPM. Score = `min(signal_strength * 2, 1)` if
`signal_strength > 0.3` else 0.2 (`rppg_analyzer.py:43-95`).

**Limitation.** Confirmed false-pulse on screens
(`spoof-detector/README.md:144` — "rPPG: anti-correlated, detects screen
flicker as false pulse, disabled"). Currently weighted at 0.15 in
`live_camera_analysis.py:37`. **This is a known issue and rPPG should
be either removed from `/live_analysis` or only weighted when fused with
device-replay-low-risk gating.**

### 6. Active light challenge (prod modules, no route)

**Algorithm.** `LightChallengeService.generate_challenge()` returns a
random color from {red, green, blue, white, yellow}, expects screen flash
within `[50, 500] ms`, verifies BGR mean shift in expected channel
exceeds `min_color_shift=0.05`
(`light_challenge_service.py:1-90`). Aysenur's `flash_spoof_analyzer.py`
adds spatial verification: per-region (forehead, cheeks, nose) diffuse-vs-specular
response classifies skin (3D, region-correlated) vs replay media (planar).

**Limitation.** Browser flash requires viewport overlay control. The
`web-app` widget needs a corresponding step component, which doesn't
exist in main. This is plumbing-blocked, not algorithmically blocked.

---

## Convergence map

Spoof-attack coverage matrix (✅ covered, ⚠ partial, ❌ uncovered):

| Attack class | UniFace MiniFASNet (prod) | Screen-replay 5-signal (prod) | Aysenur hybrid (branches) | spoof-detector session engine |
|---|:-:|:-:|:-:|:-:|
| Printed photo | ✅ | ⚠ | ✅ | ✅ |
| Static digital photo on screen | ⚠ | ✅ | ✅ | ✅ |
| Video replay (screen) | ⚠ | ⚠ | ✅ (flash + flicker) | ⚠ (acknowledged FAIL — README target Phase 3.7) |
| 3D mask (silicone/latex) | ⚠ | ❌ | ⚠ (flash specular helps) | ⚠ |
| Heavy makeup | ❌ | ❌ | ❌ | ❌ (Phase 5 planned) |
| AR filter (Snapchat/IG/FaceApp) | ❌ | ❌ | ❌ | ⚠ (heuristic; Phase 5 ONNX planned) |
| Deepfake injection (virtual cam) | ❌ | ❌ | ❌ | ⚠ (active illumination Phase 5) |
| Cutout / hole-mask | ❌ | ❌ | ✅ (cutout_anomaly_detector) | ❌ |
| Hijab/headscarf occlusion (legitimate) | n/a | n/a | ✅ (face_usability_gate hijab fix) | n/a |

The composite "coverage if we ship everything" leaves only AR filter,
deepfake injection, and heavy makeup uncovered. AR filter is the user's
chosen paper novelty (Phase 5).

---

## Extraction proposal

### Subproject A: Academic paper

**Working title.** "Session-Based Multi-Method Face Presentation Attack
Detection with Calibrated Multi-Class Fusion" — already drafted at
`practice-and-test/spoof-detector/paper/outline.md`.

**Likely contribution / novelty.**
1. Session-based verdict engine (vs per-frame classification — most FAS
   literature is single-frame).
2. Peak-sensitive verdict computation that prevents spoof dilution in
   mixed sessions (concretely: 10 % cheating = SPOOF, not LIVE).
3. Calibrated fusion weights derived from ground-truth testing showing
   that texture and moire are *anti-correlated* with screen attacks
   (a non-obvious empirical finding contradicting LBP-based FAS papers).
4. AR-filter detection dataset (Phase 5, via amispoof.com).

**Baseline comparisons available.** OULU-NPU, SiW, CASIA-SURF,
CelebA-Spoof are namechecked but **not yet run**. The current numbers
(BPCER 0.00 / APCER 30 / ACER 15) are on the user's own 4-scenario set,
not a public benchmark. This is the single biggest publishability gap.

**What's missing for first submission.**
- Run on at least one public benchmark (OULU-NPU is the cheapest start —
  4 protocols, ~2k videos, free academic license).
- Collect ≥ 500 AR-filter samples (Phase 5).
- Ablation: session engine vs averaged per-frame; calibrated vs equal
  weights; with/without peak-sensitive verdict.
- Cross-validation: ≥ 100 samples per scenario, not 4 sessions.
- Session-engine throughput numbers on CX43 CPU (the user's only
  available hardware).

**Estimated effort to first draft.** 6–10 weeks if user does it solo,
3–5 weeks with a co-author handling OULU-NPU evaluation. The text
scaffolding is already in `paper/outline.md`. The blocker is **data**
(public benchmark + AR-filter set).

### Subproject B: Professional working module

**Architecture.** Two-tier extraction:

1. **Library tier**: lift `practice-and-test/spoof-detector/src/` into a
   pip-installable package `fivucsas-antispoof` (single namespace
   `fas/`) with the public API:
   ```python
   from fas import SpoofDetectionPipeline, SessionEngine
   pipeline = SpoofDetectionPipeline.from_config("config.yaml")
   engine = SessionEngine()
   engine.start()
   while frame := camera.read():
       analysis = pipeline.process(frame)
       engine.ingest(analysis, frame)
   verdict = engine.conclude()  # SessionVerdict
   ```
   Already structured this way — minimal refactor needed. Tests already
   pass (60 unit tests).

2. **Sidecar microservice tier**: wrap as `antispoof-processor` FastAPI
   service exposing:
   - `POST /sessions` → `{session_id, expires_at}`
   - `POST /sessions/{id}/frames` (multipart frame + face_bbox) →
     `{frame_index, p_real, classification, incidents[]}`
   - `POST /sessions/{id}/conclude` → `SessionVerdict` JSON
   - `GET /sessions/{id}/verdict` (poll)
   - X-API-Key auth (mirror `biometric-processor` pattern)

**Plug-back into FIVUCSAS prod.** Two integration points:
- `biometric-processor` `/verify` and `/enroll` open a session, push the
  single submitted frame, conclude, take verdict — degrades gracefully
  to per-frame mode for one-shot APIs.
- `verify-app` web widget opens a session, streams frames over the 5s
  enrollment window, surfaces incidents in real-time UI, blocks the
  flow if SPOOF verdict.

**Dependencies it'd need.** Same as `spoof-detector/requirements.txt`:
`opencv-python`, `mediapipe>=0.10.9`, `uniface>=3.0`, `numpy`,
`scipy`, `onnxruntime>=1.18`. All already present in `biometric-processor`
prod image — zero new system deps.

**Estimated effort to MVP.** 2–3 weeks for library tier + 1 week for
microservice wrapper + 2 weeks for FIVUCSAS integration (api-side
client + web-side step component) = ~5–6 weeks.

### Subproject C: Surgical donor-branch upstream

Independent of A/B, the highest-ROI immediate work is upstreaming five
files from Aysenur's `feat/anti-spoof-pipeline` (the cleanest of her
branches):

1. **`face_usability_gate.py` + `critical_region_visibility_gate.py` +
   `face_quality_illumination_gate.py`** as one PR — adds
   pre-liveness occlusion/illumination gating. Hijab-aware (already
   tested by Aysenur). Reduces FRR for Marmara users.
2. **`flash_spoof_analyzer.py` + `light_challenge_service.py` route
   exposure** — `/liveness/challenge` endpoint that issues a color flash
   and verifies response. Web-app widget needs a corresponding step
   component, but the backend is ready.
3. **`hybrid_fusion_evaluator.py`** — once 1 and 2 land, wire it into
   `check_liveness.py:130-200` as the new fusion layer behind a
   `LIVENESS_FUSION_ENABLED` feature flag.
4. **`device_spoof_risk_evaluator.py` invocation** — already in main but
   dead code; light wiring into `check_liveness.py`.
5. **`cutout_anomaly_detector.py` invocation** — same: live in main, no
   call site.

Each can ship as its own PR with its own tests, against `main`, with
**zero `requirements.txt` changes** (avoiding the regression issue).

---

## Risks & open questions

- **Authorship / attribution.** `working_spoof_detection` has 6 commits
  by **Ayşe Gülsüm EREN** and 21 by **Aysenur15** (hotmail email). Are
  these the same person under different commit identities, or two
  collaborators? If the latter, both deserve paper authorship.
  `practice-and-test/spoof-detector/` is 100 % the user (Ahmet) per
  `git log --pretty='%an'` — 42 commits with no other contributors to
  that subtree. The paper should credit Aysenur(s) only if their
  branches' techniques are integrated.
- **Memory overstated `liveness_capture` content.** The audit memory
  (project_aysenur_liveness_branch.md) claims `liveness_capture` has
  rPPG + screen-replay + MRZ. The actual diff shows: enhanced/passive
  scoring + face bbox + color-shaded screen + liveness score. The rPPG
  analyzer is in `main`, not in `liveness_capture`. Screen-replay
  veto is in `main`. MRZ work is in `practice-and-test/`, not in
  `biometric-processor` at all. Memory needs an update.
- **License compatibility.** UniFace is Apache-2.0. MediaPipe is
  Apache-2.0. DeepFace is MIT. Resemblyzer is BSD-3. scikit-image is
  BSD-3. scipy is BSD-3. sklearn is BSD-3. ONNX Runtime is MIT.
  **No GPL / AGPL surface in the inventory** — paper + commercial
  productization are both safe. Verify Aysenur's branch licenses
  before merge if she pulled in any external snippets.
- **Performance budget on CX43 (no GPU).** UniFace MiniFASNet ONNX is
  ~30-50 ms per frame on CPU. Aysenur's hybrid fusion is ~+30-60 ms
  (Gabor + flash + cutout). Session-engine adds ~5 ms per frame (mostly
  bookkeeping). At 30 fps live, the budget is 33 ms/frame — **the
  full pipeline cannot run synchronously on CX43 at 30 fps**. Either
  (a) downsample to 10-15 fps, (b) async pipeline (acceptable since
  session engine doesn't need strict ordering), or (c) tier analyzers
  by cost (MiniFASNet every frame, Gabor every 3rd, rPPG every 5th).
- **Aysenur's `requirements.txt` regression.** Every push of her branches
  reverts security pins. This is the single biggest review blocker. If
  upstreaming her work, **rebase commits onto `main`'s `requirements.txt`
  before merging** — easy mechanically, but worth flagging on each PR.
- **Bundle bloat.** `working_spoof_detection` ships a 6.5 MB
  `yolov8n.pt` binary at root. Strip before merge.
- **Empty datasets.** `practice-and-test/spoof-detector/data/captures/`
  and `data/annotations/` are empty in git. The README's measured
  numbers are not reproducible from HEAD. Either commit the dataset
  (with KVKK/GDPR consent paperwork) or document the protocol +
  publish via amispoof.com.
- **Branch hygiene.** Seven liveness-related remote branches with
  significant overlap. Once a path forward is chosen, fold the
  superseded branches into `archived/` namespace and delete the
  duplicates (`liveness_capture` ≡ `liveness_capture2`).

---

## Appendix: file-path quick reference

Prod-active liveness/anti-spoof code (`biometric-processor` `main`):

- `app/application/use_cases/check_liveness.py:130-200` — verdict policy + DeepFace veto
- `app/application/use_cases/live_camera_analysis.py:32,134-147,515` — rPPG wiring
- `app/infrastructure/ml/liveness/uniface_liveness_detector.py` — MiniFASNet ONNX
- `app/infrastructure/ml/liveness/enhanced_liveness_detector.py:31,152-201` — screen-replay veto
- `app/infrastructure/ml/liveness/hybrid_liveness_detector.py` — enhanced+UniFace fusion
- `app/infrastructure/ml/liveness/screen_replay_anti_spoof.py` — 5-signal layered detector
- `app/infrastructure/ml/liveness/moire_pattern_analysis.py` — Gabor + FFT
- `app/infrastructure/ml/liveness/rppg_analyzer.py` — pulse detection
- `app/application/services/light_challenge_service.py` — flash challenge (no route)
- `app/application/services/flash_spoof_analyzer.py` — flash response analysis (orphan)
- `app/application/services/device_spoof_risk_evaluator.py` — multi-signal fusion (orphan)
- `app/application/services/cutout_anomaly_detector.py` — cutout/focal-blur (orphan)

Aysenur's branches (additions on top of `main`):

- `app/infrastructure/ml/liveness/critical_region_visibility_gate.py` (901 LoC)
- `app/infrastructure/ml/liveness/face_quality_illumination_gate.py` (241 LoC)
- `app/infrastructure/ml/liveness/face_usability_gate.py` (341 LoC)
- `app/application/services/hybrid_fusion_evaluator.py` (190 LoC)
- `app/application/services/preview_biometric_puzzle.py` (218 LoC)
- `app/tools/live_liveness_preview.py` (4 803 LoC, dev tuner)
- `app/tools/test_data_collector.py` (789 LoC)
- `app/tools/train_spoof_classifier.py` (268 LoC, sklearn)
- `app/tools/export_training_data.py` (148 LoC)

User's standalone work (`practice-and-test/spoof-detector/`):

- `src/domain/{models,session,interfaces,taxonomy}.py` — 7-class spoof taxonomy
- `src/application/session_engine.py` (494 LoC) — session verdict engine
- `src/application/liveness_prover.py` (338 LoC) — guilty-until-proven prover
- `src/application/pipeline.py` (118 LoC) — per-frame orchestrator
- `src/infrastructure/analyzers/{minifasnet,device_boundary,blink,rppg,screen_replay,screen_flicker,moire,texture,temporal,landmark_variance,micro_tremor,background_grid,ar_filter}_analyzer.py` (~3 000 LoC, 14 analyzers)
- `src/infrastructure/fusion/multi_class_fuser.py` — calibrated 7-class fusion
- `tests/test_{analyzers,domain,session}.py` — 60 unit tests
- `paper/outline.md` — BIOSIG/IJCB 2026 paper draft
- `ROADMAP.md` — Phase 1–8 plan, current state v1
