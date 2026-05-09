# Branch: working_spoof_detection

- **Source**: `Rollingcat-Software/biometric-processor` `origin/working_spoof_detection`
- **Tip SHA**: `cbdbe0b4d13496306aa63e85689a1d20883ca7f4`
- **Authors**: Ayşe Gülsüm EREN <aysegulsumeren@gmail.com>, Aysenur15 <aysenurarici@hotmail.com>
- **Date range**: 2026-03-31 → 2026-05-06
- **Unique commits vs main**: 27
- **Diff stat vs main**: 195 files changed, 27 373 insertions(+), 359 deletions(-)
- **Source files snapshotted**: 65 (under `unique-source/`)

## Summary

Flagship branch: full anti-spoof + liveness pipeline. Net change ≈ +27k lines.
Substantively adds (per `INVESTIGATION_2026-05-09.md` §Inventory):

- `app/infrastructure/ml/liveness/critical_region_visibility_gate.py` (901 LoC) — per-region (eyes, nose, mouth, lower-face) pixel-based occlusion gate with hijab-aware token exclusions.
- `app/infrastructure/ml/liveness/face_quality_illumination_gate.py` (241 LoC) — brightness uniformity + shadow asymmetry + over/underexposure detection.
- `app/infrastructure/ml/liveness/face_usability_gate.py` (341 LoC) — composes the two gates above with per-frame confirmation streaks.
- `app/application/services/hybrid_fusion_evaluator.py` (190 LoC) — fuses pretrained MiniFASNet score with flash/moire/device signals (weights 0.30/0.30/0.20/0.20).
- `app/tools/live_liveness_preview.py` (4 803 LoC) — standalone OpenCV desktop tuner with frame metrics, temporal aggregator, baseline calibrator.
- `app/tools/test_data_collector.py` (789 LoC) — interactive cv2 capture tool that saves `(frame, metrics_json, label)` triples.
- `app/tools/train_spoof_classifier.py` (268 LoC) — sklearn `GradientBoostingClassifier` / `RandomForest` / `LogisticRegression` / `SVC` 5-fold CV trainer.

## Tests added (9 files)

- `tests/integration/test_hybrid_fusion_real_data.py`
- `tests/unit/application/test_hybrid_fusion_evaluator.py`
- `tests/unit/application/test_light_challenge_service.py`
- `tests/unit/infrastructure/ml/liveness/test_critical_region_visibility_gate.py`
- `tests/unit/infrastructure/ml/liveness/test_face_quality_illumination_gate.py`
- `tests/unit/infrastructure/ml/liveness/test_face_usability_gate.py`
- `tests/unit/infrastructure/test_deepface_detector.py`
- `tests/unit/infrastructure/test_live_liveness_preview.py`
- `tests/unit/infrastructure/test_liveness_runtime_wiring.py`

## Regressions to be aware of (DO NOT propagate)

- **Reverts Dependabot security pins in `requirements.txt`** — drops `tensorflow-cpu` from `2.21.0` to `2.15.0` and removes pinned transitives. **Not in this snapshot** (excluded). Would need to be rebased onto `main`'s `requirements.txt` before any merge.
- Bundles a 6.5 MB `yolov8n.pt` binary at root. **Not in this snapshot** (excluded by 1 MB cap).
- The flagship branch also adds a `.debug-snapshots/` directory that mirrors several `app/` files. **Not in this snapshot** (excluded by path filter).

## Commit log (full)

```
cbdbe0b Aysenur15 2026-05-06 Spoof Detection
100b8de Aysenur15 2026-05-06 Live Update
af01130 Aysenur15 2026-05-06 Liveness Update
1229f48 Aysenur15 2026-05-06 P3 Phase
167969a Aysenur15 2026-05-06 Live Update
13243a1 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): P0 FRR reduction — freeze EMA on skipped frames + re-enable decision guards
7c58e5f Aysenur15 2026-05-06 ML Model
729531e Aysenur15 2026-05-05 No Face Update
e83f0d9 Aysenur15 2026-05-05 Spoof Detection
a134418 Aysenur15 2026-05-04 Remove debug 'Last key' overlay from test data collector UI
d0dbace Aysenur15 2026-05-04 Fix key detection: waitKeyEx -> waitKey & 0xFF
152de78 Aysenur15 2026-05-04 Fix false occlusion for hijab/headscarf users
8cfcaab Aysenur15 2026-05-04 Lower mouth visibility threshold: 0.65 → 0.45
7132bb6 Aysenur15 2026-05-04 Rewrite test data collector: use background thread for detector
d54ab6a Aysenur15 2026-05-04 Fix test data collector: detect every 3rd frame to prevent event loop blocking
ae0876d Aysenur15 2026-05-04 Fix detector confidence flickering causing 'No Face' stuck state
0ee8f2c Aysenur15 2026-05-04 Add comprehensive test data collection and integration testing framework
96908ef Aysenur15 2026-05-04 Add comprehensive edge case tests for hybrid fusion evaluator
e3e5084 Aysenur15 2026-05-04 Spoof Detection
9fd36c8 Aysenur15 2026-05-02 No Face Update
1c5e1c8 Aysenur15 2026-05-02 No Face Update
504067e Aysenur15 2026-04-18 Color Shaded Screen
7ef9638 Aysenur15 2026-04-14 Face Bbox
73ee6e9 Aysenur15 2026-04-09 Liveness Score Addition
1b91b10 Aysenur15 2026-04-01 Improve enhanced liveness confidence and passive scoring
c2f6f26 Aysenur15 2026-04-01 Set enhanced as default liveness baseline
d270c50 Aysenur15 2026-03-31 New Feature Additions
```

## How to fetch full branch (read-only)

```bash
cd /path/to/biometric-processor
git fetch origin working_spoof_detection
git show cbdbe0b:<path>   # use `git show <ref>:<path>`, NOT git checkout
```
