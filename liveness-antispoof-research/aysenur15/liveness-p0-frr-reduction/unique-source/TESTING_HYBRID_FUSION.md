# Hybrid Fusion Testing Guide

Complete workflow for validating hybrid liveness detection with real test data.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Testing Workflow                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1️⃣  Collect Test Data                                      │
│      └─ Capture frames: live, phone screen, printed photo   │
│                                                              │
│  2️⃣  Analyze Test Data                                      │
│      └─ Run integration tests                               │
│      └─ Check accuracy metrics                              │
│                                                              │
│  3️⃣  Validate Results                                       │
│      └─ Review confusion matrix                             │
│      └─ Identify failure cases                              │
│                                                              │
│  4️⃣  Production Deployment                                  │
│      └─ Enable LIVENESS_FUSION_ENABLED=true                 │
│      └─ Monitor metrics in production                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Collect Test Frames

### Run the Interactive Collector

```bash
python app/tools/test_data_collector.py
```

### Instructions

```
SPACE  → Capture frame from camera
1      → Label current frame as LIVE
2      → Label current frame as SPOOF
Q      → Quit

Example flow:
  1. Start script
  2. Point camera at your face
  3. Press SPACE → frame captured
  4. Press 1 → saved as "live_frame_001"
  5. Point camera at phone screen playing video
  6. Press SPACE → frame captured
  7. Press 2 → saved as "spoof_frame_001"
  ... repeat for more examples
```

### Test Scenarios to Collect

**✅ LIVE Frames (aim for 10-15 examples)**
- Normal face, good lighting
- Face with glasses
- Face with different angles (yaw, pitch, roll)
- Low lighting
- High lighting (overexposed)
- Different skin tones

**❌ SPOOF Frames (aim for 10-15 examples)**
- Phone screen: video playing
- Phone screen: high-quality deepfake
- Printed photo (A4, B&W)
- Printed photo (color)
- Printed photo (high-quality)
- Tablet/iPad screen

### Output Location

```
data/test_frames/
├── live_frame_001.jpg
├── live_frame_001.json
├── live_frame_002.jpg
├── live_frame_002.json
├── spoof_frame_001.jpg
├── spoof_frame_001.json
├── spoof_frame_002.jpg
├── spoof_frame_002.json
└── summary.jsonl              ← All frames + metrics
```

---

## Step 2: Check Collected Data

### View Summary

```bash
# Count frames
jq -s 'group_by(.label) | map({label: .[0].label, count: length})' \
  data/test_frames/summary.jsonl

# Output:
# [
#   {"label": "live", "count": 12},
#   {"label": "spoof", "count": 13}
# ]
```

### Review Individual Frame Metrics

```bash
# Pretty-print first frame
head -1 data/test_frames/summary.jsonl | jq '.'

# Check hybrid fusion scores
jq '.liveness.scores | {liveness_score, hybrid_fusion_spoof_score}' \
  data/test_frames/summary.jsonl
```

### Sample JSON Structure

```json
{
  "timestamp": "2026-05-04T18:30:45.123456",
  "label": "spoof",
  "frame_number": 1,
  "image_path": "data/test_frames/spoof_frame_001.jpg",
  "face_detected": true,
  "liveness": {
    "is_live": false,
    "confidence": 0.82,
    "method": "hybrid_fusion",
    "scores": {
      "liveness_score": 16.0,
      "hybrid_fusion_spoof_score": 84.0
    },
    "checks": {
      "hybrid_fusion_enabled": true,
      "hybrid_fusion_is_spoof": true
    }
  },
  "quality": {
    "score": 78.5,
    "is_acceptable": true
  }
}
```

---

## Step 3: Run Integration Tests

### Run All Tests

```bash
pytest tests/integration/test_hybrid_fusion_real_data.py -v
```

### Run Individual Test

```bash
# Test accuracy on LIVE frames
pytest tests/integration/test_hybrid_fusion_real_data.py::test_hybrid_fusion_accuracy_on_collected_live_frames -v

# Test accuracy on SPOOF frames
pytest tests/integration/test_hybrid_fusion_real_data.py::test_hybrid_fusion_accuracy_on_collected_spoof_frames -v

# Check metadata presence
pytest tests/integration/test_hybrid_fusion_real_data.py::test_hybrid_fusion_enabled_metadata -v

# Generate summary report
pytest tests/integration/test_hybrid_fusion_real_data.py::test_summary_report -v -s
```

### Expected Results

```
test_hybrid_fusion_accuracy_on_collected_live_frames PASSED     [33%]
test_hybrid_fusion_accuracy_on_collected_spoof_frames PASSED    [66%]
test_hybrid_fusion_enabled_metadata PASSED                      [100%]

======================== 3 passed in 12.34s =========================
```

**Acceptance criteria:**
- ✅ LIVE accuracy ≥ 80%
- ✅ SPOOF accuracy ≥ 80%
- ✅ Metadata checks pass

---

## Step 4: Analyze Results

### Generate Confusion Matrix

```bash
python << 'EOF'
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load data
summary_file = Path("data/test_frames/summary.jsonl")
frames = []
with open(summary_file) as f:
    for line in f:
        if line.strip():
            frames.append(json.loads(line))

# Extract labels and predictions
true_labels = [1 if f["label"] == "live" else 0 for f in frames]
pred_labels = []

for f in frames:
    if f.get("liveness"):
        pred = 1 if f["liveness"]["is_live"] else 0
        pred_labels.append(pred)

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:")
print("                Predicted")
print("                LIVE  SPOOF")
print(f"Actual LIVE  [{cm[1,1]:3d}  {cm[1,0]:3d}]")
print(f"       SPOOF [{cm[0,1]:3d}  {cm[0,0]:3d}]")

# Metrics
print("\n" + classification_report(
    true_labels, pred_labels,
    target_names=["SPOOF", "LIVE"]
))
EOF
```

### Interpret Results

**Good metrics:**
```
              precision    recall  f1-score   support
       SPOOF       0.95      0.92      0.94        12
        LIVE       0.93      0.95      0.94        13
```

**Bad metrics - investigate:**
```
              precision    recall  f1-score   support
       SPOOF       0.60      0.50      0.55        12
        LIVE       0.70      0.75      0.72        13
```

If accuracy < 80%, check:
1. **False positives** (SPOOF detected as LIVE):
   - Review which SPOOF frames failed
   - Check `hybrid_fusion_spoof_score` for those frames
   - Might need to lower threshold (0.55 → 0.50)

2. **False negatives** (LIVE detected as SPOOF):
   - Review which LIVE frames failed
   - Check lighting, face angle, device quality
   - Might need to raise threshold (0.55 → 0.60)

---

## Step 5: Tune Parameters

If accuracy is below 80%, adjust configuration:

### Option A: Change Threshold

```bash
# .env or environment variable
LIVENESS_FUSION_THRESHOLD=0.50    # More sensitive to spoof
# or
LIVENESS_FUSION_THRESHOLD=0.60    # More lenient to live faces
```

### Option B: Adjust Weights

Edit `HybridFusionEvaluator` initialization:

```python
from app.application.services.hybrid_fusion_evaluator import FusionWeights

# If MiniFASNet is too conservative
weights = FusionWeights(
    pretrained_model=0.20,  # Lower (was 0.25)
    flash_response=0.30,    # Higher
    rppg_signal=0.20,
    moire_pattern=0.15,
    device_replay=0.15,
)

evaluator = HybridFusionEvaluator(weights=weights)
```

### Option C: Improve Test Data

Collect more examples:
- More diverse lighting
- Different cameras/phones
- Higher quality prints
- Different screen types

---

## Step 6: Production Deployment

### Enable Hybrid Fusion

```bash
# .env.prod
LIVENESS_FUSION_ENABLED=true
LIVENESS_FUSION_THRESHOLD=0.55
```

### Monitor in Production

```bash
# Check fusion decisions in logs
cat logs/liveness_calibration.jsonl | jq '.metadata.hybrid_fusion_reasoning'

# Monitor accuracy over time
jq 'select(.method=="hybrid_fusion") | .is_live' logs/liveness_calibration.jsonl \
  | sort | uniq -c
```

---

## Troubleshooting

### "Camera not available"
```bash
# Check camera is connected
ls -la /dev/video*  # Linux
# or use obs-studio to test camera
```

### "No face detected"
- Move closer to camera
- Ensure good lighting
- For SPOOF frames: ensure face is visible on screen/photo

### Low accuracy on SPOOF frames

**Check:**
1. Quality of printed photo
2. Phone screen brightness
3. Distance from camera
4. Frame rate (try to capture at different speeds)

**Fix:**
- Lower threshold: `LIVENESS_FUSION_THRESHOLD=0.50`
- Increase flash weight in FusionWeights
- Collect more SPOOF examples

### Low accuracy on LIVE frames

**Check:**
1. Lighting quality
2. Face position (too far, too close, odd angle)
3. Makeup, glasses, mask

**Fix:**
- Raise threshold: `LIVENESS_FUSION_THRESHOLD=0.60`
- Collect diverse LIVE examples
- Check that rPPG analyzer is working

---

## Appendix: File Structure

```
biometric-processor/
├── app/tools/
│   ├── live_liveness_preview.py         # Original debug tool (read-only)
│   └── test_data_collector.py           # NEW: Collect labeled frames
├── data/test_frames/                    # NEW: Generated test data
│   ├── live_frame_*.jpg                 # LIVE frames
│   ├── live_frame_*.json                # Metrics
│   ├── spoof_frame_*.jpg                # SPOOF frames
│   ├── spoof_frame_*.json               # Metrics
│   └── summary.jsonl                    # All frames + metrics
├── tests/
│   ├── unit/application/
│   │   └── test_hybrid_fusion_evaluator.py        # Logic tests (10 cases)
│   └── integration/
│       └── test_hybrid_fusion_real_data.py        # NEW: Real data tests
└── TESTING_HYBRID_FUSION.md             # This file
```

---

## Next Steps

1. **This week:**
   - [ ] Run test data collector
   - [ ] Collect 20+ frames (10+ LIVE, 10+ SPOOF)
   - [ ] Run integration tests
   - [ ] Check accuracy

2. **Next week:**
   - [ ] Tune threshold if needed
   - [ ] Collect edge cases (low light, glasses, etc.)
   - [ ] Re-test

3. **Before production:**
   - [ ] Validate accuracy ≥ 80%
   - [ ] Test with real users
   - [ ] Monitor false positive/negative rates
   - [ ] Enable in staging environment

---

## Questions?

- Review code: `app/tools/test_data_collector.py`
- Check tests: `tests/integration/test_hybrid_fusion_real_data.py`
- Debug: `data/test_frames/summary.jsonl`
