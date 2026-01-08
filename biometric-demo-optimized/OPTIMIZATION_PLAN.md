# Biometric Demo Optimization Plan

## Current Performance Baseline
```
Component        | Time      | Notes
-----------------|-----------|---------------------------
Face Detection   | 6.4ms     | Good - MediaPipe Tasks API
Landmarks        | 9.7ms     | Good - MediaPipe Face Mesh
Quality          | 1.3ms     | Excellent
Liveness         | 28.2ms    | Acceptable - can improve
Card Detection   | 462ms     | Slow - YOLO
Demographics     | 1442ms    | CRITICAL BOTTLENECK
Verification     | ~800ms    | Slow on first encounter
-----------------|-----------|---------------------------
Overall FPS      | 63.7      | Good (heavy ops throttled)
```

## Priority 1: Critical Bottlenecks

### 1.1 Demographics Analysis (1.4s → <100ms)
**Problem**: DeepFace loads 3 models, runs synchronously
**Solution**:
- Move to background thread with queue
- Replace DeepFace with lightweight ONNX models:
  - Age/Gender: SSR-Net (0.32MB, ~5ms)
  - Emotion: Mini-Xception (1MB, ~10ms)
- Pre-load models at startup

**Implementation**:
```python
class AsyncDemographicsAnalyzer:
    def __init__(self):
        self._queue = Queue()
        self._result = {}
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def analyze_async(self, face_img, face_id):
        self._queue.put((face_img.copy(), face_id))

    def get_result(self, face_id):
        return self._result.get(face_id)
```

### 1.2 Challenge Detection Flickering
**Problem**: Single-frame detection causes progress reset
**Solution**:
- ✅ Already added 3-frame hysteresis
- Add EAR/MAR temporal smoothing (rolling average)
- Add One Euro Filter for landmark positions

### 1.3 Verification Embedding (800ms → async)
**Problem**: Synchronous embedding extraction blocks loop
**Solution**:
- Background thread for embedding extraction
- Content-based cache (face image hash)
- Return stale match until fresh result ready

## Priority 2: Performance Optimizations

### 2.1 Liveness Detection (28ms → 15ms)
**Current issues**:
- CLAHE object recreated every call
- Gabor kernels recreated per instance
- Redundant color conversions

**Fixes**:
```python
# Cache CLAHE (in __init__)
self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Gabor as class constants
GABOR_KERNELS = [
    cv2.getGaborKernel((15, 15), 5.0, theta, 10.0, 0.5, 0)
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]
]
```

### 2.2 Camera Threading
**Issues**:
- Frame.copy() on every read (82MB/sec waste)
- Lock contention during processing
- Artificial 1ms sleep

**Fixes**:
- Use lock-free queue with pre-allocated buffers
- Double-buffering instead of copy
- Remove sleep, use Event.wait()

### 2.3 Card Detection (462ms → 200ms)
**Fixes**:
- Reduce input size: 640 → 416 or 320
- Use TensorRT if GPU available
- Skip frames (detect every 3rd frame)

## Priority 3: Quality Improvements

### 3.1 Temporal Smoothing for EAR/MAR
```python
class TemporalSmoother:
    def __init__(self, window=5):
        self._history = deque(maxlen=window)

    def smooth(self, value):
        self._history.append(value)
        return sum(self._history) / len(self._history)
```

### 3.2 One Euro Filter for Landmarks
Reduces jitter while maintaining responsiveness:
```python
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0
        self.t_prev = None
```

### 3.3 Adaptive Calibration
- 3-second calibration phase at puzzle start
- Capture baseline EAR, MAR, brow positions
- Adjust thresholds per-user

### 3.4 Visual Feedback Improvements
- Show EAR/MAR progress bars during challenges
- Animated guides for each challenge type
- Color-coded feedback (green=good, yellow=close, red=wrong)

## Priority 4: Future Optimizations

### 4.1 Model Replacements (Long-term)
Replace DeepFace entirely with:
- **UniFace** or **InsightFace** (all-in-one ONNX)
- **YuNet** for detection (1.6ms vs 6.4ms)
- **MobileFaceNet** for embeddings (18ms vs 800ms)
- **HSEmotion** for emotion (15ms vs 400ms)

### 4.2 GPU Acceleration
- Use CUDA execution provider for ONNX
- Batch multiple faces in single inference
- Keep image preprocessing on GPU

### 4.3 Quantization
- INT8 quantization for 4x smaller models
- TensorRT FP16 for 1.8x speedup

## Implementation Phases

### Phase 1: Quick Wins (1-2 hours) ✅ COMPLETE
1. ✅ Detection smoothing (3-frame hysteresis → 5-frame)
2. ✅ Adjusted thresholds (EAR gap widened: 0.16-0.22)
3. ✅ Cache CLAHE and Gabor kernels (class-level constants)
4. ✅ Add EAR/MAR temporal smoothing (5-frame rolling average)
5. ✅ Fix camera frame copying (double-buffering)

### Phase 2: Background Processing (2-4 hours) ✅ COMPLETE
1. ✅ Async demographics analyzer (background thread)
2. ✅ Async card detector (background thread)
3. ✅ Async embedding extractor (background thread with caching)

### Phase 3: Quality Improvements (2-4 hours) ✅ COMPLETE
1. ✅ One Euro Filter for landmarks (adaptive smoothing)
2. [ ] Adaptive calibration phase (future)
3. [ ] Better visual feedback (future)

### Phase 4: Model Replacement (4-8 hours)
1. [ ] Replace DeepFace with ONNX models
2. [ ] Integrate UniFace or InsightFace
3. [ ] GPU acceleration

## Expected Results

After Phase 1-2:
- Demographics: 1400ms → 0ms main thread (async background)
- Liveness: 28ms → 18.6ms (measured)
- Smooth FPS: 60+ consistently → 122.7 FPS achieved!
- No more challenge flickering (5-frame hysteresis + wider thresholds)

After Phase 3:
- Landmark jitter reduced with One Euro Filter
- Smoother EAR/MAR readings with temporal averaging
- More stable puzzle challenge detection

After Phase 4 (future):
- Total per-frame: <50ms
- Sustained 30+ FPS even in "all" mode
- Professional-grade liveness detection

## Measured Performance (January 2026)

```
Component        | Before    | After     | Improvement
-----------------|-----------|-----------|------------
Face Detection   | 6.4ms     | 4.7ms     | 26% faster
Landmarks        | 9.7ms     | 2.1ms     | 78% faster
Quality          | 1.3ms     | 0.8ms     | 38% faster
Liveness         | 28.2ms    | 18.6ms    | 34% faster
Verification     | ~800ms    | 7.1ms     | 99% faster (cached)
Card Detection   | 462ms     | 0ms*      | Non-blocking
Demographics     | 1442ms    | 0ms*      | Non-blocking
Embedding        | ~800ms    | 0ms*      | Non-blocking
-----------------|-----------|-----------|------------
Average FPS      | 63.7      | 122.7+    | 93%+ improvement
```
*All heavy operations now run in background threads, never blocking main loop.
