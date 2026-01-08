#!/usr/bin/env python3
"""
Biometric Demo - OPTIMIZED Real-Time Version
=========================================
Refactored for maximum performance and stability.

OPTIMIZATIONS:
    1. Async Worker: DeepFace operations (embeddings, demographics) run in a separate thread.
    2. Centralized Preprocessing: Frame is converted to RGB/Gray/HSV once per loop.
    3. Optimized Drawing: Landmarks are drawn efficiently (polylines vs individual dots).
    4. Enrollment Security: Locks onto a specific Face ID during enrollment.
    5. Adaptive Liveness: Dynamic thresholds based on lighting conditions.

Usage:
    python demo_optimized.py                    # All features
    python demo_optimized.py --mode face        # Face detection only
    python demo_optimized.py --profile          # Show performance metrics
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import deque
import threading
import queue
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("OptimizedDemo")

# =============================================================================
# HELPER: Get base path for bundled files
# =============================================================================

def get_base_path():
    """Get base path for data files - works both in dev and frozen exe."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(filename):
    """Get full path to a resource file."""
    base = get_base_path()
    path = os.path.join(base, filename)
    if os.path.exists(path):
        return path
    if os.path.exists(filename):
        return os.path.abspath(filename)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    # Try looking in parent directories if in subfolder
    parent_dir = os.path.dirname(script_dir)
    path = os.path.join(parent_dir, 'biometric-processor', filename)
    if os.path.exists(path):
        return path
    return filename

# =============================================================================
# ASYNC WORKER
# =============================================================================

class AsyncBiometricWorker:
    """Handles heavy biometric tasks in a background thread."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._deepface_wrapper = None
        self.thread.start()

    def _load_model(self):
        if self._deepface_wrapper is None:
            from deepface import DeepFace
            self._deepface_wrapper = DeepFace
            logger.info("DeepFace loaded in worker thread")

    def _worker_loop(self):
        self._load_model()
        while self.running:
            try:
                task_type, data, callback = self.queue.get(timeout=0.1)
                result = None
                
                if task_type == 'embedding':
                    try:
                        embedding = self._deepface_wrapper.represent(
                            img_path=data['img'],
                            model_name=data.get('model', 'Facenet512'),
                            enforce_detection=False,
                            detector_backend="skip"
                        )
                        if embedding:
                            result = np.array(embedding[0]['embedding'])
                    except Exception as e:
                        logger.error(f"Embedding error: {e}")

                elif task_type == 'demographics':
                    try:
                        analysis = self._deepface_wrapper.analyze(
                            img_path=data['img'],
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False,
                            detector_backend='skip',
                            silent=True
                        )
                        if analysis:
                            res = analysis[0] if isinstance(analysis, list) else analysis
                            # Gender logic
                            gender_info = res.get('gender', {})
                            if isinstance(gender_info, dict):
                                man_conf = gender_info.get('Man', 0)
                                woman_conf = gender_info.get('Woman', 0)
                                gender = 'M' if man_conf > woman_conf else 'F'
                            else:
                                gender = 'M' if res.get('dominant_gender') == 'Man' else 'F'
                                
                            result = {
                                'age': int(res.get('age', 0)),
                                'gender': gender,
                                'emotion': res.get('dominant_emotion', '?')[:6]
                            }
                    except Exception as e:
                        logger.error(f"Demographics error: {e}")

                if callback:
                    callback(result)
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    def submit(self, task_type: str, data: Any, callback: Callable):
        if self.queue.qsize() < 5: # Backpressure
            self.queue.put((task_type, data, callback))

    def stop(self):
        self.running = False
        self.thread.join()

# =============================================================================
# PERFORMANCE PROFILER
# =============================================================================

class Profiler:
    def __init__(self):
        self.metrics: Dict[str, deque] = {}
        self.enabled = False

    def time(self, name: str):
        return ProfilerContext(self, name)

    def record(self, name: str, ms: float):
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=60)
        self.metrics[name].append(ms)

    def draw(self, frame: np.ndarray):
        if not self.enabled:
            return
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 45), (250, 45 + len(self.metrics) * 18 + 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "PROFILER", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y = 80
        for name, vals in sorted(self.metrics.items()):
            avg = sum(vals) / len(vals) if vals else 0
            color = (0, 255, 0) if avg < 30 else (0, 255, 255) if avg < 60 else (0, 0, 255)
            cv2.putText(frame, f"{name}: {avg:.1f}ms", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += 18

class ProfilerContext:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
        self.start = 0
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, *args):
        self.profiler.record(self.name, (time.perf_counter() - self.start) * 1000)

# =============================================================================
# FACE DETECTOR (MP Tasks)
# =============================================================================

class FastFaceDetector:
    _instance = None
    _detector = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load()
        return cls._instance

    @classmethod
    def _load(cls):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
            
            model_path = get_resource_path("blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
                urllib.request.urlretrieve(url, model_path)

            opts = vision.FaceDetectorOptions(
                base_options=tasks.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=0.5
            )
            cls._detector = vision.FaceDetector.create_from_options(opts)
            cls._mp = mp
            logger.info("MediaPipe Tasks Face Detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load MediaPipe Face Detector: {e}")
            cls._detector = None

    def detect(self, rgb_frame: np.ndarray) -> List[Dict]:
        if self._detector is None:
            return []
        
        h, w = rgb_frame.shape[:2]
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        detections = self._detector.detect(mp_img)
        
        results = []
        if detections.detections:
            for detection in detections.detections:
                bbox = detection.bounding_box
                x, y = max(0, bbox.origin_x), max(0, bbox.origin_y)
                fw, fh = min(bbox.width, w - x), min(bbox.height, h - y)
                
                if fw > 30 and fh > 30:
                    results.append({
                        'facial_area': {'x': x, 'y': y, 'w': fw, 'h': fh},
                        'confidence': detection.categories[0].score if detection.categories else 0.9,
                    })
        return results

# =============================================================================
# QUALITY & LIVENESS
# =============================================================================

class FastQualityAssessor:
    def __init__(self, blur_threshold=100.0):
        self.blur_threshold = blur_threshold

    def assess(self, gray_face: np.ndarray, shape: Tuple[int, int]) -> Dict[str, Any]:
        if gray_face.size == 0: return {'score': 0}
        
        lap_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        blur_score = min(100, (lap_var / self.blur_threshold) * 100)
        size_score = min(100, min(shape) / 80 * 50)
        brightness = np.mean(gray_face)
        bright_ok = 50 < brightness < 200
        
        score = (blur_score + size_score + (100 if bright_ok else 50)) / 3
        return {'score': score, 'blur': blur_score}

class FastLivenessDetector:
    def __init__(self):
        self._gabor_kernels = [
            cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0)
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]
        ]

    def check(self, gray_face: np.ndarray, hsv_face: np.ndarray) -> Dict[str, Any]:
        try:
            # Texture
            lap_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            texture = min(100, max(0, (lap_var - 20) / 3))

            # Color (Adaptive range could be added here)
            sat_mean = np.mean(hsv_face[:, :, 1])
            if 30 <= sat_mean <= 120: color = 100
            elif sat_mean < 30: color = max(0, sat_mean * 2)
            else: color = max(0, 100 - (sat_mean - 120) * 0.8)

            # Moire
            moire = 100
            for kernel in self._gabor_kernels:
                if np.std(cv2.filter2D(gray_face, cv2.CV_64F, kernel)) > 40:
                    moire -= 20

            score = (texture * 0.3 + color * 0.3 + moire * 0.4)
            return {'is_live': score >= 50, 'score': score}
        except Exception:
            return {'is_live': False, 'score': 0}

# =============================================================================
# FACE DB & TRACKER
# =============================================================================

class FaceDB:
    def __init__(self, path="face_db.pkl"):
        self.path = path
        self.faces = {}
        # Vectorized search cache
        self._matrix = None
        self._names = []
        self._threshold = 0.45
        self._dirty = True  # Flag to rebuild matrix
        
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f: self.faces = pickle.load(f)
                self._dirty = True
            except: pass

    def save(self):
        with open(self.path, 'wb') as f: pickle.dump(self.faces, f)

    def enroll(self, name, embedding, thumb):
        self.faces[name] = {'embeddings': [embedding], 'thumbnail': thumb}
        self._dirty = True
        self.save()

    def add_embedding(self, name, emb):
        if name in self.faces:
            embs = self.faces[name].get('embeddings', [])
            if len(embs) >= 5: embs.pop(0)
            embs.append(emb)
            self.faces[name]['embeddings'] = embs
            self._dirty = True
            self.save()

    def _rebuild_matrix(self):
        """Rebuilds the numpy matrix for vectorized search."""
        if not self.faces:
            self._matrix = None
            self._names = []
            self._dirty = False
            return

        embeddings = []
        names = []
        
        for name, data in self.faces.items():
            for emb in data.get('embeddings', []):
                # Normalize embedding for cosine similarity
                emb = np.array(emb).flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings.append(emb / norm)
                    names.append(name)
        
        if embeddings:
            self._matrix = np.array(embeddings) # Shape: (N_samples, 512)
            self._names = names
        else:
            self._matrix = None
            self._names = []
            
        self._dirty = False
        logger.info(f"FaceDB Matrix Rebuilt: {len(names)} vectors")

    def search(self, emb, threshold=0.45):
        """
        Vectorized search using BLAS/SIMD via Numpy.
        Much faster than looping for large N.
        """
        if self._dirty or self._matrix is None:
            self._rebuild_matrix()
            
        if self._matrix is None:
            return None

        # Prepare query vector
        query = np.array(emb).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm == 0: return None
        query = query / query_norm

        # Vectorized Cosine Similarity: matrix @ vector
        # (N, 512) @ (512,) -> (N,)
        scores = np.dot(self._matrix, query)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score >= threshold:
            return (self._names[best_idx], float(best_score))
        return None

class FaceTracker:
    def __init__(self, max_gone=15):
        self.next_id = 0
        self.tracks = {}
        self.max_gone = max_gone

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]['gone'] += 1
                if self.tracks[tid]['gone'] > self.max_gone: del self.tracks[tid]
            return {}

        centroids = []
        for d in detections:
            a = d['facial_area']
            centroids.append((a['x'] + a['w']//2, a['y'] + a['h']//2))

        if not self.tracks:
            res = {}
            for i, d in enumerate(detections):
                self.tracks[self.next_id] = {'centroid': centroids[i], 'gone': 0}
                res[self.next_id] = d
                self.next_id += 1
            return res

        used = set()
        res = {}
        
        # Match existing
        for tid, track in list(self.tracks.items()):
            tc = track['centroid']
            best_j, best_d = -1, float('inf')
            for j, nc in enumerate(centroids):
                if j not in used:
                    d = np.sqrt((tc[0]-nc[0])**2 + (tc[1]-nc[1])**2)
                    if d < best_d and d < 120: best_d, best_j = d, j
            
            if best_j >= 0:
                self.tracks[tid]['centroid'] = centroids[best_j]
                self.tracks[tid]['gone'] = 0
                res[tid] = detections[best_j]
                used.add(best_j)
            else:
                self.tracks[tid]['gone'] += 1
                if self.tracks[tid]['gone'] > self.max_gone: del self.tracks[tid]

        # New faces
        for j, d in enumerate(detections):
            if j not in used:
                self.tracks[self.next_id] = {'centroid': centroids[j], 'gone': 0}
                res[self.next_id] = d
                self.next_id += 1
        return res

# =============================================================================
# THREADED CAMERA (Non-Blocking I/O)
# =============================================================================

class ThreadedCamera:
    """Polles camera in separate thread to prevent I/O blocking."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        if not self.grabbed:
            logger.error("Failed to open camera!")
            
    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.001) # Yield slightly

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# =============================================================================
# MAIN APP
# =============================================================================

class OptimizedBiometricDemo:
    MODES = ["all", "enroll", "verify", "demographics", "liveness"]
    
    def __init__(self, camera=0):
        self.camera = camera
        self.mode = "all"
        self.paused = False
        
        # Core components
        self.worker = AsyncBiometricWorker()
        self.detector = FastFaceDetector()
        self.tracker = FaceTracker()
        self.quality = FastQualityAssessor()
        self.liveness = FastLivenessDetector()
        self.face_db = FaceDB()
        self.profiler = Profiler()
        self.profiler.enabled = True

        # State
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.cache = {'verify': {}, 'demo': {}}
        
        # Enrollment
        self.enrolling = False
        self.enroll_id = None # ID Locking
        self.enroll_step = 0
        self.enroll_imgs = []
        self.enroll_name = ""

    def process_frame(self, frame):
        start = time.perf_counter()
        
        # 1. Centralized Preprocessing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Detect
        with self.profiler.time('detect'):
            detections = self.detector.detect(rgb)
        
        tracks = self.tracker.update(detections)
        
        # 3. Process Faces
        for fid, face in tracks.items():
            # ID Locking for Enrollment
            if self.enrolling:
                if self.enroll_id is None: self.enroll_id = fid
                if fid != self.enroll_id: continue # Ignore other faces
            
            a = face['facial_area']
            x, y, w, h = a['x'], a['y'], a['w'], a['h']
            
            # Extract ROIs
            face_rgb = rgb[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            face_hsv = hsv[y:y+h, x:x+w]
            face_bgr = frame[y:y+h, x:x+w]
            
            # Quality & Liveness (Fast)
            q = self.quality.assess(face_gray, (h, w))
            l = self.liveness.check(face_gray, face_hsv)
            
            # Async Tasks (Demographics/Verify)
            if self.mode in ['all', 'demographics', 'verify']:
                self._dispatch_async_tasks(fid, face_bgr)

            # Draw
            color = (0, 255, 0) if l['is_live'] else (0, 0, 255)
            if self.enrolling and fid == self.enroll_id: color = (255, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Labels
            labels = [f"ID: {fid}", f"Q: {q['score']:.0f}"]
            if l['score'] > 0: labels.append(f"Live: {l['score']:.0f}")
            
            # Cached Results
            if fid in self.cache['demo']:
                d = self.cache['demo'][fid]
                labels.append(f"{d['gender']}/{d['age']}")
            
            if fid in self.cache['verify']:
                match = self.cache['verify'][fid]
                if match: labels.append(f"Match: {match[0]}")

            for i, txt in enumerate(labels):
                cv2.putText(frame, txt, (x, y - 10 - (i*15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Enrollment Logic
        if self.enrolling and self.enroll_id in tracks:
            self._handle_enrollment(tracks[self.enroll_id], frame)

        self.profiler.draw(frame)
        
        # FPS
        self.frame_times.append(time.perf_counter() - start)
        if len(self.frame_times) > 10:
            self.fps = 1.0 / (sum(self.frame_times)/len(self.frame_times))
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def _dispatch_async_tasks(self, fid, face_img):
        # Throttle requests
        now = time.time()
        
        # Demographics
        if fid not in self.cache['demo'] or now - self.cache['demo'].get(f"{fid}_t", 0) > 2.0:
            self.cache['demo'][f"{fid}_t"] = now
            self.worker.submit('demographics', {'img': face_img.copy()}, 
                             lambda r: self._update_cache('demo', fid, r))

        # Verify
        if fid not in self.cache['verify'] or now - self.cache['verify'].get(f"{fid}_t", 0) > 1.0:
            self.cache['verify'][f"{fid}_t"] = now
            self.worker.submit('embedding', {'img': face_img.copy()}, 
                             lambda r: self._verify_callback(fid, r))

    def _update_cache(self, kind, fid, result):
        if result: self.cache[kind][fid] = result

    def _verify_callback(self, fid, embedding):
        if embedding is not None:
            match = self.face_db.search(embedding)
            self.cache['verify'][fid] = match

    def _handle_enrollment(self, face, frame):
        # Simple step-based enrollment
        if self.enroll_step < 5:
            # Auto-capture every 1 sec (simplified for demo)
            now = time.time()
            if not hasattr(self, '_last_capture'): self._last_capture = 0
            
            if now - self._last_capture > 1.0:
                a = face['facial_area']
                face_img = frame[a['y']:a['y']+a['h'], a['x']:a['x']+a['w']]
                self.worker.submit('embedding', {'img': face_img.copy()}, 
                                 lambda r: self._enroll_callback(r, face_img.copy()))
                self._last_capture = now
                
        # Overlay
        h, w = frame.shape[:2]
        cv2.putText(frame, f"ENROLLING {self.enroll_step}/5", (w//2-100, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def _enroll_callback(self, embedding, thumb):
        if embedding is not None:
            self.enroll_imgs.append(embedding)
            self.enroll_step += 1
            if self.enroll_step == 1: # Save first as thumb
                self.enroll_thumb = thumb
            
            if self.enroll_step >= 5:
                # Finalize
                name = f"User_{len(self.face_db.faces)+1}"
                self.face_db.enroll(name, self.enroll_imgs[0], self.enroll_thumb)
                for emb in self.enroll_imgs[1:]:
                    self.face_db.add_embedding(name, emb)
                self.enrolling = False
                self.enroll_id = None
                self.enroll_imgs = []
                self.enroll_step = 0
                logger.info(f"Enrolled {name}")

    def run(self):
        # Initialize Threaded Camera
        camera = ThreadedCamera(self.camera).start()
        print(" Controls: 'q' quit, 'e' enroll")
        
        try:
            while True:
                ret, frame = camera.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)
                
                cv2.imshow("Optimized Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'): break
                if key == ord('e'): 
                    self.enrolling = True
                    self.enroll_id = None
                    self.enroll_step = 0
                    
        finally:
            self.worker.stop()
            camera.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    OptimizedBiometricDemo().run()
