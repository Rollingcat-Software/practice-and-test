"""
Batch video PAD evaluation against the FIVUCSAS spoof-detector session engine.

Feeds each video file (genuine + replay/attack) frame-by-frame into the SAME
session engine that powers amispoof.fivucsas.com, gets a session-level verdict
(LIVE / SPOOF), compares to ground truth, and computes ISO/IEC 30107-3 metrics
(APCER, BPCER, ACER) — fully automated, no webcam needed.

Why this is methodologically valid
-----------------------------------
Public PAD datasets store the ATTACK videos as *recordings of a screen/print
shown to a camera* — the moiré / bezel / double-capture artefacts are already
baked into the file. So feeding the attack video directly into the detector is
exactly equivalent to performing the replay attack. This is how academic PAD
papers evaluate.

Pipeline reuse
--------------
We import build_pipeline() from the spoof-detector's main.py so the analyzer
set + fusion weights are IDENTICAL to the live demo. A fresh pipeline + fresh
SessionEngine is built per video so stateful analyzers don't leak across videos.

Usage
-----
    python batch_video_eval.py \
        --dataset "C:/path/to/antispoofing-replay-dataset" \
        --max-videos 200 \
        --frame-stride 2

Label detection
---------------
A video is GENUINE if its path contains any genuine keyword
(genuine/real/live/bonafide/client), ATTACK otherwise if it contains any attack
keyword (attack/spoof/replay/fake/print/photo). Override with
--genuine-keyword / --attack-keyword. Use --dry-run to preview the label split
before running the (slow) inference.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

# ---------- locate the spoof-detector package ----------
SPOOF_DIR = Path(
    r"C:\Users\hp\Documents\GitHub\Rollingcat-Software\FIVUCSAS\practice-and-test\spoof-detector"
)
sys.path.insert(0, str(SPOOF_DIR))

OUT_DIR     = Path(__file__).parent
OUT_CSV     = OUT_DIR / "video_results.csv"
OUT_SUMMARY = OUT_DIR / "summary.txt"

GENUINE_KEYWORDS = ["genuine", "real", "live", "bonafide", "bona_fide", "client"]
ATTACK_KEYWORDS  = ["attack", "spoof", "replay", "fake", "print", "photo", "imposter"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def label_of(path: Path, gkw, akw):
    s = str(path).lower()
    if any(k in s for k in gkw):
        return "genuine"
    if any(k in s for k in akw):
        return "attack"
    return None  # unknown — skipped


def discover(dataset: Path, gkw, akw):
    vids = []
    for p in dataset.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            lab = label_of(p, gkw, akw)
            if lab:
                vids.append((p, lab))
    return vids


def eval_one_video(video_path: Path, frame_stride: int, max_frames: int):
    """Run one video through a fresh pipeline + session engine. Return verdict dict."""
    # Imported lazily so --dry-run doesn't need the ML stack
    from main import build_pipeline, load_config
    from src.application.session_engine import SessionEngine

    config = load_config(str(SPOOF_DIR / "config.yaml"))
    pipeline, _camera, _logger = build_pipeline(config)
    engine = SessionEngine(
        pipeline_analyzers=getattr(pipeline, "_face_analyzers", []),
    )
    engine.start()

    cap = cv2.VideoCapture(str(video_path))
    n_in = n_used = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n_in += 1
            if n_in % frame_stride != 0:
                continue
            analysis = pipeline.process(frame)
            engine.ingest(analysis, frame)
            n_used += 1
            if max_frames and n_used >= max_frames:
                break
    finally:
        cap.release()

    verdict = engine.get_verdict()
    return {
        "frames_used": n_used,
        "is_live": bool(verdict.is_live),
        "verdict": "LIVE" if verdict.is_live else "SPOOF",
        "confidence": round(float(verdict.confidence), 4),
        "dominant_threat": (verdict.dominant_threat.name
                            if getattr(verdict, "dominant_threat", None) else ""),
        "blink_count": getattr(engine, "_blink_count", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root folder of the video dataset")
    ap.add_argument("--max-videos", type=int, default=0, help="Cap total videos (0 = all)")
    ap.add_argument("--max-per-class", type=int, default=0, help="Cap videos per class")
    ap.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=300, help="Max frames per video (0=all)")
    ap.add_argument("--genuine-keyword", action="append", default=[])
    ap.add_argument("--attack-keyword", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="Only show label split, no inference")
    args = ap.parse_args()

    gkw = GENUINE_KEYWORDS + [k.lower() for k in args.genuine_keyword]
    akw = ATTACK_KEYWORDS + [k.lower() for k in args.attack_keyword]

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"Dataset path not found: {dataset}")
        return

    vids = discover(dataset, gkw, akw)
    genuine = [v for v in vids if v[1] == "genuine"]
    attack  = [v for v in vids if v[1] == "attack"]
    print(f"Discovered {len(vids)} labeled videos: {len(genuine)} genuine, {len(attack)} attack")

    if args.max_per_class:
        genuine = genuine[:args.max_per_class]
        attack  = attack[:args.max_per_class]
    vids = genuine + attack
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"Will evaluate {len(vids)} videos "
          f"({sum(1 for _,l in vids if l=='genuine')} genuine, "
          f"{sum(1 for _,l in vids if l=='attack')} attack)")

    if args.dry_run:
        print("\n--dry-run: sample paths:")
        for p, l in vids[:10]:
            print(f"  [{l:8}] {p}")
        return

    rows = []
    t0 = time.time()
    for i, (path, gt) in enumerate(vids, 1):
        try:
            res = eval_one_video(path, args.frame_stride, args.max_frames)
        except Exception as e:
            res = {"frames_used": 0, "is_live": None, "verdict": "ERROR",
                   "confidence": "", "dominant_threat": "", "blink_count": ""}
            print(f"  [{i}/{len(vids)}] ERROR {path.name}: {e}")
        expected = "LIVE" if gt == "genuine" else "SPOOF"
        correct = 1 if res["verdict"] == expected else 0
        rows.append({
            "video": path.name, "path": str(path), "ground_truth": gt,
            "expected": expected, **res, "correct": correct,
        })
        if i % 10 == 0 or res["verdict"] == "ERROR":
            print(f"  [{i}/{len(vids)}] {gt:8} -> {res['verdict']:6} "
                  f"(conf={res.get('confidence','')}) {path.name}")

    # write per-video CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    compute_metrics(rows, time.time() - t0)


def compute_metrics(rows, elapsed):
    scored = [r for r in rows if r["verdict"] in ("LIVE", "SPOOF")]
    genuine = [r for r in scored if r["ground_truth"] == "genuine"]
    attack  = [r for r in scored if r["ground_truth"] == "attack"]

    bpcer = (sum(1 for r in genuine if r["verdict"] == "SPOOF") / len(genuine)) if genuine else float("nan")
    apcer = (sum(1 for r in attack if r["verdict"] == "LIVE") / len(attack)) if attack else float("nan")
    acer = (apcer + bpcer) / 2 if genuine and attack else float("nan")

    def grade(a):
        if a < 0.05: return "A"
        if a < 0.10: return "B"
        if a < 0.20: return "C"
        return "D"

    lines = []
    lines.append("Video Replay PAD Evaluation (ISO 30107-3)")
    lines.append("=" * 45)
    lines.append(f"Videos scored : {len(scored)}  ({len(genuine)} genuine, {len(attack)} attack)")
    lines.append(f"Errors        : {sum(1 for r in rows if r['verdict']=='ERROR')}")
    lines.append(f"Elapsed       : {elapsed:.0f}s")
    lines.append("")
    lines.append(f"BPCER (genuine called SPOOF) : {bpcer:.2%}")
    lines.append(f"APCER (attack called LIVE)   : {apcer:.2%}")
    lines.append(f"ACER  = (APCER+BPCER)/2      : {acer:.2%}")
    lines.append(f"ISO Grade (by ACER)          : {grade(acer)}")
    text = "\n".join(lines)
    print("\n" + text)
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    print(f"\nPer-video: {OUT_CSV}\nSummary:   {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
