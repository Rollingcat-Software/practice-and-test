"""
Parse amispoof session Report JSONs into a clean comparison table.

amispoof's "Download Report" button exports a full per-session JSON (verdict +
7-category probabilities + ~25 analyzer scores + incidents + liveness proof +
occlusion gate). This script reads every reports/amispoof-session-*.json and
emits:
  - amispoof_reports_summary.csv   one row per session
  - prints a readable table
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).parent
REPORTS = sorted((BASE / "reports").glob("amispoof-session-*.json"))
OUT = BASE / "amispoof_reports_summary.csv"

ANALYZERS = ["minifasnet", "device_boundary", "texture", "moire", "screen_replay",
             "screen_flicker", "micro_tremor", "rppg", "blink", "temporal"]
CATS = ["real", "static_image", "video_replay", "mask_3d", "heavy_makeup",
        "ar_filter", "deepfake_inject"]

rows = []
for p in REPORTS:
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d.get("verdict", {})
    cs = v.get("category_scores", {})
    an = d.get("latest_analyzer_scores", {})
    gate = d.get("latest_gate_result") or {}
    proof = d.get("latest_liveness_proof") or {}

    def ascore(name):
        a = an.get(name)
        return round(a["score"], 1) if isinstance(a, dict) and a.get("score") is not None else ""

    row = {
        "file": p.stem.replace("amispoof-session-", ""),
        "generated_at": d.get("generated_at", "")[:19],
        "verdict": "LIVE" if v.get("is_live") else "SPOOF",
        "confidence": round(v.get("confidence", 0) * 100, 1),
        "dominant_threat": v.get("dominant_threat") or "-",
        "duration_s": round(v.get("session_duration_sec", 0), 1),
        "frames": v.get("frames_analyzed", ""),
        "blinks": v.get("blink_count", ""),
        "incidents": len(v.get("incidents", [])),
        "face_ratio": round(v.get("face_detected_ratio", 0), 3),
        "liveness_proof": proof.get("score", ""),
        "gate_usable": gate.get("usable", ""),
        "gate_reason": gate.get("reason", "") or "",
    }
    for c in CATS:
        row[f"p_{c}"] = round(cs.get(c, 0), 3)
    for a in ANALYZERS:
        row[a] = ascore(a)
    rows.append(row)

# write CSV
fields = (["file", "generated_at", "verdict", "confidence", "dominant_threat",
           "duration_s", "frames", "blinks", "incidents", "face_ratio",
           "liveness_proof", "gate_usable", "gate_reason"]
          + [f"p_{c}" for c in CATS] + ANALYZERS)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# print readable table
print(f"{len(rows)} sessions parsed -> {OUT}\n")
print(f"{'time':<20}{'verdict':<7}{'conf':>5}  {'threat':<13}{'dur':>6}{'blink':>6}{'inc':>4}  "
      f"{'MFN':>5}{'devB':>5}{'replay':>7}{'moire':>6}{'flick':>6}  gate")
print("-" * 120)
for r in rows:
    print(f"{r['generated_at']:<20}{r['verdict']:<7}{r['confidence']:>4}%  "
          f"{r['dominant_threat']:<13}{r['duration_s']:>6}{r['blinks']:>6}{r['incidents']:>4}  "
          f"{str(r['minifasnet']):>5}{str(r['device_boundary']):>5}{str(r['screen_replay']):>7}"
          f"{str(r['moire']):>6}{str(r['screen_flicker']):>6}  {r['gate_reason']}")
