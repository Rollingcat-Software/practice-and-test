"""
Parse the LABELED amispoof session reports (reports_final/) and compute
ISO/IEC 30107-3 PAD metrics. Ground truth is taken from the filename suffix
(e.g. ...-printedphoto.json, ...-liveperson.json).

Outputs:
  - amispoof_final_summary.csv   one row per session (with ground_truth + correct)
  - amispoof_final_metrics.txt   APCER per PAI species, BPCER, ACER
  - prints a readable table
"""
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
REPORTS = sorted((BASE / "reports_final").glob("amispoof-session-*.json"))
OUT_CSV = BASE / "amispoof_final_summary.csv"
OUT_TXT = BASE / "amispoof_final_metrics.txt"

# label suffix -> (ground_truth class, expected verdict, is_attack)
#   bona_fide  : a genuine live person  -> expect LIVE
#   attack     : a presentation attack  -> expect SPOOF
#   special    : not scored in APCER/BPCER (multi-face, no-face absence)
LABELMAP = {
    "printedphoto":   ("print",        "SPOOF", "attack"),
    "photofromphone": ("screen_photo", "SPOOF", "attack"),
    "videoreplay":    ("video_replay", "SPOOF", "attack"),
    "videoattack":    ("video_replay", "SPOOF", "attack"),
    "whatsapp":       ("video_replay", "SPOOF", "attack"),
    "fakecam":        ("video_injection", "SPOOF", "attack"),
    "injection":      ("video_injection", "SPOOF", "attack"),
    "liveperson":     ("genuine",      "LIVE",  "bona_fide"),
    "2faces":         ("two_faces",    "LIVE",  "special"),
    "noface":         ("no_face",      "SPOOF", "special"),
}

ANALYZERS = ["minifasnet", "device_boundary", "texture", "moire", "screen_replay",
             "screen_flicker", "micro_tremor", "rppg", "blink", "temporal"]
CATS = ["real", "static_image", "video_replay", "mask_3d", "heavy_makeup",
        "ar_filter", "deepfake_inject"]


def label_of(stem):
    m = re.search(r"\d{13}-(.+)$", stem)
    suffix = m.group(1) if m else stem
    key = re.sub(r"[0-9]+$", "", suffix.split("-")[0]).lower()  # liveperson2 -> liveperson
    for k, v in LABELMAP.items():
        if key.startswith(k) or suffix.lower().startswith(k):
            return suffix, v
    return suffix, ("unknown", "?", "special")


rows = []
for p in REPORTS:
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d.get("verdict", {})
    cs = v.get("category_scores", {})
    an = d.get("latest_analyzer_scores", {})
    gate = d.get("latest_gate_result") or {}

    def ascore(name):
        a = an.get(name)
        return round(a["score"], 1) if isinstance(a, dict) and a.get("score") is not None else ""

    suffix, (gt_class, expected, kind) = label_of(p.stem)
    verdict = "LIVE" if v.get("is_live") else "SPOOF"
    correct = (verdict == expected) if expected in ("LIVE", "SPOOF") else ""

    row = {
        "label": suffix,
        "ground_truth": gt_class,
        "kind": kind,
        "expected": expected,
        "verdict": verdict,
        "correct": correct,
        "confidence": round(v.get("confidence", 0) * 100, 1),
        "dominant_threat": v.get("dominant_threat") or "-",
        "duration_s": round(v.get("session_duration_sec", 0), 1),
        "blinks": v.get("blink_count", ""),
        "incidents": len(v.get("incidents", [])),
    }
    for c in CATS:
        row[f"p_{c}"] = round(cs.get(c, 0), 3)
    for a in ANALYZERS:
        row[a] = ascore(a)
    rows.append(row)

fields = (["label", "ground_truth", "kind", "expected", "verdict", "correct",
           "confidence", "dominant_threat", "duration_s", "blinks", "incidents"]
          + [f"p_{c}" for c in CATS] + ANALYZERS)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

# ---- metrics ----
attacks = [r for r in rows if r["kind"] == "attack"]
bona = [r for r in rows if r["kind"] == "bona_fide"]

bpcer = (sum(1 for r in bona if r["verdict"] == "SPOOF") / len(bona)) if bona else float("nan")

species = sorted(set(r["ground_truth"] for r in attacks))
apcer_by = {}
for sp in species:
    sp_rows = [r for r in attacks if r["ground_truth"] == sp]
    apcer_by[sp] = sum(1 for r in sp_rows if r["verdict"] == "LIVE") / len(sp_rows)
apcer_overall = max(apcer_by.values()) if apcer_by else float("nan")
acer = (apcer_overall + bpcer) / 2 if bona and attacks else float("nan")

lines = []
lines.append("amispoof Live PAD Metrics (ISO/IEC 30107-3) — labeled session reports")
lines.append("=" * 60)
lines.append(f"Sessions: {len(rows)}  ({len(bona)} bona-fide, {len(attacks)} attack, "
             f"{len(rows)-len(bona)-len(attacks)} special)")
lines.append("")
lines.append(f"BPCER (genuine called SPOOF) : {bpcer:.0%}")
lines.append("APCER by attack type (attack called LIVE = bypass):")
for sp, val in apcer_by.items():
    n = sum(1 for r in attacks if r["ground_truth"] == sp)
    lines.append(f"  {sp:<14}: {val:.0%}  (n={n})")
lines.append(f"APCER (overall, worst-case)  : {apcer_overall:.0%}")
lines.append(f"ACER  = (APCER+BPCER)/2      : {acer:.0%}")
lines.append("")
lines.append("Special (not in APCER/BPCER):")
for r in rows:
    if r["kind"] == "special":
        lines.append(f"  {r['label']:<22} -> {r['verdict']} (expected {r['expected']})")
txt = "\n".join(lines)
OUT_TXT.write_text(txt, encoding="utf-8")

# print table
print(f"{len(rows)} sessions -> {OUT_CSV}\n")
hdr = f"{'label':<24}{'ground_truth':<14}{'verdict':<7}{'exp':<6}{'ok':<3}{'conf':>5} {'MFN':>5}{'replay':>7}{'inc':>4}{'blink':>6}"
print(hdr); print("-" * len(hdr))
for r in rows:
    ok = "Y" if r["correct"] is True else ("N" if r["correct"] is False else "-")
    print(f"{r['label']:<24}{r['ground_truth']:<14}{r['verdict']:<7}{r['expected']:<6}{ok:<3}"
          f"{r['confidence']:>4}%{str(r['minifasnet']):>5}{str(r['screen_replay']):>7}"
          f"{r['incidents']:>4}{str(r['blinks']):>6}")
print("\n" + txt)
