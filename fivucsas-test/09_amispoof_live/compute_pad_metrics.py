"""
ISO/IEC 30107-3 PAD metrics for the amispoof.fivucsas.com live test.

Reads results.csv (filled in manually after running the 20 trials at
https://amispoof.fivucsas.com/) and computes:

  BPCER  = bona-fide presentations classified as SPOOF / total bona-fide
  APCER  = attack presentations classified as LIVE / total attacks   (per PAI species)
  APCER (overall) = max over PAI species  (ISO worst-case convention)
  ACER   = (APCER_overall + BPCER) / 2

PAI species in this test: print, screen_photo, video_replay.

Usage:
    python compute_pad_metrics.py
Writes summary.txt next to this script.
"""
import csv
from pathlib import Path

RESULTS = Path(__file__).parent / "results.csv"
SUMMARY = Path(__file__).parent / "summary.txt"


def grade(acer: float) -> str:
    if acer < 0.05:  return "A (strong for production)"
    if acer < 0.10:  return "B (good, improvable)"
    if acer < 0.20:  return "C (borderline)"
    return "D (insufficient)"


def main():
    rows = []
    with open(RESULTS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("verdict") or "").strip().upper()
            if v not in ("LIVE", "SPOOF"):
                continue  # not filled in yet — skip
            rows.append(r)

    if not rows:
        print("No filled-in rows yet. Run the 20 trials at "
              "https://amispoof.fivucsas.com/ and fill the 'verdict' column.")
        return

    genuine = [r for r in rows if r["category"] == "genuine"]
    attacks = [r for r in rows if r["category"] != "genuine"]

    # BPCER: genuine wrongly called SPOOF
    bpcer = (sum(1 for r in genuine if r["verdict"].strip().upper() == "SPOOF")
             / len(genuine)) if genuine else float("nan")

    # APCER per PAI species: attack wrongly called LIVE
    species = sorted(set(r["category"] for r in attacks))
    apcer_by_species = {}
    for sp in species:
        sp_rows = [r for r in attacks if r["category"] == sp]
        apcer_by_species[sp] = (
            sum(1 for r in sp_rows if r["verdict"].strip().upper() == "LIVE")
            / len(sp_rows)
        )

    apcer_overall = max(apcer_by_species.values()) if apcer_by_species else float("nan")
    acer = (apcer_overall + bpcer) / 2 if apcer_by_species and genuine else float("nan")

    lines = []
    lines.append("amispoof.fivucsas.com — ISO 30107-3 PAD Metrics")
    lines.append("=" * 50)
    lines.append(f"Trials scored : {len(rows)}  ({len(genuine)} genuine, {len(attacks)} attack)")
    lines.append("")
    lines.append(f"BPCER (bona-fide rejected)   : {bpcer:.2%}")
    lines.append("")
    lines.append("APCER by PAI species (attack accepted as LIVE):")
    for sp, val in apcer_by_species.items():
        n = sum(1 for r in attacks if r["category"] == sp)
        lines.append(f"  {sp:<14}: {val:.2%}   (n={n})")
    lines.append("")
    lines.append(f"APCER (overall, worst-case)  : {apcer_overall:.2%}")
    lines.append(f"ACER  = (APCER+BPCER)/2      : {acer:.2%}")
    lines.append(f"ISO Grade                    : {grade(acer)}")
    lines.append("")
    # Per-trial misses
    misses = [r for r in rows
              if r["verdict"].strip().upper() != r["expected"].strip().upper()]
    if misses:
        lines.append("Misclassified trials:")
        for r in misses:
            lines.append(f"  [{r['trial_id']}] {r['category']}: "
                         f"expected {r['expected']}, got {r['verdict']}  "
                         f"({r.get('notes','').strip()})")
    else:
        lines.append("No misclassifications — all trials correct.")

    text = "\n".join(lines)
    print("\n" + text + "\n")
    SUMMARY.write_text(text, encoding="utf-8")
    print(f"Summary written -> {SUMMARY}")


if __name__ == "__main__":
    main()
