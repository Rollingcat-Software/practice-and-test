"""
Step 04b — Synthetic Spoof Attack Test (FIVUCSAS)

Strategy
--------
We take real enrolled faces from the lfw-test tenant and create synthetic
spoof versions using Pillow. Each spoof image simulates a different real-world
presentation attack, then we send it to /api/v1/verify to see if the system
accepts it.

Attack types
------------
  PRINT_MILD    : JPEG recompress q=70 + light blur  (weak printer, close range)
  PRINT_STRONG  : JPEG recompress q=45 + heavier blur + vignette  (crumpled/far print)
  SCREEN_REPLAY : scan-line overlay + warm color shift + recompress  (phone screen replay)
  BORDER_PRINT  : print + white border frame  (photo held up to camera)

Metrics
-------
  APCER (Attack Presentation Classification Error Rate)
      = accepted spoof attacks / total spoof attempts
      Low is good. 0 means every spoof was rejected.

  BPCER (Bona Fide Presentation Classification Error Rate)
      = rejected genuine images / total genuine attempts
      We include this as a sanity check using real images.

  Score distributions: plot genuine vs attack score histograms.

Outputs
-------
  spoof_results.csv        per-image: attack type, verified, distance
  spoof_summary.txt        APCER/BPCER per attack type + overall
  score_distribution.png   histogram of genuine vs attack distances
  spoof_attacks/           sample generated spoof images (first 5 per type)
"""
import csv
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageFilter

# ---------- CONFIG ----------
API_BASE       = "http://localhost:8001"
DATASET        = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
ENROLL_CSV     = Path(__file__).parent.parent / "01_bulk_enroll" / "enroll_results.csv"
TENANT         = "lfw-test"

N_USERS        = 60      # how many enrolled users to attack
ATTACKS_PER_USER = 1     # spoof attempts per attack type per user (keep 1 for speed)
GENUINE_PER_USER = 1     # genuine verify calls per user (for BPCER baseline)
WORKERS        = 4
TIMEOUT        = 30
SEED           = 42

OUT_DIR        = Path(__file__).parent
SAMPLE_DIR     = OUT_DIR / "spoof_attacks"
OUT_CSV        = OUT_DIR / "spoof_results.csv"
OUT_SUM        = OUT_DIR / "spoof_summary.txt"
OUT_PLOT       = OUT_DIR / "score_distribution.png"
# ----------------------------

random.seed(SEED)
SAMPLE_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────
# 1. ATTACK GENERATORS
# ───────────────────────────────────────────────

def attack_print_mild(img: Image.Image) -> Image.Image:
    """Mild print attack: slight blur + JPEG recompress at q=70."""
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return Image.open(buf).copy()


def attack_print_strong(img: Image.Image) -> Image.Image:
    """Strong print attack: heavier blur + low quality + vignette."""
    img = img.filter(ImageFilter.GaussianBlur(radius=1.8))
    # Reduce contrast slightly (simulate faded printout)
    from PIL import ImageEnhance
    img = ImageEnhance.Contrast(img).enhance(0.85)
    img = ImageEnhance.Brightness(img).enhance(0.92)
    # Vignette: darken corners
    w, h = img.size
    vignette = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(vignette)
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.55)
    for r in range(radius, 0, -1):
        alpha = int(220 * (1 - r / radius))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
    vignette_rgb = Image.merge("RGB", [vignette, vignette, vignette])
    img = Image.blend(img.convert("RGB"), vignette_rgb, alpha=0.18)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=45)
    buf.seek(0)
    return Image.open(buf).copy()


def attack_screen_replay(img: Image.Image) -> Image.Image:
    """Screen replay: scan lines + warm color shift + JPEG recompress."""
    img = img.convert("RGB")
    pixels = img.load()
    w, h = img.size
    # Horizontal scan lines every 3 rows — reduce brightness by 12%
    for y in range(0, h, 3):
        for x in range(w):
            r, g, b = pixels[x, y]
            pixels[x, y] = (int(r * 0.88), int(g * 0.88), int(b * 0.88))
    # Warm color shift (boost red slightly, reduce blue slightly)
    from PIL import ImageEnhance
    r_ch, g_ch, b_ch = img.split()
    r_ch = r_ch.point(lambda p: min(255, int(p * 1.06)))
    b_ch = b_ch.point(lambda p: int(p * 0.92))
    img = Image.merge("RGB", (r_ch, g_ch, b_ch))
    # Slight blur (monitor pixel bleed)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return Image.open(buf).copy()


def attack_border_print(img: Image.Image) -> Image.Image:
    """Border print: simulate photo held up — add white border + slight blur."""
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    w, h = img.size
    border = 20
    canvas = Image.new("RGB", (w + border * 2, h + border * 2), (240, 240, 240))
    canvas.paste(img, (border, border))
    # Resize back to original so the face is smaller (zoomed out)
    canvas = canvas.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=65)
    buf.seek(0)
    return Image.open(buf).copy()


ATTACK_TYPES = {
    "PRINT_MILD":    attack_print_mild,
    "PRINT_STRONG":  attack_print_strong,
    "SCREEN_REPLAY": attack_screen_replay,
    "BORDER_PRINT":  attack_border_print,
}


# ───────────────────────────────────────────────
# 2. DATA LOADING
# ───────────────────────────────────────────────

def load_enrolled_users(n: int):
    """
    Return {user_id: img_path} for the first N users with at least
    one successfully enrolled image.
    """
    users = {}
    with open(ENROLL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] != "True":
                continue
            uid = row["user_id"]
            if uid in users:
                continue
            img = DATASET / row["folder"] / row["image"]
            if img.exists():
                users[uid] = img
            if len(users) >= n:
                break
    return users


# ───────────────────────────────────────────────
# 3. VERIFY CALL
# ───────────────────────────────────────────────

def do_verify(user_id: str, img_buf: io.BytesIO, label: str, filename: str = "spoof.jpg"):
    img_buf.seek(0)
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/verify",
            files={"file": (filename, img_buf, "image/jpeg")},
            data={"user_id": user_id, "tenant_id": TENANT},
            timeout=TIMEOUT,
        )
        body = r.json() if r.ok or r.status_code == 400 else {"raw": r.text[:200]}
        return {
            "label":    label,
            "user_id":  user_id,
            "status":   r.status_code,
            "verified": body.get("verified"),
            "distance": body.get("distance"),
            "error":    body.get("message", "") if not r.ok else "",
        }
    except Exception as e:
        return {"label": label, "user_id": user_id, "status": -1,
                "verified": None, "distance": None, "error": str(e)[:200]}


def img_to_buf(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


# ───────────────────────────────────────────────
# 4. MAIN TEST LOOP
# ───────────────────────────────────────────────

def run_tests(users: dict):
    tasks = []  # (user_id, img_buf, label)

    sample_saved = {k: 0 for k in list(ATTACK_TYPES) + ["GENUINE"]}

    for uid, img_path in users.items():
        try:
            real_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Genuine verify (BPCER baseline)
        for _ in range(GENUINE_PER_USER):
            tasks.append((uid, img_to_buf(real_img), "GENUINE", img_path.name))
            if sample_saved["GENUINE"] < 3:
                real_img.save(SAMPLE_DIR / f"genuine_{sample_saved['GENUINE']:02d}.jpg")
                sample_saved["GENUINE"] += 1

        # Each attack type
        for attack_name, attack_fn in ATTACK_TYPES.items():
            for _ in range(ATTACKS_PER_USER):
                try:
                    spoofed = attack_fn(real_img.copy())
                    buf = img_to_buf(spoofed)
                    tasks.append((uid, buf, attack_name, f"{attack_name.lower()}_{img_path.name}"))
                    if sample_saved[attack_name] < 3:
                        spoofed.save(SAMPLE_DIR / f"{attack_name.lower()}_{sample_saved[attack_name]:02d}.jpg")
                        sample_saved[attack_name] += 1
                except Exception as e:
                    print(f"  [WARN] attack {attack_name} failed for {uid}: {e}")

    random.shuffle(tasks)
    print(f"\nTotal tasks: {len(tasks)} "
          f"({len(users)} users × {GENUINE_PER_USER} genuine + {len(ATTACK_TYPES) * ATTACKS_PER_USER} attacks)")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(do_verify, uid, buf, label, fname): (uid, label)
            for uid, buf, label, fname in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i:4d}/{len(tasks)}] done ...")

    return results


# ───────────────────────────────────────────────
# 5. METRICS & OUTPUT
# ───────────────────────────────────────────────

def compute_apcer_bpcer(results):
    """
    APCER per attack type: accepted attacks / total attacks of that type
    BPCER: rejected genuine / total genuine
    """
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        if r["verified"] is not None:  # skip errors
            by_label[r["label"]].append(r)

    stats = {}
    for label, rows in by_label.items():
        total = len(rows)
        accepted = sum(1 for r in rows if r["verified"] is True)
        rejected = sum(1 for r in rows if r["verified"] is False)
        if label == "GENUINE":
            rate = rejected / total if total else None
            stats[label] = {"total": total, "accepted": accepted, "rejected": rejected,
                            "BPCER": rate}
        else:
            rate = accepted / total if total else None
            stats[label] = {"total": total, "accepted": accepted, "rejected": rejected,
                            "APCER": rate}
    return stats


def write_csv(results):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "user_id", "status",
                                           "verified", "distance", "error"])
        w.writeheader()
        w.writerows(results)
    print(f"  CSV -> {OUT_CSV}")


def write_summary(stats):
    lines = [
        "Synthetic Spoof Attack Test Summary\n",
        "=" * 45 + "\n\n",
        f"Tenant : {TENANT}\n",
        f"Users  : {N_USERS}\n",
        f"{'Label':<16} {'Total':>6} {'Accept':>7} {'Reject':>7} {'Rate':>8}  {'Metric'}\n",
        "-" * 55 + "\n",
    ]
    for label, s in stats.items():
        if label == "GENUINE":
            rate = s.get("BPCER")
            metric = "BPCER"
        else:
            rate = s.get("APCER")
            metric = "APCER"
        rate_str = f"{rate:.4f}" if rate is not None else "N/A"
        lines.append(
            f"{label:<16} {s['total']:>6} {s['accepted']:>7} {s['rejected']:>7} "
            f"{rate_str:>8}  {metric}\n"
        )

    lines.append("\n")
    lines.append("Interpretation\n")
    lines.append("-" * 55 + "\n")
    lines.append("APCER = 0.00: no spoof images accepted -> perfect anti-spoof\n")
    lines.append("APCER = 1.00: all spoof images accepted -> no anti-spoof at /verify\n")
    lines.append("BPCER = 0.00: all genuine images accepted -> no false rejections\n")
    lines.append("\n")

    # Verdict
    all_apcer = [s.get("APCER", 1.0) for k, s in stats.items() if k != "GENUINE"]
    bpcer = stats.get("GENUINE", {}).get("BPCER")
    avg_apcer = sum(all_apcer) / len(all_apcer) if all_apcer else None

    lines.append("Verdict\n")
    lines.append("-" * 55 + "\n")
    if avg_apcer is not None:
        if avg_apcer >= 0.9:
            lines.append(f"CRITICAL: avg APCER={avg_apcer:.3f} — /verify has NO spoof resistance.\n")
            lines.append("The model accepts spoof images as genuine because /verify only compares\n")
            lines.append("face embeddings — it does not check liveness. Anti-spoofing requires\n")
            lines.append("fixing the /liveness endpoint (BUG-02) and integrating it into /verify.\n")
        elif avg_apcer >= 0.5:
            lines.append(f"HIGH RISK: avg APCER={avg_apcer:.3f} — partial spoof resistance.\n")
        else:
            lines.append(f"MODERATE: avg APCER={avg_apcer:.3f} — some spoof resistance in embedding space.\n")
    if bpcer is not None:
        lines.append(f"BPCER={bpcer:.4f} — {bpcer*100:.1f}% of genuine images were falsely rejected.\n")

    text = "".join(lines)
    OUT_SUM.write_text(text, encoding="utf-8")
    print("\n" + "=" * 60)
    print(text)
    print(f"  Summary -> {OUT_SUM}")


def plot_distributions(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from collections import defaultdict
        by_label = defaultdict(list)
        for r in results:
            if r["distance"] is not None:
                by_label[r["label"]].append(float(r["distance"]))

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {
            "GENUINE":      ("#2196F3", "-"),
            "PRINT_MILD":   ("#FF5722", "--"),
            "PRINT_STRONG": ("#F44336", "-."),
            "SCREEN_REPLAY":("#FF9800", ":"),
            "BORDER_PRINT": ("#9C27B0", "--"),
        }
        for label, dists in sorted(by_label.items()):
            color, ls = colors.get(label, ("#999", "-"))
            ax.hist(dists, bins=20, alpha=0.55, label=label,
                    color=color, linestyle=ls, density=True, edgecolor="white")

        ax.axvline(x=0.45, color="red", linewidth=1.5, linestyle="--",
                   label="Threshold=0.45 (accept if <)")
        ax.set_xlabel("Distance (lower = more similar)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("FIVUCSAS — Genuine vs Spoof Distance Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOT, dpi=130)
        print(f"  Plot -> {OUT_PLOT}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ───────────────────────────────────────────────
# 6. ENTRY POINT
# ───────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print(" FIVUCSAS  ·  Synthetic Spoof Attack Test")
    print("=" * 70)
    print(f"\nTenant : {TENANT}")
    print(f"Users  : {N_USERS}")
    print(f"Attacks: {list(ATTACK_TYPES.keys())}")

    users = load_enrolled_users(N_USERS)
    if not users:
        print("ERROR: No enrolled users found. Run step 01 first."); return
    print(f"\nLoaded {len(users)} enrolled users")

    print("\nGenerating synthetic spoof images + running verify calls ...")
    results = run_tests(users)

    valid = [r for r in results if r["verified"] is not None]
    errors = len(results) - len(valid)
    if errors:
        print(f"\n  Skipped {errors} results with API errors")

    write_csv(results)
    stats = compute_apcer_bpcer(valid)
    write_summary(stats)
    plot_distributions(valid)

    print(f"\nSample spoof images saved to: {SAMPLE_DIR}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
