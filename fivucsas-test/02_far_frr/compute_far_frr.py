"""
FAR / FRR / EER / ROC for the FIVUCSAS face-verification pipeline.

Strategy
--------
For unbiased pair scores, we use a SEPARATE tenant ("lfw-pairs") and enroll
only photo #1 for each user. Photos #2..N are then test queries.

  Genuine pair (label=1):  query photo K of user U  →  /verify against U
  Imposter pair (label=0): query photo K of user U  →  /verify against random V (V != U)

We sweep thresholds over the returned 'distance' field, which is what /verify
already uses internally (lower = more similar). ROC, EER, AUC are computed
from those distances.

Outputs
-------
- pair_scores.csv      one row per pair (label, user, query, target, distance, confidence)
- far_frr_summary.txt  EER, AUC, operating thresholds
- roc.png              ROC curve plot (optional, only if matplotlib is installed)
"""
import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE        = "http://localhost:8001"
DATASET         = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
ENROLL_RESULTS  = Path(__file__).parent.parent / "01_bulk_enroll" / "enroll_results.csv"
TEST_TENANT     = "lfw-pairs"
WORKERS         = 4
TIMEOUT         = 60
MIN_PHOTOS      = 3                  # need ≥3 successful: 1 enroll + ≥2 tests
MAX_USERS       = 80                 # cap for runtime
MAX_GENUINE     = 1000               # cap genuine pairs sampled
MAX_IMPOSTER    = 5000               # cap imposter pairs sampled
SEED            = 42
OUT_PAIRS_CSV   = Path(__file__).parent / "pair_scores.csv"
OUT_SUMMARY     = Path(__file__).parent / "far_frr_summary.txt"
OUT_PLOT        = Path(__file__).parent / "roc.png"
# ----------------------------

random.seed(SEED)


# ---------- 1. Load enrollment results, group successful images by user ----------
def load_successful_users():
    """Return {user_id: [Path, Path, ...]} from enroll_results.csv (only ok=True)."""
    by_user = {}
    with open(ENROLL_RESULTS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] != "True":
                continue
            user = row["user_id"]
            img  = DATASET / row["folder"] / row["image"]
            if img.exists():
                by_user.setdefault(user, []).append(img)
    return {u: sorted(p) for u, p in by_user.items() if len(p) >= MIN_PHOTOS}


# ---------- 2. Re-enroll one photo per user under TEST_TENANT ----------
def enroll_baseline(user_id: str, img_path: Path):
    try:
        with open(img_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/v1/enroll",
                files={"file": (img_path.name, f, "image/jpeg")},
                data={"user_id": user_id, "tenant_id": TEST_TENANT},
                timeout=TIMEOUT,
            )
        return user_id, r.ok, r.status_code, r.text[:120]
    except Exception as e:
        return user_id, False, -1, str(e)[:120]


def enroll_baselines(users):
    print(f"\nStep 1/3 — Enrolling 1 photo / user under tenant '{TEST_TENANT}' ...")
    success = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(enroll_baseline, u, p[0]): u for u, p in users.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            user, ok, status, body = fut.result()
            if ok:
                success[user] = users[user]
            if i % 20 == 0 or not ok:
                mark = "OK" if ok else f"FAIL({status})"
                print(f"  [{i:3d}/{len(users)}] {mark:>10} {user}")
    print(f"  Baseline enrolled: {len(success)} / {len(users)}")
    return success


# ---------- 3. Build pair lists ----------
def build_pairs(users):
    """users = {user_id: [enroll_img, test_img1, test_img2, ...]}"""
    user_list = list(users.keys())
    genuine, imposter = [], []

    for u, imgs in users.items():
        for q in imgs[1:]:                 # skip imgs[0] (enrolled baseline)
            genuine.append((u, q, u))      # query vs own user_id

    if len(genuine) > MAX_GENUINE:
        genuine = random.sample(genuine, MAX_GENUINE)

    while len(imposter) < MAX_IMPOSTER:
        u = random.choice(user_list)
        v = random.choice(user_list)
        if u == v:
            continue
        q = random.choice(users[u][1:])    # any test photo of u
        imposter.append((u, q, v))         # photo of u, claim to be v

    return genuine, imposter


# ---------- 4. Score one pair ----------
def score_pair(label, query_user, query_img, target_user):
    try:
        with open(query_img, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/v1/verify",
                files={"file": (query_img.name, f, "image/jpeg")},
                data={"user_id": target_user, "tenant_id": TEST_TENANT},
                timeout=TIMEOUT,
            )
        if not r.ok:
            return None
        j = r.json()
        return {
            "label":      label,
            "query_user": query_user,
            "query_img":  query_img.name,
            "target":     target_user,
            "distance":   float(j.get("distance", 1.0)),
            "confidence": float(j.get("confidence", 0.0)),
            "verified":   bool(j.get("verified", False)),
        }
    except Exception:
        return None


def score_all(genuine, imposter):
    print(f"\nStep 2/3 — Scoring {len(genuine)} genuine + {len(imposter)} imposter pairs ...")
    rows = []
    tasks = ([("genuine", *g) for g in genuine] +
             [("imposter", *i) for i in imposter])
    random.shuffle(tasks)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(score_pair, 1 if t[0] == "genuine" else 0, t[1], t[2], t[3])
                   for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                rows.append(res)
            if i % 200 == 0:
                print(f"  [{i:5d}/{len(tasks)}] scored")

    with open(OUT_PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows)} pair scores -> {OUT_PAIRS_CSV}")
    return rows


# ---------- 5. Compute FAR/FRR/EER/AUC ----------
def compute_metrics(rows):
    print("\nStep 3/3 — Computing FAR/FRR/EER/AUC ...")
    # Lower distance = better match. Convert to "score" = 1 - distance for ROC convention.
    labels    = [r["label"] for r in rows]
    distances = [r["distance"] for r in rows]
    scores    = [1.0 - d for d in distances]   # higher score = same person

    # threshold sweep on distance
    n_genuine  = sum(labels)
    n_imposter = len(labels) - n_genuine
    print(f"  {n_genuine} genuine, {n_imposter} imposter pairs")

    thresholds = sorted(set(distances))
    best_eer, best_thr = 1.0, None
    sweep = []
    for t in thresholds:
        # accept pair if distance < t
        far = sum(1 for d, l in zip(distances, labels) if d <  t and l == 0) / max(n_imposter, 1)
        frr = sum(1 for d, l in zip(distances, labels) if d >= t and l == 1) / max(n_genuine, 1)
        sweep.append((t, far, frr))
        if abs(far - frr) < abs(best_eer * 2):
            if abs(far - frr) < 0.005 or best_thr is None:
                best_eer = (far + frr) / 2
                best_thr = t

    # AUC via trapezoidal rule on sorted (FPR, TPR)
    pts = sorted(set((s, l) for s, l in zip(scores, labels)))
    auc = compute_auc(scores, labels)

    # Pick operating points
    op_high_sec = next(((t, far, frr) for t, far, frr in sweep if far <= 0.001), None)
    op_balanced = next(((t, far, frr) for t, far, frr in sweep if far <= 0.01), None)

    summary = []
    summary.append(f"Pairs: {n_genuine} genuine, {n_imposter} imposter\n")
    summary.append(f"AUC: {auc:.4f}\n")
    summary.append(f"EER: {best_eer:.4f} at threshold (distance) = {best_thr:.4f}\n")
    if op_high_sec:
        t, far, frr = op_high_sec
        summary.append(f"High-security (FAR<=0.1%): threshold={t:.4f}  FAR={far:.4f}  FRR={frr:.4f}\n")
    if op_balanced:
        t, far, frr = op_balanced
        summary.append(f"Balanced     (FAR<=1%):   threshold={t:.4f}  FAR={far:.4f}  FRR={frr:.4f}\n")
    summary.append("\nDefault production threshold (from /verify): 0.45\n")
    at_default = next((s for s in sweep if abs(s[0] - 0.45) < 0.005), None)
    if at_default:
        t, far, frr = at_default
        summary.append(f"  At distance=0.45: FAR={far:.4f}  FRR={frr:.4f}\n")

    text = "".join(summary)
    print("\n" + text)
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    print(f"  Summary -> {OUT_SUMMARY}")

    try:
        plot_roc(sweep, auc)
    except Exception as e:
        print(f"  (skipping ROC plot: {e})")


def compute_auc(scores, labels):
    """Trapezoidal AUC."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    P = sum(labels); N = len(labels) - P
    if P == 0 or N == 0:
        return 0.5
    tp = fp = 0
    prev_score = None
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    for s, l in pairs:
        if s != prev_score:
            fpr = fp / N; tpr = tp / P
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr, prev_score = fpr, tpr, s
        if l == 1: tp += 1
        else:      fp += 1
    fpr = fp / N; tpr = tp / P
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return auc


def plot_roc(sweep, auc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fars = [s[1] for s in sweep]
    tars = [1 - s[2] for s in sweep]
    plt.figure(figsize=(6, 6))
    plt.plot(fars, tars, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FAR  (False Accept Rate)")
    plt.ylabel("TAR  (True Accept Rate)")
    plt.title("FIVUCSAS Face Verification — ROC (LFW)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=130)
    print(f"  ROC plot -> {OUT_PLOT}")


# ---------- main ----------
def main():
    t0 = time.time()
    users = load_successful_users()
    if MAX_USERS and len(users) > MAX_USERS:
        keep = list(users.keys())[:MAX_USERS]
        users = {k: users[k] for k in keep}
    print(f"Eligible users (>= {MIN_PHOTOS} successful photos): {len(users)}")
    print(f"Total photos available: {sum(len(v) for v in users.values())}")

    users = enroll_baselines(users)
    if not users:
        print("No users enrolled, abort."); return

    genuine, imposter = build_pairs(users)
    rows = score_all(genuine, imposter)
    if not rows:
        print("No pair scores, abort."); return

    compute_metrics(rows)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
