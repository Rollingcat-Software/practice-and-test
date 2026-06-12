"""
Step 08 — CFP-FP Profile/Frontal Angle Robustness Test (FIVUCSAS)

Veri Seti
---------
CFP-FP (Celebrities in Frontal-Profile):
  500 kisi x (10 frontal + 4 profil) = 5.000 goruntu
  7.000 cift:
    3.500 genuine  -> ayni kisinin frontal + profil fotografi (90 derece fark)
    3.500 imposter -> farkli kisiler
Kaynak: archive (1).zip > eval/cfp_fp.bin

Neden Onemli?
-------------
Gercek hayatta kullanicilar kameraya tam bakmiyor olabilir — yana donuk,
hafif asagiya egilmis vb. Bu test:
  "Kullanici 30-90 derece yana bakiyorsa sistem onu taniyor mu?"
sorusunu cevaplar. Kimlik dogrulama sistemleri icin kritik bir senaryo.

Endpoint
--------
POST /api/v1/compare  ->  iki resim gonder, similarity/distance al

Ciktilar
--------
  pair_scores.csv    her cift: label, similarity, distance, match
  summary.txt        AUC, EER, LFW ve AgeDB-30 karsilastirmasi
  roc.png            ROC egrisi (LFW referans dahil)
"""

import csv
import io
import pickle
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# ---------- CONFIG ----------
API_BASE    = "http://localhost:8001"
DATASET_ZIP = Path(r"C:\Users\hp\Documents\GitHub\Dataset\archive (1).zip")
BIN_PATH    = "eval/cfp_fp.bin"
MAX_PAIRS   = 7000          # tumuyle calistir (3500 genuine + 3500 imposter)
WORKERS     = 5
TIMEOUT     = 30
SEED        = 42
OUT_DIR     = Path(__file__).parent
OUT_CSV     = OUT_DIR / "pair_scores.csv"
OUT_SUM     = OUT_DIR / "summary.txt"
OUT_PLOT    = OUT_DIR / "roc.png"
# Referans degerler
LFW_AUC     = 0.9943
LFW_EER     = 0.0193
AGEDB_AUC   = None   # step 07 calistirildiktan sonra buraya girilebilir
# ----------------------------

random.seed(SEED)


# ──────────────────────────────────────────────
# 1. Bin dosyasini oku
# ──────────────────────────────────────────────

def load_pairs():
    print(f"Loading {BIN_PATH} from zip ...")
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        raw = zf.read(BIN_PATH)
    bins, issame_list = pickle.load(io.BytesIO(raw), encoding='bytes')

    pairs = []
    for i, same in enumerate(issame_list):
        pairs.append((bins[2 * i], bins[2 * i + 1], int(same)))

    if MAX_PAIRS and len(pairs) > MAX_PAIRS:
        genuine  = [p for p in pairs if p[2] == 1]
        imposter = [p for p in pairs if p[2] == 0]
        half     = MAX_PAIRS // 2
        pairs    = random.sample(genuine, min(half, len(genuine))) + \
                   random.sample(imposter, min(half, len(imposter)))
        random.shuffle(pairs)

    g = sum(1 for p in pairs if p[2] == 1)
    i = len(pairs) - g
    print(f"Pairs loaded: {len(pairs)} total  ({g} genuine frontal-profile, {i} imposter)")
    return pairs


# ──────────────────────────────────────────────
# 2. Resim donusturme
# ──────────────────────────────────────────────

def arr_to_buf(arr):
    raw = arr.flatten().tobytes()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────
# 3. Karsilastirma
# ──────────────────────────────────────────────

def compare_pair(pair_idx, arr_a, arr_b, label):
    try:
        buf_a = arr_to_buf(arr_a)
        buf_b = arr_to_buf(arr_b)
        r = requests.post(
            f"{API_BASE}/api/v1/compare",
            files={"file1": ("frontal.jpg", buf_a, "image/jpeg"),
                   "file2": ("profile.jpg", buf_b, "image/jpeg")},
            timeout=TIMEOUT,
        )
        if not r.ok:
            return {"idx": pair_idx, "label": label, "status": r.status_code,
                    "similarity": None, "distance": None, "match": None,
                    "q_frontal": None, "q_profile": None, "error": r.text[:100]}
        j = r.json()
        return {
            "idx":       pair_idx,
            "label":     label,
            "status":    r.status_code,
            "similarity": j.get("similarity"),
            "distance":  j.get("distance"),
            "match":     j.get("match"),
            "q_frontal": j.get("face1", {}).get("quality_score"),
            "q_profile": j.get("face2", {}).get("quality_score"),
            "error":     "",
        }
    except Exception as e:
        return {"idx": pair_idx, "label": label, "status": -1,
                "similarity": None, "distance": None, "match": None,
                "q_frontal": None, "q_profile": None, "error": str(e)[:120]}


def run_comparisons(pairs):
    print(f"\nScoring {len(pairs)} pairs with {WORKERS} workers ...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(compare_pair, i, a, b, lbl): i
            for i, (a, b, lbl) in enumerate(pairs)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if done % 500 == 0 or done == len(pairs):
                ok = sum(1 for r in results if r["status"] == 200)
                print(f"  [{done:5d}/{len(pairs)}]  ok={ok}  err={done-ok}")

    results.sort(key=lambda x: x["idx"])
    return results


# ──────────────────────────────────────────────
# 4. Metrik hesaplama
# ──────────────────────────────────────────────

def compute_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    P = sum(labels); N = len(labels) - P
    if P == 0 or N == 0: return 0.5
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


def compute_metrics(results):
    valid = [r for r in results if r["distance"] is not None]
    if not valid:
        print("No valid results!"); return None

    labels    = [r["label"] for r in valid]
    distances = [r["distance"] for r in valid]
    scores    = [1.0 - d for d in distances]

    n_genuine  = sum(labels)
    n_imposter = len(labels) - n_genuine

    thresholds = sorted(set(distances))
    sweep = []
    best_eer, best_thr = 1.0, None
    for t in thresholds:
        far = sum(1 for d, l in zip(distances, labels) if d < t  and l == 0) / max(n_imposter, 1)
        frr = sum(1 for d, l in zip(distances, labels) if d >= t and l == 1) / max(n_genuine,  1)
        sweep.append((t, far, frr))
        if best_thr is None or abs(far - frr) < abs(best_eer * 2 - 1e-9):
            best_eer = (far + frr) / 2
            best_thr = t

    auc = compute_auc(scores, labels)

    # Kalite karsilastirmasi: frontal vs profil quality score farki
    q_frontal = [r["q_frontal"] for r in valid if r["q_frontal"] and r["label"] == 1]
    q_profile  = [r["q_profile"]  for r in valid if r["q_profile"]  and r["label"] == 1]
    avg_q_frontal = sum(q_frontal) / len(q_frontal) if q_frontal else None
    avg_q_profile  = sum(q_profile)  / len(q_profile)  if q_profile  else None

    at_verify  = next(((t, f, r) for t, f, r in sweep if abs(t - 0.45) < 0.01), None)
    at_compare = next(((t, f, r) for t, f, r in sweep if abs(t - 0.40) < 0.01), None)

    return {
        "n_valid":       len(valid),
        "n_genuine":     n_genuine,
        "n_imposter":    n_imposter,
        "auc":           auc,
        "eer":           best_eer,
        "eer_thr":       best_thr,
        "sweep":         sweep,
        "at_verify":     at_verify,
        "at_compare":    at_compare,
        "avg_q_frontal": avg_q_frontal,
        "avg_q_profile": avg_q_profile,
        "labels":        labels,
        "distances":     distances,
    }


# ──────────────────────────────────────────────
# 5. Cikti
# ──────────────────────────────────────────────

def write_csv(results):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx","label","status","similarity",
                                           "distance","match","q_frontal","q_profile","error"])
        w.writeheader()
        w.writerows(results)
    print(f"  CSV -> {OUT_CSV}")


def write_summary(m, errors):
    lines = [
        "CFP-FP Frontal-Profile Angle Robustness Test Summary\n",
        "=" * 55 + "\n\n",
        f"Dataset : CFP-FP (frontal vs 90-degree profile pairs)\n",
        f"Pairs   : {m['n_valid']} valid  ({m['n_genuine']} genuine, {m['n_imposter']} imposter)\n",
        f"Errors  : {errors}\n\n",
        "=== Biometric Metrics ===\n",
        f"AUC  : {m['auc']:.4f}\n",
        f"EER  : {m['eer']:.4f}  at distance threshold {m['eer_thr']:.4f}\n\n",
        "=== Veri Seti Karsilastirmasi ===\n",
        f"{'Metrik':<10} {'CFP-FP':>10} {'LFW':>10} {'Fark':>10}\n",
        "-" * 42 + "\n",
        f"{'AUC':<10} {m['auc']:>10.4f} {LFW_AUC:>10.4f} {m['auc']-LFW_AUC:>+10.4f}\n",
        f"{'EER':<10} {m['eer']:>10.4f} {LFW_EER:>10.4f} {m['eer']-LFW_EER:>+10.4f}\n\n",
    ]

    # Kalite farki
    if m["avg_q_frontal"] and m["avg_q_profile"]:
        lines += [
            "=== Frontal vs Profil Kalite Farki (genuine cifler) ===\n",
            f"Ortalama frontal kalite skoru : {m['avg_q_frontal']:.1f}\n",
            f"Ortalama profil  kalite skoru : {m['avg_q_profile']:.1f}\n",
            f"Fark : {m['avg_q_frontal'] - m['avg_q_profile']:+.1f} "
            f"({'profil daha dusuk' if m['avg_q_frontal'] > m['avg_q_profile'] else 'frontal daha dusuk'})\n\n",
        ]

    if m["at_verify"]:
        t, far, frr = m["at_verify"]
        lines += [
            "=== Uretim Esiklerinde Performans ===\n",
            f"/verify esigi (distance=0.45):  FAR={far:.4f}  FRR={frr:.4f}\n",
        ]
    if m["at_compare"]:
        t, far, frr = m["at_compare"]
        lines.append(f"/compare esigi (distance=0.40): FAR={far:.4f}  FRR={frr:.4f}\n")

    lines.append("\n=== Yorum ===\n")
    delta_eer = m["eer"] - LFW_EER
    if delta_eer <= 0.03:
        lines.append("SONUC: Sistem profil acisina karsi guclu dayanim gosteriyor.\n")
        lines.append("Kullanicinin kameraya tam bakmadigi senaryolarda guvenilir.\n")
    elif delta_eer <= 0.08:
        lines.append("SONUC: Profil acisi performansi dusurmus ama kabul edilebilir.\n")
        lines.append("Oneri: Kayit sirasinda hem frontal hem hafif acili fotograf alinmali.\n")
        lines.append("Coklu goruntu kaydinin (multi-image enrollment) kullanilmasi degerlendirilmeli.\n")
    else:
        lines.append("SONUC: Profil acisi performansi ciddi sekilde dusurmus.\n")
        lines.append("Sistem yan bakan yuzlerde guvensiz — kayit kosullari standartlastirilmali.\n")
        lines.append("Kullanicilara kayit sirasinda 'kameraya tam bakin' yonlendirmesi zorunlu olmali.\n")

    text = "".join(lines)
    OUT_SUM.write_text(text, encoding="utf-8")
    print("\n" + "=" * 60)
    print(text)
    print(f"  Summary -> {OUT_SUM}")


def plot_roc(m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sweep = m["sweep"]
        fars  = [s[1] for s in sweep]
        tars  = [1 - s[2] for s in sweep]

        plt.figure(figsize=(7, 6))
        plt.plot(fars, tars, color="#FF9800", linewidth=2,
                 label=f"CFP-FP Profile (AUC={m['auc']:.3f}, EER={m['eer']:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
        plt.axvline(x=m["eer"], color="#FF9800", linestyle=":", linewidth=1,
                    label=f"EER={m['eer']:.3f}")

        # LFW referans
        plt.axhline(y=1 - LFW_EER, color="#2196F3", linestyle="--", linewidth=1,
                    label=f"LFW TAR@EER (referans)")

        # Notlar
        plt.annotate("Frontal baseline\n(LFW)", xy=(LFW_EER, 1 - LFW_EER),
                     fontsize=8, color="#2196F3",
                     xytext=(LFW_EER + 0.05, 1 - LFW_EER - 0.05),
                     arrowprops=dict(arrowstyle="->", color="#2196F3", lw=0.8))

        plt.xlabel("FAR (False Accept Rate)", fontsize=11)
        plt.ylabel("TAR (True Accept Rate)", fontsize=11)
        plt.title("FIVUCSAS — CFP-FP ROC (Frontal vs Profile 90 deg)", fontsize=12)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOT, dpi=130)
        print(f"  ROC plot -> {OUT_PLOT}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ──────────────────────────────────────────────
# 6. Ana akis
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print(" FIVUCSAS  .  CFP-FP Frontal-Profile Angle Robustness Test")
    print("=" * 70)
    print(f"  Endpoint : POST {API_BASE}/api/v1/compare")
    print(f"  Dataset  : {BIN_PATH}")
    print(f"  Pairs    : up to {MAX_PAIRS}")

    pairs   = load_pairs()
    results = run_comparisons(pairs)

    write_csv(results)
    errors = sum(1 for r in results if r["status"] != 200)

    m = compute_metrics(results)
    if m:
        write_summary(m, errors)
        plot_roc(m)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
