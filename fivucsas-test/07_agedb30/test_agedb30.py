"""
Step 07 — AgeDB-30 Age-Invariance Test (FIVUCSAS)

Veri Seti
---------
AgeDB-30: Aynı kişinin 30 yıl arayla çekilmiş fotoğraflarını içerir.
6.000 çift — 3.000 gerçek (genuine), 3.000 sahte (imposter).
Kaynak: archive (1).zip > eval/agedb_30.bin

Neden Önemli?
-------------
Bir üniversite sisteminde öğrenci kayıt olurken çekilen fotoğraf,
4 yıl sonra mezuniyette hâlâ geçerli mi? Bu test bunu ölçer.

Temel soru: Yaş farkı sistemi yanıltıyor mu?

Endpoint
--------
POST /api/v1/compare  →  iki resim gönder, similarity/distance al
(Enrollment gerekmez — direkt görüntü karşılaştırması)

Çıktılar
--------
  pair_scores.csv    her çift: label, similarity, distance, match
  summary.txt        AUC, EER, FAR/FRR, LFW karşılaştırması
  roc.png            ROC eğrisi
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
BIN_PATH    = "eval/agedb_30.bin"
MAX_PAIRS   = 6000          # tümünü çalıştır (3000 genuine + 3000 imposter)
WORKERS     = 5
TIMEOUT     = 30
SEED        = 42
OUT_DIR     = Path(__file__).parent
OUT_CSV     = OUT_DIR / "pair_scores.csv"
OUT_SUM     = OUT_DIR / "summary.txt"
OUT_PLOT    = OUT_DIR / "roc.png"
# LFW sonuçlarıyla karşılaştırma için referans değerler
LFW_AUC     = 0.9943
LFW_EER     = 0.0193
# ----------------------------

random.seed(SEED)


# ──────────────────────────────────────────────
# 1. Bin dosyasını oku
# ──────────────────────────────────────────────

def load_pairs():
    """
    Bin dosyasından (image_a, image_b, label) listesi dondurur.
    label=1 -> genuine (ayni kisi), label=0 -> imposter (farkli kisi)
    """
    print(f"Loading {BIN_PATH} from zip ...")
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        raw = zf.read(BIN_PATH)
    bins, issame_list = pickle.load(io.BytesIO(raw), encoding='bytes')

    pairs = []
    for i, same in enumerate(issame_list):
        img_a = bins[2 * i]
        img_b = bins[2 * i + 1]
        pairs.append((img_a, img_b, int(same)))

    if MAX_PAIRS and len(pairs) > MAX_PAIRS:
        genuine   = [p for p in pairs if p[2] == 1]
        imposter  = [p for p in pairs if p[2] == 0]
        half      = MAX_PAIRS // 2
        pairs     = random.sample(genuine, min(half, len(genuine))) + \
                    random.sample(imposter, min(half, len(imposter)))
        random.shuffle(pairs)

    genuine_n  = sum(1 for p in pairs if p[2] == 1)
    imposter_n = sum(1 for p in pairs if p[2] == 0)
    print(f"Pairs loaded: {len(pairs)} total  ({genuine_n} genuine, {imposter_n} imposter)")
    return pairs


# ──────────────────────────────────────────────
# 2. Resim donusturme
# ──────────────────────────────────────────────

def arr_to_buf(arr):
    """Numpy uint8 array -> JPEG BytesIO."""
    raw = arr.flatten().tobytes()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────
# 3. Karsilastirma cagrisi
# ──────────────────────────────────────────────

def compare_pair(pair_idx, arr_a, arr_b, label):
    """POST /api/v1/compare -> dict ile sonuc dondurur."""
    try:
        buf_a = arr_to_buf(arr_a)
        buf_b = arr_to_buf(arr_b)
        r = requests.post(
            f"{API_BASE}/api/v1/compare",
            files={"file1": ("a.jpg", buf_a, "image/jpeg"),
                   "file2": ("b.jpg", buf_b, "image/jpeg")},
            timeout=TIMEOUT,
        )
        if not r.ok:
            return {"idx": pair_idx, "label": label, "status": r.status_code,
                    "similarity": None, "distance": None, "match": None,
                    "q1": None, "q2": None, "error": r.text[:100]}
        j = r.json()
        return {
            "idx":        pair_idx,
            "label":      label,
            "status":     r.status_code,
            "similarity": j.get("similarity"),
            "distance":   j.get("distance"),
            "match":      j.get("match"),
            "q1":         j.get("face1", {}).get("quality_score"),
            "q2":         j.get("face2", {}).get("quality_score"),
            "error":      "",
        }
    except Exception as e:
        return {"idx": pair_idx, "label": label, "status": -1,
                "similarity": None, "distance": None, "match": None,
                "q1": None, "q2": None, "error": str(e)[:120]}


def run_comparisons(pairs):
    print(f"\nScoring {len(pairs)} pairs with {WORKERS} workers ...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(compare_pair, i, a, b, lbl): i
            for i, (a, b, lbl) in enumerate(pairs)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if done % 500 == 0 or done == len(pairs):
                ok = sum(1 for r in results if r["status"] == 200)
                print(f"  [{done:5d}/{len(pairs)}]  ok={ok}  err={done-ok}")

    results.sort(key=lambda x: x["idx"])
    return results


# ──────────────────────────────────────────────
# 4. Metrik hesaplama
# ──────────────────────────────────────────────

def compute_metrics(results):
    valid = [r for r in results if r["distance"] is not None]
    if not valid:
        print("No valid results!"); return None

    labels    = [r["label"] for r in valid]
    distances = [r["distance"] for r in valid]
    scores    = [1.0 - d for d in distances]   # yuksek = daha benzer

    n_genuine  = sum(labels)
    n_imposter = len(labels) - n_genuine

    # Esik taramasi
    thresholds = sorted(set(distances))
    sweep = []
    for t in thresholds:
        far = sum(1 for d, l in zip(distances, labels) if d < t  and l == 0) / max(n_imposter, 1)
        frr = sum(1 for d, l in zip(distances, labels) if d >= t and l == 1) / max(n_genuine,  1)
        sweep.append((t, far, frr))

    # EER
    best_eer, best_thr = 1.0, None
    for t, far, frr in sweep:
        if best_thr is None or abs(far - frr) < abs(best_eer * 2 - 1e-9):
            best_eer = (far + frr) / 2
            best_thr = t

    # AUC
    auc = compute_auc(scores, labels)

    # Uretim esigindeki performans (/verify icin 0.45, /compare icin 0.40)
    at_verify = next(((t, f, r) for t, f, r in sweep if abs(t - 0.45) < 0.01), None)
    at_compare = next(((t, f, r) for t, f, r in sweep if abs(t - 0.40) < 0.01), None)

    return {
        "n_valid":    len(valid),
        "n_genuine":  n_genuine,
        "n_imposter": n_imposter,
        "auc":        auc,
        "eer":        best_eer,
        "eer_thr":    best_thr,
        "sweep":      sweep,
        "at_verify":  at_verify,
        "at_compare": at_compare,
        "labels":     labels,
        "distances":  distances,
    }


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


# ──────────────────────────────────────────────
# 5. Cikti
# ──────────────────────────────────────────────

def write_csv(results):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx","label","status","similarity",
                                           "distance","match","q1","q2","error"])
        w.writeheader()
        w.writerows(results)
    print(f"  CSV -> {OUT_CSV}")


def write_summary(m, errors):
    lines = [
        "AgeDB-30 Age-Invariance Test Summary\n",
        "=" * 45 + "\n\n",
        f"Dataset : AgeDB-30 (30-year age gap pairs)\n",
        f"Pairs   : {m['n_valid']} valid  ({m['n_genuine']} genuine + {m['n_imposter']} imposter)\n",
        f"Errors  : {errors}\n\n",
        "=== Biometric Metrics ===\n",
        f"AUC  : {m['auc']:.4f}\n",
        f"EER  : {m['eer']:.4f}  at distance threshold {m['eer_thr']:.4f}\n\n",
        "=== LFW ile Karsilastirma ===\n",
        f"{'Metric':<10} {'AgeDB-30':>12} {'LFW':>12} {'Fark':>10}\n",
        "-" * 46 + "\n",
        f"{'AUC':<10} {m['auc']:>12.4f} {LFW_AUC:>12.4f} {m['auc']-LFW_AUC:>+10.4f}\n",
        f"{'EER':<10} {m['eer']:>12.4f} {LFW_EER:>12.4f} {m['eer']-LFW_EER:>+10.4f}\n\n",
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
    delta_auc = m["auc"] - LFW_AUC
    delta_eer = m["eer"] - LFW_EER
    if delta_auc >= -0.01 and delta_eer <= 0.02:
        lines.append("SONUC: Sistem yas farki karsisinda guclu — LFW ile benzer performans.\n")
        lines.append("30 yillik yas farki modeli belirgin sekilde etkilememektedir.\n")
    elif delta_auc >= -0.03 and delta_eer <= 0.05:
        lines.append("SONUC: Yas farki bazi kayiplara yol acmis ama kabul edilebilir seviyede.\n")
        lines.append("FAR/FRR degerleri gozden gecirilmeli, esik ayari dusunulebilir.\n")
    else:
        lines.append("SONUC: Yas farki performansi ciddi sekilde dusurmus.\n")
        lines.append("Model yas degisikligine karsi hassas — embedding yenileme politikasi gerekebilir.\n")
        lines.append("Oneri: Kullanicilari her 1-2 yilda bir yeniden kayit ettirin.\n")

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
        plt.plot(fars, tars, color="#E91E63", linewidth=2,
                 label=f"AgeDB-30 (AUC={m['auc']:.3f}, EER={m['eer']:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
        plt.axvline(x=m["eer"], color="#E91E63", linestyle=":", linewidth=1,
                    label=f"EER={m['eer']:.3f}")

        # LFW referans cizgisi
        plt.axhline(y=1 - LFW_EER, color="#2196F3", linestyle="--", linewidth=1,
                    label=f"LFW TAR@EER={1-LFW_EER:.3f} (referans)")

        plt.xlabel("FAR (False Accept Rate)", fontsize=11)
        plt.ylabel("TAR (True Accept Rate)", fontsize=11)
        plt.title("FIVUCSAS — AgeDB-30 ROC (30-Year Age Gap)", fontsize=12)
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
    print(" FIVUCSAS  .  AgeDB-30 Age-Invariance Test")
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
