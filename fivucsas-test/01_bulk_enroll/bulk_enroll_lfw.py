"""
Bulk enroll LFW dataset to FIVUCSAS biometric-processor (local Docker).

Picks the first MAX_PEOPLE folders with >= MIN_IMAGES jpgs, enrolls every image
under user_id = folder name, tenant_id = TENANT.  Saves a CSV with results so
the FAR/FRR script can pick up where this left off.
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE   = "http://localhost:8001"
DATASET    = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
TENANT     = "lfw-test"
MIN_IMAGES = 5
MAX_PEOPLE = 100        # 100 people * ~5-15 images = 500-1500 enrollments
WORKERS    = 4
TIMEOUT    = 60
OUT_CSV    = Path(__file__).parent / "enroll_results.csv"
# ----------------------------


def select_people(root: Path, min_images: int, max_people: int):
    """Return [(user_id, [img_path, ...])] for the first max_people folders
    with at least min_images jpgs."""
    chosen = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(d.glob("*.jpg"))
        if len(imgs) >= min_images:
            chosen.append((d.name, imgs))
        if len(chosen) >= max_people:
            break
    return chosen


def enroll_one(user_id: str, img_path: Path):
    """POST /api/v1/enroll for one image. Return result dict."""
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            data  = {"user_id": user_id, "tenant_id": TENANT}
            r = requests.post(
                f"{API_BASE}/api/v1/enroll",
                files=files, data=data, timeout=TIMEOUT,
            )
        body_short = r.text[:200].replace("\n", " ")
        return {
            "user_id": user_id,
            "image":   str(img_path.name),
            "folder":  img_path.parent.name,
            "status":  r.status_code,
            "ok":      r.ok,
            "body":    body_short,
        }
    except Exception as e:
        return {"user_id": user_id, "image": img_path.name,
                "folder": img_path.parent.name,
                "status": -1, "ok": False, "body": f"EXC: {e}"}


def main():
    t0 = time.time()
    people = select_people(DATASET, MIN_IMAGES, MAX_PEOPLE)
    tasks = [(uid, img) for uid, imgs in people for img in imgs]
    print(f"Selected {len(people)} people, {len(tasks)} total images")
    print(f"Tenant: {TENANT} | Workers: {WORKERS}")
    print(f"Output: {OUT_CSV}\n")

    results = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(enroll_one, uid, img): (uid, img) for uid, img in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if res["ok"]:
                ok += 1
            else:
                fail += 1
            if i % 25 == 0 or not res["ok"]:
                mark = "OK" if res["ok"] else f"FAIL({res['status']})"
                print(f"[{i:4d}/{len(tasks)}] {mark:>10} {res['user_id']:<28} {res['image']}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "image", "folder", "status", "ok", "body"])
        w.writeheader()
        w.writerows(results)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s — {ok} ok, {fail} fail / {len(results)} total")
    print(f"Results: {OUT_CSV}")


if __name__ == "__main__":
    main()
