"""
Step 06 — Performance Tests (FIVUCSAS)

Measures
--------
P1. Single-request latency: p50, p95, p99 for /verify over 30 requests.
    Target from CLAUDE.md: p95 ≤ 1.5s.

P2. Concurrent load: 20 parallel /verify requests — checks for OOM / timeouts.
    The biometric-api is at 94% memory — watch for server errors under load.

P3. Rate limiting: hammer /verify until 429 appears (or confirm it never does).

P4. Search latency: POST /api/v1/search — how long to find top-K across
    all enrolled embeddings.

Outputs
-------
- latency_results.csv   raw per-request timings
- summary.txt           p50/p95/p99, pass/fail vs targets
"""
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE      = "http://localhost:8001"
DATASET       = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
ENROLL_CSV    = Path(__file__).parent.parent / "01_bulk_enroll" / "enroll_results.csv"
TENANT        = "lfw-test"
N_LATENCY     = 30         # sequential verify calls for latency stats
N_CONCURRENT  = 20         # parallel workers for load test
N_RATE_LIMIT  = 40         # calls in burst to detect 429
TARGET_P95_S  = 1.5        # CLAUDE.md target
TIMEOUT       = 60
OUT_CSV       = Path(__file__).parent / "latency_results.csv"
OUT_SUM       = Path(__file__).parent / "summary.txt"
# ----------------------------


def get_enrolled_users(n: int):
    """Return list of (user_id, image_path) from successful enrollments."""
    users = []
    seen = set()
    with open(ENROLL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] != "True":
                continue
            uid = row["user_id"]
            if uid in seen:
                continue
            img = DATASET / row["folder"] / row["image"]
            if img.exists():
                users.append((uid, img))
                seen.add(uid)
            if len(users) >= n:
                break
    return users


def do_verify(user_id: str, img_path: Path, label: str = ""):
    """Run one /verify call, return timing + result dict."""
    t0 = time.perf_counter()
    try:
        with open(img_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/v1/verify",
                files={"file": (img_path.name, f, "image/jpeg")},
                data={"user_id": user_id, "tenant_id": TENANT},
                timeout=TIMEOUT,
            )
        elapsed = time.perf_counter() - t0
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:200]}
        return {
            "label":    label,
            "user_id":  user_id,
            "status":   r.status_code,
            "elapsed":  round(elapsed, 4),
            "verified": body.get("verified"),
            "distance": body.get("distance"),
            "error":    "" if r.ok else body.get("message", r.text[:100]),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"label": label, "user_id": user_id, "status": -1,
                "elapsed": round(elapsed, 4), "verified": None,
                "distance": None, "error": str(e)[:200]}


def p1_sequential_latency(users):
    print(f"\n[P1] Sequential latency — {N_LATENCY} calls ...")
    import itertools
    cycle = itertools.cycle(users)
    results = []
    for i in range(N_LATENCY):
        uid, img = next(cycle)
        res = do_verify(uid, img, label="sequential")
        results.append(res)
        print(f"  [{i+1:3d}/{N_LATENCY}]  {res['elapsed']:.3f}s  status={res['status']}")
    return results


def p2_concurrent_load(users):
    print(f"\n[P2] Concurrent load — {N_CONCURRENT} parallel requests ...")
    import itertools
    cycle = itertools.cycle(users)
    tasks = [(next(cycle), next(cycle)[1]) for _ in range(N_CONCURRENT)]

    results = []
    t_wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        futures = [ex.submit(do_verify, uid, img, "concurrent") for (uid, _), img in
                   [((u, i), i) for u, i in [next(iter(cycle), ("", Path())) for _ in range(N_CONCURRENT)]]]
        # Simpler approach: just submit N_CONCURRENT tasks
        futures2 = []
        cycle2 = itertools.cycle(users)
        for _ in range(N_CONCURRENT):
            uid, img = next(cycle2)
            futures2.append(ex.submit(do_verify, uid, img, "concurrent"))
        for fut in as_completed(futures2):
            res = fut.result()
            results.append(res)
            print(f"  {res['elapsed']:.3f}s  status={res['status']}  {res.get('error','')[:60]}")
    t_wall = time.perf_counter() - t_wall_start
    print(f"  Wall time for {N_CONCURRENT} concurrent: {t_wall:.2f}s")
    n_err = sum(1 for r in results if r["status"] not in (200, 400))
    print(f"  Errors / non-200: {n_err}")
    return results, t_wall


def p3_rate_limit(users):
    print(f"\n[P3] Rate limit burst — {N_RATE_LIMIT} rapid calls ...")
    uid, img = users[0]
    results = []
    for i in range(N_RATE_LIMIT):
        res = do_verify(uid, img, label="rate-burst")
        results.append(res)
        if res["status"] == 429:
            print(f"  [{i+1:3d}]  429 RATE LIMITED after {i+1} requests")
            break
        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}]  {res['status']}  {res['elapsed']:.3f}s")
    rate_limited = any(r["status"] == 429 for r in results)
    at_request = next((i+1 for i, r in enumerate(results) if r["status"] == 429), None)
    print(f"  Rate limit triggered: {rate_limited}"
          + (f" at request #{at_request}" if at_request else f" (not triggered in {N_RATE_LIMIT} calls)"))
    return results, rate_limited, at_request


def p4_search_latency(users):
    print(f"\n[P4] Search latency — /api/v1/search ...")
    uid, img = users[0]
    timings = []
    for i in range(5):
        t0 = time.perf_counter()
        try:
            with open(img, "rb") as f:
                r = requests.post(
                    f"{API_BASE}/api/v1/search",
                    files={"file": (img.name, f, "image/jpeg")},
                    data={"tenant_id": TENANT, "top_k": 5},
                    timeout=TIMEOUT,
                )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            try:
                body = r.json()
                n_results = len(body.get("results", body.get("matches", [])))
            except Exception:
                n_results = "?"
            print(f"  [{i+1}]  {elapsed:.3f}s  status={r.status_code}  results={n_results}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}]  {elapsed:.3f}s  ERROR: {e}")
    if timings:
        print(f"  Search p50={statistics.median(timings):.3f}s  "
              f"max={max(timings):.3f}s")
    return timings


def compute_stats(results):
    timings = [r["elapsed"] for r in results if r["status"] != -1]
    if not timings:
        return None
    return {
        "n":   len(timings),
        "p50": round(statistics.median(timings), 3),
        "p95": round(sorted(timings)[int(0.95 * len(timings))], 3),
        "p99": round(sorted(timings)[int(0.99 * len(timings))], 3),
        "min": round(min(timings), 3),
        "max": round(max(timings), 3),
    }


def write_summary(seq_results, conc_results, conc_wall, rl_results, rl_hit, rl_at,
                  search_timings):
    all_rows = seq_results + conc_results + rl_results
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label","user_id","status","elapsed",
                                           "verified","distance","error"])
        w.writeheader()
        w.writerows(all_rows)

    seq_stats  = compute_stats(seq_results)
    conc_stats = compute_stats(conc_results)

    lines = ["Performance Test Summary\n", "=" * 40 + "\n\n"]

    lines.append("=== P1. Sequential Latency ===\n")
    if seq_stats:
        p95_pass = seq_stats["p95"] <= TARGET_P95_S
        lines.append(f"  n={seq_stats['n']}  min={seq_stats['min']}s  "
                     f"p50={seq_stats['p50']}s  p95={seq_stats['p95']}s  "
                     f"p99={seq_stats['p99']}s  max={seq_stats['max']}s\n")
        lines.append(f"  Target p95 <= {TARGET_P95_S}s:  {'PASS' if p95_pass else 'FAIL'}\n")
    else:
        lines.append("  No valid timings\n")
    lines.append("\n")

    lines.append(f"=== P2. Concurrent Load ({N_CONCURRENT} workers) ===\n")
    if conc_stats:
        lines.append(f"  Wall time: {conc_wall:.2f}s\n")
        lines.append(f"  Per-request: p50={conc_stats['p50']}s  p95={conc_stats['p95']}s\n")
        n_err_conc = sum(1 for r in conc_results if r["status"] not in (200, 400))
        lines.append(f"  Errors: {n_err_conc}/{N_CONCURRENT}"
                     + ("  (possible OOM / thread contention)" if n_err_conc else "") + "\n")
    lines.append("\n")

    lines.append(f"=== P3. Rate Limiting ===\n")
    if rl_hit:
        lines.append(f"  PASS — 429 triggered at request #{rl_at}\n")
    else:
        lines.append(f"  WARN — rate limit NOT triggered in {N_RATE_LIMIT} rapid calls\n")
        lines.append(f"  Check RATE_LIMIT_ENABLED env var (README: set to False for testing)\n")
    lines.append("\n")

    lines.append(f"=== P4. Search Latency ===\n")
    if search_timings:
        lines.append(f"  median={statistics.median(search_timings):.3f}s  "
                     f"max={max(search_timings):.3f}s\n")
        lines.append(f"  Note: pgvector HNSW index gap flagged in TODO — "
                     f"latency may grow with scale\n")
    else:
        lines.append("  No timings collected\n")

    text = "".join(lines)
    OUT_SUM.write_text(text, encoding="utf-8")
    print("\n" + "=" * 70)
    print(text)
    print(f"CSV    -> {OUT_CSV}")
    print(f"Summary-> {OUT_SUM}")


def main():
    t0 = time.time()
    print("=" * 70)
    print(" FIVUCSAS  ·  Performance Tests")
    print("=" * 70)

    users = get_enrolled_users(30)
    print(f"\n{len(users)} enrolled users available for testing")
    if not users:
        print("No enrolled users found — run step 01 first"); return

    seq_results = p1_sequential_latency(users)
    conc_results, conc_wall = p2_concurrent_load(users)
    rl_results, rl_hit, rl_at = p3_rate_limit(users)
    search_timings = p4_search_latency(users)

    write_summary(seq_results, conc_results, conc_wall, rl_results, rl_hit, rl_at,
                  search_timings)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
