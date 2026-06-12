"""
Cross-Tenant Isolation Test for FIVUCSAS biometric-processor.

Goal
----
Prove that data enrolled in tenant A is NEVER reachable from tenant B,
even when both tenants share the same user_id.

Test setup
----------
We use 3 distinct LFW identities (different real people):

    P1 = Aaron_Eckhart
    P2 = Adam_Sandler
    P3 = Adrien_Brody

We enroll them in two separate tenants with overlapping user_ids:

    tenant "ct-test-A"        tenant "ct-test-B"
    ------------------        ------------------
    user_id=user001 -> P1     user_id=user001 -> P3   <-- same id, DIFFERENT person
    user_id=user002 -> P2     (no user001 not present here either way)

Cases (each one gets a clear PASS/FAIL):

    1. Control positive: P1 photo  vs  ct-test-A / user001  -> verified=true   (sanity)
    2. ISOLATION:        P1 photo  vs  ct-test-B / user001  -> verified=false  (B/user001 is P3)
    3. CROSS-PERSON:     P3 photo  vs  ct-test-A / user001  -> verified=false  (A/user001 is P1)
    4. UNKNOWN USER:     P1 photo  vs  ct-test-A / ghost    -> 404
    5. UNKNOWN TENANT:   P1 photo  vs  ct-ghost  / user001  -> 404
    6. CROSS-USER SAME-TENANT: P1 photo vs ct-test-A / user002 -> verified=false

Each case writes a row to results.csv with the verdict.
A summary.txt lists overall PASS/FAIL counts.
"""
import csv
import json
import time
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE = "http://localhost:8001"
DATASET  = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
TENANT_A = "ct-test-A"
TENANT_B = "ct-test-B"

# Pick three identities with at least 2 photos each
P1 = ("Aaron_Eckhart",  DATASET / "Aaron_Eckhart"  / "Aaron_Eckhart_0001.jpg")
P2 = ("Adam_Sandler",   DATASET / "Adam_Sandler"   / "Adam_Sandler_0001.jpg")
P3 = ("Adrien_Brody",   DATASET / "Adrien_Brody"   / "Adrien_Brody_0001.jpg")
P1_TEST = DATASET / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"
P3_TEST = DATASET / "Adrien_Brody"  / "Adrien_Brody_0002.jpg"

OUT_RESULTS = Path(__file__).parent / "results.csv"
OUT_SUMMARY = Path(__file__).parent / "summary.txt"
# ----------------------------


def enroll(user_id: str, tenant_id: str, image: Path):
    """POST /api/v1/enroll. Returns (ok, status, body_text)."""
    with open(image, "rb") as f:
        r = requests.post(
            f"{API_BASE}/api/v1/enroll",
            files={"file": (image.name, f, "image/jpeg")},
            data={"user_id": user_id, "tenant_id": tenant_id},
            timeout=60,
        )
    return r.ok, r.status_code, r.text[:200]


def verify(user_id: str, tenant_id: str, image: Path):
    """POST /api/v1/verify. Returns (status_code, json_or_text)."""
    with open(image, "rb") as f:
        r = requests.post(
            f"{API_BASE}/api/v1/verify",
            files={"file": (image.name, f, "image/jpeg")},
            data={"user_id": user_id, "tenant_id": tenant_id},
            timeout=60,
        )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text[:200]}


def cleanup(user_id: str, tenant_id: str):
    """Best-effort delete to keep the test idempotent."""
    try:
        requests.delete(
            f"{API_BASE}/api/v1/enroll",
            params={"user_id": user_id, "tenant_id": tenant_id},
            timeout=15,
        )
    except Exception:
        pass


# ---------- run ----------
def main():
    print("=" * 70)
    print(" FIVUCSAS  ·  Cross-Tenant Isolation Test")
    print("=" * 70)

    # Step 0: cleanup any prior run
    print("\n[0] Pre-cleanup ...")
    for tid in (TENANT_A, TENANT_B):
        for uid in ("user001", "user002"):
            cleanup(uid, tid)

    # Step 1: enroll baselines
    print("\n[1] Enrolling baselines ...")
    enrollments = [
        (TENANT_A, "user001", P1[1], "P1 (Aaron_Eckhart)"),
        (TENANT_A, "user002", P2[1], "P2 (Adam_Sandler)"),
        (TENANT_B, "user001", P3[1], "P3 (Adrien_Brody)"),
    ]
    for tid, uid, img, who in enrollments:
        ok, status, body = enroll(uid, tid, img)
        print(f"    {tid:<12} / {uid:<10} <- {who}  ->  {status} {'OK' if ok else 'FAIL'}")
        if not ok:
            print("    ABORT: baseline enrollment failed:", body)
            return

    # Step 2: run test cases
    print("\n[2] Running test cases ...\n")
    cases = [
        # (case_id, description, query_image, target_user, target_tenant, expectation)
        ("01_control_positive",
         "P1 photo vs A/user001",
         P1_TEST, "user001", TENANT_A,
         "should match (verified=true)"),
        ("02_isolation_same_id_diff_person",
         "P1 photo vs B/user001 (B/user001 is actually P3)",
         P1_TEST, "user001", TENANT_B,
         "should NOT match (cross-tenant data leak risk)"),
        ("03_cross_person_same_id",
         "P3 photo vs A/user001 (A/user001 is P1)",
         P3_TEST, "user001", TENANT_A,
         "should NOT match"),
        ("04_unknown_user_in_tenant",
         "P1 photo vs A/ghost-user-zzz",
         P1_TEST, "ghost-user-zzz", TENANT_A,
         "should 404"),
        ("05_unknown_tenant",
         "P1 photo vs ct-ghost-tenant/user001",
         P1_TEST, "user001", "ct-ghost-tenant",
         "should 404"),
        ("06_cross_user_same_tenant",
         "P1 photo vs A/user002 (different user, same tenant)",
         P1_TEST, "user002", TENANT_A,
         "should NOT match"),
    ]

    rows = []
    for case_id, desc, img, uid, tid, expect in cases:
        status, body = verify(uid, tid, img)
        verified = body.get("verified") if isinstance(body, dict) else None
        distance = body.get("distance") if isinstance(body, dict) else None

        # Determine PASS/FAIL based on expectation
        verdict = decide(case_id, status, verified)
        symbol = "PASS" if verdict == "PASS" else "FAIL"

        print(f"  [{case_id}] {desc}")
        print(f"      expect:   {expect}")
        print(f"      got:      status={status}  verified={verified}  distance={distance}")
        print(f"      verdict:  {symbol}\n")

        rows.append({
            "case_id":   case_id,
            "desc":      desc,
            "tenant":    tid,
            "user_id":   uid,
            "image":     img.name,
            "expected":  expect,
            "status":    status,
            "verified":  verified,
            "distance":  distance,
            "verdict":   verdict,
            "raw":       json.dumps(body)[:300] if isinstance(body, dict) else str(body)[:300],
        })

    # Step 3: write CSV + summary
    with open(OUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fail = len(rows) - n_pass

    summary = (
        f"Cross-Tenant Isolation Test Summary\n"
        f"{'='*40}\n"
        f"Total cases : {len(rows)}\n"
        f"PASS        : {n_pass}\n"
        f"FAIL        : {n_fail}\n\n"
    )
    for r in rows:
        summary += f"  [{r['verdict']}] {r['case_id']}  status={r['status']}  verified={r['verified']}\n"

    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print("=" * 70)
    print(summary)
    print(f"Detailed results: {OUT_RESULTS}")
    print(f"Summary:          {OUT_SUMMARY}")

    # Cleanup so re-running is idempotent
    print("\n[cleanup] removing test enrollments ...")
    for tid in (TENANT_A, TENANT_B):
        for uid in ("user001", "user002"):
            cleanup(uid, tid)

    print("\nDone.")


def decide(case_id: str, status: int, verified):
    """Return 'PASS' or 'FAIL <reason>' based on case expectation."""
    if case_id == "01_control_positive":
        return "PASS" if (status == 200 and verified is True) else f"FAIL expected verified=true, got status={status} verified={verified}"
    if case_id == "02_isolation_same_id_diff_person":
        return "PASS" if (status == 200 and verified is False) else f"FAIL expected verified=false, got status={status} verified={verified}"
    if case_id == "03_cross_person_same_id":
        return "PASS" if (status == 200 and verified is False) else f"FAIL expected verified=false, got status={status} verified={verified}"
    if case_id == "04_unknown_user_in_tenant":
        return "PASS" if status in (404, 400) else f"FAIL expected 404, got {status}"
    if case_id == "05_unknown_tenant":
        return "PASS" if status in (404, 400) else f"FAIL expected 404, got {status}"
    if case_id == "06_cross_user_same_tenant":
        return "PASS" if (status == 200 and verified is False) else f"FAIL expected verified=false, got status={status} verified={verified}"
    return "FAIL unknown case"


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
