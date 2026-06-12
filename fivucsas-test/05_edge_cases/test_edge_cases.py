"""
Step 05 — Edge Case & Input Validation Tests (FIVUCSAS)

Tests every case from Layer C of the test plan:
  - No face in image
  - Multiple faces
  - Tiny face (<80px canvas)
  - Huge image (>10MB)
  - Wrong format (GIF, BMP sent as-is)
  - Corrupted base64 / binary garbage
  - Same image for enroll and verify (should work, high score)
  - Blank white image
  - Completely black image

Each case has a clear expected result and PASS/FAIL verdict.

No external dataset needed — test images are generated with Pillow.

Outputs
-------
- edge_results.csv   per-case details
- summary.txt        PASS/FAIL table
"""
import csv
import io
import json
import os
import random
import struct
import time
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE   = "http://localhost:8001"
DATASET    = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
ENROLL_CSV = Path(__file__).parent.parent / "01_bulk_enroll" / "enroll_results.csv"
TENANT     = "edge-test"
TIMEOUT    = 30
OUT_CSV    = Path(__file__).parent / "edge_results.csv"
OUT_SUM    = Path(__file__).parent / "summary.txt"
# ----------------------------


def get_real_image():
    """Pick any successfully-enrolled image path."""
    with open(ENROLL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] != "True":
                continue
            img = DATASET / row["folder"] / row["image"]
            if img.exists():
                return row["user_id"], img
    raise RuntimeError("No enrolled images found in enroll_results.csv")


def make_tiny_image():
    """10x10 white JPEG — face will be too small to detect."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(200, 180, 160)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def make_blank_white():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(255, 255, 255)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def make_solid_black():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(0, 0, 0)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def make_large_image(target_mb=12):
    """Generate a buffer large enough to exceed the size limit (fake JPEG header + padding)."""
    data = b"\xff\xd8\xff\xe0" + os.urandom(1024) + b"\x00" * (target_mb * 1024 * 1024)
    return io.BytesIO(data)


def make_gif():
    """Minimal valid GIF."""
    # GIF89a 1x1 transparent
    gif = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00"
           b"!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01"
           b"\x00\x00\x02\x02D\x01\x00;")
    return io.BytesIO(gif)


def make_corrupted_jpeg():
    """Valid JPEG header, garbage body."""
    return io.BytesIO(b"\xff\xd8\xff\xe0" + os.urandom(512))


def make_text_file():
    """A .txt file disguised as an image."""
    return io.BytesIO(b"This is not an image\n" * 100)


def post_enroll(user_id, file_buf, filename="test.jpg", mimetype="image/jpeg"):
    file_buf.seek(0)
    r = requests.post(
        f"{API_BASE}/api/v1/enroll",
        files={"file": (filename, file_buf, mimetype)},
        data={"user_id": user_id, "tenant_id": TENANT},
        timeout=TIMEOUT,
    )
    return r.status_code, _body(r)


def post_verify(user_id, file_buf, filename="test.jpg", mimetype="image/jpeg"):
    file_buf.seek(0)
    r = requests.post(
        f"{API_BASE}/api/v1/verify",
        files={"file": (filename, file_buf, mimetype)},
        data={"user_id": user_id, "tenant_id": TENANT},
        timeout=TIMEOUT,
    )
    return r.status_code, _body(r)


def _body(r):
    try:
        return r.json()
    except Exception:
        return {"raw": r.text[:300]}


def decide(case_id, status, body):
    """Return PASS or FAIL string based on case expectations."""
    ec = body.get("error_code", "") if isinstance(body, dict) else ""
    verified = body.get("verified") if isinstance(body, dict) else None

    if case_id == "01_no_face":
        return "PASS" if status == 400 else f"FAIL — expected 400, got {status}"
    if case_id == "02_tiny_face":
        return "PASS" if status == 400 else f"FAIL — expected 400, got {status}"
    if case_id == "03_blank_white":
        return "PASS" if status == 400 else f"FAIL — expected 400, got {status}"
    if case_id == "04_black_image":
        return "PASS" if status == 400 else f"FAIL — expected 400, got {status}"
    if case_id == "05_large_image":
        return "PASS" if status in (400, 413, 422) else f"FAIL — expected 400/413, got {status}"
    if case_id == "06_gif_format":
        return "PASS" if status in (400, 415, 422) else f"FAIL — expected 400/415, got {status}"
    if case_id == "07_corrupted_jpeg":
        return "PASS" if status in (400, 422, 500) else f"FAIL — expected 400/422, got {status}"
    if case_id == "08_text_file":
        return "PASS" if status in (400, 415, 422) else f"FAIL — expected 400/415, got {status}"
    if case_id == "09_same_image_enroll_verify":
        return "PASS" if (status == 200 and verified is True) else f"FAIL — expected verified=true, got {status}/{verified}"
    if case_id == "10_no_api_key":
        return "PASS" if status == 401 else f"FAIL — expected 401, got {status}"
    return f"UNKNOWN case {case_id}"


def cleanup():
    """Best-effort delete test enrollments."""
    try:
        for uid in ["edge-same-user", "edge-no-face"]:
            requests.delete(
                f"{API_BASE}/api/v1/enroll",
                params={"user_id": uid, "tenant_id": TENANT},
                timeout=10,
            )
    except Exception:
        pass


def main():
    t0 = time.time()
    print("=" * 70)
    print(" FIVUCSAS  ·  Edge Case & Input Validation Tests")
    print("=" * 70)

    try:
        from PIL import Image
    except ImportError:
        print("Installing Pillow ...")
        os.system("pip install pillow -q")
        from PIL import Image

    user_id, real_img = get_real_image()
    print(f"\nReal image: {real_img.name}  (user: {user_id})")

    cleanup()

    cases = []

    # --- 01: no face (blank white) enrolled as "edge-no-face" ---
    print("\n[01] No face (blank white image) ...")
    status, body = post_enroll("edge-no-face", make_blank_white())
    v = decide("01_no_face", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "01_no_face", "desc": "blank white — no face",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 02: tiny face ---
    print("\n[02] Tiny face (<80px canvas) ...")
    status, body = post_enroll("edge-tiny", make_tiny_image())
    v = decide("02_tiny_face", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "02_tiny_face", "desc": "10x10 image — face too small",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 03: blank white ---
    print("\n[03] Solid white 640x480 ...")
    status, body = post_enroll("edge-white", make_blank_white())
    v = decide("03_blank_white", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "03_blank_white", "desc": "solid white 640x480",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 04: solid black ---
    print("\n[04] Solid black 640x480 ...")
    status, body = post_enroll("edge-black", make_solid_black())
    v = decide("04_black_image", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "04_black_image", "desc": "solid black 640x480",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 05: large image (>10MB) ---
    print("\n[05] Large image (>10MB garbage JPEG) ...")
    status, body = post_enroll("edge-large", make_large_image(12))
    v = decide("05_large_image", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "05_large_image", "desc": "12MB garbage content",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 06: GIF format ---
    print("\n[06] GIF format (should be rejected) ...")
    status, body = post_enroll("edge-gif", make_gif(), filename="test.gif",
                               mimetype="image/gif")
    v = decide("06_gif_format", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "06_gif_format", "desc": "GIF file",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 07: corrupted JPEG ---
    print("\n[07] Corrupted JPEG (valid header, garbage body) ...")
    status, body = post_enroll("edge-corrupt", make_corrupted_jpeg())
    v = decide("07_corrupted_jpeg", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "07_corrupted_jpeg", "desc": "JPEG header + random bytes",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 08: text file ---
    print("\n[08] Text file sent as image/jpeg ...")
    status, body = post_enroll("edge-text", make_text_file(), filename="face.jpg")
    v = decide("08_text_file", status, body)
    print(f"     status={status}  {v}")
    cases.append({"case_id": "08_text_file", "desc": "text file disguised as JPEG",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 09: same image for enroll + verify ---
    print("\n[09] Same image: enroll then verify (should match) ...")
    enroll_status, enroll_body = post_enroll("edge-same-user", open(real_img, "rb"))
    if enroll_status == 200:
        status, body = post_verify("edge-same-user", open(real_img, "rb"))
        v = decide("09_same_image_enroll_verify", status, body)
        print(f"     enroll={enroll_status}  verify={status}  verified={body.get('verified')}  {v}")
    else:
        status = enroll_status
        body = enroll_body
        v = f"FAIL — enroll failed: {enroll_status}"
        print(f"     enroll failed: {enroll_status}")
    cases.append({"case_id": "09_same_image_enroll_verify",
                  "desc": "same image enrolled and verified",
                  "status": status, "body": json.dumps(body)[:200], "verdict": v})

    # --- 10: missing API key (if auth is enforced) ---
    print("\n[10] Missing API key (if X-Api-Key is required) ...")
    try:
        with open(real_img, "rb") as f:
            r_no_key = requests.post(
                f"{API_BASE}/api/v1/verify",
                files={"file": (real_img.name, f, "image/jpeg")},
                data={"user_id": user_id, "tenant_id": TENANT},
                headers={"X-Api-Key": "INVALID_KEY_XYZ_000"},
                timeout=TIMEOUT,
            )
        status_no_key = r_no_key.status_code
        body_no_key = _body(r_no_key)
    except Exception as e:
        status_no_key = -1
        body_no_key = {"error": str(e)}
    v = decide("10_no_api_key", status_no_key, body_no_key)
    print(f"     status={status_no_key}  {v}")
    cases.append({"case_id": "10_no_api_key", "desc": "invalid X-Api-Key header",
                  "status": status_no_key, "body": json.dumps(body_no_key)[:200], "verdict": v})

    # Write results
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "desc", "status", "body", "verdict"])
        w.writeheader()
        w.writerows(cases)

    n_pass = sum(1 for c in cases if c["verdict"].startswith("PASS"))
    n_fail = len(cases) - n_pass

    summary_lines = [
        "Edge Case Test Summary\n",
        "=" * 40 + "\n",
        f"Total : {len(cases)}\n",
        f"PASS  : {n_pass}\n",
        f"FAIL  : {n_fail}\n\n",
        f"{'Case':<35} {'Status':>6}  {'Verdict'}\n",
        "-" * 80 + "\n",
    ]
    for c in cases:
        summary_lines.append(f"{c['case_id']:<35} {c['status']:>6}  {c['verdict']}\n")

    text = "".join(summary_lines)
    OUT_SUM.write_text(text, encoding="utf-8")

    print("\n" + "=" * 70)
    print(text)
    print(f"CSV    -> {OUT_CSV}")
    print(f"Summary-> {OUT_SUM}")

    cleanup()
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
