"""
Step 04c — Puzzle Liveness Verification Tests (FIVUCSAS)

Endpoints under test
--------------------
  POST /api/v1/liveness/generate-puzzle   -> get puzzle_id + steps
  POST /api/v1/liveness/verify            -> submit step evidence

What this tests
---------------
T1  FAKE_BYPASS       Generate puzzle, immediately submit fake evidence with
                      high confidence (no real actions performed).
                      EXPECTED: liveness_confirmed=False
                      IF TRUE:  server trusts client-supplied confidence blindly — critical bug.

T2  CORRECT_FAKE      Same bypass but with correct step action names from the puzzle.
                      Isolates whether action-name matching or confidence is the weak point.

T3  REPLAY_ATTACK     Verify the same puzzle_id a second time.
                      EXPECTED: reason_code=PUZZLE_ALREADY_COMPLETED

T4  EXPIRED_PUZZLE    Submit to a puzzle_id that doesn't exist / is expired.
                      EXPECTED: reason_code=PUZZLE_NOT_FOUND

T5  WRONG_STEPS       Submit correct confidence but wrong action sequence.
                      EXPECTED: failure with STEP_ACTION_MISMATCH or similar

T6  LOW_CONFIDENCE    Submit correct steps but confidence=0.3 (below 0.6 threshold).
                      EXPECTED: failure with STEP_x_LOW_CONFIDENCE

T7  SHORT_DURATION    Submit steps with end-start < 0.5s (below MIN_STEP_DURATION).
                      EXPECTED: failure with STEP_x_TOO_SHORT

T8  TIMESTAMP_REPLAY  Submit timestamps from the past (before puzzle creation).
                      EXPECTED: failure with TIMESTAMP_BEFORE_PUZZLE

T9  SPOT_FRAME_SPOOF  Include base64-encoded static photo as spot_frames.
                      IF spot-check is active, this should fail.
                      IF broken (MiniFASNet path issue), check passes by default.

T10 DIFFICULTY_SWEEP  Generate puzzles at easy/standard/hard — verify step counts match spec.

Outputs
-------
  puzzle_results.csv    per-test result
  puzzle_summary.txt    PASS/FAIL table + critical findings
"""

import base64
import csv
import json
import time
from pathlib import Path

import requests

# ---------- CONFIG ----------
API_BASE   = "http://localhost:8001"
DATASET    = Path(r"C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled")
TIMEOUT    = 30
OUT_CSV    = Path(__file__).parent / "puzzle_results.csv"
OUT_SUM    = Path(__file__).parent / "puzzle_summary.txt"
# ----------------------------

GENERATE_URL = f"{API_BASE}/api/v1/liveness/generate-puzzle"
VERIFY_URL   = f"{API_BASE}/api/v1/liveness/verify"


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def generate_puzzle(difficulty="standard", min_steps=3, max_steps=4):
    r = requests.post(GENERATE_URL,
                      json={"difficulty": difficulty,
                            "min_steps": min_steps,
                            "max_steps": max_steps},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def verify_puzzle(puzzle_id, results, spot_frames=None, client_meta=None):
    payload = {"puzzle_id": puzzle_id, "results": results}
    if spot_frames:
        payload["spot_frames"] = spot_frames
    if client_meta:
        payload["client_meta"] = client_meta
    r = requests.post(VERIFY_URL, json=payload, timeout=TIMEOUT)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text[:300]}


def fake_evidence(steps, confidence=0.95, start_offset=0.0, duration=1.5,
                  override_actions=None):
    """
    Build fake StepEvidence list.
    steps     : list of step dicts from generate-puzzle response
    confidence: confidence to claim for each step
    duration  : claimed duration per step in seconds
    """
    now = time.time() + start_offset
    results = []
    for i, step in enumerate(steps):
        action = (override_actions[i] if override_actions and i < len(override_actions)
                  else step["action"])
        step_start = now + i * (duration + 0.2)
        step_end   = step_start + duration
        results.append({
            "action":          action,
            "start_timestamp": step_start,
            "end_timestamp":   step_end,
            "confidence":      confidence,
            "metrics":         {"fake": True},
        })
    return results


def get_sample_frame_b64():
    """Return a base64-encoded real photo to use as a fake spot_frame."""
    img_path = DATASET / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"
    if img_path.exists():
        return base64.b64encode(img_path.read_bytes()).decode()
    return None


def decide(test_id, status, body):
    """Return PASS or FAIL with a reason string."""
    lc   = body.get("liveness_confirmed")
    succ = body.get("success")
    rc   = body.get("reason_codes", [])

    if test_id == "T1_FAKE_BYPASS":
        # We expect the server to REJECT fake confidence — liveness_confirmed must be False
        if lc is True:
            return "CRITICAL — server accepted fake evidence (client confidence trusted blindly)"
        return "PASS — fake bypass correctly rejected"

    if test_id == "T2_CORRECT_FAKE":
        if lc is True:
            return "CRITICAL — correct-action fake bypass accepted (no server-side frame check)"
        return "PASS — correct-action fake bypass rejected"

    if test_id == "T3_REPLAY_ATTACK":
        if "PUZZLE_ALREADY_COMPLETED" in rc:
            return "PASS — replay correctly blocked"
        return f"FAIL — replay not blocked; reason_codes={rc}"

    if test_id == "T4_EXPIRED_PUZZLE":
        if status in (404, 400) or "PUZZLE_NOT_FOUND" in rc or "PUZZLE_EXPIRED" in rc:
            return "PASS — nonexistent puzzle rejected"
        return f"FAIL — expected 404/PUZZLE_NOT_FOUND, got status={status} rc={rc}"

    if test_id == "T5_WRONG_STEPS":
        if lc is False or succ is False:
            return "PASS — wrong step sequence rejected"
        return "FAIL — wrong steps accepted"

    if test_id == "T6_LOW_CONFIDENCE":
        if lc is False or any("LOW_CONFIDENCE" in c for c in rc):
            return "PASS — low confidence rejected"
        return f"FAIL — low confidence accepted; rc={rc}"

    if test_id == "T7_SHORT_DURATION":
        if lc is False or any("TOO_SHORT" in c or "OVERLAP" in c for c in rc):
            return "PASS — short duration rejected"
        return f"WARN — short steps not caught; rc={rc}"

    if test_id == "T8_TIMESTAMP_REPLAY":
        if lc is False or any("TIMESTAMP_BEFORE_PUZZLE" in c for c in rc):
            return "PASS — past timestamps rejected"
        return f"WARN — past timestamps accepted; rc={rc}"

    if test_id == "T9_SPOT_FRAME_SPOOF":
        if lc is True:
            return "WARN — static photo accepted as spot_frame (detector may be disabled)"
        return "PASS — static photo spot_frame rejected"

    return "UNKNOWN"


# ──────────────────────────────────────────────
# test cases
# ──────────────────────────────────────────────

def run_all():
    rows = []

    # ── T1: FAKE_BYPASS ──────────────────────────────────────────
    print("\n[T1] FAKE_BYPASS — fake confidence, wrong action names ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    # Use totally wrong action names
    fake_steps = len(steps)
    ev = fake_evidence(steps, confidence=0.99,
                       override_actions=["fake_action"] * fake_steps)
    status, body = verify_puzzle(pid, ev)
    v = decide("T1_FAKE_BYPASS", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  score={body.get('overall_score')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T1_FAKE_BYPASS", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T2: CORRECT_FAKE ─────────────────────────────────────────
    print("\n[T2] CORRECT_FAKE — correct action names, fake confidence ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    ev = fake_evidence(steps, confidence=0.99)  # correct actions, fake timestamps
    status, body = verify_puzzle(pid, ev)
    v = decide("T2_CORRECT_FAKE", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  score={body.get('overall_score')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T2_CORRECT_FAKE", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T3: REPLAY_ATTACK ────────────────────────────────────────
    print("\n[T3] REPLAY_ATTACK — same puzzle_id a second time ...")
    # Use pid from T2 (which was just completed/attempted)
    ev2 = fake_evidence(steps, confidence=0.99)
    status, body = verify_puzzle(pid, ev2)
    v = decide("T3_REPLAY_ATTACK", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T3_REPLAY_ATTACK", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T4: EXPIRED / NONEXISTENT PUZZLE ─────────────────────────
    print("\n[T4] EXPIRED_PUZZLE — submit to a fake puzzle_id ...")
    fake_pid = "00000000-dead-beef-dead-000000000000"
    ev_fake  = [{"action": "blink", "start_timestamp": time.time(),
                 "end_timestamp": time.time() + 2, "confidence": 0.99, "metrics": {}}]
    status, body = verify_puzzle(fake_pid, ev_fake)
    v = decide("T4_EXPIRED_PUZZLE", status, body)
    print(f"   status={status}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T4_EXPIRED_PUZZLE", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T5: WRONG_STEPS ──────────────────────────────────────────
    print("\n[T5] WRONG_STEPS — shuffled / incorrect action names ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    wrong_actions = ["turn_left", "blink", "smile", "nod"][:len(steps)]  # likely wrong
    ev = fake_evidence(steps, confidence=0.99, override_actions=wrong_actions)
    status, body = verify_puzzle(pid, ev)
    v = decide("T5_WRONG_STEPS", status, body)
    print(f"   steps={[s['action'] for s in steps]}  sent={wrong_actions}")
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T5_WRONG_STEPS", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T6: LOW_CONFIDENCE ───────────────────────────────────────
    print("\n[T6] LOW_CONFIDENCE — confidence=0.3 (below 0.6 threshold) ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    ev = fake_evidence(steps, confidence=0.30)  # correct steps, low confidence
    status, body = verify_puzzle(pid, ev)
    v = decide("T6_LOW_CONFIDENCE", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T6_LOW_CONFIDENCE", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T7: SHORT_DURATION ───────────────────────────────────────
    print("\n[T7] SHORT_DURATION — step duration = 0.1s (below 0.5s minimum) ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    ev = fake_evidence(steps, confidence=0.99, duration=0.1)
    status, body = verify_puzzle(pid, ev)
    v = decide("T7_SHORT_DURATION", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T7_SHORT_DURATION", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T8: TIMESTAMP_REPLAY ─────────────────────────────────────
    print("\n[T8] TIMESTAMP_REPLAY — timestamps from 10 minutes in the past ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    ev = fake_evidence(steps, confidence=0.99, start_offset=-600)  # 10 min ago
    status, body = verify_puzzle(pid, ev)
    v = decide("T8_TIMESTAMP_REPLAY", status, body)
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T8_TIMESTAMP_REPLAY", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T9: SPOT_FRAME_SPOOF ─────────────────────────────────────
    print("\n[T9] SPOT_FRAME_SPOOF — static photo as spot_frames ...")
    puzzle = generate_puzzle()
    pid    = puzzle["puzzle_id"]
    steps  = puzzle["steps"]
    ev = fake_evidence(steps, confidence=0.99)
    frame_b64 = get_sample_frame_b64()
    spot_frames = [frame_b64, frame_b64, frame_b64] if frame_b64 else None
    status, body = verify_puzzle(pid, ev, spot_frames=spot_frames)
    v = decide("T9_SPOT_FRAME_SPOOF", status, body)
    print(f"   spot_frames_sent={len(spot_frames) if spot_frames else 0}")
    print(f"   liveness_confirmed={body.get('liveness_confirmed')}  rc={body.get('reason_codes')}")
    print(f"   -> {v}")
    rows.append({"test": "T9_SPOT_FRAME_SPOOF", "status": status,
                 "liveness_confirmed": body.get("liveness_confirmed"),
                 "score": body.get("overall_score"), "reason_codes": str(body.get("reason_codes")),
                 "verdict": v})

    # ── T10: DIFFICULTY SWEEP ────────────────────────────────────
    print("\n[T10] DIFFICULTY_SWEEP — check step counts per difficulty ...")
    spec = {"easy": (2, 3), "standard": (3, 4), "hard": (4, 5)}
    for diff, (min_s, max_s) in spec.items():
        p = generate_puzzle(difficulty=diff, min_steps=min_s, max_steps=max_s)
        n = len(p["steps"])
        ok = min_s <= n <= max_s
        mark = "PASS" if ok else f"FAIL — got {n} steps, expected {min_s}-{max_s}"
        print(f"   {diff:<10} steps={n}  {mark}")
        rows.append({"test": f"T10_DIFFICULTY_{diff.upper()}", "status": 200,
                     "liveness_confirmed": None, "score": None,
                     "reason_codes": str(p["steps"]),
                     "verdict": mark})

    return rows


# ──────────────────────────────────────────────
# output
# ──────────────────────────────────────────────

def write_output(rows):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "status", "liveness_confirmed",
                                           "score", "reason_codes", "verdict"])
        w.writeheader()
        w.writerows(rows)

    n_pass     = sum(1 for r in rows if r["verdict"].startswith("PASS"))
    n_critical = sum(1 for r in rows if r["verdict"].startswith("CRITICAL"))
    n_warn     = sum(1 for r in rows if r["verdict"].startswith("WARN"))
    n_fail     = len(rows) - n_pass - n_critical - n_warn

    lines = [
        "Puzzle Liveness Verification Test Summary\n",
        "=" * 50 + "\n\n",
        f"Endpoint: POST {VERIFY_URL}\n\n",
        f"{'Test':<30} {'LiveConf':>8}  {'Verdict'}\n",
        "-" * 80 + "\n",
    ]
    for r in rows:
        lc = str(r["liveness_confirmed"])
        lines.append(f"{r['test']:<30} {lc:>8}  {r['verdict']}\n")

    lines.append("\n")
    lines.append(f"PASS     : {n_pass}\n")
    lines.append(f"CRITICAL : {n_critical}\n")
    lines.append(f"WARN     : {n_warn}\n")
    lines.append(f"FAIL     : {n_fail}\n")

    if n_critical > 0:
        lines.append("\nWARNING: CRITICAL FINDINGS:\n")
        for r in rows:
            if r["verdict"].startswith("CRITICAL"):
                lines.append(f"  [{r['test']}] {r['verdict']}\n")
        lines.append("\nRoot cause: liveness confirmation is based only on client-supplied\n")
        lines.append("confidence values + timestamp ordering. The server has no frame-level\n")
        lines.append("image analysis to verify actions were actually performed.\n")
        lines.append("The spot_frame check (server-side frame analysis) exists in code but\n")
        lines.append("the MiniFASNet detector is broken (BUG-02), so it always passes.\n")
        lines.append("\nRecommended fix:\n")
        lines.append("  1. Fix MiniFASNet model path (BUG-02) to enable server-side spot check.\n")
        lines.append("  2. Make spot_frames mandatory (not optional) in the verify request.\n")
        lines.append("  3. Reject any verify request without valid spot_frames.\n")

    text = "".join(lines)
    OUT_SUM.write_text(text, encoding="utf-8")
    print("\n" + "=" * 70)
    print(text)
    print(f"CSV     -> {OUT_CSV}")
    print(f"Summary -> {OUT_SUM}")


def main():
    t0 = time.time()
    print("=" * 70)
    print(" FIVUCSAS  ·  Puzzle Liveness Verification Tests")
    print("=" * 70)
    print(f"  generate-puzzle : POST /api/v1/liveness/generate-puzzle")
    print(f"  verify          : POST /api/v1/liveness/verify")

    rows = run_all()
    write_output(rows)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
