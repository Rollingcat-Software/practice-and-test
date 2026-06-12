# FIVUCSAS Face Verification — Full Test Report (English)

**Date:** 2026-04-27
**Prepared by:** QA / Biometric Test Team
**System:** FIVUCSAS Biometric Processor — Face Recognition & Liveness Pipeline
**API Base:** http://localhost:8001/api/v1
**Test Dataset:** LFW Deep-Funneled (100 identities, 1,405 images)

---

## Executive Summary

A full-stack test was performed on the FIVUCSAS face verification pipeline covering enrollment accuracy, biometric metrics (FAR/FRR), multi-tenant security isolation, liveness & anti-spoofing, input validation, and performance. The face recognition model itself performs at an excellent level (AUC 0.9943). However, **four security vulnerabilities were identified**, two of which are critical and require immediate remediation before production deployment.

| Area | Status | Details |
|---|---|---|
| Face Recognition Accuracy | ✅ Excellent | AUC 0.9943, EER 1.93% |
| Input Validation | ✅ Mostly Good | 8/10 cases pass |
| Performance (single request) | ✅ Pass | p95 = 0.423s, target ≤ 1.5s |
| Multi-Tenant Isolation | ❌ Critical | tenant_id completely ignored |
| Liveness Endpoint | ❌ Critical | Broken — 500 on all calls |
| Anti-Spoofing (verify) | ❌ High Risk | 88% of spoof images accepted |
| Puzzle Liveness Bypass | ❌ Critical | Bypass possible with fake metadata |
| API Authentication | ❌ Medium | Invalid API key accepted with 200 |

---

## 1. Enrollment — Bulk Dataset Load

**What was tested:** 100 LFW identities (1,405 images total) were enrolled via `POST /api/v1/enroll` using a parallel bulk script.

**Results:**

| Metric | Value |
|---|---|
| Total images submitted | 1,405 |
| Successfully enrolled | 1,342 (95.5%) |
| Failed (no face detected) | 63 (4.5%) |
| Enrolled users | 100 |
| Embedding dimension | 512 (Facenet512) |

**Findings:**
- The 4.5% failure rate is within the expected range for the LFW dataset, which contains some images with no clear frontal face, group photos, or occlusions.
- Quality scores ranged from 70 to 87 — all above the configured threshold of 40.
- **Notice — quality threshold is set low:** The current `QUALITY_THRESHOLD=40` is permissive. In security-critical deployments (e.g. building access, exam proctoring), raising this to **60–70** is recommended. A stricter quality gate produces better embeddings, which directly reduces FAR and FRR. Images scoring below 60 often have poor lighting, slight blur, or off-angle framing — all of which increase matching errors downstream.
- The `liveness_score` field always returned 1.0 during enrollment, which is a placeholder value and does not represent real anti-spoofing analysis (confirmed as a known issue in the codebase comment).

---

## 2. Biometric Accuracy — FAR / FRR / EER / AUC

**What was tested:** 772 genuine pairs (same person, different photos) and 4,828 imposter pairs (different people) were scored via `POST /api/v1/verify`. Distance threshold sweeps were run to produce ROC, EER, and operating points.

**Results:**

| Metric | Value |
|---|---|
| Genuine pairs | 772 |
| Imposter pairs | 4,828 |
| AUC | 0.9943 |
| EER | 1.93% at distance threshold 0.5885 |
| Production threshold (distance) | 0.45 |
| FAR at production threshold | 0.27% |
| FRR at production threshold | 4.40% |

**Interpretation:**
- **AUC = 0.9943** — near-perfect separation between genuine and imposter pairs. The Facenet512 model is performing excellently.
- **EER = 1.93%** — this is competitive with published results on the LFW benchmark (state-of-the-art is <1%, well-tuned commercial systems are 0.5–2%).
- **At the current production threshold (0.45):**
  - FAR = 0.27% → approximately 1 in 370 impostors would be wrongly accepted.
  - FRR = 4.40% → approximately 1 in 23 legitimate users would be wrongly rejected.
- The current threshold favors security (low FAR) but causes a noticeable false rejection rate. Depending on the deployment context, this threshold may benefit from tuning.

**Recommendation:** The model quality is not a concern. Consider adjusting the production threshold from 0.45 to approximately 0.50–0.55 if reducing FRR (user friction) is a priority, while keeping FAR below 1%.

---

## 3. Multi-Tenant Isolation — CRITICAL VULNERABILITY

**What was tested:** 6 test cases using 3 real LFW identities enrolled across two isolated tenants (ct-test-A and ct-test-B) with overlapping user IDs.

**Results:**

| Case | Description | Expected | Got | Verdict |
|---|---|---|---|---|
| 01 | Same person, same tenant | verified=True | verified=True | ✅ PASS |
| 02 | Person A verifies against Tenant B (where B has Person C) | verified=False | verified=True | ❌ FAIL |
| 03 | Person C verifies against Tenant A (which has Person A) | verified=False | verified=True | ❌ FAIL |
| 04 | Unknown user in valid tenant | 404 | 404 | ✅ PASS |
| 05 | Valid person verifies against a non-existent tenant | 404 | verified=True (200) | ❌ FAIL |
| 06 | Different user, same tenant | verified=False | verified=False | ✅ PASS |

**Root Cause:**
The `/verify` endpoint does not apply `tenant_id` as a filter on the embedding lookup. It performs a **global search across all tenants**. Case 05 is the most conclusive proof: submitting a request to a completely fabricated tenant name still returns `verified=True` with the same distance score (0.212) as a legitimate same-tenant match.

**Impact:**
- Any user enrolled in Tenant A can be verified against Tenant B.
- A completely non-existent tenant still returns successful verifications.
- In a production multi-tenant deployment (e.g., university departments, different organizations) this means **full cross-organizational data leakage**.
- Severity: **CRITICAL**

**Required Fix:**
Add `WHERE tenant_id = :tenant_id AND user_id = :user_id` to the embedding lookup SQL query in the verification use case. Return 404 if the (tenant_id, user_id) combination is not found.

---

## 4. Liveness & Anti-Spoofing

### 4A. MiniFASNet Liveness Endpoint — CRITICAL

**Endpoint:** `POST /api/v1/liveness`

**Result:** Returns HTTP 500 on every call.

**Error message:** `Failed to initialize MiniFASNet: [Errno 13] Permission denied: '/nonexistent'`

**Root cause:** The `MINIFAS_MODEL_PATH` environment variable is set to `/nonexistent` — likely a placeholder that was never replaced with the actual model file path in docker-compose.

**Impact:**
- The dedicated liveness endpoint is completely non-functional.
- BPCER and APCER cannot be measured until this is fixed.
- The `liveness_score: 1.0` returned during enrollment is a hardcoded placeholder (confirmed in source: `# Placeholder - actual liveness check would need anti-spoofing model`).
- Severity: **CRITICAL**

**Required Fix:** Set `MINIFAS_MODEL_PATH` to the correct mounted path of the MiniFASNet model file inside the Docker container.

---

### 4B. Synthetic Spoof Attack via /verify — HIGH RISK

**What was tested:** 60 enrolled users were attacked with 4 synthetic spoof image types generated from their real enrollment photos. Each spoofed image was sent to `/verify` against the enrolled user.

**Attack types and results:**

| Attack Type | Description | Accepted | Total | APCER |
|---|---|---|---|---|
| PRINT_MILD | Slight blur + JPEG recompress q=70 | 54 | 58 | **93.1%** |
| SCREEN_REPLAY | Scan lines + warm color shift + recompress | 54 | 57 | **94.7%** |
| BORDER_PRINT | Print with white border frame | 54 | 56 | **96.4%** |
| PRINT_STRONG | Heavy blur + vignette + q=45 | 39 | 57 | **68.4%** |
| **Average APCER** | | | | **88.2%** |
| GENUINE (BPCER) | Real photo of enrolled person | 52 | 55 | 5.5% false reject |

**Interpretation:**
- `/verify` has no anti-spoofing capability — it only compares face embeddings.
- A printed photo, phone screen replay, or a slightly degraded copy of the enrollment image is accepted as genuine in 88% of cases on average.
- This means a physical photograph of any enrolled person can be used to gain access.
- PRINT_STRONG has lower APCER (68%) because heavy blur degrades the embedding enough to push the distance past the threshold — but this is not a reliable protection mechanism.

**Required Fix:** The liveness check (BUG-02 above) must be fixed and integrated into the `/verify` flow so that every verification call also verifies the image is from a live person.

---

### 4C. Puzzle Liveness Challenge-Response — CRITICAL BYPASS

**Endpoint:** `POST /api/v1/liveness/verify` (after generating a puzzle via `/liveness/generate-puzzle`)

**What was tested:** 10 attack scenarios against the challenge-response liveness system (blink, smile, turn head puzzles).

**Results:**

| Test | Expected | Result | Verdict |
|---|---|---|---|
| T1 — Fake actions (wrong names) | rejected | rejected | ✅ PASS |
| **T2 — Correct actions, fake confidence** | **rejected** | **liveness_confirmed=True** | ❌ **CRITICAL** |
| T3 — Replay same puzzle_id | rejected | PUZZLE_ALREADY_COMPLETED | ✅ PASS |
| T4 — Non-existent puzzle_id | rejected | PUZZLE_NOT_FOUND | ✅ PASS |
| T5 — Wrong step order | rejected | rejected | ✅ PASS |
| T6 — Confidence = 0.3 (below 0.6 min) | rejected | rejected | ✅ PASS |
| T7 — Step duration 0.1s (below 0.5s min) | rejected | rejected | ✅ PASS |
| T8 — Timestamps 10 min in the past | rejected | rejected | ✅ PASS |
| T9 — Static photo as spot_frames | rejected | liveness_confirmed=True | ⚠️ WARN |
| T10 — Difficulty step counts | correct counts | correct counts | ✅ PASS |

**Critical Finding (T2):**
The puzzle system can be fully bypassed in two API calls:

```
Step 1: POST /api/v1/liveness/generate-puzzle
        Response: puzzle_id + step names (e.g. ["blink", "smile", "turn_left"])

Step 2: POST /api/v1/liveness/verify
        Body: {
          "puzzle_id": "<id from step 1>",
          "results": [
            {"action": "blink",      "start_timestamp": <now>, "end_timestamp": <now+1.5>, "confidence": 0.99},
            {"action": "smile",      "start_timestamp": <now+1.7>, "end_timestamp": <now+3.2>, "confidence": 0.99},
            {"action": "turn_left",  "start_timestamp": <now+3.4>, "end_timestamp": <now+4.9>, "confidence": 0.99}
          ]
        }
        Response: liveness_confirmed=True, overall_score=99.0
```

No camera, no real actions required. The attacker only needs to know the step names (obtained from generate-puzzle) and submit fake confidence values with valid timestamps.

**Root Cause:**
The `confidence` field is client-provided and trusted without server-side verification. The server-side `spot_frames` check (which would verify actual camera frames) exists in the code but is disabled because the MiniFASNet detector fails to initialize (same BUG-02).

**Required Fixes:**
1. Fix MiniFASNet model path (BUG-02) to enable server-side spot check.
2. Make `spot_frames` a required field — reject any verify request without it.
3. Validate spot frames server-side and reject if liveness score is below threshold.

---

## 5. Edge Cases & Input Validation

**What was tested:** 10 edge case inputs sent to `/enroll` and `/verify` — invalid formats, extreme sizes, empty images, authentication bypass attempts.

**Results:**

| Case | Input | Expected | Got | Verdict |
|---|---|---|---|---|
| 01 | Blank white image | 400 | 400 | ✅ PASS |
| 02 | 10×10 pixel image | 400 | 400 | ✅ PASS |
| 03 | Solid white 640×480 | 400 | 400 | ✅ PASS |
| 04 | Solid black 640×480 | 400 | 400 | ✅ PASS |
| 05 | 12 MB file | 413 | 413 | ✅ PASS |
| **06** | **GIF file** | **400/415** | **500** | ❌ **FAIL** |
| 07 | Corrupted JPEG | 400 | 400 | ✅ PASS |
| 08 | Text file as JPEG | 400 | 400 | ✅ PASS |
| 09 | Same image enroll + verify | verified=True | verified=True | ✅ PASS |
| **10** | **Invalid API key** | **401** | **200** | ❌ **FAIL** |

**Bug 03 — GIF Format Returns 500:**
When a GIF file is sent, the server returns HTTP 500 Internal Server Error instead of a proper 400/415. This happens because format validation occurs inside the model pipeline rather than at the API boundary. The raw traceback may be visible in logs, and no user-friendly error is returned.

**Bug 04 — API Authentication Not Enforced:**
Sending a request with `X-Api-Key: INVALID_KEY_XYZ_000` returns HTTP 200 with a valid response. Either API key validation is not implemented or is disabled in the current deployment. Any caller with knowledge of the endpoint URL can access the system without credentials.

---

## 6. Performance

**What was tested:** Single-request latency (30 sequential calls), concurrent load (20 parallel requests), rate limiting (40 burst calls), and search latency (5 calls).

**Results:**

### P1 — Single Request Latency

| Metric | Value | Target | Status |
|---|---|---|---|
| Min | 0.201s | — | — |
| p50 | 0.227s | — | — |
| p95 | 0.423s | ≤ 1.5s | ✅ PASS |
| p99 | 0.424s | — | — |
| Max | 0.424s | — | — |

Single-request performance is well within the target. The first 1-2 requests take ~0.4s (model warm-up), then stabilize at ~0.22s.

### P2 — Concurrent Load (20 Parallel Requests)

| Metric | Value |
|---|---|
| Wall time (all 20) | 6.95s |
| p50 per request | 4.257s |
| p95 per request | 5.689s |
| Errors | 0 / 20 |

No crashes or errors under concurrent load — positive. However, per-request latency increased from 0.22s to 4–5s under 20 parallel users (approximately 17–25× slowdown). This is caused by the DeepFace model being executed sequentially despite concurrent HTTP requests. Under real-world load this will produce unacceptable user wait times.

**Note:** The biometric API server is currently at **94% memory usage**. Under sustained concurrent load, OOM (Out of Memory) errors are possible.

### P3 — Rate Limiting

Rate limiting was not triggered in 40 rapid consecutive calls. The `RATE_LIMIT_ENABLED` environment variable is set to `False` in the test environment. This must be verified to be `True` in production. Without rate limiting, the system is vulnerable to brute-force enumeration attacks.

### P4 — Search Latency

| Metric | Value |
|---|---|
| Median | 0.256s |
| Max | 0.305s |

Search across 553 embeddings is fast. However, the pgvector **HNSW index is not yet configured** (flagged as a TODO). Without it, the database performs a sequential scan. Estimated latency at scale:

| Users | Estimated latency (no index) |
|---|---|
| 553 (current) | ~0.25s |
| 10,000 | ~4–5s |
| 100,000 | ~40–50s |

---

## 7. Complete Bug Register

| ID | Severity | Area | Description | Required Action |
|---|---|---|---|---|
| BUG-01 | 🔴 CRITICAL | Multi-Tenant | `tenant_id` not applied in `/verify` — global embedding search | Add `WHERE tenant_id = :tenant_id` to DB query |
| BUG-02 | 🔴 CRITICAL | Liveness | MiniFASNet model path set to `/nonexistent` — endpoint returns 500 | Set correct `MINIFAS_MODEL_PATH` in docker-compose |
| BUG-03 | 🔴 CRITICAL | Liveness | Puzzle liveness bypass: fake confidence + correct action names → `liveness_confirmed=True` | Make `spot_frames` required; fix BUG-02 |
| BUG-04 | 🟠 HIGH | Anti-Spoofing | `/verify` has no spoof resistance — avg APCER 88.2% | Integrate liveness check into verify flow after BUG-02 fix |
| BUG-05 | 🟡 MEDIUM | Input Validation | GIF input returns 500 instead of 400/415 | Add format validation before model pipeline |
| BUG-06 | 🟡 MEDIUM | Authentication | Invalid API key returns 200 | Enforce API key validation on all endpoints |
| BUG-07 | 🟡 MEDIUM | Performance | 17–25× latency increase under 20 concurrent users | Configure async model workers or request queuing |
| BUG-08 | 🟡 MEDIUM | Performance | pgvector HNSW index missing — search will degrade at scale | Add HNSW index to `face_embeddings` table |
| BUG-09 | 🟡 MEDIUM | Performance | Rate limiting disabled (`RATE_LIMIT_ENABLED=False`) | Verify production config has rate limiting enabled |
| BUG-10 | 🟢 LOW | Enrollment | `liveness_score: 1.0` is a hardcoded placeholder during enrollment | Track with a TODO — misleads callers |

---

## 8. What Is Working Well

- **Face recognition model quality** — AUC 0.9943 is excellent. Facenet512 is correctly integrated and producing reliable embeddings.
- **Threshold calibration** — The production threshold of 0.45 gives FAR=0.27% which is a reasonable operating point.
- **Input validation (most cases)** — Blank images, tiny images, black images, corrupted JPEGs, oversized files, and text files are all correctly rejected.
- **Puzzle structural integrity** — Replay attacks, expired puzzles, wrong action sequences, low confidence, and past timestamps are all correctly detected and rejected by the puzzle liveness system.
- **Cross-user isolation within a tenant** — A user in Tenant A cannot be verified as a different user within the same tenant (same-tenant user isolation works correctly).
- **Client-side bypass fields ignored** — Injecting extra fields like `liveness_passed=true` into the verify request body has no effect — the server correctly ignores them.
- **File size limit** — 12 MB files are correctly rejected with HTTP 413.

---

## 9. Priority Action Plan

| Priority | Action | Owner | Blocking |
|---|---|---|---|
| P0 — Immediate | Fix `MINIFAS_MODEL_PATH` in docker-compose (BUG-02) | DevOps | BUG-03, BUG-04 |
| P0 — Immediate | Add `WHERE tenant_id` filter to verify query (BUG-01) | Backend | Production launch |
| P0 — Immediate | Make `spot_frames` required in puzzle verify + reject without it (BUG-03) | Backend | Production launch |
| P1 — Before launch | Enforce API key validation on all endpoints (BUG-06) | Backend | Security compliance |
| P1 — Before launch | Add format validation before model pipeline for GIF/BMP etc. (BUG-05) | Backend | |
| P1 — Before launch | Verify `RATE_LIMIT_ENABLED=True` in production config (BUG-09) | DevOps | |
| P2 — Short term | Configure async workers to reduce concurrent latency (BUG-07) | Backend/DevOps | |
| P2 — Short term | Add pgvector HNSW index to `face_embeddings` table (BUG-08) | Backend/DB | |
| P3 — Ongoing | Download real spoof dataset (NUAA/Replay-Attack) and measure APCER with real photos | QA | |
| P3 — Ongoing | Monitor server memory (currently 94%) and set up OOM alerting | DevOps | |

---

*Report generated from automated test suite in `C:\Users\hp\fivucsas-test\`*
*Test scripts are reproducible — re-run any step individually after fixes are applied.*
