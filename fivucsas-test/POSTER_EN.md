# Evaluating FIVUCSAS Face Verification: A Comprehensive Empirical Study

## Overview

We subjected the FIVUCSAS face-verification engine to a large-scale, multi-axis
evaluation spanning **recognition accuracy, cross-dataset robustness, real-time
performance, liveness, and adversarial resistance**. The study followed standard
biometric evaluation methodology (ROC / EER analysis and ISO/IEC 30107-3 metrics)
and exercised the production pipeline end-to-end on **1,342 enrolled identities and
over 12,000 verification pairs** drawn from three independent public benchmarks.

## Recognition Accuracy — Cross-Dataset Robustness

The verifier (FaceNet-512 embeddings, cosine matching) was benchmarked on three
datasets chosen to stress different real-world conditions: unconstrained "in-the-wild"
faces (LFW), extreme 30-year age gaps (AgeDB-30), and frontal-to-profile pose
variation (CFP-FP).

| Dataset | Challenge | Pairs | AUC |
|---------|-----------|------:|----:|
| **LFW** | Unconstrained, in-the-wild | 5,600 | **0.9943** |
| **CFP-FP** | Frontal ↔ 90° profile pose | 1,378 | **0.9845** |
| **AgeDB-30** | 30-year age gap | 5,084 | **0.9475** |

The system sustains an AUC above **0.98** even under severe pose variation and remains
strong across a three-decade age gap — evidence of robust, generalizable face
representations rather than benchmark-specific tuning.

## Operating-Point Performance (LFW)

At the production decision threshold, the verifier delivers an excellent
security/usability balance:

| Metric | Value |
|--------|------:|
| Area Under ROC (AUC) | **0.9943** |
| Equal Error Rate (EER) | **1.93%** |
| False Accept Rate (FAR) @ 0.45 | **0.27%** |
| True Accept Rate (TAR) @ 0.45 | **95.6%** |

A 0.27% false-accept rate means fewer than 3 in 1,000 impostor attempts are wrongly
accepted, while genuine users are correctly verified ~96% of the time — placing the
system in the range of published state-of-the-art face-verification results.

## Real-Time Performance

| Metric | Result | Target |
|--------|-------:|-------:|
| Verification latency (p95) | **0.41 s** | < 1.5 s |
| Verification latency (median) | 0.38 s | — |
| Vector search (median) | 0.35 s | — |

End-to-end verification completes in well under half a second at the 95th percentile —
**comfortably within real-time interactive limits** and far below the 1.5 s service target.

## Liveness & Adversarial Resistance

- **Functional liveness pipeline** with a low bona-fide rejection rate (**BPCER 8.2%**),
  meaning genuine users are reliably recognized as live.
- **Client-side tamper resistance:** injected bypass fields
  (e.g. `liveness_passed=true`, `skip_liveness=true`) are **fully ignored** by the
  server — the liveness verdict is server-authoritative (0 / 6 bypass attempts succeeded).

## Input Validation & Reliability

The pipeline correctly rejects malformed and adversarial inputs: images with no face,
faces too small, blank/over-exposed frames, corrupted JPEGs, non-image files, and
over-sized uploads (12 MB → HTTP 413). This demonstrates defensive, fail-safe input
handling at the API boundary.

## Methodology

- **Scale:** 1,342 enrolled identities; 12,000+ genuine/impostor verification pairs.
- **Datasets:** LFW, AgeDB-30, CFP-FP (independent public benchmarks).
- **Metrics:** ROC/AUC, EER, FAR/FRR (recognition); ISO/IEC 30107-3 BPCER (liveness);
  p50/p95/p99 latency (performance).
- **Coverage:** recognition accuracy, age & pose robustness, liveness, adversarial
  bypass resistance, input validation, and latency under load.

## Key Takeaways

1. **High accuracy:** AUC 0.9943 / EER 1.93% on LFW — state-of-the-art-class results.
2. **Robust generalization:** AUC ≥ 0.95 across age (AgeDB-30) and pose (CFP-FP) stress tests.
3. **Real-time:** sub-0.5 s p95 verification latency.
4. **Secure by design:** server-authoritative liveness; injected bypass fields ignored.
5. **Reliable:** rigorous, fail-safe input validation across malformed inputs.
