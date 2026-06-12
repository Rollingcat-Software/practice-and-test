# amispoof Live Anti-Spoofing Evaluation

**System under test:** amispoof.fivucsas.com — real-time, fully client-side
(WASM + MediaPipe) face anti-spoofing + liveness demo, the browser-deployed build
of the FIVUCSAS spoof-detector (session-based, 25-analyzer pipeline).
**Method:** ISO/IEC 30107-3 presentation-attack detection (PAD). Each case was run
as a live webcam session; the verdict, 7-category probabilities, ~25 analyzer scores
and incident log were exported from amispoof's own session report (JSON).
**Sessions:** 9 labeled live sessions (2 genuine, 5 attacks, 2 special).

## Results

| Case | Ground truth | Verdict | Correct | Conf | MiniFASNet | Screen-replay | Incidents | Blinks |
|------|--------------|---------|:------:|:----:|:----------:|:-------------:|:---------:|:------:|
| liveperson | genuine | LIVE | ✅ | 86% | 100 | 53 | 0 | 9 |
| liveperson2 | genuine | LIVE | ✅ | 71% | 100 | 61 | 0 | 1 |
| printedphoto | print attack | SPOOF | ✅ | 52% | 100 | 71 | 5 | 0 |
| photofromphone | screen-photo attack | SPOOF | ✅ | 53% | 100 | 40 | 4 | 0 |
| videoreplay-me | video replay | **LIVE** | ❌ | 87% | 100 | 57 | 0 | 19 |
| videoattack-yt | video replay | **LIVE** | ❌ | 86% | 100 | 67 | 0 | 8 |
| whatsapp-replay | video replay | SPOOF | ✅ | 70% | 95 | 70 | 3 | 1 |
| **fakecam-injection-yt** | **video injection** | **LIVE** | ❌ | 72% | 100 | 62 | 0 | 19 |
| 2faces | two live people | LIVE | ✅* | 73% | 100 | 69 | 1 | 1 |
| noface | no face / absence | SPOOF | ✅ | 71% | — | — | 787 | 11 |

*video injection* = the YouTube video fed directly into the browser as a virtual
webcam (Chrome `--use-file-for-fake-video-capture`), with no screen and no physical
re-capture — the purest form of injection attack.

\* "2faces" reported `Faces detected = 1` even with two people present — only the
primary face is tracked; multi-presence is not flagged.

## ISO/IEC 30107-3 Metrics

| Metric | Value |
|--------|------:|
| **BPCER** (genuine wrongly rejected) | **0%** (0/2) |
| APCER — print | 0% (0/1) |
| APCER — screen photo | 0% (0/1) |
| **APCER — video replay** | **67% (2/3)** |
| **APCER — video injection (virtual cam)** | **100% (3/3)** |
| **APCER** (overall, worst-case) | **100%** |
| **ACER** = (APCER+BPCER)/2 | **50%** |

## Findings

### 1. Static attacks are reliably caught ✅
Printed photo and a still photo shown on a phone were both classified **SPOOF**.
The catch is liveness-driven: with no blinks/motion the "guilty-until-proven"
liveness prover flags `static_image`, even though MiniFASNet alone still rated the
texture as live (100). **APCER for static attacks = 0%.**

### 2. Genuine users are accepted ✅
Both live-person sessions returned **LIVE** with 0 incidents. **BPCER = 0%** — no
false rejection of real users in this sample.

### 3. Video replay is the key vulnerability 🚨
Two of three replay attacks (a recorded selfie video and a YouTube face video shown
on a screen) **passed as LIVE at 86–87% confidence with 0 incidents**. The replayed
video carries real blinks and motion (8–19 blinks), which satisfy the liveness
prover, and **MiniFASNet rated the screen texture as live (100)** — so the dominant,
high-weight signals voted LIVE. The low-weight screen-replay analyzer (57–67) sensed
something but could not flip the verdict. The one replay that *was* caught
(whatsapp-replay) only flipped because MiniFASNet dropped to 95 and incidents
accumulated. **Replay detection is inconsistent and presentation-dependent
(distance, brightness, bezel, moiré).** APCER for replay = **67%**.

### 4. Direct video injection is undetectable (worst case) 🔴
Three different face videos were fed straight into the browser as a virtual webcam
(Chrome `--use-file-for-fake-video-capture`), with **no screen and no physical
re-capture** — a YouTube talking-head, a frontal vlog clip, and a real NATO news
broadcast. **All three were classified LIVE (78–97% confidence) with 0–1 incidents;
none flipped — APCER = 100% (3/3).** One report rated the injected video **76.6%
"real"** (video_replay probability only 6%). Unlike physical screen replay, injection
**never flips to SPOOF over time** (sustained >135 s in one run), because there are
zero screen artefacts (no moiré, flicker, bezel) for the over-time analyzers to
accumulate. This is the most dangerous attack vector (virtual-camera /
deepfake-injection) and the system has **no passive defense** against it.

### 5. Time dependence (observed during testing)
A *physical* replay held briefly passes as LIVE, but the same replay held much longer
can flip to SPOOF as the over-time analyzers accumulate incidents (peak-sensitive
verdict). **Direct injection does not flip at all.** Implication: short interactive
verification (auth) is vulnerable to both; a long proctoring session may eventually
catch a physical replay but will **never** catch a clean virtual-camera injection.

### 6. Multi-face is not flagged ⚠️
With two people in frame, `Faces detected` stayed at 1 and the verdict remained LIVE
— no multi-presence incident. Relevant for proctoring use cases.

## Conclusion

amispoof is **strong against static presentation attacks (print, screen photo) and
does not reject genuine users** (BPCER 0%), but is **vulnerable to motion-bearing
attacks**: video replay bypasses 67% of the time, and **direct virtual-camera
injection bypasses 100% of the time and is never caught** (sustained LIVE, 0
incidents, rated 76.6% "real"). The root cause is the same — MiniFASNet accepts
on-screen / injected video as a live face and the dedicated replay/injection signals
are under-weighted. The peak-sensitive design eventually catches a *physical* replay
(screen artefacts accumulate) but has **no passive defense against a clean injection**,
making short interactive verification the most exposed scenario.

**Recommended hardening:**
1. Increase the weight / sensitivity of the screen-replay and moiré analyzers, or
   require their corroboration before a LIVE verdict on motion-rich input.
2. Add an active-illumination (random screen-color flash) challenge — the strongest
   defense against replay/injection that real-time playback cannot react to.
3. Flag multi-face presence as an incident for proctoring scenarios.
4. For short auth flows, require a quick active challenge rather than relying on
   passive over-time accumulation.

---
*Data: `reports_final/` (9 session JSONs), `amispoof_final_summary.csv`,
`amispoof_final_metrics.txt`. Parser: `parse_labeled.py`.*
