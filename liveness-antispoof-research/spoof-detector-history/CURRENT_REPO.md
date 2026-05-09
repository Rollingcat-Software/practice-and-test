# spoof-detector — current standalone repo

After extraction on 2026-05-09 (`practice-and-test` commit `70c5216`),
the production-track liveness/anti-spoof library lives here:

- **Repo**: <https://github.com/Rollingcat-Software/spoof-detector>
- **Owner**: Rollingcat-Software
- **Author**: Ahmet Abdullah Gultekin (sole)
- **Latest tagged version**: `v0.2.0`

## Architecture

Per `INVESTIGATION_2026-05-09.md` §1.4 + memory `feedback_spoof_detector_architecture.md`:

- 14 analyzers in 3 layers (per-frame, 1-5s, 5-30s, 30s-3hr)
- Calibrated 7-class fusion via `MultiClassFuser`
- Session engine with peak-sensitive verdict
  (`0.5 * average_p_real + 0.5 * worst_window_p_real`)
- Liveness prover: blink (EAR), motion, rotation, expression
  → max 75 gold-proof points
- 60+ unit tests
- ISO 30107-3 measured: BPCER 0.00% / APCER 30% / ACER 15%, Grade C, 4 scenarios
- Paper outline at `paper/outline.md` (BIOSIG/IJCB 2026 target)

## How it integrates into FIVUCSAS prod

The production wiring lives in `biometric-processor`'s `feat/depend-on-spoof-detector-and-rewire`
branch (and successors). `spoof-detector` is consumed as an external Python
dependency — biometric-processor pins to a specific tag and calls
`SpoofDetectionPipeline` + `SessionEngine` from its `/verify` and `/enroll`
use cases.

## Why this folder is separate from production

This `liveness-antispoof-research/` collection is the **archival research
record**: Aysenur's branches, Ayşe Gülsüm Eren's contributions, and the
pre-extraction `spoof-detector` history. The **production track** is the
standalone repo above plus the biometric-processor wiring branch, governed
by the architecture decision recorded in
`memory/feedback_spoof_detector_architecture.md`.
