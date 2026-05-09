# Liveness & Anti-Spoof Research Collection

This folder collects all liveness / anti-spoof R&D work that went into
FIVUCSAS, organized for archival and reference. Created 2026-05-09 in
response to the user's directive: *"aysegulsum and aysenur have some
branches and works, it might be better to move all work and running
examples etc."*

## Production vs research split

> **The CURRENT PRODUCTION libraries live at**
> <https://github.com/Rollingcat-Software/spoof-detector> (v0.2.0)
> with `biometric-processor`'s `feat/depend-on-spoof-detector-and-rewire`
> branch as the consumer wiring. **This folder is the wider research
> record** — it does NOT contain production code paths, just the
> historical R&D artifacts plus pointers.

The architecture decision is captured in
`memory/feedback_spoof_detector_architecture.md`: production algorithms
live in the `spoof-detector` standalone repo; this `liveness-antispoof-research/`
folder is the parallel archival collection of branches that did not
survive the production extraction.

## Folder map

```
liveness-antispoof-research/
├── README.md                              ← this file
├── INVESTIGATION_2026-05-09.md            ← canonical file-by-file analysis (copy)
├── ATTRIBUTION.md                         ← who did what, GitHub handles, dates
├── aysenur15/                             ← Aysenur15's work + Ayşe Gülsüm EREN's
│   ├── working_spoof_detection/           ← flagship (27 commits, 65 source files)
│   ├── liveness_capture/                  ← earliest line (6 commits, 50 source files)
│   ├── Spoof-Detection/                   ← intermediate subset (9 commits)
│   ├── feat-anti-spoof-pipeline-local/    ← clean squash (LOCAL only, never pushed)
│   ├── liveness-cascade-frr-reduction/    ← hijab fix on top of flagship
│   ├── liveness-p0-frr-reduction/         ← P0/P1/P2 EMA tuning
│   └── liveness-p3-frr-reduction/         ← P3 phase iteration
├── ayse-gulsum-eren/                      ← (empty — her commits are mixed into branches above)
└── spoof-detector-history/
    ├── PRE_EXTRACTION.md                  ← 18 commits in practice-and-test before 70c5216
    └── CURRENT_REPO.md                    ← pointer to standalone Rollingcat-Software/spoof-detector
```

Each branch folder contains:
- `BRANCH_INFO.md` — tip SHA, commit log, diff stat vs main, summary, regressions
- `unique-source/` — read-only snapshots of unique `.py` and `.md` files added/modified vs `main`, organized by their original path under `app/` so it's still browsable

## Per-branch summary table

| Folder | Branch | Tip SHA | Commits | Files | TLDR | Authors |
|---|---|---|---:|---:|---|---|
| `aysenur15/working_spoof_detection/` | `origin/working_spoof_detection` | `cbdbe0b` | 27 | 65 | Flagship: face-usability + critical-region + hybrid-fusion + sklearn trainer + 4 803-LoC tuner | Aysenur15 + Ayşe Gülsüm EREN |
| `aysenur15/liveness_capture/` | `origin/liveness_capture` | `504067e` | 6 | 50 | Enhanced backend default + color-shaded screen heuristic | Aysenur15 |
| `aysenur15/Spoof-Detection/` | `origin/Spoof-Detection` | `0685f05` | 9 | 56 | No-face handling + face bbox subset of flagship | Aysenur15 |
| `aysenur15/feat-anti-spoof-pipeline-local/` | LOCAL `feat/anti-spoof-pipeline` | `9ca51a2` | 6 | 24 | Cleanest review-friendly squash of anti-spoof modules | Aysenur15 + Ahmet |
| `aysenur15/liveness-cascade-frr-reduction/` | `origin/fix/liveness-cascade-frr-reduction` | `b730a6d` | 27 | 65 | Hijab/head-turn FRR fix on top of flagship | Ayşe Gülsüm EREN + Aysenur15 |
| `aysenur15/liveness-p0-frr-reduction/` | `origin/fix/liveness-p0-frr-reduction` | `00bf4d7` | 28 | 65 | P0/P1/P2 EMA tuning with revert iterations | Ayşe Gülsüm EREN + Aysenur15 |
| `aysenur15/liveness-p3-frr-reduction/` | `origin/fix/liveness-p3-frr-reduction` | `1229f48` | 24 | 65 | P3 phase tuning | Aysenur15 + Ayşe Gülsüm EREN |
| `spoof-detector-history/` | n/a | n/a | 18 | n/a | Pre-extraction commits in `practice-and-test/spoof-detector/` | Ahmet Abdullah Gultekin |

`liveness_capture2` is intentionally absent — it is identical to
`liveness_capture` (both at `504067e…`) per `git rev-parse`.

## How to read this collection

1. **Start at `INVESTIGATION_2026-05-09.md`** — single source of truth on
   what every artifact does, with cross-references to file paths.
2. **For any branch**, open the corresponding `BRANCH_INFO.md` for the
   commit log + diff stat + regressions warning. Then browse
   `unique-source/` for read-only file content.
3. **For author attribution**, see `ATTRIBUTION.md`.
4. **For the production path**, see `spoof-detector-history/CURRENT_REPO.md`.

## What's intentionally NOT here

Per the constraints in the original task:

- No `demo-ui/out/` Next.js build output (per investigation: 112 of 166
  files in some branches were build artifacts).
- No `__pycache__/` or `*.pyc`.
- No binaries > 1 MB (e.g. the 6.5 MB `yolov8n.pt` in
  `working_spoof_detection`).
- No reverted Dependabot security pin changes from `requirements.txt`
  (those are anti-features, flagged in each `BRANCH_INFO.md`).
- No `.debug-snapshots/` directories that mirror `app/` files.
- No git refs or working trees from `biometric-processor` itself
  — that submodule is read-only from this collection's perspective.

To access anything excluded, fetch the original branch from
`Rollingcat-Software/biometric-processor` directly.

## Provenance

- Created: 2026-05-09
- Source repos: `Rollingcat-Software/biometric-processor`,
  `Rollingcat-Software/practice-and-test`
- Method: `git show <ref>:<path>` (read-only, no `git checkout` of
  `biometric-processor` branches at any point during creation).
- Investigation doc copied verbatim from
  `/opt/projects/fivucsas/LIVENESS_ANTISPOOF_INVESTIGATION_2026-05-09.md`.
