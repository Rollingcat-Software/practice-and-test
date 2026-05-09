# spoof-detector — pre-extraction history (in `practice-and-test`)

The `spoof-detector/` subtree was developed inside this submodule between
2026-05-02 and 2026-05-09 before being extracted to a standalone repository
at `Rollingcat-Software/spoof-detector` (commit `70c5216`).

## Author

- **Ahmet Abdullah Gultekin** <ahmetabdullahgultekin@gmail.com> (sole author of all 18 commits)

## Pre-extraction commit log (in `practice-and-test`)

```
70c5216 2026-05-09  chore(spoof-detector): extract to standalone repo
e8d0a50 2026-05-02  feat(spoof-detector): session-based multi-layer face spoof detection engine
25ece8a 2026-05-02  docs(spoof-detector): final session docs update — Grade C baseline
4abb516 2026-05-02  feat(spoof-detector): three-layer detection — flicker, tremor, environment grid
9afcb31 2026-05-02  docs(spoof-detector): add pixel-level screen forensics to roadmap
93a6e47 2026-05-02  docs(spoof-detector): add grid environment + behavioral signal processing to roadmap
da20801 2026-05-02  fix(spoof-detector): passive-only liveness — disable active challenges
5a507c3 2026-05-02  feat(spoof-detector): guilty-until-proven liveness architecture + active challenges
83fce65 2026-05-02  feat(spoof-detector): landmark variance analyzer + remove images from repo
cb45660 2026-05-02  milestone(spoof-detector): Grade D -> C (ACER 23% -> 15%)
aac41c6 2026-05-02  fix(spoof-detector): data-driven weight recalibration from ground truth
5e5e83e 2026-05-02  feat(spoof-detector): MiniFASNet instability detection for screen attacks
9995e46 2026-05-02  fix(spoof-detector): rebalance blink EAR threshold to middle ground
6477a58 2026-05-02  fix(spoof-detector): blink false positives on screens + verdict threshold
8f186f6 2026-05-02  fix(spoof-detector): fullscreen content scaling, rPPG FPS, roadmap doc
2a7ff6d 2026-05-02  feat(spoof-detector): complete all phases — blink, rPPG, AR filter, evaluation
051e76e 2026-05-02  fix(spoof-detector): fullscreen windows for all executables
84ee521 2026-05-02  feat(spoof-detector): session-based multi-method face spoof detection engine
```

## Original location at extraction

- Path: `practice-and-test/spoof-detector/`
- Removed by: `70c5216 chore(spoof-detector): extract to standalone repo` (2026-05-09)
- Removed files included: `README.md`, `ROADMAP.md`, `config.yaml`, `data/annotations/.gitkeep`, `data/captures/.gitkeep`, `data/protocol/report_*.json`, plus the full `src/` and `tests/` trees.

## Reconstructing the pre-extraction tree (read-only)

From the `practice-and-test` repo:

```bash
# View any pre-extraction file at HEAD just before removal
git show e8d0a50:spoof-detector/README.md
git show e8d0a50:spoof-detector/src/application/session_engine.py
# Etc.
```

Or list everything tracked at the pre-extraction commit:

```bash
git ls-tree -r e8d0a50 -- spoof-detector/
```
