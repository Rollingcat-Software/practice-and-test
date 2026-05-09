# Branch: feat/anti-spoof-pipeline (LOCAL ONLY — never pushed)

- **Source**: `Rollingcat-Software/biometric-processor` LOCAL branch `feat/anti-spoof-pipeline` (no remote)
- **Tip SHA**: `9ca51a239893b041a51aee4b74e8739bd03338fb`
- **Authors**: Aysenur15 <aysenurarici@hotmail.com>, Ahmet Abdullah Gultekin <ahmetabdullahgultekin@gmail.com>
- **Date range**: 2026-04-01 → 2026-04-25
- **Unique commits vs main**: 6
- **Diff stat vs main**: 28 files changed, 4 600 insertions(+), 87 deletions(-)
- **Source files snapshotted**: 24

## Summary

Cleaned-up squash view of the anti-spoof integration: moire + device-spoof +
reaction + baseline + flash-spoof + cutout + screen-replay veto + hybrid
backend + strict-profile config. **Restricted to anti-spoof modules only**;
drops the test-deletion noise + the `requirements.txt` regression that
plagues `working_spoof_detection`.

Per investigation doc §Inventory: *"This is the most review-friendly version
of Aysenur's contribution and is the right starting point if we want to
upstream her work."*

## Commit log

```
9ca51a2 Ahmet Abdullah Gultekin 2026-04-25 fix(main): remove duplicate /ping route
db0faaa Aysenur15 2026-04-25 feat(anti-spoof): flash-spoof analyzer + cutout anomaly + strict-profile config
4d1cbaf Aysenur15 2026-04-25 feat(anti-spoof): face bbox refinements + hybrid backend + screen-replay veto
6da71d3 Aysenur15 2026-04-25 feat(anti-spoof): integrate moire + device-spoof + reaction + baseline pipeline
5e074e9 Aysenur15 2026-04-01 Improve enhanced liveness confidence and passive scoring
1687036 Aysenur15 2026-04-01 Set enhanced as default liveness baseline
```

## Caveat

Because this branch lives only in the working clone at `biometric-processor`'s
local refs, it can't be `git fetch`'d from a fresh clone. To rebuild:
1. Cherry-pick the 6 SHAs above on top of `main` of `biometric-processor`.
2. Or use the snapshot in `unique-source/` for read-only reference.
