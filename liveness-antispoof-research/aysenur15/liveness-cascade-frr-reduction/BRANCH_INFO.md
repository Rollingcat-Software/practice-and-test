# Branch: fix/liveness-cascade-frr-reduction

- **Source**: `Rollingcat-Software/biometric-processor` `origin/fix/liveness-cascade-frr-reduction`
- **Tip SHA**: `b730a6dbe2bf944f63f05ee80d828d39fbc39159`
- **Authors**: Ayşe Gülsüm EREN <aysegulsumeren@gmail.com>, Aysenur15 <aysenurarici@hotmail.com>
- **Date range**: 2026-03-31 → 2026-05-06
- **Unique commits vs main**: 27
- **Diff stat vs main**: 195 files changed, 27 312 insertions(+), 359 deletions(-)
- **Source files snapshotted**: 65

## Summary

Built on top of `working_spoof_detection`. Adds the hijab/head-turn FRR
fix: "nose-alone physical block is not critical occlusion". Two
top-of-branch commits by Ayşe Gülsüm EREN address head-turn flagged as
occlusion in `FaceUsabilityGate`.

Cannot land without first landing the parent `working_spoof_detection`.
Same `requirements.txt` regression. Same 6.5 MB binary excluded.

## Commit log (top of branch)

```
b730a6d Ayşe Gülsüm EREN 2026-05-06 fix(liveness): nose-alone physical block is not critical occlusion (head-turn FRR)
1715986 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): don't flag head turns as occlusion in FaceUsabilityGate
af01130 Aysenur15 2026-05-06 Liveness Update
1229f48 Aysenur15 2026-05-06 P3 Phase
167969a Aysenur15 2026-05-06 Live Update
13243a1 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): P0 FRR reduction — freeze EMA on skipped frames + re-enable decision guards
…(20 more commits inherited from working_spoof_detection — see that branch's BRANCH_INFO.md)
```
