# Branch: fix/liveness-p0-frr-reduction

- **Source**: `Rollingcat-Software/biometric-processor` `origin/fix/liveness-p0-frr-reduction`
- **Tip SHA**: `00bf4d76ecd00170c130cfb88280a353d91a825d`
- **Authors**: Ayşe Gülsüm EREN <aysegulsumeren@gmail.com>, Aysenur15 <aysenurarici@hotmail.com>
- **Date range**: 2026-03-31 → 2026-05-06
- **Unique commits vs main**: 28
- **Diff stat vs main**: 195 files changed, 27 173 insertions(+), 359 deletions(-)
- **Source files snapshotted**: 65

## Summary

Most disciplined of the FRR-tuning branches: each fix is followed by a
revert if pushback came in (suggesting Ahmet was shepherding the FRR
knob). Commits include EMA freeze on skipped frames, decision-guard
re-enable, P1 flash_replay_strong tightening (reverted), P2 recency-weighted
score mean + illumination tiers (reverted), screen_frame cascade geom
confirmation (reverted).

Built on top of `working_spoof_detection`.

## Commit log (top of branch)

```
00bf4d7 Ayşe Gülsüm EREN 2026-05-06 revert(liveness): roll back P2 and cascade-guard changes (a0fbc1a + b44e1c9)
b44e1c9 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): require geom confirmation for screen_frame cascade + flash_planar min 2 samples
a0fbc1a Ayşe Gülsüm EREN 2026-05-06 fix(liveness): P2 — recency-weighted score mean + illumination tiers + faster RECOVERING exit
05cc03f Ayşe Gülsüm EREN 2026-05-06 revert(liveness): roll back P1 flash_replay_strong + live_active_ready changes
45c8914 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): P1 FRR — tighten flash_replay_strong + lower live_active_ready threshold
3b93e44 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): guard ema_score None when first frame is liveness-skipped
13243a1 Ayşe Gülsüm EREN 2026-05-06 fix(liveness): P0 FRR reduction — freeze EMA on skipped frames + re-enable decision guards
…(21 more commits inherited from working_spoof_detection)
```
