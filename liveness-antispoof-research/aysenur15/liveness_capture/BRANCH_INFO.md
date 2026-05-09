# Branch: liveness_capture (≡ liveness_capture2)

- **Source**: `Rollingcat-Software/biometric-processor` `origin/liveness_capture`
- **Tip SHA**: `504067ed621d37580030b45cbffaa28f03137203`
- **`liveness_capture2` tip SHA**: identical (`504067e…`) — superseded duplicate, no separate folder.
- **Authors**: Aysenur15 <aysenurarici@hotmail.com> (the 6 commits below); investigation doc credits "Ayşe Gülsüm EREN" but `git log` shows only Aysenur15 on this isolated branch — Ayşe's commits land on `working_spoof_detection` & `fix/liveness-*-frr-reduction`.
- **Date range**: 2026-03-31 → 2026-04-18
- **Unique commits vs main**: 6
- **Diff stat vs main**: 171 files changed, 13 177 insertions(+), 349 deletions(-)
- **Source files snapshotted**: 50

## Summary

Earliest of Aysenur's lines. Promotes `enhanced` to default liveness backend and
adjusts `EnhancedLivenessDetector` confidence/passive scoring. Adds a
"color-shaded-screen" heuristic (display-characteristic colour macro flicker
check). Limited scope vs the flagship.

> **Memory correction (per investigation doc §Risks).** The audit memory
> `project_aysenur_liveness_branch.md` claims this branch already has rPPG +
> screen-replay + MRZ. The actual diff shows: enhanced/passive scoring + face
> bbox + color-shaded screen + liveness score. rPPG and screen-replay land in
> `main` proper; MRZ work lives in `practice-and-test/`, not in
> `biometric-processor` at all.

## Commit log

```
504067e Aysenur15 2026-04-18 Color Shaded Screen
7ef9638 Aysenur15 2026-04-14 Face Bbox
73ee6e9 Aysenur15 2026-04-09 Liveness Score Addition
1b91b10 Aysenur15 2026-04-01 Improve enhanced liveness confidence and passive scoring
c2f6f26 Aysenur15 2026-04-01 Set enhanced as default liveness baseline
d270c50 Aysenur15 2026-03-31 New Feature Additions
```

## Regressions

- Same `requirements.txt` revert pattern as the flagship.
- Several existing tests deleted (per investigation doc).
