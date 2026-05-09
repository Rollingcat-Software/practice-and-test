# Attribution — Liveness & Anti-Spoof R&D

This collection aggregates contributions from three people across multiple
branches. Authorship was extracted via `git log --format='%an <%ae>' | sort -u`
on the relevant ranges in `Rollingcat-Software/biometric-processor` and
`Rollingcat-Software/practice-and-test`.

## Contributors

### Aysenur15
- **Email**: aysenurarici@hotmail.com
- **GitHub handle**: `@Aysenur15` (per memory + commit attribution)
- **Primary work areas**: full anti-spoof + liveness pipeline on `biometric-processor`
- **Branches**:
  - `working_spoof_detection` (21 commits) — flagship
  - `liveness_capture` / `liveness_capture2` (6 commits — sole author)
  - `Spoof-Detection` (9 commits — sole author)
  - co-author on `feat/anti-spoof-pipeline` local branch (4 commits)
  - inherited base on FRR-reduction sibling branches
- **Modules originated**: `face_usability_gate`, `critical_region_visibility_gate`,
  `face_quality_illumination_gate`, `hybrid_fusion_evaluator`,
  `train_spoof_classifier`, `test_data_collector`, `live_liveness_preview`
  (4 803 LoC tuner)
- **Approximate total commits in this collection**: 30+

### Ayşe Gülsüm EREN
- **Email**: aysegulsumeren@gmail.com
- **GitHub handle**: not yet confirmed in this audit. The investigation doc
  flags this as an open question (§Risks): "Are these the same person under
  different commit identities, or two collaborators?" The split email
  domains (`hotmail.com` vs `gmail.com`) plus the distinct commit-message
  style (Conventional Commits with subsystem scopes for Ayşe vs free-form
  for Aysenur15) strongly suggest **two separate collaborators**.
- **Primary work areas**: FRR-reduction tuning + hijab/head-turn occlusion
  fixes + decision-guard EMA work
- **Branches**:
  - lead author on `fix/liveness-cascade-frr-reduction` (head-turn fix, nose-occlusion)
  - lead author on `fix/liveness-p0-frr-reduction` (P0/P1/P2/cascade-guard iterations + reverts)
  - co-author on `working_spoof_detection` (1 commit: P0 FRR reduction)
- **Approximate total commits in this collection**: 8+

### Ahmet Abdullah Gultekin
- **Email**: ahmetabdullahgultekin@gmail.com (also seen as `Ahmet Abdullah Gültekin` on the spoof-detector commits)
- **GitHub handle**: project owner; commits as user
- **Primary work areas**:
  - sole author of `practice-and-test/spoof-detector/` (18 commits, 2026-05-02 → 2026-05-09)
  - sole author of standalone `Rollingcat-Software/spoof-detector` repo (post-extraction)
  - co-author on `feat/anti-spoof-pipeline` local branch (1 commit)
  - shepherding role on FRR-reduction branches (per investigation doc §Inventory)

## Citation guidance for future paper(s)

Per `INVESTIGATION_2026-05-09.md` §Risks:

> The paper should credit Aysenur(s) only if their branches' techniques are integrated.

If `working_spoof_detection`'s `face_usability_gate` /
`critical_region_visibility_gate` / `hybrid_fusion_evaluator` are upstreamed
into the production session-engine architecture, both Aysenur15 and Ayşe
Gülsüm EREN must be co-authors on the resulting paper.

If only the standalone `spoof-detector` session-engine work is published
without integrating Aysenur's modules, single-author attribution to Ahmet
is appropriate (matches the actual commit log).

## Verification command

To re-verify any author claim above:

```bash
cd /path/to/biometric-processor
git log main..origin/<branch> --format='%an <%ae>' | sort -u

cd /path/to/practice-and-test
git log -- spoof-detector/ --format='%an <%ae>' | sort -u
```
