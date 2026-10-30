# Current state

## Current position

This is the standalone two-site thermal purification/iTEBD crate, with real and
complex backends, solve CLI, checkpoint/restart, reduced-density-matrix APIs and
verification tools. It is published at
[AtsushiIwaki/thermal-imps-purification](https://github.com/AtsushiIwaki/thermal-imps-purification).
Project-authored material uses MIT; retained upstream sources keep their notices.
The [usage guide](docs/usage.md) owns current CLI/library contracts.

The approved T0.3 continuation completed668 new fine-grid steps to absolute1014,
acquiring verified states/physical/variance rows atT0.2 andT0.1. Original14 plus
prior4 plus new2 gives20/20 temperatures. Observed maximum bond146 and complete
cap coverage ended1014; cap256 was not reached. Native elapsed1297.058s and
supervised wall1305.466s are inclusive, not additive. Production kept Native
defaults (Faer/thread_hint24, visibleCPU24, five overrides absent). No old variance
pass, optimizer work, time-grid change or automatic retry ran.

New output is `results/aklt-projector-dmax256-resume-bracket-repair/fine/`, with
only two new HDF5 snapshots. Raw beta/norm and boundary-row cap evidence came
from snapshot3 atT0.3/step346 of the prior v4 continuation, which in turn pins
the original v3 fourteen-state bundle. All27 old reference/figure/matrix hashes
and the old consumed launch guard are unchanged. Native verification confirms
two checkpoints, scalar consistency and variance_remeasured=false.

The complete six-page controlling report preserves all three producer/index
sources; frozen-only regeneration matches every page stream and all six pages
were visually inspected. Maximum cap256-cap128 energy difference is6.50347e-9
atT0.8; matched C_FD difference is4.67674e-8 atT0.7. These do not certify absolute
thermal accuracy. The controlling note is Infinite_MST
`notes/2026-10-30-aklt-purification-t03-continuation.md`.

The reviewed T0.3 extension is locally integrated into main at f1e109b
(controlling Infinite_MST main d3660ee), with merged trees identical to their
verified feature branches. Production source2c7a6c0/worker
`d7d4909b9492105b9df786d49d0188e2220fc485070239532bbedf199ec87b6c`
remain recorded in conditions/HDF5. Native routine passed497 tests/42 binaries
(16 optional ignored, one known ARM64 warning); Python324 passed with zero
skips; main native HDF514 passed (4 optional exporters ignored); doctests0.
Source implementation was in ordinary external worktrees, with stable Dropbox
outputs and this same Local controlling chat. Later changes are documentation.

The earlier Faer terminal2x2 and secular-root repairs are already integrated
into local main; the main MST vendor was not changed. The prior T0.7 continuation
retained four points throughT0.3 then failed at402 with an all-nonfinite2304²
matrix. A separate diagnostic at371/call1 captured a finite963² input yielding
NaN SVD vectors and localized bracket inversion. The repaired factorization has
relative reconstruction3.93e-15 and no apparent healthy-input timing slowdown.
The new continuation succeeded; the original402 first corruption remains
unobserved. Diagnostic branches b751a9e/cd9a468 and old partial publications stay
retained. See the [repair record](docs/investigations/2026-10-29-faer-secular-bracket.md).

V1/schema1 keeps3600 seconds; v2/schema2 keeps explicit null. The old cap128 fine
reference completed20 points/1014 steps and reached cap128 at T0.1/0.2; its
step-size variant retained18 through T0.3 and failed SVD. Both are unchanged.
The earlier mixed-field Ising study and canonical repair remain as documented
in [validation](docs/validation.md) and [limitations](docs/limitations.md).

## Next actions and limits

- Local integration and merged-primary verification are complete: Native497
  passed/16 optional ignored, Python324 passed/zero skips, main native HDF514
  passed/4 optional exporters ignored; doctests discovered0. Both ordinary
  worktrees and merged feature branches are retired after SHA256 source/workflow
  preservation and unchanged36-output-hash checks. Diagnostic branches remain.
- `/private/tmp/imst-current-build` remains pending a fresh explicit normal-Trash
  review. No build removal, Trash approval or physical-space reclamation is
  inferred. Thermal time-step/cutoff convergence and original402 attribution
  remain unproved; no further trajectory is selected.
- The extra malformed-CSV-row finalizer panic remains a deferred fail-closed
  minor; it did not affect this completed run.
- Checkpoint publication is exclusive, not transactional across every format or
  arbitrary power loss. Use the documented completed-prefix recovery.
- Native HDF5 in-process Git launcher support remains scoped to documented
  macOS/GNU Linux (glibc>=2.34). Local Windows smoke evidence is not a full-suite
  platform guarantee. Hosted CI does not attest these new local numerical data.
- Release tags/registry publication remain undecided. Retained third-party notices
  do not independently establish complete binary redistribution clearance.
