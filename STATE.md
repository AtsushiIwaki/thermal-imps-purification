# Current state

## Current position

This is the standalone two-site thermal purification/iTEBD crate, with real and
complex backends, solve CLI, checkpoint/restart, reduced-density-matrix APIs and
verification tools. It is published at
[AtsushiIwaki/thermal-imps-purification](https://github.com/AtsushiIwaki/thermal-imps-purification).
Project-authored material uses MIT; retained upstream sources keep their notices.
The [usage guide](docs/usage.md) owns current CLI/library contracts.

The Faer terminal2x2 SVD repair and v4 saved-MPS continuation are integrated into
local main. The single continuation from parent snapshot13/T0.7/step154 retained
four new HDF5/physical/variance targets T=0.6,0.5,0.4,0.3, then failed at fullstep402,
third SVD (beta_before_step3.878968253968235, temperature context~0.2578).
The captured2304x2304 matrix has5,308,416 nonfinite entries. The first corrupting
operation within that full step is unobserved. This is not proof of another
finite-input Faer defect or the uncaptured original failure's root cause.

Original14 plus new4 gives18/20 available temperatures; T0.2/T0.1 remains missing.
Production preserved Native defaults (visibleCPU24/thread_hint24; five overrides
absent), original fine grid, Dmax256/cutoff1e-13 and no elapsed cap. No retry or
extra variance evaluation ran. Original producer f43b496/workerf877e84 is retained
in conditions/HDF5; source d6dbce7 fixes only interrupted finalization.

Native output is `results/aklt-projector-dmax256-resume-svd-repair/fine/`, with
four new snapshots and a partial summary at retained_lower_bound346; its sibling
`svd-failure.h5` is diagnostic only. Retained cap maximum102/observed346 has an
unobserved tail and does not certify the failed interval. Original parent bundle
`results/aklt-projector-dmax256-reference/fine/` and its14 states are unchanged.
A compatible reader recovered the new four completed rows and states unchanged,
without variance remeasurement. Exact untouched pending placeholders inherit
retained progress/cap bounds; changed seeds are rejected and CSV-lag recovery
cannot invent physical/heat measurements or cap coverage.

The controlling Infinite_MST report is published with18 available points and two
gaps; all six one-axes pages were inspected and frozen-only regeneration matches
page content streams. Its original launch marker stays consumed/unknown after
failed original finalization; separate rescue records describe partial publication.
Do not reset the guard or rerun production for report recovery. The controlling
note is `notes/2026-10-29-aklt-purification-svd-resume.md`, with ignored records
in `codes/imst_rs/results/records/aklt-purification-dmax256-resume-svd-repair/`.

Software gates passed495 Native tests/42 binaries (16 optional ignored),320
controlling Python tests,14 main HDF5 tests (4 optional exporters ignored),54
tooling tests and selected vendor SVD5+3. One exact known upstream ARM64 warning
remains; no unknown Native warnings. Doctests discovered0. Final rescue Native
wall250.194s includes compilation/setup; compatible release rebuild25.518s.
Fresh review found no Critical/Important issue. An extra malformed CSV row can
panic before typed row-count validation but before publication (deferred minor).

V1/schema1 keeps3600 seconds; v2/schema2 keeps explicit null. The old cap128 fine
reference completed20 points/1014 steps and reached cap128 at T0.1/0.2; its
step-size variant retained18 through T0.3 and failed SVD. Both are unchanged.
The earlier mixed-field Ising study and canonical repair remain as documented
in [validation](docs/validation.md) and [limitations](docs/limitations.md).

## Next actions and limits

- Cold cap256 completion and the first nonfinite operation require a separately
  approved diagnostic scope. No new trajectory is authorized. Bond/time-step
  convergence remains unestablished.
- Both integrated ordinary worktrees/registrations and repair branches are retired
  with unchanged retained hashes. The earlier diagnostic branch b751a9e remains.
  Task build `/private/tmp/imst-current-build` awaits fresh normal-Trash review;
  its old manifest is superseded.
- Checkpoint publication is exclusive, not transactional across every format or
  arbitrary power loss. Use the documented completed-prefix recovery.
- Native HDF5 in-process Git launcher support remains scoped to documented
  macOS/GNU Linux (glibc>=2.34). Local Windows smoke evidence is not a full-suite
  platform guarantee. Hosted CI does not attest these new local numerical data.
- Release tags/registry publication remain undecided. Retained third-party notices
  do not independently establish complete binary redistribution clearance.
