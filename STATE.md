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
operation within that production full step remains unobserved.

The approved secular-root bracket repair is implemented on branch
codex/faer-secular-bracket-fix, pending integration. It rejects invalid secant
candidates before use and preserves the fallback bracket. An independent small
polynomial-root regression failed before the change and passes after it; prior
terminal2x2 regressions also pass. The captured963x963 matrix is now finite and
accurate at controlled24-way Faer parallelism (relative reconstruction3.93e-15).
ABBA before/after timing found healthy-input median changes−0.23% at1 and−1.11%
at24, with overlapping IQRs; no significant speedup or thermal-time claim follows.
The old captured24 result is invalid; its timing cannot certify a speed ratio.
See the [repair record](docs/investigations/2026-10-29-faer-secular-bracket.md).

The diagnostic root trace identified out-of-range candidates reversing the
fallback bracket, skipped bisection, a below-pole root and negative sqrt. It
first trapped diagnostic371/call1, distinct from the unobserved corrupting
operation in production402. Diagnostic branches b751a9e/cd9a468 remain retained;
no diagnostic instrumentation is merged into the repair. No new thermal
trajectory, variance measurement or reference publication ran in the repair.

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

- Integrate the reviewed root-bracket repair after final gates and the user
  selects integration. Final gates passed495 Native/42 binaries (16 optional
  ignored), six vendor plus three selected SVD checks and14 main native HDF5
  checks (4 optional exporters ignored); doctests discovered0. Original402 attribution and cold cap256 completion remain
  open; no further trajectory is selected. Bond/time-step convergence is unproved.
- Both integrated ordinary worktrees/registrations and repair branches are retired
  with unchanged retained hashes. The earlier diagnostic branch b751a9e remains.
  Ordinary `/private/tmp/thermal-purification-nonfinite-worktree` now hosts the
  repair branch and awaits integration/disposition, with no production merge. Task build `/private/tmp/imst-current-build`
  was reactivated for diagnosis; its pending Trash manifest is superseded with no
  actionable rows. Fresh review is needed before Trash.
- Checkpoint publication is exclusive, not transactional across every format or
  arbitrary power loss. Use the documented completed-prefix recovery.
- Native HDF5 in-process Git launcher support remains scoped to documented
  macOS/GNU Linux (glibc>=2.34). Local Windows smoke evidence is not a full-suite
  platform guarantee. Hosted CI does not attest these new local numerical data.
- Release tags/registry publication remain undecided. Retained third-party notices
  do not independently establish complete binary redistribution clearance.
