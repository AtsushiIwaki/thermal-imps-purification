# Current state

## Current position

This repository is a standalone two-site thermal purification/iTEBD crate with real and complex
backends, a solve CLI, checkpoint/restart and reduced-density-matrix APIs, examples, verification
tools, and a macOS/Linux CI configuration. It is published at
[AtsushiIwaki/thermal-imps-purification](https://github.com/AtsushiIwaki/thermal-imps-purification).
Its development guidance and selected
knowledge pages are self-contained within this checkout. The authoritative numerical contracts,
measured evidence, and qualified limits remain in the linked public documents.
MIT is applied to project-authored material; copied upstream material retains its own terms.
The README is a short finite-temperature quickstart; advanced CLI and library details live in
the linked usage and API guides.

The approved cap256 extension is implemented on the ordinary external worktree
`/private/tmp/thermal-purification-worktree`, branch
`codex/aklt-purification-dmax256`; the controlling chat remains in Infinite_MST
Dropbox Local. V3 retains20 target states in one native HDF5, evaluates variance
heat with the existing defaults and preserves physical values on heat failure.
It reuses the old fine1014-step grid without a deadline, and records native
thread defaults independently from single-thread software checks.
See the [v3 usage contract](docs/usage.md#aklt-cap256-target-snapshots-and-variance-heat).
The routine gate passed485 tests across40 binaries,16 optional ignored,
0 failures and0 warnings, wall231.7s including incremental compilation.
Final release/controlling Python gates and fresh review are still in progress.
No new thermal trajectory has started; integration remains a separate choice.

V1/schema1 retains its3600-second contract; v2/schema2 retains explicit null.
The old cap128 fine run completed20 targets/1014 steps in1096.053s, reached
cap128 at T=0.1/0.2 and retained a verified endpoint. Its time-step variant
retained18 targets through T=0.3, then failed in SVD without a certified endpoint.
That partial outcome and all old bytes remain unchanged; no retry or old endpoint
variance evaluation is part of the new task.

The published canonical repair and mixed-field Ising worker remain the baseline.
The [successful hosted follow-up](https://github.com/AtsushiIwaki/thermal-imps-purification/actions/runs/38651815754)
verified the prior code at `e063778`, not this AKLT change. The completed mixed-field
study retains three cold reference bundles under results/mixed-field-ising-g105-h05-reference/.
Its fine values were adopted by explicit user choice despite time-step differences and
slightly negative low-temperature entropy; that is not refinement qualification.
The controlling study retains its detailed numerical and cleanup evidence.

## Unresolved constraints

- Release tags and registry publication remain undecided.
- AKLT validation covers a measured finite temperature window with active bond caps; it does not
  establish low-temperature or bond-dimension convergence. The validation and limitations pages
  retain the other model-specific numerical bounds.
- JSON and HDF5 outputs use exclusive creation. Checkpoint publication is independent rather than
  transactional across formats and does not guarantee recovery after an arbitrary process or
  power loss. Use fresh output destinations.
- HDF5 launch coordination on macOS and GNU/Linux is scoped to the documented in-process Git
  launcher and the pinned shared mutex. GNU/Linux requires glibc >=2.34. Windows has local
  build/smoke evidence; this does not establish full-suite coverage on every platform or behavior
  for independent launchers.
- Documentation-only verification does not establish new physics, fresh full-suite success, or
  hosted CI coverage.
- Selected source notices do not establish complete legal clearance for a built binary artifact.

## Next action

Complete software verification and one fresh review, select integration for both
repositories, then bind stable sources and the exact release worker for the single
cap256 operation. The task-owned build root is `/private/tmp/imst-current-build`.
Preserve workflow records and results before ordinary worktree retirement; build
Trash review and cleanup remain. No build/worktree has been retired in this task.
The time-step SVD root cause and low-temperature time-step/cap convergence remain
unresolved. Registry publication and additional numerical studies remain unselected.
