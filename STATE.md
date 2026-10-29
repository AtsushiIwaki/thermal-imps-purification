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

The cap256 extension is integrated into local main. The single authorized run
retained14 verified snapshots/physical/variance measurements (T=2.0,...,0.7),
then exited101 in `src/tensor.rs:143` with an SVD decomposition failure.
T=0.6,...,0.1 remains unavailable; no retry or fallback ran. The native retained
progress154steps/beta1.4285714285714268 is a lower bound, with unobserved cap
tail and unknown trajectory-only time. Retained maximum dimension68 is not a
certificate for the failed interval. The exact failing step/matrix is unavailable;
a common root cause with prior SVD failures is unproved.

The four-file bundle is retained at
`results/aklt-projector-dmax256-reference/fine/`; states.h5 holds14 snapshots.
Interrupted finalization verified all14 associations and physical values without
repeating variance sums. The controlling Infinite_MST report is published with
reference_status=partial. This is not a full-temperature cap256 reference.
Faer/native defaults were preserved (visibleCPU24/thread_hint24; five thread
overrides absent). The controlling chat remains Dropbox Local Infinite_MST.
See the [v3 contract](docs/usage.md#aklt-cap256-target-snapshots-and-variance-heat).
Post-integration verification passed488 native tests across40 binaries,
16 optional ignored,0 failures/warnings, wall251.8s including compile/setup.
The controlling314 Python and14 main native HDF5 tests passed. The single fresh
review's two Important integrity findings were fixed by RED→GREEN tests.

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

A separately scoped investigation is needed for the cap256 SVD failure before
any new attempt. Cold cap256 values and low-temperature cap/time-step convergence
remain unresolved. The ordinary task worktree/branch were retired with Git without force after
verified integration and SHA-256 preservation; directory/registration are absent. `/private/tmp/imst-current-build` awaits explicit
normal-Trash review; required source, vendor patches, locks and terminal results
remain. No extra calculation, retry or registry publication is selected.
