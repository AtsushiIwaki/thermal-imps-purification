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

The approved Dense AKLT continuation adds `aklt-projector-reference-v2`/schema2
with required `max_seconds:null`; v1/schema1 retains its exact3600-second contract.
The cap128 fine/time-step variants keep cutoff1e-13, max_dtau0.005/0.01 and
1014/507 complete Strang steps. Both share the existing trajectory engine; Ising
behavior is unchanged. Native retention and verification preserve contract version,
cap history, actual stop reasons and uncertainty after mutating failure/interruption.
See the [worker contract](docs/usage.md#aklt-reference-without-a-deadline).

Final qualification passed466 routine tests across38 binaries (16 optional tests
unselected,0 failures,0 warnings),0 selected doc tests and6 release native CLI
tests, with0 required skips. Routine wall224.5s includes debug compilation and
execution; doc0.496s, release build22.067s and release CLI26.203s (0.74s execution)
were recorded separately. The existing contraction AST audit required lifting
three JSON `.contract` field reads outside macros; the audit itself is unchanged.
One fresh whole-branch review found two Important frozen-report consistency gaps
in the controlling Infinite_MST Python code; both were fixed by RED-GREEN tests
and its full affected Python gate.
No deferred minor finding remains. The single controlling operation ended on
2026-10-27 JST. Fine purification completed all20 targets/1014 steps in1096.053s;
its checkpoint and terminal consistency were natively verified. Cap128 was reached
at T=0.1/0.2. The time-step variant retained18 targets through T=0.3, then exited101
on an SVD decomposition error at src/tensor.rs:143. Native finalization retained
its actual reason, T=0.1/0.2 gaps and no certified endpoint. Its173 retained steps
are a lower bound; maximum observed dimension120 does not establish that the cap
was never reached in the unobserved tail. No retry occurred. The controlling
study remains completed=false despite all60 thermal endpoints being verified.
See [the observed limits](docs/limitations.md#new-aklt-reference-adapter-2026-10-24-implementation).
The original bounded studies and their unattempted references are unchanged.

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

The approved single continuation attempt and its report are finished. A separate
scoped investigation is needed for the time-step SVD failure; the exact failing
matrix/backtrace was not retained, so no common root cause with earlier Faer
failures is established. Low-temperature time-step and cap convergence remain
unresolved. Launch bindings retain revision b61ae0f and the exact tested worker;
closing documentation does not change the numerical source or rebuild it.
After explicit review on 2026-10-27, the shared /private/tmp/imst-current-build
parent (including purification targets) was moved to normal recoverable Trash.
The original path is absent, the Trash inventory and tested worker hash match,
and the Trash was not emptied. Infinite_MST retains the cleanup receipt, path
mapping, rebuild record and successful strict inventory; retained reference
results and source remain here. After a Local handoff on 2026-10-28, Codex
archived the controlling Infinite_MST worktree with recovery available; its
directory and Git registration are absent. Git branches remain retained.
Future native execution requires rebuilding. Registry publication, extra cap,
retry or numerical refinement remains unselected.
