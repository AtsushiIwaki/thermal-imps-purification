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

The canonical eigendecomposition/SVD repair is published on `main` at `af67529`, following
the GNU/Linux HDF5 launch repair. The [2026-09-08 hosted CI run](https://github.com/AtsushiIwaki/thermal-imps-purification/actions/runs/34237297719)
passed build and verification on both Ubuntu and macOS at that commit. The
[numerical investigation](docs/investigations/2026-09-08-complex-heat-reality.md) retains the
local Linux routine and fixed specific-heat evidence; its pending-publication and platform
statements describe the earlier verification checkpoint and are superseded by this status.

The fixed mixed-field Ising adapter now provides exact target schedules, ordinary norm-ledger
thermodynamics, exclusive compact bundles, native verification, and explicit dead-process
finalization. Scheduled checkpoints use schema2; fixed schema1 remains supported and `solve`
rejects scheduled restart. See the [worker contract](docs/usage.md#fixed-mixed-field-ising-reference-worker).
The reviewed implementation and final corrections (`57ae151`) are now integrated into
the stable local `main`; no public push or new hosted CI run was performed. Required
workflow evidence was copied and hash-verified under `.superpowers/sdd/` before the
temporary worktree was moved to normal Trash and its merged branch retired.
The controlling study subsequently completed the three approved cold reference runs
at source HEAD `da92cfb`: fine (delta_tau<=0.005, Dmax=128), time-step (0.01,128)
and bond-cap (0.005,64), cutoff 1e-13. All 60 target samples and all three native
checkpoint checks completed. Bundles are retained under
`results/mixed-field-ising-g105-h05-reference/`. Ordinary entropy/free energy
failed the 1e-5 time-step sensitivity diagnostic at every target, and fine entropy
is slightly negative at T=0.1–0.3. Fine values are adopted by explicit user choice;
this is not a physical refinement qualification. The main study records exact
outcomes in `Infinite_MST/notes/2026-10-14-mixed-field-ising-cg-fr-heating.md`.

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

The approved reference production scope is complete; no additional refinement is
scheduled. The controlling Infinite_MST study completed final report QA and reviewed
retirement of `/private/tmp/imst-current-build` to normal recoverable Trash;
rebuild information and receipts remain in that study’s result records. Production reused the reviewed
worker from `57ae151`; later documentation changes do not alter that source binding.
No public push or new hosted CI run was performed.

## Relevant links

- [README](README.md) — build, usage, model, output, and verification entry points.
- [Usage guide](docs/usage.md) — configuration, matrix input, output and restart details.
- [Library APIs](docs/library-api.md) — evolution, specific heat, checkpoints and RDM contracts.
- [Contributing](CONTRIBUTING.md) — scoped development and review procedure.
- [Development workflow](docs/development-workflow.md) — handoff, checks, evidence, and cleanup.
- [Knowledge index](knowledge/index.md) — focused numerical and engineering background.
- [Numerical conventions](docs/numerical-conventions.md) — authoritative index and normalization contracts.
- [Validation](docs/validation.md) — measured evidence and exact scientific commands.
- [Limitations](docs/limitations.md) — scientific, resource, persistence, and platform bounds.
- [Provenance](docs/provenance.md) — extraction, dependencies, and later adaptation history.
- [License](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) — project and upstream terms.
- [References](docs/references.md) — upstream software and method sources.
