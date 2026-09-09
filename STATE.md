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

Select the next bounded task from the documented numerical limitations or the release
and registry-publication decisions. Publication and hosted macOS/Linux verification of
the canonical matrix-decomposition repair are complete; no next implementation task is selected.

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
