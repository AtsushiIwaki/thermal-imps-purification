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

## Unresolved constraints

- The first hosted CI run passed on macOS and failed on Ubuntu during HDF5 checkpoint creation.
  The GNU/Linux process-launch repair passes focused local regressions, but a repaired hosted
  run is outstanding. Local full-suite verification also exposes a separate phase-twisted XX
  specific-heat failure, reproduced unchanged on the original commit. See the
  [investigation](docs/investigations/2026-09-08-linux-hdf5-spawn.md) for evidence and scope.
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

Publish the reviewed GNU/Linux HDF5 process-launch repair, then observe a new hosted
macOS/Linux workflow. Investigate the independently reproduced phase-twisted XX specific-heat
failure as a separate numerical task, preserving its existing assertions. Full Linux CI success
remains outstanding.

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
