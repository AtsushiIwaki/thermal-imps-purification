# Current state

## Current position

This repository is a standalone two-site thermal purification/iTEBD crate with real and complex
backends, a solve CLI, checkpoint/restart and reduced-density-matrix APIs, examples, verification
tools, and a prepared macOS/Linux CI configuration. Its development guidance and selected
knowledge pages are self-contained within this checkout. The authoritative numerical contracts,
measured evidence, and qualified limits remain in the linked public documents.
MIT is applied to project-authored material; copied upstream material retains its own terms.

## Unresolved constraints

- A repository owner/remote, release tags, and registry publication remain undecided. The prepared
  workflow has not run on GitHub, and Linux execution is unverified.
- AKLT validation covers a measured finite temperature window with active bond caps; it does not
  establish low-temperature or bond-dimension convergence. The validation and limitations pages
  retain the other model-specific numerical bounds.
- JSON and HDF5 outputs use exclusive creation. Checkpoint publication is independent rather than
  transactional across formats and does not guarantee recovery after an arbitrary process or
  power loss. Use fresh output destinations.
- The macOS HDF5 launch coordination is scoped to the documented in-process Git launcher and the
  pinned shared mutex. It does not establish behavior for every platform or independent launcher.
- Documentation-only verification does not establish new physics, fresh full-suite success, or
  hosted CI coverage.
- Selected source notices do not establish complete legal clearance for a built binary artifact.

## Next action

For repository development, choose a bounded task through the contributor workflow and validate
the affected interface. For publication, select the hosting destination, configure a
remote, and observe the hosted macOS/Linux workflow before claiming that platform coverage.

## Relevant links

- [README](README.md) — build, usage, model, output, and verification entry points.
- [Contributing](CONTRIBUTING.md) — scoped development and review procedure.
- [Development workflow](docs/development-workflow.md) — handoff, checks, evidence, and cleanup.
- [Knowledge index](knowledge/index.md) — focused numerical and engineering background.
- [Numerical conventions](docs/numerical-conventions.md) — authoritative index and normalization contracts.
- [Validation](docs/validation.md) — measured evidence and exact scientific commands.
- [Limitations](docs/limitations.md) — scientific, resource, persistence, and platform bounds.
- [Provenance](docs/provenance.md) — extraction, dependencies, and later adaptation history.
- [License](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) — project and upstream terms.
- [References](docs/references.md) — upstream software and method sources.
