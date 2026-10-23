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

The new coefficient-one AKLT projector reference worker is implemented on
`codex/aklt-reference` in the isolated checkout, based on
`bf4d627b19f5ed785608c050efbad1be8efca400`; reviewed code is `f0e5be0`.
It shares the scheduled trajectory engine with the unchanged Ising public contract.
The strict AKLT worker supports fine/time-step cap128 variants, cooperative trajectory
budgets, cumulative cap-hit evidence, retained scalar prefixes and one native terminal
checkpoint. See the [worker contract](docs/usage.md#fixed-aklt-projector-reference-worker).
No new AKLT production trajectory has launched. Local integration remains pending;
no push, PR or hosted CI is authorized for this task.

One independent whole-branch review across this repository and Infinite_MST found one
Important Python/Rust schedule-rounding mismatch in the main Python bridge. It was fixed
there with native-wire regression evidence; no reference source change was needed.
Final local routine verification passes462 tests in38 binaries,0 failures,16 optional
ignored tests and0 warnings; doc tests select0 and pass. The debug no-run build took273.31s,
routine wall192.70s, doc-test wall0.53s. The standalone release worker build passed in290.05s.
The controlling Infinite_MST implementation record owns preparation cost, integration,
launch and recoverable cleanup; builds are under /private/tmp/imst-current-build/purification.

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

The controlling study qualified all three chi24 initial states with identical raw/admitted
hashes,0 updates and1 AD evaluation each. Obtain the required local integration choice, then
finish the stable-source supervisor preflight before the two approved AKLT trajectories from the stable
Dropbox results root. No extra bond-cap, retry or refinement study is authorized. Preserve
both task checkouts/builds until integration and remaining use, then perform only explicitly
reviewed recoverable cleanup. The earlier Ising publication/cleanup remains complete.

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
