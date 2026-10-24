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

The coefficient-one AKLT projector reference worker is integrated into local main
by the user-approved fast-forward to `b764331` on2026-10-24 JST. Its base is
`bf4d627b19f5ed785608c050efbad1be8efca400`; reviewed code is `f0e5be0`.
It shares the scheduled trajectory engine with the unchanged Ising public contract.
The strict AKLT worker supports fine/time-step cap128 variants, cooperative trajectory
budgets, cumulative cap-hit evidence, retained scalar prefixes and one native terminal
checkpoint. See the [worker contract](docs/usage.md#fixed-aklt-projector-reference-worker).
The controlling Infinite_MST study passed stable-source preflight, then its single
bounded run stopped after20 thermal endpoints (m1/m2 through T0.2; m3 through T1.6).
Its shared verification/publication/report scope reached3972.609327127s against3600s,
including372.609327127s overrun to finish the in-flight invocation. Both AKLT reference
trajectories were unattempted. No new reference samples/checkpoints, time-step differences
or cap-hit evidence exist for this study. Infinite_MST retains a six-page report with
explicit reference gaps; `completed=false` and no integrity debt. No retry, changed
budget, push, PR or hosted CI was performed.

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

The controlling study has preserved its bounded outcome and verified frozen-report
regeneration/visuals. Cleanup review and final record commits remain. Task builds and
worktrees are retained pending explicit review. Completing the two unattempted AKLT
trajectories now requires a separately authorized scope/budget decision; the prior
controller refuses automatic restart. No additional bond-cap, retry, refinement or
algorithm change is authorized. The earlier Ising publication/cleanup remains complete.

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
