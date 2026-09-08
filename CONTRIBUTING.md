# Contributing

Use the [README](README.md) for setup and runnable examples. Before changing numerical code, read
the authoritative [numerical conventions](docs/numerical-conventions.md),
[validation record and commands](docs/validation.md), and [known limitations](docs/limitations.md).
Repository-specific engineering rules are in [AGENTS.md](AGENTS.md).
Contributions to project-authored code, documentation, test harnesses, and knowledge material are
accepted under the repository's [MIT License](LICENSE). Preserve applicable upstream notices and
license texts described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and keep scientific
references current in [docs/references.md](docs/references.md) when dependency or method use changes.
The current standalone position and unresolved constraints are in [STATE.md](STATE.md); use the
[knowledge index](knowledge/index.md) to find focused background for the part of the system being
changed.

For substantial work, record the objective, exact path scope, captured base, pre-existing changes,
caller contracts, early checks, completion evidence, and cleanup owner. The
[development workflow](docs/development-workflow.md) gives the Git and verification procedure.
Use the [task handoff template](docs/superpowers/templates/task-handoff.md) when transferring an
implementation task and the
[investigation-note template](docs/superpowers/templates/investigation-note.md) for unresolved or
evidence-driven diagnosis. Replace every labeled prompt before publishing an instantiated document.

Run commands from the repository root. Python harness changes start with:

```sh
python3 -m unittest discover -s scripts/tests
```

The ordinary full Rust verification is:

```sh
CARGO_INCREMENTAL=0 python3 scripts/verify.py routine
CARGO_INCREMENTAL=0 cargo test --doc
```

The wrapper retains the full `cargo test --all-targets` scope and fails on unknown project
warnings. Select focused tests for the interface being changed and confirm that filters select a
nonzero number of tests. Scientific and ignored release checks are separate; run only the exact
drivers identified in [validation](docs/validation.md), with fresh output paths where required.

Before committing, inspect all four Git views described in the workflow, review newly created
files directly, run `git diff --check`, and stage explicit paths only. Report what each check
establishes and what it does not establish.
