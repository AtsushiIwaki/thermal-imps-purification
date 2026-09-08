# Verification policy

Use `CARGO_INCREMENTAL=0 python3 scripts/verify.py routine` for ordinary full routine verification,
as specified in the [development workflow](../../docs/development-workflow.md). The
[wrapper](../../scripts/verify.py) runs the fixed command `cargo test --all-targets --color never`.
It summarizes successful output without reducing target or test scope. A nonzero Cargo exit
retains its status and full diagnostic output; a successful Cargo run with an unknown project
warning fails routine verification. Invalid baseline/input or unparseable successful output also
fails rather than being reported as a passing run.

The public [warning baseline](../../scripts/verify_known_warnings.json) is empty. An intentional
future entry must identify the exact lint, normalized message and project-relative path. Review
and fix warnings where practical; do not add entries merely to make a check pass. Missing known
warnings are review candidates, while external/location-less warnings are reported separately.
The [wrapper tests](../../scripts/tests/test_verify.py) cover parsing, warning classification,
fixed scope and failure reporting.

The routine command does not replace documentation tests, ignored release benchmarks, scientific
evidence or task-specific checks. Run those directly when required, using `CARGO_INCREMENTAL=0`
for Cargo. After wrapper failure, use focused direct Cargo commands to diagnose the affected
target. For documentation-only additions, scope/link/privacy/whitespace review can be the
appropriate fresh evidence without claiming a fresh Rust or Python suite result.

[Validation](../../docs/validation.md) owns numerical and build measurements;
[numerical validation guidance](../topics/numerical-validation.md) explains how to qualify new
scientific claims. Test listings and documentation checks do not establish fresh physics,
Linux coverage or hosted-CI success.
