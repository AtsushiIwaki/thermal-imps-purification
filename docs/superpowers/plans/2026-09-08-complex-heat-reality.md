# Complex heat reality investigation and repair

## Scope and acceptance

- Objective: repair the cause of the phase-twisted XX heat reality failure without relaxing any existing assertions, projecting away imaginary values, changing dependency pins, or reducing verification scope.
- Captured base: `45007b4c26bc7e13525088fa0d1a359324b8611a` on `main`.
- Pre-existing change: `scripts/smoke.py` mode 100755 to 100644; preserve it.
- Initial editable paths: this plan; `docs/investigations/2026-09-08-complex-heat-reality.md`; `tests/specific_heat.rs`; `src/itebd_complex/variance.rs`; `src/itebd_complex/canonicalize.rs`; `src/specific_heat.rs`; `STATE.md`; `docs/validation.md`; `docs/limitations.md`; `knowledge/topics/complex-evolution.md`; ignored diagnostic artifacts under `results/complex-heat-reality-20260908/`. Narrow the actual implementation to the established cause; record any scope expansion before edits.
- Caller contracts: `specific_heat_complex_with_options(&ComplexPurifiedMps, &ComplexLocalHamiltonian, f64, &SpecificHeatOptions) -> Result<SpecificHeatReport, ItebdError>`; `canonicalize_complex(&mut ComplexPurifiedMps, f64) -> Result<f64, ItebdError>`; unchanged real and automatic facades, report fields, reality and tail errors.
- Preserve physical/ancilla split, first-site-fastest indices, conjugated bra, alternating bond orientation, and independent positive/negative correlation streams. Keep products matrix-free and the pairwise contraction boundary green.
- Primary agent owns all final verification, review, evidence and cleanup. Work in the current checkout; use the existing WSL toolchain/cache for Linux execution. No pre-existing WSL checkout or ignored artifacts are owned for deletion.

## Steps

Scope expansion during diagnosis: include `src/canonicalize.rs`. The real reference uses the
same failing nalgebra PSD decomposition; correcting only the complex decomposition removes the
imaginary failure but exposes a real/complex variance discrepancy of about 5.6e-7. A 3x3
positive matrix with diagonal [1, 1e-6, 1e-3] and 1e-18 coupling reproduces factor reconstruction
error 1.41e-3. Use the already pinned tensor4all eigensolver for both paths, preserving floors,
normalization and public signatures. Evaluate SVD residuals separately before changing SVD.

The dtau=.025 control then reproduced SVD reconstruction error 6.27e-3 and parity imaginary
residual -3.22e-6 even with the corrected eigen solver. Include
`tests/fixtures/canonical-svd-xx.json` for the captured 10x10 column-major matrix and replace the
complex gauge SVD with tensor4all full-rank factorization. Full rank must retain zero singular
slots and ignore global truncation defaults. Check exact reconstruction and active-subspace
orthogonality. Retain full U, including null columns used by the unweighted canonical Gram
scalar; only zero V^H rows are safe because they become incoming zero-Schmidt Gamma rows.
The reviewed zero-support product fixture checks zero log normalization, physical cell norm,
direct periodic norm and product-state probabilities.

- [x] Reproduce the exact existing test at the captured base and retain command, environment, exit and output.
- [x] Measure parity/direction/onsite/overlap/tail values and canonical residuals. Form and test one causal hypothesis at a time; compare real, untwisted complex, and twisted complex cases.
- [x] Add a focused regression that fails on the original code and isolates the established cause. Apply the minimum correction with unchanged numerical tolerances.
- [x] Verify affected callers, independent real/complex oracles for any contraction change, canonical invariants, and relevant step/cutoff/bond/tail sensitivities and independent heat estimator.
- [x] Run the focused specific-heat tests and AST boundary audit early, then final `CARGO_INCREMENTAL=0 python3 scripts/verify.py routine`, `CARGO_INCREMENTAL=0 cargo test --locked --doc`, and Python harness tests if changed.
- [x] Review actual diff and affected callers; record evidence and qualified limitations; update STATE.md; inspect committed, unstaged, staged and untracked views and `git diff --check`. Retain only intentional source/docs and ignored evidence. Do not publish changes as part of diagnosis.
