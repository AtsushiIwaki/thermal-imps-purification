# Complex evolution

The [automatic facade](../../src/itebd_auto.rs) validates both Hamiltonian matrices before state
creation. Only exactly zero imaginary components, including signed zero, select real storage;
any nonzero component selects complex storage. A complex observable alone does not select the
state backend. Stored backend identity survives checkpoints, including complex storage containing
all-real values. Crossed state/Hamiltonian variants return a typed backend mismatch.

[Hamiltonian validation](../../src/itebd_complex/hamiltonian.rs) checks shape, finiteness and
scaled Frobenius Hermiticity. It does not silently symmetrize input. Complex contractions conjugate
bra legs, and Hermitian observables must produce an expectation with an acceptable imaginary
residual. The [public conventions](../../docs/numerical-conventions.md) define these contracts;
[complex API tests](../../tests/itebd_complex.rs) cover exact-zero classification, exact-real
compatibility, explicit Strang evolution, canonicalization and typed Hermiticity/reality failures.
These are implementation coverage, not evidence that arbitrary complex models converge.

The [complex variance estimator](../../src/itebd_complex/variance.rs) contracts positive and
negative correlation streams independently for both parities, including overlap and tail terms.
It does not reconstruct the negative direction from a conjugated positive result. Paired shells
are reality-checked before later shells can cancel their imaginary residual. Failure of the
bidirectional tail criterion returns `SpecificHeatTailNonConvergence`; the scalar API does not
return a partial heat estimate. The source contains independent direction oracles and mutation
guards; the [heat validation driver](../../tests/specific_heat_benchmark.rs) also compares
variance heat with symmetric energy derivatives and checks tail sensitivity.

The [published matrix CLI refinement](../../docs/validation.md#phase-tfim-matrix-cli-refinement)
uses consistently rotated Hamiltonian and observable matrices and compares direct evolution with
TOML/JSON CLI paths. Consult its finite grid and reference-resolution diagnostics for measured
claims. Unitary equivalence in these controls does not qualify a general complex Hamiltonian.

The [limitations](../../docs/limitations.md) retain the excluded finest TFIM energy window,
phase-twisted-XX heat reality failures and absence of off-canonical qualification. They also
record the beta=.2 real/phase free-energy difference despite much closer agreement of other
observables. Its normalization cause remains unresolved; these pages neither relax tolerances
nor diagnose it. For basis, beta advancement and new-input versus legacy-order defaults, use
[numerical conventions](../../docs/numerical-conventions.md).
