# Limitations of the numerical evidence

The checked-in quickstart and smoke cases verify usability and finite output only. They are not
convergence measurements.

At `J=1`, `g=0.7`, `beta=0.2`, `dtau=0.05`, cutoff `1e-12`, cap 32, and canonicalization every
step, independently evolved real and phase-rotated TFIM runs showed a free-energy difference of
`8.9327509389391935e-8` for second order (`2.4726140735436109e-8` on the recorded scaled metric).
Their energy, specific heat, and local-observable differences were at most
`4.4408920985006262e-16`. The normalization cause of the free-energy discrepancy is unresolved;
fresh complex and real gauges or UUIDs are not expected to match.

The coefficient-one AKLT projector comparison covers a finite grid ending at projector
`beta=0.4`. Bond cap 32 was active at the `dtau=0.05`, cutoff `1e-12`, `beta=0.4` endpoint. This
does not establish cap convergence or approach to the ground state. Its `beta=0.01` control
supports only the high-temperature limit.

The complex-Hermitian evidence is for stated phase-rotated TFIM and phase-twisted XX controls.
The earlier finest TFIM energy refinement window was excluded because its observed order fell
to about `0.829`; that energy qualification has not been rerun for the canonical decomposition
repair. The [2026-09-08 heat investigation](investigations/2026-09-08-complex-heat-reality.md)
resolves the reproduced XX reality failures: all 28 rows in the fixed heat driver return valid
reports after the repair. Its selected `.1 -> .05` heat windows pass the existing criteria,
but the `.025 -> .0125` windows remain rejected (orders about `1.511` for TFIM and `.756`
for XX, below `1.6`). The grid is limited to beta1 TFIM and beta.6 XX with infinite-temperature
initial states, canonicalization every step, and the stated cutoff/cap/tail controls. It does
not establish arbitrary-model, low-temperature or off-canonical convergence, or sensitivity
to other initial states. No tolerance was relaxed.

Interval RDMs are approximate fixed-point reconstructions. Degenerate dominant transfer sectors
can make identity-seeded boundaries sector-dependent even when residual checks pass. Resource
caps bound planned tensor/output elements and occurrence-index metadata; they exclude aggregate
allocations, metadata storage, hidden backend workspace, and dependency allocation that may occur
while rejecting corrupted underreported dimensions. No clipping or Hermitian projection repairs
an RDM that fails positivity or Hermiticity checks.

JSON and HDF5 outputs are separate publications. A crash can leave one ahead of the other, and an
HDF5 completion marker plus flush is not a filesystem transaction or power-loss guarantee.
Concurrent writers to the same destination are unsupported. Scientific use for a new model or
regime requires independent references or limiting cases, step-size refinement, cutoff and bond
cap sensitivity, canonical checks, and model-specific invariants.
