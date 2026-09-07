#[derive(Debug, thiserror::Error)]
pub enum ItebdError {
    #[error("invalid run configuration: {0}")]
    InvalidRunConfig(String),
    #[error("{field} must be square, got {rows}x{cols}")]
    NonSquare {
        field: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "Hamiltonian matrix dimensions do not match: two_site={two_site}, site_energy={site_energy}"
    )]
    MatrixDimensionMismatch { two_site: usize, site_energy: usize },
    #[error("two-site dimension {matrix_dim} is not a positive perfect square")]
    InvalidPhysicalDimension { matrix_dim: usize },
    #[error("{field} contains a non-finite value at ({row}, {column})")]
    NonFiniteMatrix {
        field: &'static str,
        row: usize,
        column: usize,
    },
    #[error("invalid {name}: expected finite non-negative value, got {value}")]
    InvalidTolerance { name: &'static str, value: f64 },
    #[error("invalid imaginary-time step: expected a finite value, got {value}")]
    InvalidTimeStep { value: f64 },
    #[error("{field} is not Hermitian: residual {residual} exceeds tolerance {tolerance}")]
    NonHermitian {
        field: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error("Hamiltonian eigendecomposition produced a non-finite eigenvalue at index {index}")]
    NonFiniteEigenvalue { index: usize },
    #[error("Trotter gate contains a non-finite value at ({row}, {column})")]
    NonFiniteGate { row: usize, column: usize },
    #[error("tensor operation failed at {stage}: {message}")]
    TensorOperation {
        stage: &'static str,
        message: String,
    },
    #[error("zero Schmidt norm at {stage}")]
    ZeroSchmidtNorm { stage: &'static str },
    #[error("{observable} expectation has imaginary part {imaginary} above tolerance {tolerance}")]
    NonRealObservable {
        observable: &'static str,
        imaginary: f64,
        tolerance: f64,
    },
    #[error("iTEBD backend mismatch: state={state}, Hamiltonian={hamiltonian}")]
    BackendMismatch {
        state: &'static str,
        hamiltonian: &'static str,
    },
    #[error("invalid specific-heat beta: expected a finite non-negative value, got {value}")]
    InvalidSpecificHeatBeta { value: f64 },
    #[error("invalid specific-heat {name}: expected at least {minimum}, got {value}")]
    InvalidSpecificHeatCount {
        name: &'static str,
        value: usize,
        minimum: usize,
    },
    #[error(
        "non-finite specific-heat value at {stage}, parity {parity}, distance {distance}: {real}+{imaginary}i"
    )]
    NonFiniteSpecificHeatValue {
        stage: &'static str,
        parity: usize,
        distance: usize,
        real: f64,
        imaginary: f64,
    },
    #[error(
        "specific-heat direction mismatch at parity {parity}, distance {distance}: residual {residual} exceeds tolerance {tolerance}"
    )]
    SpecificHeatDirectionMismatch {
        parity: usize,
        distance: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error("specific-heat {stage} has imaginary part {imaginary} above tolerance {tolerance}")]
    NonRealSpecificHeat {
        stage: &'static str,
        imaginary: f64,
        tolerance: f64,
    },
    #[error(
        "specific-heat tail did not converge by distance {max_distance}: positive={last_positive}, negative={last_negative}, consecutive small shells required={consecutive_small_shells}"
    )]
    SpecificHeatTailNonConvergence {
        max_distance: usize,
        last_positive: f64,
        last_negative: f64,
        consecutive_small_shells: usize,
    },
    #[error("energy variance per site {value} is negative beyond tolerance {tolerance}")]
    NegativeEnergyVariance { value: f64, tolerance: f64 },
}
