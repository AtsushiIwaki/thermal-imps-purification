#[path = "support/solve_matrix.rs"]
mod support;

use thermal_imps_purification::config::{
    MatrixInput, MatrixModelInput, ModelSpec, ObservableInput, RunConfig, TrotterOrder,
};
use thermal_imps_purification::itebd_auto::ItebdHamiltonian;
use thermal_imps_purification::itebd_error::ItebdError;
use thermal_imps_purification::runner::{read_result, run_sweep, write_result, Metadata, SweepResult};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct TrackingAllocator;

thread_local! {
    static TRACK_LARGE_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static LARGE_ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

const DENSE_SIZED_ALLOCATION: usize = 32 * 1024;

fn record_large_allocation(size: usize) {
    let tracking = TRACK_LARGE_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
    if tracking && size >= DENSE_SIZED_ALLOCATION {
        let _ = LARGE_ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_large_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_large_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_large_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn track_large_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    LARGE_ALLOCATION_COUNT.with(|count| count.set(0));
    TRACK_LARGE_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = operation();
    TRACK_LARGE_ALLOCATIONS.with(|tracking| tracking.set(false));
    let count = LARGE_ALLOCATION_COUNT.with(Cell::get);
    (result, count)
}

fn zeros(n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; n]
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = zeros(n);
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    matrix
}

fn error_text(error: ItebdError) -> String {
    error.to_string()
}

#[test]
fn json_matrix_dispatches_complex() {
    let input = support::phase_config();
    let cfg = RunConfig::from_json_str(&input.to_string()).unwrap();
    let resolved = cfg.model.resolve().unwrap();
    let ItebdHamiltonian::Complex(h) = resolved.hamiltonian else {
        panic!("nonzero imaginary Hamiltonian must select complex storage")
    };
    assert_eq!(h.site_energy()[(0, 1)], Complex64::new(0.0, 0.7));
    assert_eq!(h.site_energy()[(1, 0)], Complex64::new(0.0, -0.7));
    assert_eq!(resolved.observable[(0, 1)], Complex64::new(0.0, -1.0));
    assert_eq!(resolved.observable_name, "sigma_y");
}

#[test]
fn row_arrays_do_not_transpose_or_conjugate() {
    let mut input = support::phase_config();
    let mut h_real = zeros(9);
    let mut h_imag = zeros(9);
    h_real[0][0] = -2.0;
    h_imag[1][7] = 0.25;
    h_imag[7][1] = -0.25;
    input["model"]["local_dim"] = json!(3);
    input["model"]["two_site_h"] = json!({"real": h_real, "imag": h_imag});
    input["model"]["site_energy"] = json!({"real": identity(9)});
    input["model"]["observable"] =
        json!({"name": "qutrit projector", "matrix": {"real": identity(3)}});

    let cfg = RunConfig::from_json_str(&input.to_string()).unwrap();
    let resolved = cfg.model.resolve().unwrap();
    let ItebdHamiltonian::Complex(h) = resolved.hamiltonian else {
        panic!("imaginary off-diagonal pair must select complex storage")
    };
    assert_eq!(h.dim(), 3);
    assert_eq!(h.two_site_h()[(1, 7)], Complex64::new(0.0, 0.25));
    assert_eq!(h.two_site_h()[(7, 1)], Complex64::new(0.0, -0.25));
}

#[test]
fn rejects_missing_matrix_fields_versions_basis_unknown_and_mixed_keys() {
    for key in [
        "version",
        "local_dim",
        "basis_order",
        "two_site_h",
        "site_energy",
        "observable",
    ] {
        let mut input = support::phase_config();
        input["model"].as_object_mut().unwrap().remove(key);
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains(key), "missing {key}: {error}");
    }
    for version in [0, 2] {
        let mut input = support::phase_config();
        input["model"]["version"] = json!(version);
        assert!(RunConfig::from_json_str(&input.to_string())
            .unwrap_err()
            .contains("version"));
    }
    let mut wrong_basis = support::phase_config();
    wrong_basis["model"]["basis_order"] = json!("second_site_fastest");
    assert!(RunConfig::from_json_str(&wrong_basis.to_string())
        .unwrap_err()
        .contains("basis_order"));

    for (key, value) in [("surprise", json!(true)), ("j", json!(1.0))] {
        let mut input = support::phase_config();
        input["model"][key] = value;
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(
            error.contains(key),
            "accepted/reported wrong key {key}: {error}"
        );
    }
}

#[test]
fn rejects_null_preset_only_keys_by_presence() {
    let mut accepted = Vec::new();
    for key in ["j", "g", "gamma", "h", "j1", "j2"] {
        let mut input = support::phase_config();
        input["model"][key] = Value::Null;
        match RunConfig::from_json_str(&input.to_string()) {
            Ok(_) => accepted.push(key),
            Err(error) => assert!(
                error.contains(key) && error.contains("preset-only key"),
                "reported wrong null preset key {key}: {error}"
            ),
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted null preset keys: {accepted:?}"
    );
}

#[test]
fn rejects_zero_overflow_and_malformed_matrix_shapes_before_conversion() {
    let mut zero = support::phase_config();
    zero["model"]["local_dim"] = json!(0);
    assert!(RunConfig::from_json_str(&zero.to_string())
        .unwrap_err()
        .contains("local_dim"));

    let mut overflow = support::phase_config();
    overflow["model"]["local_dim"] = json!(u64::MAX);
    assert!(RunConfig::from_json_str(&overflow.to_string()).is_err());

    let cases: Vec<(&str, Value)> = vec![
        ("real row count", json!([[1., 0.], [0., 1.]])),
        (
            "ragged real row",
            json!([
                [1., 0., 0., 0.],
                [0., 1.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]),
        ),
    ];
    for (name, matrix) in cases {
        let mut input = support::phase_config();
        input["model"]["two_site_h"]["real"] = matrix;
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains("model.two_site_h.real"), "{name}: {error}");
    }

    for imag in [
        json!([[0., 0.], [0., 0.]]),
        json!([
            [0., 0., 0., 0.],
            [0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ]),
    ] {
        let mut input = support::phase_config();
        input["model"]["site_energy"]["imag"] = imag;
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains("model.site_energy.imag"), "{error}");
    }

    for (path, component) in [("two_site_h", "imag"), ("observable.matrix", "real")] {
        let mut input = support::phase_config();
        if path == "two_site_h" {
            input["model"]["two_site_h"][component] = json!([[0.0]]);
        } else {
            input["model"]["observable"]["matrix"][component] = json!([[0.0]]);
        }
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains(component), "{path}.{component}: {error}");
    }

    for path in ["two_site_h", "observable"] {
        let mut input = support::phase_config();
        input["model"][path]["unexpected"] = json!(1);
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains("unexpected"), "{path}: {error}");
    }
}

#[test]
fn rejects_explicit_null_imaginary_matrix_component() {
    let mut input = support::phase_config();
    input["model"]["two_site_h"]["imag"] = Value::Null;
    assert!(RunConfig::from_json_str(&input.to_string()).is_err());
}

#[test]
fn rejects_blank_observable_names_and_non_hermitian_inputs() {
    for name in ["", " \t\n"] {
        let mut input = support::phase_config();
        input["model"]["observable"]["name"] = json!(name);
        assert!(RunConfig::from_json_str(&input.to_string())
            .unwrap_err()
            .contains("observable.name"));
    }

    for path in ["two_site_h", "site_energy"] {
        let mut input = support::phase_config();
        input["model"][path]["real"][0][1] = json!(0.2);
        input["model"][path]["real"][1][0] = json!(0.0);
        let error = RunConfig::from_json_str(&input.to_string()).unwrap_err();
        assert!(error.contains(&format!("model.{path}")), "{path}: {error}");
    }
    let mut observable = support::phase_config();
    observable["model"]["observable"]["matrix"]["real"][0][0] = json!(0.3);
    observable["model"]["observable"]["matrix"]["imag"][0][0] = json!(0.2);
    let error = RunConfig::from_json_str(&observable.to_string()).unwrap_err();
    assert!(error.contains("model.observable.matrix"), "{error}");
}

#[test]
fn rejects_nonfinite_toml_and_direct_matrix_values() {
    let toml = phase_toml().replacen("0.0,0.35,0.35", "0.0,inf,0.35", 1);
    let error = RunConfig::from_toml_str(&toml).unwrap_err();
    assert!(error.contains("model.two_site_h.imag[0][1]"), "{error}");

    let matrix = MatrixInput {
        real: vec![vec![f64::NAN]],
        imag: None,
    };
    let error = matrix.to_matrix(1, "direct.matrix").unwrap_err();
    assert!(error_text(error).contains("direct.matrix.real[0][0]"));

    let mut cfg = RunConfig::from_json_str(&exact_real_config().to_string()).unwrap();
    let ModelSpec::Matrix(matrix) = &mut cfg.model else {
        panic!("wrong variant")
    };
    matrix.site_energy.real[2][1] = f64::INFINITY;
    let error = run_sweep(&cfg).unwrap_err();
    assert!(matches!(error, ItebdError::InvalidRunConfig(_)));
    assert!(error.to_string().contains("model.site_energy.real[2][1]"));
}

#[test]
fn real_helpers_reject_only_the_complex_value_they_return() {
    let mut input = support::phase_config();
    input["model"]["two_site_h"]
        .as_object_mut()
        .unwrap()
        .remove("imag");
    input["model"]["site_energy"]
        .as_object_mut()
        .unwrap()
        .remove("imag");
    let cfg = RunConfig::from_json_str(&input.to_string()).unwrap();
    assert!(cfg.model.hamiltonian().is_ok());
    assert!(matches!(
        cfg.model.magnetization_op(),
        Err(ItebdError::InvalidRunConfig(_))
    ));
    assert!(matches!(
        cfg.model.resolve().unwrap().hamiltonian,
        ItebdHamiltonian::Real(_)
    ));

    let mut complex_h = support::phase_config();
    complex_h["model"]["observable"] =
        json!({"name": "sigma_z", "matrix": {"real": [[1., 0.], [0., -1.]]}});
    let cfg = RunConfig::from_json_str(&complex_h.to_string()).unwrap();
    assert!(cfg.model.magnetization_op().is_ok());
    assert!(matches!(
        cfg.model.hamiltonian(),
        Err(ItebdError::InvalidRunConfig(_))
    ));

    let mut directly_mutated =
        RunConfig::from_json_str(&support::phase_config().to_string()).unwrap();
    let ModelSpec::Matrix(matrix) = &mut directly_mutated.model else {
        panic!("wrong variant")
    };
    matrix.two_site_h.imag.as_mut().unwrap()[1][0] = 0.0;
    assert!(matches!(
        directly_mutated.model.hamiltonian(),
        Err(ItebdError::InvalidRunConfig(_))
    ));

    let mut directly_mutated =
        RunConfig::from_json_str(&support::phase_config().to_string()).unwrap();
    let ModelSpec::Matrix(matrix) = &mut directly_mutated.model else {
        panic!("wrong variant")
    };
    matrix.observable.matrix.imag.as_mut().unwrap()[1][0] = 0.0;
    assert!(matches!(
        directly_mutated.model.magnetization_op(),
        Err(ItebdError::InvalidRunConfig(_))
    ));
}

#[test]
fn exact_real_signed_zeros_stay_real_but_any_nonzero_imaginary_selects_complex() {
    for explicit_zeros in [false, true] {
        let mut input = exact_real_config();
        if explicit_zeros {
            let mut h_zeros = zeros(4);
            h_zeros[0][1] = -0.0;
            input["model"]["two_site_h"]["imag"] = json!(h_zeros);
            input["model"]["site_energy"]["imag"] = json!(zeros(4));
            input["model"]["observable"]["matrix"]["imag"] = json!([[0., -0.], [0., 0.]]);
        }
        let cfg = RunConfig::from_json_str(&input.to_string()).unwrap();
        assert!(matches!(
            cfg.model.resolve().unwrap().hamiltonian,
            ItebdHamiltonian::Real(_)
        ));
    }

    let mut tiny = support::phase_config();
    let mut imag = zeros(4);
    imag[0][1] = f64::MIN_POSITIVE;
    imag[1][0] = -f64::MIN_POSITIVE;
    tiny["model"]["two_site_h"] = json!({"real": identity(4), "imag": imag});
    tiny["model"]["site_energy"] = json!({"real": identity(4)});
    tiny["model"]["observable"] = json!({"name": "z", "matrix": {"real": [[1., 0.], [0., -1.]]}});
    let cfg = RunConfig::from_json_str(&tiny.to_string()).unwrap();
    assert!(matches!(
        cfg.model.resolve().unwrap().hamiltonian,
        ItebdHamiltonian::Complex(_)
    ));
}

#[test]
fn toml_and_json_parse_the_same_matrix_fixture() {
    let json_cfg = RunConfig::from_json_str(&support::phase_config().to_string()).unwrap();
    let toml_cfg = RunConfig::from_toml_str(&phase_toml()).unwrap();
    assert_eq!(
        serde_json::to_value(toml_cfg).unwrap(),
        serde_json::to_value(json_cfg).unwrap()
    );
}

#[test]
fn model_json_round_trip_preserves_adversarial_finite_float_bits() {
    let mut input = support::phase_config();
    input["model"]["two_site_h"]["real"][0][0] = json!(f64::from_bits(1));
    input["model"]["two_site_h"]["real"][3][3] = json!(-0.0);
    input["model"]["site_energy"]["real"][0][0] = json!(f64::MAX);
    input["model"]["site_energy"]["real"][3][3] = json!(-f64::MAX);
    input["model"]["two_site_h"]
        .as_object_mut()
        .unwrap()
        .remove("imag");
    input["model"]["site_energy"]
        .as_object_mut()
        .unwrap()
        .remove("imag");
    input["model"]["observable"] = json!({"name": "z", "matrix": {"real": [[1., 0.], [0., -1.]]}});
    let model: ModelSpec = serde_json::from_value(input["model"].clone()).unwrap();
    let encoded = serde_json::to_string(&model).unwrap();
    let decoded: ModelSpec = serde_json::from_str(&encoded).unwrap();
    let ModelSpec::Matrix(decoded) = decoded else {
        panic!("wrong variant")
    };
    assert_eq!(decoded.two_site_h.real[0][0].to_bits(), 1);
    assert_eq!(
        decoded.two_site_h.real[3][3].to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(decoded.site_energy.real[0][0].to_bits(), f64::MAX.to_bits());
    assert_eq!(
        decoded.site_energy.real[3][3].to_bits(),
        (-f64::MAX).to_bits()
    );
}

#[test]
fn result_reader_preserves_complete_matrix_metadata_and_adversarial_float_bits() {
    let cfg = RunConfig::from_json_str(&support::phase_config().to_string()).unwrap();
    let mut model = cfg.model.clone();
    let ModelSpec::Matrix(matrix) = &mut model else {
        panic!("wrong variant")
    };
    matrix.two_site_h.real[0][0] = f64::from_bits(1);
    matrix.two_site_h.real[3][3] = -0.0;
    matrix.site_energy.real[0][0] = f64::MAX;
    matrix.site_energy.real[3][3] = -f64::MAX;
    let result = SweepResult {
        metadata: Metadata {
            model,
            evolution: cfg.evolution,
            truncation: cfg.truncation,
            canonicalize_every: cfg.run.canonicalize_every,
            local_dim: 2,
            git_revision: None,
        },
        records: Vec::new(),
        segment: None,
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("adversarial.json");
    write_result(&path, &result).unwrap();
    let decoded = read_result(&path).unwrap();
    let ModelSpec::Matrix(decoded) = decoded.metadata.model else {
        panic!("wrong variant")
    };
    assert_eq!(decoded.two_site_h.real[0][0].to_bits(), 1);
    assert_eq!(
        decoded.two_site_h.real[3][3].to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(decoded.site_energy.real[0][0].to_bits(), f64::MAX.to_bits());
    assert_eq!(
        decoded.site_energy.real[3][3].to_bits(),
        (-f64::MAX).to_bits()
    );
    assert_eq!(decoded.basis_order, "first_site_fastest");
    assert_eq!(decoded.observable.name, "sigma_y");
    assert_eq!(decoded.observable.matrix.imag.unwrap()[0][1], -1.0);
}

#[test]
fn all_builtin_presets_parse_from_json() {
    for model in [
        json!({"type": "tfim", "j": 1.0, "g": 0.7}),
        json!({"type": "xy", "gamma": 0.5, "h": 0.7}),
        json!({"type": "bilinear_biquadratic", "j1": 1.0, "j2": 0.25}),
        json!({"type": "aklt"}),
        json!({"type": "heisenberg"}),
    ] {
        let mut input = support::phase_config();
        input["model"] = model.clone();
        let cfg = RunConfig::from_json_str(&input.to_string()).unwrap();
        assert_eq!(serde_json::to_value(cfg.model).unwrap(), model);
    }
}

#[test]
fn matrix_configs_reject_exact_references_and_legacy_results_stay_first_order() {
    let mut input = support::phase_config();
    input["output"]["include_exact"] = json!(true);
    assert!(RunConfig::from_json_str(&input.to_string())
        .unwrap_err()
        .contains("include_exact"));

    let cfg = RunConfig::from_json_str(&exact_real_config().to_string()).unwrap();
    assert!(!cfg.model.has_exact());
    assert!(cfg.model.exact(1.0).is_none());

    let result: SweepResult = serde_json::from_str(
        r#"{"metadata":{"model":{"type":"tfim","j":1.0,"g":0.7},"evolution":{"dtau":0.05,"beta_max":0.4,"record_every_beta":0.2},"truncation":{"epsilon":1e-12,"max_bond":16},"canonicalize_every":1,"local_dim":2,"git_revision":null},"records":[]}"#,
    ).unwrap();
    assert_eq!(result.metadata.evolution.trotter_order, TrotterOrder::First);
    assert!(result.segment.is_none());
}

#[test]
fn automatic_runner_accepts_complex_hamiltonians() {
    let cfg = RunConfig::from_json_str(&support::phase_config().to_string()).unwrap();
    let result = run_sweep(&cfg).unwrap();
    assert_eq!(result.records.len(), 1);
    let record = &result.records[0];
    assert!([record.u, record.c, record.f, record.magnetization]
        .iter()
        .all(|value| value.is_finite()));
    assert!(record.exact.is_none());
}

fn exact_real_config() -> Value {
    let mut input = support::phase_config();
    input["model"]["two_site_h"] = json!({"real": identity(4)});
    input["model"]["site_energy"] = json!({"real": identity(4)});
    input["model"]["observable"] = json!({"name": "z", "matrix": {"real": [[1., 0.], [0., -1.]]}});
    input
}

fn phase_toml() -> String {
    r#"
[model]
type = "matrix"
version = 1
local_dim = 2
basis_order = "first_site_fastest"
label = "phase TFIM"

[model.two_site_h]
real = [[-1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,-1.0]]
imag = [[0.0,0.35,0.35,0.0],[-0.35,0.0,0.0,0.35],[-0.35,0.0,0.0,0.35],[0.0,-0.35,-0.35,0.0]]

[model.site_energy]
real = [[-1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,-1.0]]
imag = [[0.0,0.7,0.0,0.0],[-0.7,0.0,0.0,0.0],[0.0,0.0,0.0,0.7],[0.0,0.0,-0.7,0.0]]

[model.observable]
name = "sigma_y"

[model.observable.matrix]
real = [[0.0,0.0],[0.0,0.0]]
imag = [[0.0,-1.0],[1.0,0.0]]

[evolution]
dtau = 0.05
beta_max = 0.2
record_every_beta = 0.2
trotter_order = 2

[truncation]
epsilon = 1e-12
max_bond = 32

[run]
canonicalize_every = 1

[output]
path = "unused.json"
include_exact = false
"#
    .to_owned()
}

#[test]
fn direct_matrix_conversion_uses_rows_then_columns() {
    let input = MatrixInput {
        real: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        imag: Some(vec![vec![0.0, 0.5], vec![-0.5, -0.0]]),
    };
    assert_eq!(
        input.to_matrix(2, "direct.matrix").unwrap(),
        DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.5),
                Complex64::new(3.0, -0.5),
                Complex64::new(4.0, -0.0),
            ],
        )
    );
}

fn allocation_order_model() -> MatrixModelInput {
    let local_dim = 8;
    let matrix_dim = local_dim * local_dim;
    MatrixModelInput {
        version: 1,
        local_dim,
        basis_order: "first_site_fastest".into(),
        label: None,
        two_site_h: MatrixInput {
            real: zeros(matrix_dim),
            imag: None,
        },
        site_energy: MatrixInput {
            real: zeros(matrix_dim),
            imag: None,
        },
        observable: ObservableInput {
            name: "ordering probe".into(),
            matrix: MatrixInput {
                real: zeros(local_dim),
                imag: None,
            },
        },
    }
}

#[test]
fn later_validation_errors_precede_any_dense_matrix_allocation() {
    let mut malformed_energy = allocation_order_model();
    malformed_energy.site_energy.real.pop();
    let model = ModelSpec::Matrix(malformed_energy);
    let (result, large_allocations) = track_large_allocations(|| model.hamiltonian());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("model.site_energy.real"));
    assert_eq!(
        large_allocations, 0,
        "real Hamiltonian validation allocated before checking site_energy"
    );

    let mut malformed_observable = allocation_order_model();
    malformed_observable.observable.matrix.real[7][7] = f64::NAN;
    let model = ModelSpec::Matrix(malformed_observable);
    let (result, large_allocations) = track_large_allocations(|| model.resolve());
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("malformed observable was accepted"),
    };
    assert!(error
        .to_string()
        .contains("model.observable.matrix.real[7][7]"));
    assert_eq!(
        large_allocations, 0,
        "automatic resolution allocated before checking the observable"
    );
}
