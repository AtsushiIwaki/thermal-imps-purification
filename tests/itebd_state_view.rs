use thermal_imps_purification::itebd_auto::ItebdState;
use thermal_imps_purification::itebd_complex::{ComplexPurifiedMps, ComplexSite};
use thermal_imps_purification::itebd_state_view::{
    ItebdScalarType, ItebdStateRef, ItebdStateValidationError,
};
use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps, Site};
use thermal_imps_purification::tensor::{new_index, Tensor};
use num_complex::{Complex32, Complex64};
use tensor4all_core::AnyScalar;

#[derive(Debug, thiserror::Error)]
#[error("state validation failed")]
struct StateConsumerError {
    #[source]
    source: ItebdStateValidationError,
}

#[test]
fn validates_without_owning_or_changing_the_state() {
    let s = infinite_temperature(2);
    let before = s.a.gamma.to_vec::<f64>().unwrap();
    let summary = ItebdStateRef::from(&s).validate().unwrap();
    assert_eq!(summary.bond_dimensions, [1, 1]);
    assert_eq!(summary.a_role_axes, [0, 1, 2, 3]);
    assert_eq!(before, s.a.gamma.to_vec::<f64>().unwrap());
}

#[test]
fn equal_identity_with_wrong_declared_dimension_is_rejected() {
    let mut s = infinite_temperature(2);
    s.a.phys.dim = 3;
    assert!(ItebdStateRef::from(&s).validate().is_err());
}

#[test]
fn borrowed_accessors_cover_direct_and_automatic_backends() {
    let real = infinite_temperature(2);
    let real_view = ItebdStateRef::from(&real);
    assert_eq!(real_view.scalar_type(), ItebdScalarType::Real);
    assert!(std::ptr::eq(real_view.a().gamma, &real.a.gamma));
    assert!(std::ptr::eq(real_view.b().right, &real.b.right));
    assert_eq!(real_view.lambda_ab(), real.lambda_ab);
    assert_eq!(real_view.lambda_ba(), real.lambda_ba);

    let automatic = ItebdState::Complex(ComplexPurifiedMps::infinite_temperature(2).unwrap());
    let complex_view = ItebdStateRef::from(&automatic);
    assert_eq!(complex_view.scalar_type(), ItebdScalarType::Complex);
    assert_eq!(
        complex_view.validate().unwrap().scalar_type,
        ItebdScalarType::Complex
    );

    let encoded = serde_json::to_string(&ItebdScalarType::Complex).unwrap();
    assert_eq!(
        serde_json::from_str::<ItebdScalarType>(&encoded).unwrap(),
        ItebdScalarType::Complex
    );
}

#[test]
fn validation_error_participates_in_a_standard_error_source_chain() {
    let error = StateConsumerError {
        source: ItebdStateValidationError::InvalidField {
            field: "a.gamma".to_string(),
            reason: "expected rank 4".to_string(),
        },
    };

    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(
        source.to_string(),
        "invalid iTEBD state field `a.gamma`: expected rank 4"
    );
}

#[derive(Debug, Clone, Copy)]
enum Backend {
    Real,
    Complex,
}

#[derive(Debug, Clone, Copy)]
enum Malformation {
    RepeatedRoles,
    RepeatedGammaRole,
    CrossSitePhysicalCollision,
    CrossSiteAncillaCollision,
    CrossSiteMixedCollision,
    IdentityMetadataMismatch,
    WrongRank,
    WrongStorage,
    WrongPrecision,
    NonDenseStorage,
    StructuredStorage,
    WrongCount,
    ZeroDimension,
    NonFiniteGamma,
    NegativeSchmidt,
    NanSchmidt,
    InfiniteSchmidt,
    ZeroSchmidt,
    MismatchedAbBond,
    MismatchedBaBond,
}

const MALFORMATIONS: &[Malformation] = &[
    Malformation::RepeatedRoles,
    Malformation::RepeatedGammaRole,
    Malformation::CrossSitePhysicalCollision,
    Malformation::CrossSiteAncillaCollision,
    Malformation::CrossSiteMixedCollision,
    Malformation::IdentityMetadataMismatch,
    Malformation::WrongRank,
    Malformation::WrongStorage,
    Malformation::WrongPrecision,
    Malformation::NonDenseStorage,
    Malformation::StructuredStorage,
    Malformation::WrongCount,
    Malformation::ZeroDimension,
    Malformation::NonFiniteGamma,
    Malformation::NegativeSchmidt,
    Malformation::NanSchmidt,
    Malformation::InfiniteSchmidt,
    Malformation::ZeroSchmidt,
    Malformation::MismatchedAbBond,
    Malformation::MismatchedBaBond,
];

#[test]
fn malformed_states_are_rejected_for_both_backends() {
    for backend in [Backend::Real, Backend::Complex] {
        for &malformation in MALFORMATIONS {
            let result = match backend {
                Backend::Real => {
                    let mut state = infinite_temperature(2);
                    malform_real(&mut state, malformation);
                    ItebdStateRef::from(&state).validate()
                }
                Backend::Complex => {
                    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
                    malform_complex(&mut state, malformation);
                    ItebdStateRef::from(&state).validate()
                }
            };
            if matches!(malformation, Malformation::StructuredStorage) {
                assert!(
                    matches!(
                        result,
                        Err(ItebdStateValidationError::InvalidField { field, reason })
                            if field == "a.gamma" && reason == "expected dense storage"
                    ),
                    "{backend:?} structured fixture failed for the wrong reason"
                );
            } else {
                assert!(
                    result.is_err(),
                    "{backend:?} backend accepted {malformation:?}"
                );
            }
        }
    }
}

#[test]
fn equal_product_storage_shape_mismatch_is_rejected_for_both_backends() {
    let mut real = infinite_temperature(2);
    expand_all_bonds_real(&mut real);
    assert!(ItebdStateRef::from(&real).validate().is_ok());
    corrupt_equal_product_bond_dimensions_real(&mut real);
    assert_eq!(real.a.gamma.dims(), vec![4, 2, 2, 1]);
    assert_eq!(
        real.a.gamma.storage().unwrap().logical_dims(),
        &[2, 2, 2, 2]
    );
    assert_eq!(real.a.gamma.dims().iter().product::<usize>(), 16);
    assert!(matches!(
        ItebdStateRef::from(&real).validate(),
        Err(ItebdStateValidationError::InvalidField { field, reason })
            if field == "a.gamma"
                && reason == "storage shape [2, 2, 2, 2] does not match gamma index dimensions [4, 2, 2, 1]"
    ));

    let mut complex = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    expand_all_bonds_complex(&mut complex);
    assert!(ItebdStateRef::from(&complex).validate().is_ok());
    corrupt_equal_product_bond_dimensions_complex(&mut complex);
    assert_eq!(complex.a.gamma.dims(), vec![4, 2, 2, 1]);
    assert_eq!(
        complex.a.gamma.storage().unwrap().logical_dims(),
        &[2, 2, 2, 2]
    );
    assert_eq!(complex.a.gamma.dims().iter().product::<usize>(), 16);
    assert!(matches!(
        ItebdStateRef::from(&complex).validate(),
        Err(ItebdStateValidationError::InvalidField { field, reason })
            if field == "a.gamma"
                && reason == "storage shape [2, 2, 2, 2] does not match gamma index dimensions [4, 2, 2, 1]"
    ));
}

#[test]
fn reordered_axes_with_permuted_payload_validate_for_both_backends() {
    const ORDER: [usize; 4] = [2, 0, 3, 1];
    const EXPECTED_ROLE_AXES: [u8; 4] = [1, 3, 0, 2];

    let mut real = infinite_temperature(2);
    real.a.gamma =
        Tensor::from_dense(real.a.gamma.indices.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    real.b.gamma =
        Tensor::from_dense(real.b.gamma.indices.clone(), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    permute_real(&mut real.a.gamma, ORDER);
    permute_real(&mut real.b.gamma, ORDER);
    assert_eq!(
        real.a.gamma.to_vec::<f64>().unwrap(),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    let real_summary = ItebdStateRef::from(&real).validate().unwrap();
    assert_eq!(real_summary.a_role_axes, EXPECTED_ROLE_AXES);
    assert_eq!(real_summary.b_role_axes, EXPECTED_ROLE_AXES);

    let mut complex = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let a_values = [1.0, 2.0, 3.0, 4.0].map(|value| Complex64::new(value, -value));
    let b_values = [5.0, 6.0, 7.0, 8.0].map(|value| Complex64::new(value, value + 0.5));
    complex.a.gamma =
        Tensor::from_dense(complex.a.gamma.indices.clone(), a_values.to_vec()).unwrap();
    complex.b.gamma =
        Tensor::from_dense(complex.b.gamma.indices.clone(), b_values.to_vec()).unwrap();
    permute_complex(&mut complex.a.gamma, ORDER);
    permute_complex(&mut complex.b.gamma, ORDER);
    assert_eq!(
        complex.a.gamma.to_vec::<Complex64>().unwrap(),
        vec![a_values[0], a_values[2], a_values[1], a_values[3]]
    );
    let complex_summary = ItebdStateRef::from(&complex).validate().unwrap();
    assert_eq!(complex_summary.a_role_axes, EXPECTED_ROLE_AXES);
    assert_eq!(complex_summary.b_role_axes, EXPECTED_ROLE_AXES);
}

#[test]
fn schmidt_norms_are_stable_for_large_finite_values() {
    let mut real = infinite_temperature(2);
    expand_ab_bond_real(&mut real);
    real.lambda_ab = vec![f64::MAX, f64::MIN_POSITIVE];
    assert_eq!(
        ItebdStateRef::from(&real).validate().unwrap().schmidt_norms[0],
        f64::MAX
    );

    let mut complex = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    expand_ab_bond_complex(&mut complex);
    complex.lambda_ab = vec![f64::MAX, f64::MIN_POSITIVE];
    assert_eq!(
        ItebdStateRef::from(&complex)
            .validate()
            .unwrap()
            .schmidt_norms[0],
        f64::MAX
    );
}

fn malform_real(state: &mut PurifiedMps, malformation: Malformation) {
    match malformation {
        Malformation::RepeatedRoles => state.a.anc = state.a.phys.clone(),
        Malformation::RepeatedGammaRole => {
            state.a.gamma.indices[2] = state.a.gamma.indices[1].clone()
        }
        Malformation::CrossSitePhysicalCollision => state.b.phys = state.a.phys.clone(),
        Malformation::CrossSiteAncillaCollision => state.b.anc = state.a.anc.clone(),
        Malformation::CrossSiteMixedCollision => state.b.anc = state.a.phys.clone(),
        Malformation::IdentityMetadataMismatch => state.a.phys.plev += 1,
        Malformation::WrongRank => {
            state.a.gamma.indices.pop();
        }
        Malformation::WrongStorage => {
            state.a.gamma = Tensor::from_dense(
                state.a.gamma.indices.clone(),
                vec![Complex64::new(0.0, 0.0); 4],
            )
            .unwrap();
        }
        Malformation::WrongPrecision => {
            state.a.gamma =
                Tensor::from_dense(state.a.gamma.indices.clone(), vec![0.0_f32; 4]).unwrap();
        }
        Malformation::NonDenseStorage => {
            expand_all_bonds_real(state);
            state.a.gamma =
                Tensor::from_diag(state.a.gamma.indices.clone(), vec![1.0_f64; 2]).unwrap();
        }
        Malformation::StructuredStorage => {
            expand_all_bonds_real(state);
            let structured = Tensor::delta(
                &[state.a.left.clone(), state.a.phys.clone()],
                &[state.a.right.clone(), state.a.anc.clone()],
            )
            .unwrap();
            assert_eq!(
                structured.indices,
                vec![
                    state.a.left.clone(),
                    state.a.right.clone(),
                    state.a.phys.clone(),
                    state.a.anc.clone(),
                ]
            );
            assert_eq!(structured.dims(), vec![2, 2, 2, 2]);
            assert!(!structured.storage().unwrap().is_dense());
            assert!(!structured.is_diag());
            state.a.gamma = structured;
        }
        Malformation::WrongCount => {
            set_real_site_dimensions_without_resizing(&mut state.a, 3);
            set_real_site_dimensions_without_resizing(&mut state.b, 3);
        }
        Malformation::ZeroDimension => state.a.phys.dim = 0,
        Malformation::NonFiniteGamma => {
            let mut values = state.a.gamma.to_vec::<f64>().unwrap();
            values[0] = f64::NAN;
            state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), values).unwrap();
        }
        Malformation::NegativeSchmidt => state.lambda_ab[0] = -1.0,
        Malformation::NanSchmidt => state.lambda_ab[0] = f64::NAN,
        Malformation::InfiniteSchmidt => state.lambda_ab[0] = f64::INFINITY,
        Malformation::ZeroSchmidt => state.lambda_ab[0] = 0.0,
        Malformation::MismatchedAbBond => state.b.left = new_index(state.b.left.dim),
        Malformation::MismatchedBaBond => state.b.right = new_index(state.b.right.dim),
    }
}

fn malform_complex(state: &mut ComplexPurifiedMps, malformation: Malformation) {
    match malformation {
        Malformation::RepeatedRoles => state.a.anc = state.a.phys.clone(),
        Malformation::RepeatedGammaRole => {
            state.a.gamma.indices[2] = state.a.gamma.indices[1].clone()
        }
        Malformation::CrossSitePhysicalCollision => state.b.phys = state.a.phys.clone(),
        Malformation::CrossSiteAncillaCollision => state.b.anc = state.a.anc.clone(),
        Malformation::CrossSiteMixedCollision => state.b.anc = state.a.phys.clone(),
        Malformation::IdentityMetadataMismatch => state.a.phys.plev += 1,
        Malformation::WrongRank => {
            state.a.gamma.indices.pop();
        }
        Malformation::WrongStorage => {
            state.a.gamma =
                Tensor::from_dense(state.a.gamma.indices.clone(), vec![0.0_f64; 4]).unwrap();
        }
        Malformation::WrongPrecision => {
            state.a.gamma = Tensor::from_dense(
                state.a.gamma.indices.clone(),
                vec![Complex32::new(0.0, 0.0); 4],
            )
            .unwrap();
        }
        Malformation::NonDenseStorage => {
            expand_all_bonds_complex(state);
            state.a.gamma = Tensor::from_diag(
                state.a.gamma.indices.clone(),
                vec![Complex64::new(1.0, 0.5); 2],
            )
            .unwrap();
        }
        Malformation::StructuredStorage => {
            expand_all_bonds_complex(state);
            let structured = Tensor::delta(
                &[state.a.left.clone(), state.a.phys.clone()],
                &[state.a.right.clone(), state.a.anc.clone()],
            )
            .unwrap()
            .scale(AnyScalar::new_complex(1.0, 0.5))
            .unwrap();
            assert_eq!(
                structured.indices,
                vec![
                    state.a.left.clone(),
                    state.a.right.clone(),
                    state.a.phys.clone(),
                    state.a.anc.clone(),
                ]
            );
            assert_eq!(structured.dims(), vec![2, 2, 2, 2]);
            assert!(!structured.storage().unwrap().is_dense());
            assert!(!structured.is_diag());
            state.a.gamma = structured;
        }
        Malformation::WrongCount => {
            set_complex_site_dimensions_without_resizing(&mut state.a, 3);
            set_complex_site_dimensions_without_resizing(&mut state.b, 3);
        }
        Malformation::ZeroDimension => state.a.phys.dim = 0,
        Malformation::NonFiniteGamma => {
            let mut values = state.a.gamma.to_vec::<Complex64>().unwrap();
            values[0] = Complex64::new(0.0, f64::INFINITY);
            state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), values).unwrap();
        }
        Malformation::NegativeSchmidt => state.lambda_ab[0] = -1.0,
        Malformation::NanSchmidt => state.lambda_ab[0] = f64::NAN,
        Malformation::InfiniteSchmidt => state.lambda_ab[0] = f64::INFINITY,
        Malformation::ZeroSchmidt => state.lambda_ab[0] = 0.0,
        Malformation::MismatchedAbBond => state.b.left = new_index(state.b.left.dim),
        Malformation::MismatchedBaBond => state.b.right = new_index(state.b.right.dim),
    }
}

fn set_real_site_dimensions_without_resizing(site: &mut Site, dimension: usize) {
    site.phys.dim = dimension;
    site.anc.dim = dimension;
    for actual in &mut site.gamma.indices {
        if actual == &site.phys || actual == &site.anc {
            actual.dim = dimension;
        }
    }
}

fn set_complex_site_dimensions_without_resizing(site: &mut ComplexSite, dimension: usize) {
    site.phys.dim = dimension;
    site.anc.dim = dimension;
    for actual in &mut site.gamma.indices {
        if actual == &site.phys || actual == &site.anc {
            actual.dim = dimension;
        }
    }
}

fn expand_ab_bond_real(state: &mut PurifiedMps) {
    state.lambda_bond_ab.dim = 2;
    state.a.right.dim = 2;
    state.b.left.dim = 2;
    state.a.gamma.indices[3].dim = 2;
    state.b.gamma.indices[0].dim = 2;
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), vec![1.0; 8]).unwrap();
    state.b.gamma = Tensor::from_dense(state.b.gamma.indices.clone(), vec![1.0; 8]).unwrap();
}

fn expand_ab_bond_complex(state: &mut ComplexPurifiedMps) {
    state.lambda_bond_ab.dim = 2;
    state.a.right.dim = 2;
    state.b.left.dim = 2;
    state.a.gamma.indices[3].dim = 2;
    state.b.gamma.indices[0].dim = 2;
    state.a.gamma = Tensor::from_dense(
        state.a.gamma.indices.clone(),
        vec![Complex64::new(1.0, 0.0); 8],
    )
    .unwrap();
    state.b.gamma = Tensor::from_dense(
        state.b.gamma.indices.clone(),
        vec![Complex64::new(1.0, 0.0); 8],
    )
    .unwrap();
}

fn expand_all_bonds_real(state: &mut PurifiedMps) {
    expand_ab_bond_real(state);
    state.lambda_ab = vec![1.0; 2];
    state.lambda_bond_ba.dim = 2;
    state.a.left.dim = 2;
    state.b.right.dim = 2;
    state.a.gamma.indices[0].dim = 2;
    state.b.gamma.indices[3].dim = 2;
    state.lambda_ba = vec![1.0; 2];
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), vec![1.0; 16]).unwrap();
    state.b.gamma = Tensor::from_dense(state.b.gamma.indices.clone(), vec![1.0; 16]).unwrap();
}

fn expand_all_bonds_complex(state: &mut ComplexPurifiedMps) {
    expand_ab_bond_complex(state);
    state.lambda_ab = vec![1.0; 2];
    state.lambda_bond_ba.dim = 2;
    state.a.left.dim = 2;
    state.b.right.dim = 2;
    state.a.gamma.indices[0].dim = 2;
    state.b.gamma.indices[3].dim = 2;
    state.lambda_ba = vec![1.0; 2];
    state.a.gamma = Tensor::from_dense(
        state.a.gamma.indices.clone(),
        vec![Complex64::new(1.0, 0.0); 16],
    )
    .unwrap();
    state.b.gamma = Tensor::from_dense(
        state.b.gamma.indices.clone(),
        vec![Complex64::new(1.0, 0.0); 16],
    )
    .unwrap();
}

fn corrupt_equal_product_bond_dimensions_real(state: &mut PurifiedMps) {
    state.lambda_bond_ab.dim = 1;
    state.a.right.dim = 1;
    state.b.left.dim = 1;
    state.a.gamma.indices[3].dim = 1;
    state.b.gamma.indices[0].dim = 1;
    state.lambda_ab = vec![1.0];

    state.lambda_bond_ba.dim = 4;
    state.a.left.dim = 4;
    state.b.right.dim = 4;
    state.a.gamma.indices[0].dim = 4;
    state.b.gamma.indices[3].dim = 4;
    state.lambda_ba = vec![1.0; 4];
}

fn corrupt_equal_product_bond_dimensions_complex(state: &mut ComplexPurifiedMps) {
    state.lambda_bond_ab.dim = 1;
    state.a.right.dim = 1;
    state.b.left.dim = 1;
    state.a.gamma.indices[3].dim = 1;
    state.b.gamma.indices[0].dim = 1;
    state.lambda_ab = vec![1.0];

    state.lambda_bond_ba.dim = 4;
    state.a.left.dim = 4;
    state.b.right.dim = 4;
    state.a.gamma.indices[0].dim = 4;
    state.b.gamma.indices[3].dim = 4;
    state.lambda_ba = vec![1.0; 4];
}

fn permute_real(tensor: &mut Tensor, order: [usize; 4]) {
    let old_indices = tensor.indices.clone();
    let old_values = tensor.to_vec::<f64>().unwrap();
    let offsets = permuted_old_offsets(&old_indices, order);
    let new_indices = order.map(|old_axis| old_indices[old_axis].clone()).to_vec();
    let new_values = offsets
        .into_iter()
        .map(|offset| old_values[offset])
        .collect();
    *tensor = Tensor::from_dense(new_indices, new_values).unwrap();
}

fn permute_complex(tensor: &mut Tensor, order: [usize; 4]) {
    let old_indices = tensor.indices.clone();
    let old_values = tensor.to_vec::<Complex64>().unwrap();
    let offsets = permuted_old_offsets(&old_indices, order);
    let new_indices = order.map(|old_axis| old_indices[old_axis].clone()).to_vec();
    let new_values = offsets
        .into_iter()
        .map(|offset| old_values[offset])
        .collect();
    *tensor = Tensor::from_dense(new_indices, new_values).unwrap();
}

fn permuted_old_offsets(
    old_indices: &[thermal_imps_purification::tensor::Idx],
    order: [usize; 4],
) -> Vec<usize> {
    let old_dims: Vec<_> = old_indices.iter().map(|index| index.dim).collect();
    let new_dims = order.map(|old_axis| old_dims[old_axis]);
    let element_count = new_dims.iter().product();
    (0..element_count)
        .map(|new_offset| {
            let mut remainder = new_offset;
            let mut old_coordinates = [0_usize; 4];
            for (new_axis, dimension) in new_dims.into_iter().enumerate() {
                old_coordinates[order[new_axis]] = remainder % dimension;
                remainder /= dimension;
            }
            let mut old_offset = 0;
            let mut stride = 1;
            for (coordinate, dimension) in old_coordinates.into_iter().zip(old_dims.iter()) {
                old_offset += coordinate * stride;
                stride *= dimension;
            }
            old_offset
        })
        .collect()
}
