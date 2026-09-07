// Historical TDVP names below occur only in negative policy fixtures.
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use syn::File;

mod deterministic_boundary;

const PAIRWISE_ADAPTER_OWNER: &str = "src/contraction_pairwise.rs";

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    fn collect(path: &Path, output: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(path).expect("read Rust source directory") {
            let path = entry.expect("read Rust source entry").path();
            if path.is_dir() {
                collect(&path, output);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                output.push(path);
            }
        }
    }

    let mut output = Vec::new();
    for directory in ["src", "tests"] {
        let path = root.join(directory);
        if path.exists() {
            collect(&path, &mut output);
        }
    }
    output.sort();
    output
}

fn parse_source(path: &Path) -> File {
    syn::parse_file(&fs::read_to_string(path).expect("read Rust source"))
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

#[test]
fn repository_contraction_boundary_is_ast_audited() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    let mut inventory_seen = BTreeSet::new();
    for path in rust_sources(root) {
        let relative = path.strip_prefix(root).unwrap().to_string_lossy();
        let outcome =
            deterministic_boundary::audit_repository_file(&relative, &parse_source(&path));
        violations.extend(outcome.violations);
        inventory_seen.extend(outcome.inventory_seen);
    }
    assert_eq!(
        inventory_seen.len(),
        deterministic_boundary::inventory_len(),
        "every handwritten-contraction inventory identity must name a current source item"
    );
    assert!(
        violations.is_empty(),
        "repository contraction boundary violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ast_audit_rejects_unused_restricted_imports_outside_their_owners() {
    let source =
        "use tensor4all_core::{contract, contract_pair, PairwiseContractionOptions}; fn production() {}";
    let violations = audit_fixture("src/fixture.rs", source);
    assert_eq!(violations.len(), 3, "{violations:?}");

    let pairwise_adapter_source =
        "use tensor4all_core::{contract_pair, contract_pair_with_operand_options, PairwiseContractionOptions};";
    assert!(
        audit_fixture(PAIRWISE_ADAPTER_OWNER, pairwise_adapter_source).is_empty(),
        "the production adapter owns the pairwise imports"
    );
}

fn audit_fixture(relative_path: &str, source: &str) -> Vec<String> {
    deterministic_boundary::audit_fixture(relative_path, source)
}

#[test]
fn ast_audit_rejects_production_contract_and_unclassified_outer_product() {
    for expression in [
        "tensor4all_core::contract(&[lhs, rhs])",
        "tensor4all_core::contract(&[lhs, rhs])",
        "wrapper!(contract(dynamic))",
        "tensor4all_core::outer_product(lhs, rhs)",
    ] {
        let source = format!("fn production() {{ let _ = {expression}; }}");
        assert_eq!(audit_fixture("src/fixture.rs", &source).len(), 1);
    }
}

#[test]
fn ast_audit_allows_only_exact_pairwise_and_test_candidate_owners() {
    for (path, source) in [
        (
            "src/contraction_pairwise.rs",
            "fn pairwise() { tensor4all_core::contract_pair(lhs, rhs); }",
        ),
        (
            "src/contraction_pairwise.rs",
            "fn pairwise_with_conjugation() { tensor4all_core::contract_pair_with_operand_options(lhs, rhs, options); }",
        ),
        (
            "src/uniform_hamiltonian_kernel.rs",
            "fn hamiltonian_action_outer_product() { outer_product(lhs, rhs); }",
        ),
        (
            "src/tensor.rs",
            "mod tests { #[test] fn relabel_changes_contraction_partner() { tensor4all_core::outer_product(lhs, rhs); } }",
        ),
        (
            "src/contraction_contract_candidate.rs",
            "#[cfg(test)] fn contract_candidate() { tensor4all_core::contract(dynamic); }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "impl UniformTransferKernel { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } }",
        ),
        (
            "src/contraction_boundary_audit.rs",
            "#[cfg(test)] fn audit_one_call_nary_candidate() { tensor4all_core::contract(dynamic); }",
        ),
        (
            "src/observable.rs",
            "fn production() { crate::tensor::contract(lhs, rhs); }",
        ),
        (
            "src/contraction_pairwise.rs",
            "fn r#pairwise() { tensor4all_core::r#contract_pair(lhs, rhs); }",
        ),
    ] {
        let violations = audit_fixture(path, source);
        if path.starts_with("src/uniform_") {
            // Negative fixtures: removed TDVP owners grant no exception.
            assert!(!violations.is_empty(), "removed owner {path} escaped");
        } else {
            assert!(violations.is_empty(), "{path}: {violations:?}");
        }
    }
}

#[test]
fn ast_audit_rejects_boundary_bypasses_and_near_matches() {
    for (path, source) in [
        (
            "src/contraction_pairwise.rs",
            "fn another_function() { tensor4all_core::contract_pair(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "fn pairwise() { tensor4all_core::contract_pair(lhs, rhs); }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "#[cfg(test)] fn one_call_nary_transfer_candidate_extra() { tensor4all_core::contract(dynamic); }",
        ),
        (
            "src/fixture.rs",
            "#[cfg(test)] fn contract_candidate() { tensor4all_core::contract(&[lhs, rhs]); }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "#[cfg(any(feature = \"fixture\", test))] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "#[cfg(test)] fn one_call_nary_transfer_candidate() { wrapper!(outer_product(lhs, rhs)); }",
        ),
        (
            "src/fixture.rs",
            "use tensor4all_core::contract as pairwise; fn production() { pairwise(dynamic); }",
        ),
        (
            "src/fixture.rs",
            "use tensor4all_core::contract; fn production() {}",
        ),
        (
            "src/observable.rs",
            "fn production() { crate::tensor::contract(&[lhs, rhs]); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tenferro::contract(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tenferro::einsum!(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tensor4all_core::r#contract(dynamic); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tensor4all_core::r#contract_pair(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tensor4all_core::r#outer_product(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "fn production() { tenferro::r#einsum!(lhs, rhs); }",
        ),
        (
            "src/fixture.rs",
            "use tensor4all_core::r#contract as call; fn production() { call(dynamic); }",
        ),
        (
            "src/observable.rs",
            "fn production() { crate::tensor::r#contract(lhs, rhs, third); }",
        ),
    ] {
        let violations = audit_fixture(path, source);
        assert!(!violations.is_empty(), "{path}: {violations:?}");
    }
}

#[test]
fn ast_audit_rejects_parenthesized_and_aliased_restricted_apis() {
    for source in [
        "fn production() { (tensor4all_core::contract)(&[lhs, rhs]); }",
        "fn production() { let call = tensor4all_core::contract; call(&[lhs, rhs]); }",
        "const CALL: () = tensor4all_core::contract; fn production() { CALL(&[lhs, rhs]); }",
        "static CALL: () = tensor4all_core::contract; fn production() { CALL(&[lhs, rhs]); }",
        "fn production() { (tensor4all_core::contract_pair)(lhs, rhs); }",
        "fn production() { let call = tensor4all_core::contract_pair; call(lhs, rhs); }",
        "fn production() { (tensor4all_core::outer_product)(lhs, rhs); }",
        "fn production() { let call = tensor4all_core::outer_product; call(lhs, rhs); }",
    ] {
        let violations = audit_fixture("src/fixture.rs", source);
        assert!(!violations.is_empty(), "{source}: {violations:?}");
    }
}

#[test]
fn ast_audit_resolves_recursive_module_aliases_before_classifying_restricted_calls() {
    let mutations = [
        "use tensor4all_core as t; fn production() { t::contract(dynamic); }",
        "use tensor4all_core as t; fn production() { (t::contract)(dynamic); }",
        "use tensor4all_core as t; fn production() { let call = t::contract; call(dynamic); }",
        "use tensor4all_core as t; use t::contract as c; fn production() { c(dynamic); }",
        "use tensor4all_core as t; use t as u; fn production() { u::contract(dynamic); }",
        "use tensor4all_core as t; mod nested { use super::t as u; fn production() { u::contract(dynamic); } }",
        "use tensor4all_core as t; fn production() { t::contract_owned(dynamic); }",
        "use tensor4all_core as t; fn production() { t::contract_pair(lhs, rhs); }",
        "use tensor4all_core as t; fn production() { let call = t::contract_pair; call(lhs, rhs); }",
        "use tensor4all_core as t; fn production() { t::contract_pair_with_operand_options(lhs, rhs, options); }",
        "use tensor4all_core as t; fn production() { t::outer_product(lhs, rhs); }",
        "use tensor4all_core as t; fn production() { let call = t::outer_product; call(lhs, rhs); }",
        "use tensor4all_core as t; fn production() { t::einsum!(lhs, rhs); }",
        "use tenferro as t; fn production() { t::contract(lhs, rhs); }",
        "use tenferro as t; use t as u; fn production() { u::contract(lhs, rhs); }",
        "use tenferro as t; fn production() { t::einsum!(lhs, rhs); }",
        "mod nested { #[cfg(test)] fn contract_candidate() { tensor4all_core::contract(dynamic); } }",
        "impl crate::UniformTransferKernel { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } }",
    ];
    assert_eq!(mutations.len(), 18);
    for source in mutations {
        let violations = audit_fixture("src/fixture.rs", source);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("restricted tensor API")
                    || violation.contains("restricted contraction tokens")
            }),
            "{source}: {violations:?}"
        );
    }
}

#[test]
fn ast_audit_resolves_lexical_ancestry_and_rejects_unresolved_restricted_paths() {
    let rejected = [
        "extern crate tensor4all_core as t; fn production() { t::contract(dynamic); }",
        "use crate as project; fn production() { project::tensor::contract(lhs, rhs, extra); }",
        "use tensor4all_core as t; mod first { mod second { use super::super::t as u; fn production() { u::contract(dynamic); } } }",
        "use tensor4all_core as t; fn production() { let call = t::contract as fn(_); call(dynamic); }",
        "fn production() { lookalike::contract(dynamic); }",
        "fn production() { lookalike::outer_product(lhs, rhs); }",
    ];
    for source in rejected {
        assert!(
            !audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }

    let local_shadows = [
        "fn production() { let callback = |contract| contract; callback(value); }",
        "fn production() { for contract in values { contract(value); } }",
        "mod local { fn contract(value: usize) -> usize { value } fn production() { contract(value); } }",
        "mod local { fn contract(value: usize) -> usize { value } } fn production() { local::contract(value); }",
        "use self as project; fn production() { project::tensor::contract(lhs, rhs); }",
        "fn r#contract(value: usize) -> usize { value } fn production() { r#contract(value); }",
    ];
    for source in local_shadows {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }
}

#[test]
fn ast_audit_is_independent_of_module_item_declaration_order() {
    let rejected = [
        "fn production() { binary(lhs, rhs, third); } use crate::tensor::contract as binary;",
        "fn production() { BINARY(lhs, rhs, third); } const BINARY: fn(_, _, _) = crate::tensor::contract;",
        r#"
            fn third_helper() {
                hamiltonian_action_outer_product(stage, lhs, rhs);
            }
            fn hamiltonian_action_outer_product(stage: usize, lhs: usize, rhs: usize) {}
        "#,
    ];
    for source in rejected {
        assert!(
            !audit_fixture("src/uniform_hamiltonian_kernel.rs", source).is_empty(),
            "later declaration escaped: {source}"
        );
    }

    let accepted = [
        "fn production() { contract(value); } fn contract(value: usize) -> usize { value }",
        "fn production() { tensor4all_core::contract(value); } mod tensor4all_core { pub fn contract(value: usize) -> usize { value } }",
    ];
    for source in accepted {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "later local item was not pre-collected: {source}"
        );
    }
}

#[test]
fn ast_audit_precollects_block_item_imports() {
    let rejected = [
        r#"
            fn production() {
                binary(lhs, rhs, third);
                use crate::tensor::contract as binary;
            }
        "#,
        r#"
            fn production() {
                binary(lhs, rhs, third);
                use crate::tensor as tensor_api;
                use tensor_api::contract as binary;
            }
        "#,
        r#"
            fn production() {
                binary(lhs, rhs, third);
                use tensor_api::contract as binary;
                use crate::tensor as tensor_api;
            }
        "#,
    ];
    for source in rejected {
        assert!(
            !audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }

    let accepted = r#"
        fn production() {
            local(value);
            fn local(value: usize) -> usize { value }
        }
    "#;
    assert!(audit_fixture("src/fixture.rs", accepted).is_empty());

    let chained_const = r#"
        fn production() {
            BINARY(lhs, rhs, third);
            const BINARY: fn(_, _, _) = binary;
            use crate::tensor::contract as binary;
        }
    "#;
    assert!(
        !audit_fixture("src/fixture.rs", chained_const).is_empty(),
        "block const/import chains must be resolved independent of declaration order"
    );

    for accepted in [
        r#"
            fn production() {
                binary(lhs, rhs);
                use tensor_api::contract as binary;
                use crate::tensor as tensor_api;
            }
        "#,
        r#"
            fn production() {
                contract(lhs, rhs);
                use crate::tensor::*;
            }
        "#,
        r#"
            fn production() {
                contract(value);
                use local::*;
                mod local { pub fn contract(value: usize) -> usize { value } }
            }
        "#,
    ] {
        assert!(
            audit_fixture("src/fixture.rs", accepted).is_empty(),
            "valid block alias/glob was misclassified: {accepted}"
        );
    }
}

#[test]
fn ast_audit_does_not_trust_cfg_disabled_local_shadows() {
    for source in [
        r#"
            #[cfg(any())]
            mod tensor4all_core { pub fn contract(value: usize) -> usize { value } }
            fn production() { tensor4all_core::contract(dynamic); }
        "#,
        r#"
            #[cfg(any())]
            fn contract(value: usize) -> usize { value }
            fn production() { contract(dynamic); }
        "#,
        r#"
            fn production() {
                #[cfg(any())]
                extern crate self as tensor4all_core;
                tensor4all_core::contract(dynamic);
            }
        "#,
        r#"
            fn production() {
                contract(lhs, rhs, third);
                use local::*;
                use crate::tensor::*;
                mod local {
                    #[cfg(any())]
                    pub fn contract(value: usize) -> usize { value }
                }
            }
        "#,
    ] {
        assert!(
            !audit_fixture("src/fixture.rs", source).is_empty(),
            "a cfg-disabled local symbol hid a production API: {source}"
        );
    }

    for source in [
        r#"
            #[cfg(feature = "local")]
            fn contract(value: usize) -> usize { value }
            #[cfg(feature = "local")]
            fn production() { contract(value); }
        "#,
        r#"
            #[cfg(feature = "local")]
            mod tensor4all_core { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            fn production() { tensor4all_core::contract(value); }
        "#,
        r#"
            #[cfg(feature = "local")]
            fn production() {
                #[cfg(feature = "local")]
                fn contract(value: usize) -> usize { value }
                contract(value);
            }
        "#,
        r#"
            mod local { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            use local::contract;
            #[cfg(feature = "local")]
            fn production() { contract(value); }
        "#,
        r#"
            mod local { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            use local as tensor4all_core;
            #[cfg(feature = "local")]
            fn production() { tensor4all_core::contract(value); }
        "#,
        r#"
            mod local { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            fn production() {
                #[cfg(feature = "local")]
                use crate::local::contract;
                contract(value);
            }
        "#,
        r#"
            mod local { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            fn production() {
                #[cfg(feature = "local")]
                use crate::local as tensor4all_core;
                tensor4all_core::contract(value);
            }
        "#,
        r#"
            mod local { pub fn contract(value: usize) -> usize { value } }
            #[cfg(feature = "local")]
            use local::*;
            #[cfg(feature = "local")]
            fn production() { contract(value); }
        "#,
    ] {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "a cfg-correlated local symbol is an active lexical shadow: {source}"
        );
    }
}

#[test]
fn ast_audit_resolves_exact_module_ancestors_and_crate_self_aliases() {
    let rejected = [
        r#"
            use tenferro as t;
            mod first {
                use crate::tensor as t;
                mod second {
                    use super::super::t as selected;
                    fn production() { selected::contract(lhs, rhs); }
                }
            }
        "#,
        r#"
            use crate::tensor as t;
            mod first {
                use tenferro as t;
                mod second {
                    use super::super::t as selected;
                    fn production() { selected::contract(lhs, rhs, third); }
                }
            }
        "#,
    ];
    for source in rejected {
        assert!(!audit_fixture("src/lib.rs", source).is_empty(), "{source}");
    }

    let accepted = [
        "extern crate self as project; fn production() { project::tensor::contract(lhs, rhs); }",
        r#"
            mod outer {
                mod tensor { pub fn contract(lhs: usize, rhs: usize, third: usize) {} }
                mod inner {
                    use super as project;
                    fn production() { project::tensor::contract(lhs, rhs, third); }
                }
            }
        "#,
    ];
    for source in accepted {
        assert!(audit_fixture("src/lib.rs", source).is_empty(), "{source}");
    }
}

#[test]
fn ast_audit_tracks_all_lexical_pattern_shadows() {
    for source in [
        "fn production() { let (contract, value) = callbacks; contract(value); }",
        "fn production() { values.map(|(contract, value)| contract(value)); }",
        "fn production() { for (contract, value) in callbacks { contract(value); } }",
        "fn production() { match callback { (contract, value) => contract(value) } }",
        "fn production() { if let Some((contract, value)) = callback { contract(value); } }",
        "fn production() { while let Some((contract, value)) = callbacks.next() { contract(value); } }",
    ] {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }
}

#[test]
fn ast_audit_requires_inherent_not_trait_owner_identities() {
    let rejected = [
        r#"
            trait Candidate { fn one_call_nary_transfer_candidate(); }
            impl Candidate for UniformTransferKernel {
                #[cfg(test)]
                fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); }
            }
        "#,
        r#"
            trait Actions { fn apply_left_ac(&self); }
            impl Actions for HamiltonianKernelCache {
                fn apply_left_ac(&self) {
                    crate::uniform_hamiltonian_kernel::hamiltonian_action_outer_product(stage, lhs, rhs);
                }
            }
        "#,
        r#"
            impl Other {
                fn hamiltonian_action_outer_product() {
                    tensor4all_core::outer_product(lhs, rhs);
                }
            }
        "#,
    ];
    for source in rejected {
        assert!(
            !audit_fixture("src/uniform_hamiltonian_kernel.rs", source).is_empty(),
            "forged owner escaped: {source}"
        );
    }
}

#[test]
fn ast_audit_requires_full_item_owners_and_rejects_outer_helper_values() {
    let nested_candidate = r#"
        impl UniformTransferKernel {
            #[cfg(test)]
            fn wrapper() {
                #[cfg(test)]
                fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); }
                one_call_nary_transfer_candidate();
            }
        }
    "#;
    assert!(!audit_fixture("src/uniform_transfer_kernel.rs", nested_candidate).is_empty());

    let nested_pairwise = r#"
        mod nested {
            fn pairwise() { tensor4all_core::contract_pair(lhs, rhs); }
        }
    "#;
    assert!(!audit_fixture(PAIRWISE_ADAPTER_OWNER, nested_pairwise).is_empty());

    let outer_helper_cast = r#"
        fn hamiltonian_action_outer_product(stage: usize, lhs: usize, rhs: usize) {}
        fn third_helper() {
            let action = hamiltonian_action_outer_product as fn(usize, usize, usize);
            action(stage, lhs, rhs);
        }
    "#;
    let violations = audit_fixture("src/uniform_hamiltonian_kernel.rs", outer_helper_cast);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("function value outside AC actions")),
        "{violations:?}"
    );

    let owner_function_values = [
        (
            PAIRWISE_ADAPTER_OWNER,
            "fn pairwise() { consume(tensor4all_core::contract_pair); }",
        ),
        (
            "src/uniform_hamiltonian_kernel.rs",
            "fn hamiltonian_action_outer_product() { let call = tensor4all_core::outer_product as fn(_, _); consume(call); }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "impl UniformTransferKernel { #[cfg(test)] fn one_call_nary_transfer_candidate() { let call = tensor4all_core::contract; call(dynamic); } }",
        ),
    ];
    for (path, source) in owner_function_values {
        assert!(
            !audit_fixture(path, source).is_empty(),
            "restricted function value escaped from exact owner: {source}"
        );
    }
}

#[test]
fn ast_audit_does_not_treat_local_names_as_tensor_api_imports() {
    let local_bindings = [
        "fn production() { let contract = |value| value; contract(value); }",
        "fn contract(value: usize) -> usize { value } fn production() { contract(value); }",
        "const contract: usize = 1; fn production() { contract(value); }",
        "fn production() { let outer_product = |value| value; outer_product(value); }",
        "fn outer() { mod local { fn contract(value: usize) -> usize { value } fn production() { contract(value); } } }",
        "fn outer() { mod local { mod api { pub fn contract(value: usize) -> usize { value } } fn production() { api::contract(value); } } }",
        "fn outer() { mod local { mod sibling { pub fn contract(value: usize) -> usize { value } } use sibling as api; fn production() { api::contract(value); } } }",
        "fn outer() { mod local { mod sibling { pub fn contract(value: usize) -> usize { value } } use sibling::*; fn production() { contract(value); } } }",
        "fn outer() { mod local { mod sibling { pub fn contract(value: usize) -> usize { value } } use self::sibling as api; fn production() { api::contract(value); } } }",
        "mod local { pub fn contract(value: usize) -> usize { value } } use local::*; fn production() { contract(value); }",
    ];
    assert_eq!(local_bindings.len(), 10);
    for source in local_bindings {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }

    let associated = r#"
        impl Local {
            fn contract(value: usize) -> usize { value }
            fn production() { Self::contract(value); }
        }
    "#;
    assert!(audit_fixture("src/fixture.rs", associated).is_empty());

    let adapter_aliases = [
        "use crate::tensor as x; fn production() { x::contract(lhs, rhs); }",
        "use crate::tensor as x; use x::contract as pairwise; fn production() { pairwise(lhs, rhs); }",
    ];
    for source in adapter_aliases {
        assert!(
            audit_fixture("src/fixture.rs", source).is_empty(),
            "{source}"
        );
    }
}

#[test]
fn ast_audit_tracks_outer_helper_function_items_and_module_aliases() {
    let function_item = r#"
        fn hamiltonian_action_outer_product(stage: usize, lhs: usize, rhs: usize) {}
        fn third_helper() {
            let action = hamiltonian_action_outer_product;
            action(stage, lhs, rhs);
        }
    "#;
    let module_alias = r#"
        use crate::uniform_hamiltonian_kernel as kernel;
        fn third_helper() {
            let action = kernel::hamiltonian_action_outer_product;
            (action)(stage, lhs, rhs);
        }
    "#;
    for source in [function_item, module_alias] {
        let violations = audit_fixture("src/uniform_hamiltonian_kernel.rs", source);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("outside AC actions")),
            "{source}: {violations:?}"
        );
    }
}

#[test]
fn ast_audit_resolves_binary_contracts_only_through_their_import_or_path() {
    let imported = r#"
        use crate::tensor::contract;
        fn production() { contract(lhs, rhs); }
    "#;
    assert!(audit_fixture("src/fixture.rs", imported).is_empty());

    let tenferro = r#"
        use tenferro::contract;
        fn production() { contract(lhs, rhs); }
    "#;
    assert!(!audit_fixture("src/tensor.rs", tenferro).is_empty());
}

#[test]
fn ast_audit_requires_exact_candidate_location_and_outer_product_callers() {
    for (path, source) in [
        (
            "src/uniform_transfer_kernel.rs",
            "mod nested { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "impl Other { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "impl crate::UniformTransferKernel { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } }",
        ),
        (
            "src/uniform_transfer_kernel.rs",
            "mod lookalike { impl UniformTransferKernel { #[cfg(test)] fn one_call_nary_transfer_candidate() { tensor4all_core::contract(dynamic); } } }",
        ),
        (
            "src/uniform_hamiltonian_kernel.rs",
            "fn third_helper() { crate::uniform_hamiltonian_kernel::hamiltonian_action_outer_product(stage, lhs, rhs); }",
        ),
    ] {
        let violations = audit_fixture(path, source);
        assert!(!violations.is_empty(), "{path}: {violations:?}");
    }
}

#[test]
fn ast_audit_rejects_handwritten_production_reductions() {
    let production = r#"
        fn production() {
            let mut sum = 0.0;
            for index in 0..count { sum += lhs[index] * rhs[index]; }
        }
    "#;
    assert!(!audit_fixture("src/fixture.rs", production).is_empty());

    let test_oracle = r#"
        #[cfg(test)]
        fn oracle() {
            let mut sum = 0.0;
            for index in 0..count { sum += lhs[index] * rhs[index]; }
        }
    "#;
    assert!(audit_fixture("src/fixture.rs", test_oracle).is_empty());

    let nested_product = r#"
        fn production() {
            let mut sum = 0.0;
            for index in 0..count { sum = sum + bias + scale * lhs[index] * rhs[index]; }
        }
    "#;
    assert!(!audit_fixture("src/fixture.rs", nested_product).is_empty());

    let control_loop = r#"
        fn production() {
            let mut sum = 0.0;
            for index in 0..count { sum += weights[index] * scale; }
        }
    "#;
    assert!(audit_fixture("src/fixture.rs", control_loop).is_empty());

    for reduction in [
        "fn production() { while ready { total += lhs[index] * rhs[index]; } }",
        "fn production() { loop { total += lhs[index] * rhs[index]; break; } }",
        "fn production() { for index in 0..count { total -= lhs[index] * rhs[index]; } }",
        "fn production() { let total: f64 = values.iter().map(|index| lhs[index] * rhs[index]).sum(); }",
        "fn production() { for index in 0..count { let product = lhs[index] * rhs[index]; total += product; } }",
        "fn production() { for index in 0..count { let left = lhs[index]; let right = rhs[index]; total += left * right; } }",
        "fn production() { for index in 0..count { let mut left = 0.0; left = lhs[index]; total += left * rhs[index]; } }",
        "fn production() { let mut total = 0.0; for index in 0..count { { let total = 0.0; consume(total); } total += lhs[index] * rhs[index]; } }",
        "fn production() { for (left, right) in lhs.iter().zip(rhs) { total += left * right; } }",
        "fn production() { let total = values.iter().fold(0.0, |sum, index| sum + lhs[index] * rhs[index]); }",
        "fn production() { let total = values.iter().map(|index| lhs[index] * rhs[index]).reduce(|sum, value| sum + value); }",
        "fn production() { let total = lhs.iter().zip(rhs).fold(0.0, |sum, (left, right)| sum + left * right); }",
        "fn production() { let total = values.iter().fold(0.0, |sum, index| sum - lhs[index] * rhs[index]); }",
        "fn production() { let total = values.iter().fold(0.0, |mut sum, index| { sum += lhs[index] * rhs[index]; sum }); }",
        "fn production() { let total = Iterator::fold(values.iter(), 0.0, |sum, index| sum + lhs[index] * rhs[index]); }",
        "fn production() { let total = Iterator::reduce(values.iter().map(|index| lhs[index] * rhs[index]), |sum, value| sum + value); }",
        "fn production() { let total: f64 = Iterator::sum(values.iter().map(|index| lhs[index] * rhs[index])); }",
        "fn production() { values.iter().fold(out, |mut out, index| { out[row] += lhs[row][index] * rhs[index]; out }); }",
        "fn production() { for index in 0..count { let output_index = index % 2; out[output_index] += lhs[index] * rhs[index]; } }",
    ] {
        assert!(!audit_fixture("src/fixture.rs", reduction).is_empty(), "{reduction}");
    }

    for elementwise in [
        "fn production() { for index in 0..count { out[index] = lhs[index] * rhs[index]; } }",
        "fn production() { for index in 0..count { out[index] += lhs[index] * rhs[index]; } }",
        "fn production() { while ready { out[index] = lhs[index] * rhs[index]; } }",
        "fn production() { total += lhs[index] * rhs[index]; }",
        "fn production() { for index in 0..count { out[index] = lhs[offset(index)] * rhs[offset(other)]; } }",
        "fn production() { let product = lhs[index] * rhs[index]; for product in values { total += product; } }",
        "fn production() { let product = lhs[index] * rhs[index]; values.iter().fold(0.0, |product, value| product + value); }",
        "fn production() { for (index, (left, right)) in lhs.iter().zip(rhs).enumerate() { out[index] += left * right; } }",
        "fn production() { for index in 0..count { let destination = &mut out[index]; *destination += lhs[index] * rhs[index]; } }",
        "fn production() { for index in 0..count { let output_index = index; out[output_index] += lhs[index] * rhs[index]; } }",
        "fn production() { for index in 0..count { let mut value = 0.0; value += lhs[index] * rhs[index]; out[index] = value; } }",
        "fn production() { values.iter().enumerate().fold(out, |mut out, (index, _)| { out[index] += lhs[index] * rhs[index]; out }); }",
    ] {
        assert!(
            audit_fixture("src/fixture.rs", elementwise).is_empty(),
            "elementwise assignment is not a reduction: {elementwise}"
        );
    }
}

#[test]
fn ast_audit_propagates_contraction_provenance_through_multiply_assignment() {
    for reduction in [
        r#"
            fn production() {
                for k in 0..n {
                    let mut product = lhs[k];
                    product *= rhs[k];
                    total += product;
                }
            }
        "#,
        r#"
            fn production() {
                values.fold(0.0, |mut total, k| {
                    let mut product = lhs[k];
                    product *= rhs[k];
                    total += product;
                    total
                });
            }
        "#,
    ] {
        assert!(
            !audit_fixture("src/fixture.rs", reduction).is_empty(),
            "mutable staged contraction escaped: {reduction}"
        );
    }

    let elementwise = r#"
        fn production() {
            for k in 0..n {
                let mut product = lhs[k];
                product *= rhs[k];
                out[k] += product;
            }
        }
    "#;
    assert!(
        audit_fixture("src/fixture.rs", elementwise).is_empty(),
        "mutable staged elementwise operation is not a reduction"
    );
}

#[test]
fn ast_audit_distinguishes_referenced_output_slot_granularity() {
    for reduction in [
        r#"
            fn production() {
                for k in 0..n {
                    let slot = &mut out[row];
                    *slot += lhs[k] * rhs[k];
                }
            }
        "#,
        r#"
            fn production() {
                for k in 0..n {
                    let slot = &mut out[k % 2];
                    *slot += lhs[k] * rhs[k];
                }
            }
        "#,
        r#"
            fn production() {
                values.fold(out, |mut out, k| {
                    let slot = &mut out[row];
                    *slot += lhs[k] * rhs[k];
                    out
                });
            }
        "#,
        r#"
            fn production() {
                values.fold(out, |mut out, k| {
                    let slot = &mut out[k % 2];
                    *slot += lhs[k] * rhs[k];
                    out
                });
            }
        "#,
    ] {
        assert!(
            !audit_fixture("src/fixture.rs", reduction).is_empty(),
            "fixed or coarser output-slot reduction escaped: {reduction}"
        );
    }

    for elementwise in [
        r#"
            fn production() {
                for k in 0..n {
                    let slot = &mut out[k];
                    *slot += lhs[k] * rhs[k];
                }
            }
        "#,
        r#"
            fn production() {
                values.fold(out, |mut out, k| {
                    let slot = &mut out[k];
                    *slot += lhs[k] * rhs[k];
                    out
                });
            }
        "#,
    ] {
        assert!(
            audit_fixture("src/fixture.rs", elementwise).is_empty(),
            "referenced elementwise output is not a reduction: {elementwise}"
        );
    }
}

#[test]
fn ast_audit_rejects_all_removed_handwritten_reduction_owners() {
    let dual_density = r#"
        fn canonical_dual_density() {
            for index in 0..count { total += lhs[index] * rhs[index]; }
        }
    "#;
    assert!(!audit_fixture("src/uniform_environment.rs", dual_density).is_empty());
    assert!(!audit_fixture("src/fixture.rs", dual_density).is_empty());

    let iterator_reduction = r#"
        fn inner() {
            let total: f64 = values.iter().map(|index| lhs[index] * rhs[index]).sum();
        }
    "#;
    assert!(!audit_fixture("src/uniform_tensor.rs", iterator_reduction).is_empty());
    assert!(!audit_fixture(
        "src/uniform_tensor.rs",
        &iterator_reduction.replacen("inner", "other", 1)
    )
    .is_empty());

    let schmidt_reduction = iterator_reduction.replacen("inner", "reconstruct_dense", 1);
    assert!(!audit_fixture("src/uniform_operator_schmidt.rs", &schmidt_reduction).is_empty());

    let procrustes_reduction = iterator_reduction.replacen("inner", "mixed_transfer_procrustes", 1);
    assert!(!audit_fixture("src/uniform_gauge.rs", &procrustes_reduction).is_empty());

    let no_reduction = syn::parse_file("fn inner() { let value = 1; }").unwrap();
    assert!(
        deterministic_boundary::audit_repository_file("src/uniform_tensor.rs", &no_reduction)
            .inventory_seen
            .is_empty(),
        "an inventory identity is current only while its source still contains a detected contraction"
    );

    for stale in [
        ("src/model.rs", "local"),
        ("src/uniform_bond_expansion.rs", "embed_state"),
    ] {
        let source = format!(
            "fn {}() {{ for index in 0..count {{ total += lhs[index] * rhs[index]; }} }}",
            stale.1
        );
        assert!(
            !audit_fixture(stale.0, &source).is_empty(),
            "stale inventory identity remained active: {stale:?}"
        );
    }
}

#[test]
fn ast_audit_rejects_every_nonbinary_production_contract_form() {
    for (name, expression) in [
        ("zero", "tensor4all_core::contract(&[])"),
        ("one", "tensor4all_core::contract(&[lhs])"),
        ("three", "tensor4all_core::contract(&[lhs, rhs, third])"),
        ("dynamic-slice", "tensor4all_core::contract(tensors)"),
        (
            "borrowed-dynamic-slice",
            "tensor4all_core::contract(&tensors)",
        ),
        ("vec", "tensor4all_core::contract(&vec![lhs, rhs])"),
    ] {
        let source = format!("fn production() {{ let _ = {expression}; }}");
        let violations = audit_fixture("src/fixture.rs", &source);
        assert_eq!(violations.len(), 1, "{name}: {violations:?}");
    }

    let macro_source = "fn production() { wrapper!(contract(&[lhs, rhs])); }";
    let violations = audit_fixture("src/fixture.rs", macro_source);
    assert_eq!(violations.len(), 1, "macro: {violations:?}");

    let wrong_two_argument_api = "fn production() { tensor4all_core::contract(lhs, rhs); }";
    let violations = audit_fixture("src/fixture.rs", wrong_two_argument_api);
    assert_eq!(violations.len(), 1, "wrong API shape: {violations:?}");

    for (path, source) in [
        (
            "src/fixture.rs",
            "fn production() { crate::tensor::contract(lhs, rhs); }",
        ),
        (
            "src/observable.rs",
            "use crate::tensor::contract; fn production() { contract(lhs, rhs); }",
        ),
    ] {
        let violations = audit_fixture(path, source);
        assert!(violations.is_empty(), "binary adapter: {violations:?}");
    }
}

#[test]
fn ast_audit_checks_binary_adapter_arity_through_casts() {
    for source in [
        "fn production() { (crate::tensor::contract as fn(_, _))(lhs, rhs, third); }",
        "fn production() { let binary = crate::tensor::contract; (binary as fn(_, _))(lhs, rhs, third); }",
    ] {
        let violations = audit_fixture("src/fixture.rs", source);
        assert_eq!(violations.len(), 1, "{source}: {violations:?}");
    }

    for source in [
        "fn production() { (crate::tensor::contract as fn(_, _))(lhs, rhs); }",
        "fn production() { let binary = crate::tensor::contract; (binary as fn(_, _))(lhs, rhs); }",
    ] {
        let violations = audit_fixture("src/fixture.rs", source);
        assert!(violations.is_empty(), "{source}: {violations:?}");
    }
}

#[test]
fn ast_audit_treats_only_cfgs_that_imply_test_as_test_context() {
    for attribute in [
        "#[cfg(not(test))]",
        "#[cfg(all(unix, not(test)))]",
        "#[cfg(any(not(test), feature = \"fixture\"))]",
        "#[cfg(any(feature = \"fixture\", test))]",
        "#[cfg(any(all(test, unix), feature = \"fixture\"))]",
        "#[cfg(custom(test))]",
        "#[cfg(not(test, unix))]",
    ] {
        let source = format!(
            "{attribute} fn contract_candidate() {{ tensor4all_core::contract(&[a, b, c]); }}"
        );
        let violations = audit_fixture("src/contraction_contract_candidate.rs", &source);
        assert_eq!(violations.len(), 1, "{attribute}: {violations:?}");
    }

    for attribute in [
        "#[test]",
        "#[cfg(test)]",
        "#[cfg(all(test, unix))]",
        "#[cfg(not(not(test)))]",
        "#[cfg(all(any(test, feature = \"fixture\"), test))]",
    ] {
        let source = format!(
            "{attribute} fn contract_candidate() {{ tensor4all_core::contract(&[a, b, c]); }}"
        );
        let violations = audit_fixture("src/contraction_contract_candidate.rs", &source);
        assert!(violations.is_empty(), "{attribute}: {violations:?}");
    }

    let multiple_cfgs = r#"
        #[cfg(any(test, feature = "fixture"))]
        #[cfg(test)]
        fn contract_candidate() { tensor4all_core::contract(&[a, b, c]); }
    "#;
    assert!(
        audit_fixture("src/contraction_contract_candidate.rs", multiple_cfgs).is_empty(),
        "conjoined cfg attributes imply test when any one requires it"
    );
}

#[test]
fn ast_audit_candidate_exception_is_an_exact_path_and_function_allowlist() {
    let source = "#[cfg(test)] fn contract_candidate() { tensor4all_core::contract(dynamic); }";
    assert!(
        audit_fixture("src/contraction_contract_candidate.rs", source).is_empty(),
        "exact allowlist entry must accept its historical dynamic candidate"
    );
    for (path, function) in [
        (
            "src/not_contraction_contract_candidate.rs",
            "contract_candidate",
        ),
        (
            "src/contraction_contract_candidate.rs",
            "contract_candidate_extra",
        ),
        (
            "src/contraction_contract_candidate.rs",
            "prefix_contract_candidate",
        ),
    ] {
        let source =
            format!("#[cfg(test)] fn {function}() {{ tensor4all_core::contract(dynamic); }}");
        let violations = audit_fixture(path, &source);
        assert_eq!(violations.len(), 1, "{path}::{function}: {violations:?}");
    }
}

#[test]
fn ast_audit_exempts_contract_macros_only_for_exact_test_candidate() {
    let exact = r#"
        #[cfg(test)]
        fn contract_candidate() { wrapper!(contract(dynamic)); }
    "#;
    assert!(
        audit_fixture("src/contraction_contract_candidate.rs", exact).is_empty(),
        "exact test-only candidate macro must remain available"
    );

    for (name, path, source) in [
        (
            "ordinary-test",
            "src/fixture.rs",
            "#[test] fn helper() { wrapper!(contract(&[a, b, c])); }",
        ),
        (
            "near-name",
            "src/contraction_contract_candidate.rs",
            "#[cfg(test)] fn contract_candidate_extra() { wrapper!(contract(dynamic)); }",
        ),
        (
            "feature-or-test",
            "src/contraction_contract_candidate.rs",
            "#[cfg(any(feature = \"fixture\", test))] fn contract_candidate() { wrapper!(contract(dynamic)); }",
        ),
    ] {
        let violations = audit_fixture(path, source);
        assert_eq!(violations.len(), 1, "{name}: {violations:?}");
    }
}

#[derive(Clone, Copy, Debug)]
enum AuditTopology {
    Transfer3,
    Transfer4,
    Transfer6,
    VarianceKet3,
    VarianceOperator3,
    VarianceClose4,
}

fn benchmark_index(dimension: usize) -> tensor4all_core::DynIndex {
    tensor4all_core::DynIndex::new_dyn(dimension)
}

fn benchmark_tensor(
    indices: Vec<tensor4all_core::DynIndex>,
    complex: bool,
) -> crate::tensor::Tensor {
    let len = indices.iter().map(|index| index.dim).product::<usize>();
    if complex {
        crate::tensor::Tensor::from_dense(
            indices,
            (0..len)
                .map(|index| {
                    let value = (index % 17 + 1) as f64 / 19.0;
                    num_complex::Complex64::new(value, -0.37 * value)
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        crate::tensor::Tensor::from_dense(
            indices,
            (0..len)
                .map(|index| (index % 17 + 1) as f64 / 19.0)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
}

fn benchmark_network(
    topology: AuditTopology,
    chi: usize,
    complex: bool,
) -> Vec<crate::tensor::Tensor> {
    let ix = |dimension| benchmark_index(dimension);
    let tensor = |indices| benchmark_tensor(indices, complex);
    match topology {
        AuditTopology::Transfer3 => {
            let lb = ix(chi);
            let lk = ix(chi);
            let rk = ix(chi);
            let rb = ix(chi);
            let p = ix(2);
            let a = ix(2);
            vec![
                tensor(vec![lb.clone(), lk.clone()]),
                tensor(vec![lk, p.clone(), a.clone(), rk]),
                tensor(vec![lb, p, a, rb]),
            ]
        }
        AuditTopology::Transfer4 => {
            let lb = ix(chi);
            let lk = ix(chi);
            let rk = ix(chi);
            let rb = ix(chi);
            let pin = ix(2);
            let pout = ix(2);
            let a = ix(2);
            vec![
                tensor(vec![lb.clone(), lk.clone()]),
                tensor(vec![lk, pin.clone(), a.clone(), rk]),
                tensor(vec![pin, pout.clone()]),
                tensor(vec![lb, pout, a, rb]),
            ]
        }
        AuditTopology::Transfer6 => {
            let lb = ix(chi);
            let lk = ix(chi);
            let kmk = ix(chi);
            let rk = ix(chi);
            let bmb = ix(chi);
            let rb = ix(chi);
            let p0 = ix(2);
            let p1 = ix(2);
            let q0 = ix(2);
            let q1 = ix(2);
            let a0 = ix(2);
            let a1 = ix(2);
            vec![
                tensor(vec![lb.clone(), lk.clone()]),
                tensor(vec![lk, p0.clone(), a0.clone(), kmk.clone()]),
                tensor(vec![kmk, p1.clone(), a1.clone(), rk]),
                tensor(vec![p0, p1, q0.clone(), q1.clone()]),
                tensor(vec![lb, q0, a0, bmb.clone()]),
                tensor(vec![bmb, q1, a1, rb]),
            ]
        }
        AuditTopology::VarianceKet3 => {
            let l = ix(chi);
            let m0 = ix(chi);
            let m1 = ix(chi);
            let r = ix(chi);
            vec![
                tensor(vec![l, ix(2), ix(2), m0.clone()]),
                tensor(vec![m0, ix(2), ix(2), m1.clone()]),
                tensor(vec![m1, ix(2), ix(2), r]),
            ]
        }
        AuditTopology::VarianceOperator3 => {
            let p0 = ix(2);
            let p1 = ix(2);
            let p2 = ix(2);
            let q0 = ix(2);
            let q1m = ix(2);
            let q1 = ix(2);
            let q2 = ix(2);
            vec![
                tensor(vec![
                    ix(chi),
                    p0.clone(),
                    ix(2),
                    p1.clone(),
                    ix(2),
                    p2.clone(),
                    ix(2),
                    ix(chi),
                ]),
                tensor(vec![p0, p1, q0, q1m.clone()]),
                tensor(vec![q1m, p2, q1, q2]),
            ]
        }
        AuditTopology::VarianceClose4 => {
            let kl = ix(chi);
            let bl = ix(chi);
            let kr = ix(chi);
            let br = ix(chi);
            let p0 = ix(2);
            let p1 = ix(2);
            let p2 = ix(2);
            vec![
                tensor(vec![kl.clone(), bl.clone()]),
                tensor(vec![kl, p0.clone(), p1.clone(), p2.clone(), kr.clone()]),
                tensor(vec![bl, p0, p1, p2, br.clone()]),
                tensor(vec![kr, br]),
            ]
        }
    }
}

#[cfg(test)]
fn audit_one_call_nary_candidate(tensors: &[crate::tensor::Tensor]) -> crate::tensor::Tensor {
    tensor4all_core::contract(&tensors.iter().collect::<Vec<_>>()).unwrap()
}

#[derive(Debug, PartialEq, Eq)]
struct AuditStageShape {
    label: &'static str,
    lhs_dims: Vec<usize>,
    rhs_dims: Vec<usize>,
    output_dims: Vec<usize>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct AuditStagedEvidence {
    stages: Vec<AuditStageShape>,
    peak_dims: Vec<usize>,
    peak_elements: usize,
}

impl AuditStagedEvidence {
    fn record(
        &mut self,
        label: &'static str,
        lhs: &crate::tensor::Tensor,
        rhs: &crate::tensor::Tensor,
        output: &crate::tensor::Tensor,
    ) {
        let output_dims = output.dims();
        let elements = output_dims.iter().product::<usize>().max(1);
        if elements > self.peak_elements {
            self.peak_dims = output_dims.clone();
            self.peak_elements = elements;
        }
        self.stages.push(AuditStageShape {
            label,
            lhs_dims: lhs.dims(),
            rhs_dims: rhs.dims(),
            output_dims,
        });
    }
}

fn audit_staged_candidate(
    topology: AuditTopology,
    tensors: &[crate::tensor::Tensor],
) -> (crate::tensor::Tensor, AuditStagedEvidence) {
    let binary = |lhs: &crate::tensor::Tensor, rhs: &crate::tensor::Tensor| match topology {
        AuditTopology::Transfer3 | AuditTopology::Transfer4 | AuditTopology::Transfer6 => {
            crate::transfer::transfer_contract("Task9 transfer benchmark stage", lhs, rhs)
        }
        AuditTopology::VarianceKet3
        | AuditTopology::VarianceOperator3
        | AuditTopology::VarianceClose4 => {
            crate::variance::variance_contract("Task9 variance benchmark stage", lhs, rhs)
                .unwrap()
        }
    };
    let mut evidence = AuditStagedEvidence::default();
    let mut apply = |label, lhs: &crate::tensor::Tensor, rhs: &crate::tensor::Tensor| {
        let output = binary(lhs, rhs);
        evidence.record(label, lhs, rhs, &output);
        output
    };
    let output = match topology {
        AuditTopology::Transfer3 => {
            let first = apply("environment*ket", &tensors[0], &tensors[1]);
            apply("first*bra", &first, &tensors[2])
        }
        AuditTopology::VarianceKet3 => {
            let first = apply("site0*site1", &tensors[0], &tensors[1]);
            apply("first*site2", &first, &tensors[2])
        }
        AuditTopology::Transfer4 => {
            let first = apply("environment*ket", &tensors[0], &tensors[1]);
            let second = apply("first*operator", &first, &tensors[2]);
            apply("second*bra", &second, &tensors[3])
        }
        AuditTopology::VarianceClose4 => {
            let first = apply("left_cap*h_ket", &tensors[0], &tensors[1]);
            let second = apply("first*bra", &first, &tensors[2]);
            apply("second*right_cap", &second, &tensors[3])
        }
        AuditTopology::Transfer6 => {
            let first = apply("environment*ket0", &tensors[0], &tensors[1]);
            let second = apply("first*ket1", &first, &tensors[2]);
            let third = apply("second*operator", &second, &tensors[3]);
            let fourth = apply("third*bra0", &third, &tensors[4]);
            apply("fourth*bra1", &fourth, &tensors[5])
        }
        AuditTopology::VarianceOperator3 => {
            let first = apply("theta*operator0", &tensors[0], &tensors[1]);
            apply("first*operator1", &first, &tensors[2])
        }
    };
    (output, evidence)
}

fn dense_loop_oracle(
    tensors: &[crate::tensor::Tensor],
    output_indices: &[tensor4all_core::DynIndex],
) -> Vec<num_complex::Complex64> {
    let mut all_indices = Vec::new();
    for tensor in tensors {
        for index in &tensor.indices {
            if !all_indices
                .iter()
                .any(|known: &tensor4all_core::DynIndex| known.id == index.id)
            {
                all_indices.push(index.clone());
            }
        }
    }
    let tensor_values = tensors
        .iter()
        .map(|tensor| crate::test_tensor_support::dense_c64("Task9 dense oracle", tensor).unwrap())
        .collect::<Vec<_>>();
    let output_len = output_indices
        .iter()
        .map(|index| index.dim)
        .product::<usize>()
        .max(1);
    let mut output = vec![num_complex::Complex64::new(0.0, 0.0); output_len];
    let assignment_count = all_indices
        .iter()
        .map(|index| index.dim)
        .product::<usize>()
        .max(1);
    for flat_assignment in 0..assignment_count {
        let mut remaining = flat_assignment;
        let mut assignment = vec![0usize; all_indices.len()];
        for (slot, index) in assignment.iter_mut().zip(&all_indices) {
            *slot = remaining % index.dim;
            remaining /= index.dim;
        }
        let mut product = num_complex::Complex64::new(1.0, 0.0);
        for (tensor, values) in tensors.iter().zip(&tensor_values) {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for index in &tensor.indices {
                let position = all_indices
                    .iter()
                    .position(|known| known.id == index.id)
                    .unwrap();
                offset += assignment[position] * stride;
                stride *= index.dim;
            }
            product *= values[offset];
        }
        let mut output_offset = 0usize;
        let mut output_stride = 1usize;
        for index in output_indices {
            let position = all_indices
                .iter()
                .position(|known| known.id == index.id)
                .unwrap();
            output_offset += assignment[position] * output_stride;
            output_stride *= index.dim;
        }
        output[output_offset] += product;
    }
    output
}

#[test]
fn remaining_topologies_match_independent_real_and_complex_dense_oracles() {
    use crate::test_tensor_support::{dense_c64, relative_distance};
    for complex in [false, true] {
        for topology in [
            AuditTopology::Transfer3,
            AuditTopology::Transfer4,
            AuditTopology::Transfer6,
            AuditTopology::VarianceKet3,
            AuditTopology::VarianceOperator3,
            AuditTopology::VarianceClose4,
        ] {
            let tensors = benchmark_network(topology, 2, complex);
            let expected_indices = audit_one_call_nary_candidate(&tensors).indices;
            let (actual, _) = audit_staged_candidate(topology, &tensors);
            let actual = actual.permute_indices(&expected_indices).unwrap();
            assert_eq!(actual.indices, expected_indices);
            let oracle = dense_loop_oracle(&tensors, &expected_indices);
            let actual = dense_c64("Task9 staged oracle output", &actual).unwrap();
            let distance = relative_distance(&actual, &oracle);
            assert!(
                distance <= 3e-13,
                "{topology:?} complex={complex} distance={distance}"
            );
        }
    }
}

#[test]
fn staged_benchmark_evidence_records_actual_ordered_stage_shapes() {
    let tensors = benchmark_network(AuditTopology::Transfer3, 2, true);
    let (_, evidence) = audit_staged_candidate(AuditTopology::Transfer3, &tensors);
    assert_eq!(
        evidence.stages,
        vec![
            AuditStageShape {
                label: "environment*ket",
                lhs_dims: vec![2, 2],
                rhs_dims: vec![2, 2, 2, 2],
                output_dims: vec![2, 2, 2, 2],
            },
            AuditStageShape {
                label: "first*bra",
                lhs_dims: vec![2, 2, 2, 2],
                rhs_dims: vec![2, 2, 2, 2],
                output_dims: vec![2, 2],
            },
        ]
    );
    assert_eq!(evidence.peak_dims, vec![2, 2, 2, 2]);
    assert_eq!(evidence.peak_elements, 16);
}

#[test]
#[ignore = "release-only Task9 remaining-production topology comparison"]
fn benchmark_task9_remaining_nary_topologies() {
    use crate::contraction_benchmark::{measure_release, sample_nanos, sorted_median};
    use crate::test_tensor_support::{dense_c64, relative_distance};
    use std::hint::black_box;

    for complex in [false, true] {
        for chi in [8usize, 16, 24, 32] {
            for topology in [
                AuditTopology::Transfer3,
                AuditTopology::Transfer4,
                AuditTopology::Transfer6,
                AuditTopology::VarianceKet3,
                AuditTopology::VarianceOperator3,
                AuditTopology::VarianceClose4,
            ] {
                let tensors = benchmark_network(topology, chi, complex);
                let old = audit_one_call_nary_candidate(&tensors);
                let (new, evidence) = audit_staged_candidate(topology, &tensors);
                let new = new.permute_indices(&old.indices).unwrap();
                let old_dense = dense_c64("Task9 old benchmark output", &old).unwrap();
                let new_dense = dense_c64("Task9 staged benchmark output", &new).unwrap();
                let distance = relative_distance(&new_dense, &old_dense);
                assert!(
                    distance <= 2e-12,
                    "{topology:?} chi={chi} distance={distance}"
                );
                let old_samples = measure_release(|| {
                    let input = black_box(&tensors);
                    black_box(audit_one_call_nary_candidate(input))
                });
                let staged_samples = measure_release(|| {
                    let input = black_box(&tensors);
                    let output = audit_staged_candidate(topology, input).0;
                    black_box(output.permute_indices(black_box(&old.indices)).unwrap())
                });
                println!(
                    "task9-boundary storage={} chi={chi} topology={topology:?} one_call_nary_ns={:?} one_call_median_ns={} staged_binary_ns={:?} staged_median_ns={} oracle_distance={distance:.16e} stage_shapes={:?} stage_peak_dims={:?} stage_peak_elements={} warmups=3 sample_count=9",
                    if complex { "complex" } else { "real" },
                    sample_nanos(&old_samples),
                    sorted_median(&old_samples).as_nanos(),
                    sample_nanos(&staged_samples),
                    sorted_median(&staged_samples).as_nanos(),
                    evidence.stages,
                    evidence.peak_dims,
                    evidence.peak_elements,
                );
            }
        }
    }
    tensor4all_core::print_and_reset_native_einsum_profile();
}

#[test]
fn extraction_accepts_only_the_new_pairwise_owner() {
    let source = "use tensor4all_core::contract_pair; fn f() {}";
    assert!(audit_fixture("src/contraction_pairwise.rs", source).is_empty());
    assert!(!audit_fixture("src/uniform_contraction_pairwise.rs", source).is_empty());
    assert!(!audit_fixture("src/not_contraction_pairwise.rs", source).is_empty());
}
