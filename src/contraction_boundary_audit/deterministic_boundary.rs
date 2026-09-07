use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprAssign, ExprBinary, ExprBlock, ExprCall, ExprCast, ExprClosure,
    ExprForLoop, ExprPath, File, FnArg, ImplItemFn, ItemFn, ItemImpl, ItemMod, ItemUse, Local,
    Macro, Meta, Stmt, Token, UseTree,
};

const PAIRWISE_ADAPTER_OWNER: &str = "src/contraction_pairwise.rs";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RestrictedApi {
    Contract,
    ContractOwned,
    ContractPair,
    ContractPairWithOperandOptions,
    OuterProduct,
    Einsum,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApiBinding {
    BinaryContract,
    Restricted(RestrictedApi),
    HamiltonianOuterHelper,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResolvedValue {
    binding: ApiBinding,
    indirect: bool,
}

#[derive(Clone, Debug)]
struct AliasPath {
    leading_colon: bool,
    segments: Vec<String>,
    origin_module: Vec<String>,
    conditional: bool,
    condition: Option<CfgExpr>,
}

impl AliasPath {
    fn from_path(path: &syn::Path, origin_module: &[String]) -> Self {
        Self {
            leading_colon: path.leading_colon.is_some(),
            segments: path_segments(path),
            origin_module: origin_module.to_vec(),
            conditional: false,
            condition: None,
        }
    }

    fn guaranteed_in(&self, context: &CfgExpr) -> bool {
        !self.conditional
            || self
                .condition
                .as_ref()
                .is_some_and(|condition| context.implies(condition))
    }
}

#[derive(Clone, Debug)]
enum ValueDefinition {
    Local,
    Api(ApiBinding),
    Import(AliasPath),
    FunctionValue(AliasPath),
    Ambiguous,
}

#[derive(Clone, Debug)]
enum NamespaceDefinition {
    LocalModule(Vec<String>),
    Local,
    Import(AliasPath),
    CrateRoot,
    External(String),
    Ambiguous,
}

#[derive(Default)]
struct ModuleSymbols {
    values: HashMap<String, ValueDefinition>,
    namespaces: HashMap<String, NamespaceDefinition>,
    conditional_values: HashMap<String, Vec<(CfgExpr, ValueDefinition)>>,
    conditional_namespaces: HashMap<String, Vec<(CfgExpr, NamespaceDefinition)>>,
    glob_imports: Vec<AliasPath>,
}

struct SourceModel {
    root_module: Vec<String>,
    modules: HashMap<Vec<String>, ModuleSymbols>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ResolvedNamespace {
    Crate(Vec<String>),
    External { name: String, path: Vec<String> },
    BlockLocal(Box<ScopeFrame>),
    Local,
    Unknown,
}

#[derive(Clone, Debug)]
struct UseBinding {
    path: Vec<String>,
    local_name: String,
    renamed: bool,
}

fn path_segments(path: &syn::Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| identifier_name(&segment.ident))
        .collect()
}

fn identifier_name(identifier: &syn::Ident) -> String {
    let identifier = identifier.to_string();
    identifier
        .strip_prefix("r#")
        .unwrap_or(&identifier)
        .to_string()
}

fn path_is_identifier(path: &syn::Path, expected: &str) -> bool {
    path.segments.len() == 1 && identifier_name(&path.segments[0].ident) == expected
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CfgExpr {
    True,
    False,
    Atom(String),
    Not(Box<CfgExpr>),
    All(Vec<CfgExpr>),
    Any(Vec<CfgExpr>),
}

impl CfgExpr {
    fn all(expressions: Vec<Self>) -> Self {
        if expressions.contains(&Self::False) {
            Self::False
        } else {
            let expressions = expressions
                .into_iter()
                .filter(|expression| *expression != Self::True)
                .collect::<Vec<_>>();
            match expressions.as_slice() {
                [] => Self::True,
                [expression] => expression.clone(),
                _ => Self::All(expressions),
            }
        }
    }

    fn any(expressions: Vec<Self>) -> Self {
        if expressions.contains(&Self::True) {
            Self::True
        } else {
            let expressions = expressions
                .into_iter()
                .filter(|expression| *expression != Self::False)
                .collect::<Vec<_>>();
            match expressions.as_slice() {
                [] => Self::False,
                [expression] => expression.clone(),
                _ => Self::Any(expressions),
            }
        }
    }

    fn atoms(&self, output: &mut BTreeSet<String>) {
        match self {
            Self::Atom(atom) => {
                output.insert(atom.clone());
            }
            Self::Not(expression) => expression.atoms(output),
            Self::All(expressions) | Self::Any(expressions) => {
                for expression in expressions {
                    expression.atoms(output);
                }
            }
            Self::True | Self::False => {}
        }
    }

    fn evaluate(&self, assignment: &HashMap<String, bool>) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Atom(atom) => assignment.get(atom).copied().unwrap_or(false),
            Self::Not(expression) => !expression.evaluate(assignment),
            Self::All(expressions) => expressions
                .iter()
                .all(|expression| expression.evaluate(assignment)),
            Self::Any(expressions) => expressions
                .iter()
                .any(|expression| expression.evaluate(assignment)),
        }
    }

    fn implies(&self, required: &Self) -> bool {
        if *required == Self::True || *self == Self::False || self == required {
            return true;
        }
        let mut atoms = BTreeSet::new();
        self.atoms(&mut atoms);
        required.atoms(&mut atoms);
        if atoms.len() > 16 {
            return false;
        }
        let atoms = atoms.into_iter().collect::<Vec<_>>();
        for mask in 0..(1usize << atoms.len()) {
            let assignment = atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| (atom.clone(), mask & (1 << index) != 0))
                .collect::<HashMap<_, _>>();
            if self.evaluate(&assignment) && !required.evaluate(&assignment) {
                return false;
            }
        }
        true
    }
}

fn cfg_literal(expression: &Expr) -> Option<String> {
    let Expr::Lit(expression) = expression else {
        return None;
    };
    match &expression.lit {
        syn::Lit::Str(value) => Some(format!("{:?}", value.value())),
        syn::Lit::Bool(value) => Some(value.value.to_string()),
        syn::Lit::Int(value) => Some(value.base10_digits().to_string()),
        syn::Lit::Char(value) => Some(format!("{:?}", value.value())),
        _ => None,
    }
}

fn cfg_meta_expression(meta: &Meta) -> Option<CfgExpr> {
    match meta {
        Meta::Path(path) => Some(CfgExpr::Atom(path_segments(path).join("::"))),
        Meta::NameValue(value) => Some(CfgExpr::Atom(format!(
            "{}={}",
            path_segments(&value.path).join("::"),
            cfg_literal(&value.value)?
        ))),
        Meta::List(list) => {
            let children = list
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .ok()?;
            let expressions = children
                .iter()
                .map(cfg_meta_expression)
                .collect::<Option<Vec<_>>>()?;
            if path_is_identifier(&list.path, "all") {
                Some(CfgExpr::all(expressions))
            } else if path_is_identifier(&list.path, "any") {
                Some(CfgExpr::any(expressions))
            } else if path_is_identifier(&list.path, "not") && expressions.len() == 1 {
                Some(CfgExpr::Not(Box::new(expressions[0].clone())))
            } else {
                Some(CfgExpr::Atom(format!(
                    "{}({})",
                    path_segments(&list.path).join("::"),
                    list.tokens
                )))
            }
        }
    }
}

fn cfg_condition(attributes: &[Attribute]) -> Option<CfgExpr> {
    let mut conditions = Vec::new();
    for attribute in attributes {
        if path_is_identifier(attribute.path(), "test") {
            conditions.push(CfgExpr::Atom("test".to_string()));
        } else if path_is_identifier(attribute.path(), "cfg") {
            conditions.push(cfg_meta_expression(&attribute.parse_args::<Meta>().ok()?)?);
        } else if path_is_identifier(attribute.path(), "cfg_attr") {
            let arguments = attribute
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .ok()?;
            let mut arguments = arguments.iter();
            let predicate = cfg_meta_expression(arguments.next()?)?;
            for nested in arguments {
                if let Meta::List(list) = nested {
                    if path_is_identifier(&list.path, "cfg") {
                        let required = cfg_meta_expression(&list.parse_args::<Meta>().ok()?)?;
                        conditions.push(CfgExpr::any(vec![
                            CfgExpr::Not(Box::new(predicate.clone())),
                            required,
                        ]));
                    }
                }
            }
        }
    }
    Some(CfgExpr::all(conditions))
}

fn cfg_possibilities_with_test_false(meta: &Meta) -> (bool, bool) {
    match meta {
        Meta::Path(path) if path_is_identifier(path, "test") => (false, true),
        Meta::Path(_) | Meta::NameValue(_) => (true, true),
        Meta::List(list) => {
            let Ok(children) =
                list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return (true, true);
            };
            let possibilities = children
                .iter()
                .map(cfg_possibilities_with_test_false)
                .collect::<Vec<_>>();
            if path_is_identifier(&list.path, "all") {
                (
                    possibilities.iter().all(|value| value.0),
                    possibilities.iter().any(|value| value.1),
                )
            } else if path_is_identifier(&list.path, "any") {
                (
                    possibilities.iter().any(|value| value.0),
                    possibilities.iter().all(|value| value.1),
                )
            } else if path_is_identifier(&list.path, "not") && possibilities.len() == 1 {
                (possibilities[0].1, possibilities[0].0)
            } else {
                (true, true)
            }
        }
    }
}

fn has_cfg_test(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        if path_is_identifier(attribute.path(), "test") {
            return true;
        }
        path_is_identifier(attribute.path(), "cfg")
            && attribute
                .parse_args::<Meta>()
                .is_ok_and(|meta| !cfg_possibilities_with_test_false(&meta).0)
    })
}

fn is_conditionally_compiled(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        path_is_identifier(attribute.path(), "test")
            || path_is_identifier(attribute.path(), "cfg")
            || path_is_identifier(attribute.path(), "cfg_attr")
    })
}

fn collect_use_bindings(tree: &UseTree, prefix: &mut Vec<String>, output: &mut Vec<UseBinding>) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(identifier_name(&path.ident));
            collect_use_bindings(&path.tree, prefix, output);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let name = identifier_name(&name.ident);
            if name == "self" && !prefix.is_empty() {
                output.push(UseBinding {
                    path: prefix.clone(),
                    local_name: prefix.last().unwrap().clone(),
                    renamed: false,
                });
            } else {
                let mut path = prefix.clone();
                path.push(name.clone());
                output.push(UseBinding {
                    path,
                    local_name: name,
                    renamed: false,
                });
            }
        }
        UseTree::Rename(rename) => {
            let name = identifier_name(&rename.ident);
            let path = if name == "self" && !prefix.is_empty() {
                prefix.clone()
            } else {
                let mut path = prefix.clone();
                path.push(name);
                path
            };
            output.push(UseBinding {
                path,
                local_name: identifier_name(&rename.rename),
                renamed: true,
            });
        }
        UseTree::Group(group) => {
            for item in &group.items {
                collect_use_bindings(item, prefix, output);
            }
        }
        UseTree::Glob(_) => {
            output.push(UseBinding {
                path: prefix.clone(),
                local_name: "*".to_string(),
                renamed: false,
            });
        }
    }
}

fn transparent_path(expression: &Expr) -> Option<&ExprPath> {
    match expression {
        Expr::Path(path) => Some(path),
        Expr::Paren(expression) => transparent_path(&expression.expr),
        Expr::Group(expression) => transparent_path(&expression.expr),
        Expr::Cast(ExprCast { expr, .. }) => transparent_path(expr),
        Expr::Block(ExprBlock { block, .. }) if block.stmts.len() == 1 => {
            let Stmt::Expr(expression, None) = &block.stmts[0] else {
                return None;
            };
            transparent_path(expression)
        }
        _ => None,
    }
}

impl SourceModel {
    fn collect(relative_path: &str, file: &File, root_module: Vec<String>) -> Self {
        let mut model = Self {
            root_module: root_module.clone(),
            modules: HashMap::new(),
        };
        model.collect_items(relative_path, &file.items, &root_module);
        model
    }

    fn collect_items(&mut self, relative_path: &str, items: &[syn::Item], module: &[String]) {
        self.modules.entry(module.to_vec()).or_default();
        for item in items {
            match item {
                syn::Item::Mod(item) => {
                    let mut child = module.to_vec();
                    child.push(identifier_name(&item.ident));
                    if is_conditionally_compiled(&item.attrs) {
                        if let Some(condition) = cfg_condition(&item.attrs) {
                            self.insert_conditional_namespace_symbol(
                                module,
                                identifier_name(&item.ident),
                                condition,
                                NamespaceDefinition::LocalModule(child.clone()),
                            );
                        }
                    } else {
                        self.insert_namespace_symbol(
                            module,
                            identifier_name(&item.ident),
                            NamespaceDefinition::LocalModule(child.clone()),
                        );
                    }
                    if let Some((_, items)) = &item.content {
                        self.collect_items(relative_path, items, &child);
                    }
                }
                syn::Item::Fn(item) => {
                    let name = identifier_name(&item.sig.ident);
                    let binding = if module == self.root_module
                        && relative_path == "src/tensor.rs"
                        && name == "contract"
                    {
                        ValueDefinition::Api(ApiBinding::BinaryContract)
                    } else if module == self.root_module
                        && relative_path == "src/uniform_hamiltonian_kernel.rs"
                        && name == "hamiltonian_action_outer_product"
                    {
                        ValueDefinition::Api(ApiBinding::HamiltonianOuterHelper)
                    } else {
                        ValueDefinition::Local
                    };
                    if is_conditionally_compiled(&item.attrs) {
                        if let Some(condition) = cfg_condition(&item.attrs) {
                            self.insert_conditional_value_symbol(module, name, condition, binding);
                        }
                    } else {
                        self.insert_value_symbol(module, name, binding);
                    }
                }
                syn::Item::Const(item) => {
                    let conditional = is_conditionally_compiled(&item.attrs);
                    let Some(path) = transparent_path(&item.expr) else {
                        if conditional {
                            if let Some(condition) = cfg_condition(&item.attrs) {
                                self.insert_conditional_value_symbol(
                                    module,
                                    identifier_name(&item.ident),
                                    condition,
                                    ValueDefinition::Local,
                                );
                            }
                        } else {
                            self.insert_value_symbol(
                                module,
                                identifier_name(&item.ident),
                                ValueDefinition::Local,
                            );
                        }
                        continue;
                    };
                    let mut alias = AliasPath::from_path(&path.path, module);
                    alias.conditional = conditional;
                    alias.condition = conditional.then(|| cfg_condition(&item.attrs)).flatten();
                    self.insert_value_symbol(
                        module,
                        identifier_name(&item.ident),
                        ValueDefinition::FunctionValue(alias),
                    );
                }
                syn::Item::Static(item) => {
                    let conditional = is_conditionally_compiled(&item.attrs);
                    let Some(path) = transparent_path(&item.expr) else {
                        if conditional {
                            if let Some(condition) = cfg_condition(&item.attrs) {
                                self.insert_conditional_value_symbol(
                                    module,
                                    identifier_name(&item.ident),
                                    condition,
                                    ValueDefinition::Local,
                                );
                            }
                        } else {
                            self.insert_value_symbol(
                                module,
                                identifier_name(&item.ident),
                                ValueDefinition::Local,
                            );
                        }
                        continue;
                    };
                    let mut alias = AliasPath::from_path(&path.path, module);
                    alias.conditional = conditional;
                    alias.condition = conditional.then(|| cfg_condition(&item.attrs)).flatten();
                    self.insert_value_symbol(
                        module,
                        identifier_name(&item.ident),
                        ValueDefinition::FunctionValue(alias),
                    );
                }
                syn::Item::Use(item) => {
                    let mut bindings = Vec::new();
                    collect_use_bindings(&item.tree, &mut Vec::new(), &mut bindings);
                    for binding in bindings {
                        let conditional = is_conditionally_compiled(&item.attrs);
                        let alias = AliasPath {
                            leading_colon: item.leading_colon.is_some(),
                            segments: binding.path,
                            origin_module: module.to_vec(),
                            conditional,
                            condition: conditional.then(|| cfg_condition(&item.attrs)).flatten(),
                        };
                        if binding.local_name == "*" {
                            self.modules
                                .entry(module.to_vec())
                                .or_default()
                                .glob_imports
                                .push(alias);
                            continue;
                        }
                        self.insert_value_symbol(
                            module,
                            binding.local_name.clone(),
                            ValueDefinition::Import(alias.clone()),
                        );
                        self.insert_namespace_symbol(
                            module,
                            binding.local_name,
                            NamespaceDefinition::Import(alias),
                        );
                    }
                }
                syn::Item::ExternCrate(item) => {
                    let external_name = identifier_name(&item.ident);
                    let local_name = item.rename.as_ref().map_or_else(
                        || external_name.clone(),
                        |rename| identifier_name(&rename.1),
                    );
                    let binding = if external_name == "self" {
                        NamespaceDefinition::CrateRoot
                    } else {
                        NamespaceDefinition::External(external_name)
                    };
                    if is_conditionally_compiled(&item.attrs) {
                        if let Some(condition) = cfg_condition(&item.attrs) {
                            self.insert_conditional_namespace_symbol(
                                module, local_name, condition, binding,
                            );
                        }
                    } else {
                        self.insert_namespace_symbol(module, local_name, binding);
                    }
                }
                syn::Item::Struct(item) => {
                    self.bind_possibly_conditional_local_namespace(
                        module,
                        identifier_name(&item.ident),
                        &item.attrs,
                    );
                }
                syn::Item::Enum(item) => {
                    self.bind_possibly_conditional_local_namespace(
                        module,
                        identifier_name(&item.ident),
                        &item.attrs,
                    );
                }
                syn::Item::Trait(item) => {
                    self.bind_possibly_conditional_local_namespace(
                        module,
                        identifier_name(&item.ident),
                        &item.attrs,
                    );
                }
                syn::Item::Type(item) => {
                    self.bind_possibly_conditional_local_namespace(
                        module,
                        identifier_name(&item.ident),
                        &item.attrs,
                    );
                }
                syn::Item::Union(item) => {
                    self.bind_possibly_conditional_local_namespace(
                        module,
                        identifier_name(&item.ident),
                        &item.attrs,
                    );
                }
                _ => {}
            }
        }
    }

    fn bind_local_namespace(&mut self, module: &[String], name: String) {
        self.insert_namespace_symbol(module, name, NamespaceDefinition::Local);
    }

    fn bind_possibly_conditional_local_namespace(
        &mut self,
        module: &[String],
        name: String,
        attributes: &[Attribute],
    ) {
        if is_conditionally_compiled(attributes) {
            if let Some(condition) = cfg_condition(attributes) {
                self.insert_conditional_namespace_symbol(
                    module,
                    name,
                    condition,
                    NamespaceDefinition::Local,
                );
            }
        } else {
            self.bind_local_namespace(module, name);
        }
    }

    fn insert_conditional_value_symbol(
        &mut self,
        module: &[String],
        name: String,
        condition: CfgExpr,
        definition: ValueDefinition,
    ) {
        self.modules
            .entry(module.to_vec())
            .or_default()
            .conditional_values
            .entry(name)
            .or_default()
            .push((condition, definition));
    }

    fn insert_conditional_namespace_symbol(
        &mut self,
        module: &[String],
        name: String,
        condition: CfgExpr,
        definition: NamespaceDefinition,
    ) {
        self.modules
            .entry(module.to_vec())
            .or_default()
            .conditional_namespaces
            .entry(name)
            .or_default()
            .push((condition, definition));
    }

    fn insert_value_symbol(
        &mut self,
        module: &[String],
        name: String,
        definition: ValueDefinition,
    ) {
        let values = &mut self.modules.entry(module.to_vec()).or_default().values;
        match values.entry(name) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(definition);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.insert(ValueDefinition::Ambiguous);
            }
        }
    }

    fn insert_namespace_symbol(
        &mut self,
        module: &[String],
        name: String,
        definition: NamespaceDefinition,
    ) {
        let namespaces = &mut self.modules.entry(module.to_vec()).or_default().namespaces;
        match namespaces.entry(name) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(definition);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.insert(NamespaceDefinition::Ambiguous);
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ScopedValueDefinition {
    Local,
    ConditionalLocal(CfgExpr),
    Api(ResolvedValue),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ScopeFrame {
    values: HashMap<String, ScopedValueDefinition>,
    namespaces: HashMap<String, ResolvedNamespace>,
    item_scope: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConditionalValueResolution {
    Absent,
    Local,
    Api(ResolvedValue),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ItemContainer {
    Free,
    Inherent {
        self_path: Vec<String>,
    },
    TraitImpl {
        trait_path: Vec<String>,
        self_path: Vec<String>,
    },
    TraitDefinition {
        trait_name: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ItemIdentity {
    name: String,
    container: ItemContainer,
}

#[derive(Clone, Copy)]
struct InventoryItem {
    source: &'static str,
    modules: &'static [&'static str],
    name: &'static str,
}

const HANDWRITTEN_CONTRACTION_INVENTORY: &[InventoryItem] = &[];

#[derive(Clone, Default)]
struct ProductEvidence {
    contracted_indices: BTreeSet<String>,
    indexed_dependencies: BTreeSet<String>,
    value_dependencies: BTreeSet<String>,
    referent_retained_indices: Option<BTreeSet<String>>,
}

impl ProductEvidence {
    fn merge(&mut self, other: Self) {
        self.contracted_indices.extend(other.contracted_indices);
        self.indexed_dependencies.extend(other.indexed_dependencies);
        self.value_dependencies.extend(other.value_dependencies);
        match (
            &mut self.referent_retained_indices,
            other.referent_retained_indices,
        ) {
            (Some(retained), Some(other)) => retained.extend(other),
            (slot @ None, Some(other)) => *slot = Some(other),
            (Some(_), None) | (None, None) => {}
        }
    }

    fn is_empty(&self) -> bool {
        self.contracted_indices.is_empty()
    }
}

struct IdentifierVisitor {
    identifiers: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for IdentifierVisitor {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast ExprPath) {
        if path.qself.is_none() && path.path.segments.len() == 1 {
            self.identifiers
                .insert(identifier_name(&path.path.segments[0].ident));
        }
        visit::visit_expr_path(self, path);
    }
}

fn expression_identifiers(expression: &Expr) -> BTreeSet<String> {
    let mut visitor = IdentifierVisitor {
        identifiers: BTreeSet::new(),
    };
    visitor.visit_expr(expression);
    visitor.identifiers
}

fn transparent_value_dependencies(expression: &Expr) -> BTreeSet<String> {
    match expression {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            BTreeSet::from([identifier_name(&path.path.segments[0].ident)])
        }
        Expr::Paren(paren) => transparent_value_dependencies(&paren.expr),
        Expr::Group(group) => transparent_value_dependencies(&group.expr),
        _ => BTreeSet::new(),
    }
}

fn indexed_identifiers(expression: &Expr) -> BTreeSet<String> {
    struct IndexedVisitor {
        identifiers: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for IndexedVisitor {
        fn visit_expr_index(&mut self, expression: &'ast syn::ExprIndex) {
            self.identifiers
                .extend(expression_identifiers(&expression.index));
            visit::visit_expr_index(self, expression);
        }
    }
    let mut visitor = IndexedVisitor {
        identifiers: BTreeSet::new(),
    };
    visitor.visit_expr(expression);
    visitor.identifiers
}

fn collect_pattern_identifiers(pattern: &syn::Pat, output: &mut BTreeSet<String>) {
    struct PatternVisitor<'a> {
        output: &'a mut BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for PatternVisitor<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.output.insert(identifier_name(&pattern.ident));
            visit::visit_pat_ident(self, pattern);
        }
    }
    PatternVisitor { output }.visit_pat(pattern);
}

fn condition_pattern_identifiers(expression: &Expr) -> BTreeSet<String> {
    struct LetPatternVisitor {
        identifiers: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for LetPatternVisitor {
        fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
            collect_pattern_identifiers(&expression.pat, &mut self.identifiers);
            visit::visit_expr(self, &expression.expr);
        }
    }
    let mut visitor = LetPatternVisitor {
        identifiers: BTreeSet::new(),
    };
    visitor.visit_expr(expression);
    visitor.identifiers
}

struct HandwrittenReductionDetector {
    aliases: Vec<HashMap<String, ProductEvidence>>,
    loop_depth: usize,
    loop_scope_starts: Vec<usize>,
    next_synthetic_index: usize,
    closure_accumulators: BTreeSet<String>,
    closure_accumulator_found: bool,
    mutable_product_found: bool,
    found: bool,
}

impl HandwrittenReductionDetector {
    fn detect(block: &syn::Block) -> bool {
        let mut detector = Self {
            aliases: Vec::new(),
            loop_depth: 0,
            loop_scope_starts: Vec::new(),
            next_synthetic_index: 0,
            closure_accumulators: BTreeSet::new(),
            closure_accumulator_found: false,
            mutable_product_found: false,
            found: false,
        };
        detector.visit_block(block);
        detector.found
    }

    fn alias(&self, name: &str) -> Option<ProductEvidence> {
        self.aliases
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    fn product_evidence(&self, expression: &Expr) -> ProductEvidence {
        self.product_evidence_excluding(expression, &BTreeSet::new())
    }

    fn product_evidence_excluding(
        &self,
        expression: &Expr,
        excluded: &BTreeSet<String>,
    ) -> ProductEvidence {
        struct ProductVisitor<'a> {
            detector: &'a HandwrittenReductionDetector,
            excluded: &'a BTreeSet<String>,
            evidence: ProductEvidence,
        }
        impl<'ast> Visit<'ast> for ProductVisitor<'_> {
            fn visit_expr_path(&mut self, path: &'ast ExprPath) {
                if path.qself.is_none() && path.path.segments.len() == 1 {
                    let name = identifier_name(&path.path.segments[0].ident);
                    if !self.excluded.contains(&name) {
                        if let Some(evidence) = self.detector.alias(&name) {
                            self.evidence.merge(evidence);
                        }
                    }
                }
            }

            fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
                if matches!(expression.op, syn::BinOp::Mul(_)) {
                    let left = self
                        .detector
                        .product_evidence_excluding(&expression.left, self.excluded)
                        .indexed_dependencies;
                    let right = self
                        .detector
                        .product_evidence_excluding(&expression.right, self.excluded)
                        .indexed_dependencies;
                    self.evidence
                        .contracted_indices
                        .extend(left.intersection(&right).cloned());
                }
                visit::visit_expr_binary(self, expression);
            }
        }
        let mut visitor = ProductVisitor {
            detector: self,
            excluded,
            evidence: ProductEvidence {
                contracted_indices: BTreeSet::new(),
                indexed_dependencies: indexed_identifiers(expression),
                value_dependencies: transparent_value_dependencies(expression),
                referent_retained_indices: None,
            },
        };
        visitor.visit_expr(expression);
        visitor.evidence
    }

    fn expr_key(expression: &Expr) -> Option<String> {
        match expression {
            Expr::Path(path) if path.qself.is_none() => Some(path_segments(&path.path).join("::")),
            Expr::Field(field) => Some(format!(
                "{}.{}",
                Self::expr_key(&field.base)?,
                match &field.member {
                    syn::Member::Named(name) => identifier_name(name),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                }
            )),
            Expr::Index(index) => Some(format!(
                "{}[{}]",
                Self::expr_key(&index.expr)?,
                Self::expr_key(&index.index).unwrap_or_else(|| "_".to_string())
            )),
            Expr::Lit(literal) => match &literal.lit {
                syn::Lit::Int(value) => Some(value.base10_digits().to_string()),
                syn::Lit::Bool(value) => Some(value.value.to_string()),
                _ => None,
            },
            Expr::Paren(expression) => Self::expr_key(&expression.expr),
            Expr::Group(expression) => Self::expr_key(&expression.expr),
            _ => None,
        }
    }

    fn contains_place(expression: &Expr, place: &str) -> bool {
        struct PlaceVisitor<'a> {
            place: &'a str,
            found: bool,
        }
        impl<'ast> Visit<'ast> for PlaceVisitor<'_> {
            fn visit_expr(&mut self, expression: &'ast Expr) {
                self.found |= HandwrittenReductionDetector::expr_key(expression)
                    .is_some_and(|candidate| candidate == self.place);
                if !self.found {
                    visit::visit_expr(self, expression);
                }
            }
        }
        let mut visitor = PlaceVisitor {
            place,
            found: false,
        };
        visitor.visit_expr(expression);
        visitor.found
    }

    fn is_reduction_into(&self, left: &Expr, right: &Expr, requires_self: bool) -> bool {
        if requires_self {
            let Some(place) = Self::expr_key(left) else {
                return false;
            };
            if !Self::contains_place(right, &place) {
                return false;
            }
        }
        let mut evidence = self.product_evidence(right).contracted_indices;
        for retained in self.retained_index_dependencies(left) {
            evidence.remove(&retained);
        }
        !evidence.is_empty()
    }

    fn retained_index_dependencies(&self, expression: &Expr) -> BTreeSet<String> {
        struct RetainedVisitor<'a> {
            detector: &'a HandwrittenReductionDetector,
            dependencies: BTreeSet<String>,
        }
        impl<'ast> Visit<'ast> for RetainedVisitor<'_> {
            fn visit_expr_unary(&mut self, expression: &'ast syn::ExprUnary) {
                if matches!(expression.op, syn::UnOp::Deref(_)) {
                    if let Some(retained) = self
                        .detector
                        .product_evidence(&expression.expr)
                        .referent_retained_indices
                    {
                        self.dependencies.extend(retained);
                        return;
                    }
                }
                visit::visit_expr_unary(self, expression);
            }

            fn visit_expr_index(&mut self, expression: &'ast syn::ExprIndex) {
                self.dependencies
                    .extend(expression_identifiers(&expression.index));
                self.dependencies.extend(
                    self.detector
                        .product_evidence(&expression.index)
                        .indexed_dependencies,
                );
                self.dependencies.extend(
                    self.detector
                        .product_evidence(&expression.index)
                        .value_dependencies,
                );
                visit::visit_expr_index(self, expression);
            }
        }
        let mut visitor = RetainedVisitor {
            detector: self,
            dependencies: BTreeSet::new(),
        };
        visitor.visit_expr(expression);
        visitor.dependencies
    }

    fn reference_target(expression: &Expr) -> Option<&Expr> {
        match expression {
            Expr::Reference(reference) => Some(&reference.expr),
            Expr::Paren(paren) => Self::reference_target(&paren.expr),
            Expr::Group(group) => Self::reference_target(&group.expr),
            _ => None,
        }
    }

    fn referent_retained_indices(&self, expression: &Expr) -> BTreeSet<String> {
        struct ReferentVisitor<'a> {
            detector: &'a HandwrittenReductionDetector,
            retained: BTreeSet<String>,
        }
        impl<'ast> Visit<'ast> for ReferentVisitor<'_> {
            fn visit_expr_index(&mut self, expression: &'ast syn::ExprIndex) {
                self.retained.extend(
                    self.detector
                        .product_evidence(&expression.index)
                        .value_dependencies,
                );
                visit::visit_expr_index(self, expression);
            }
        }
        let mut visitor = ReferentVisitor {
            detector: self,
            retained: BTreeSet::new(),
        };
        visitor.visit_expr(expression);
        visitor.retained
    }

    fn source_contains_zip(expression: &Expr) -> bool {
        match expression {
            Expr::MethodCall(call) => {
                call.method == "zip" || Self::source_contains_zip(&call.receiver)
            }
            Expr::Paren(expression) => Self::source_contains_zip(&expression.expr),
            Expr::Group(expression) => Self::source_contains_zip(&expression.expr),
            _ => false,
        }
    }

    fn closure_product(&self, closure: &ExprClosure, source_is_zip: bool) -> bool {
        let mut bound = BTreeSet::new();
        for pattern in &closure.inputs {
            collect_pattern_identifiers(pattern, &mut bound);
        }
        if !self
            .product_evidence_excluding(&closure.body, &bound)
            .is_empty()
        {
            return true;
        }
        if Self::closure_mutable_product(closure, source_is_zip) {
            return true;
        }
        struct ClosureProductVisitor<'a> {
            bound: &'a BTreeSet<String>,
            source_is_zip: bool,
            found: bool,
        }
        impl<'ast> Visit<'ast> for ClosureProductVisitor<'_> {
            fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
                if matches!(expression.op, syn::BinOp::Mul(_)) {
                    let left = expression_identifiers(&expression.left)
                        .intersection(self.bound)
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    let right = expression_identifiers(&expression.right)
                        .intersection(self.bound)
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    let different_operands =
                        HandwrittenReductionDetector::expr_key(&expression.left)
                            != HandwrittenReductionDetector::expr_key(&expression.right);
                    self.found |= !left.is_empty()
                        && !right.is_empty()
                        && different_operands
                        && (self.source_is_zip || !left.is_disjoint(&right));
                }
                visit::visit_expr_binary(self, expression);
            }
        }
        let mut visitor = ClosureProductVisitor {
            bound: &bound,
            source_is_zip,
            found: false,
        };
        visitor.visit_expr(&closure.body);
        visitor.found
    }

    fn closure_mutable_product(closure: &ExprClosure, source_is_zip: bool) -> bool {
        Self::analyze_closure(closure, source_is_zip, BTreeSet::new()).mutable_product_found
    }

    fn analyze_closure(
        closure: &ExprClosure,
        source_is_zip: bool,
        closure_accumulators: BTreeSet<String>,
    ) -> Self {
        let zip_dependency = source_is_zip.then(|| "<zip:closure>".to_string());
        let mut shadows = HashMap::new();
        for pattern in &closure.inputs {
            let mut identifiers = BTreeSet::new();
            collect_pattern_identifiers(pattern, &mut identifiers);
            for identifier in identifiers {
                let mut evidence = ProductEvidence::default();
                if let Some(dependency) = &zip_dependency {
                    evidence.indexed_dependencies.insert(dependency.clone());
                }
                shadows.insert(identifier, evidence);
            }
        }
        let mut detector = Self {
            aliases: vec![shadows],
            loop_depth: 0,
            loop_scope_starts: Vec::new(),
            next_synthetic_index: 0,
            closure_accumulators,
            closure_accumulator_found: false,
            mutable_product_found: false,
            found: false,
        };
        detector.visit_expr(&closure.body);
        detector
    }

    fn receiver_map_contraction(&self, expression: &Expr) -> bool {
        let Expr::MethodCall(call) = expression else {
            return false;
        };
        if call.method == "map" {
            return call.args.first().is_some_and(|argument| {
                let Expr::Closure(closure) = argument else {
                    return false;
                };
                self.closure_product(closure, Self::source_contains_zip(&call.receiver))
            });
        }
        self.receiver_map_contraction(&call.receiver)
    }

    fn iterator_reduction(
        &self,
        method: &str,
        source: &Expr,
        closure_argument: Option<&Expr>,
    ) -> bool {
        if method == "sum" {
            return self.receiver_map_contraction(source);
        }
        if !matches!(method, "fold" | "reduce") {
            return false;
        }
        let closure_product = closure_argument.is_some_and(|argument| {
            let Expr::Closure(closure) = argument else {
                return false;
            };
            self.closure_combines_accumulator(closure)
                && self.closure_product(closure, Self::source_contains_zip(source))
        });
        closure_product || self.receiver_map_contraction(source)
    }

    fn closure_combines_accumulator(&self, closure: &ExprClosure) -> bool {
        let mut identifiers = BTreeSet::new();
        let Some(first) = closure.inputs.first() else {
            return false;
        };
        collect_pattern_identifiers(first, &mut identifiers);
        Self::analyze_closure(closure, false, identifiers).closure_accumulator_found
    }

    fn target_root(expression: &Expr) -> Option<String> {
        match expression {
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                Some(identifier_name(&path.path.segments[0].ident))
            }
            Expr::Field(field) => Self::target_root(&field.base),
            Expr::Index(index) => Self::target_root(&index.expr),
            Expr::Paren(paren) => Self::target_root(&paren.expr),
            Expr::Group(group) => Self::target_root(&group.expr),
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                Self::target_root(&unary.expr)
            }
            _ => None,
        }
    }

    fn target_is_current_loop_local(&self, expression: &Expr) -> bool {
        let Some(target) = Self::target_root(expression) else {
            return false;
        };
        let Some(loop_start) = self.loop_scope_starts.last() else {
            return false;
        };
        self.aliases
            .iter()
            .enumerate()
            .rev()
            .find(|(_, scope)| scope.contains_key(&target))
            .is_some_and(|(scope_index, scope)| {
                scope_index >= *loop_start
                    && scope
                        .get(&target)
                        .is_some_and(|evidence| evidence.referent_retained_indices.is_none())
            })
    }

    fn target_dependencies(&self, expression: &Expr) -> BTreeSet<String> {
        let mut dependencies = expression_identifiers(expression);
        for identifier in dependencies.clone() {
            if let Some(evidence) = self.alias(&identifier) {
                dependencies.extend(evidence.value_dependencies);
            }
        }
        dependencies
    }

    fn update_alias(&mut self, expression: &Expr, evidence: ProductEvidence) {
        let Some(name) = Self::target_root(expression) else {
            return;
        };
        if !matches!(expression, Expr::Path(_)) {
            return;
        }
        if let Some(scope) = self
            .aliases
            .iter_mut()
            .rev()
            .find(|scope| scope.contains_key(&name))
        {
            scope.insert(name, evidence);
        }
    }
}

impl<'ast> Visit<'ast> for HandwrittenReductionDetector {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.aliases.push(HashMap::new());
        visit::visit_block(self, block);
        self.aliases.pop();
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if let Some(initializer) = &local.init {
            self.visit_expr(&initializer.expr);
        }
        let mut evidence = local
            .init
            .as_ref()
            .map_or_else(ProductEvidence::default, |initializer| {
                self.product_evidence(&initializer.expr)
            });
        if let Some(target) = local
            .init
            .as_ref()
            .and_then(|initializer| Self::reference_target(&initializer.expr))
        {
            evidence.referent_retained_indices = Some(self.referent_retained_indices(target));
            if let Some(root) = Self::target_root(target) {
                evidence.value_dependencies.insert(root);
            }
        }
        let mut identifiers = BTreeSet::new();
        collect_pattern_identifiers(&local.pat, &mut identifiers);
        let scope = self.aliases.last_mut().expect("reduction scope");
        for identifier in identifiers {
            scope.insert(identifier, evidence.clone());
        }
    }

    fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
        if !self.closure_accumulators.is_empty() {
            if matches!(
                expression.op,
                syn::BinOp::AddAssign(_) | syn::BinOp::SubAssign(_)
            ) {
                let target = self.target_dependencies(&expression.left);
                self.closure_accumulator_found |= !target.is_disjoint(&self.closure_accumulators)
                    && (self.is_reduction_into(&expression.left, &expression.right, false)
                        || self
                            .retained_index_dependencies(&expression.left)
                            .is_empty());
            } else if matches!(expression.op, syn::BinOp::Add(_) | syn::BinOp::Sub(_)) {
                for (operand, contribution) in [
                    (&expression.left, &expression.right),
                    (&expression.right, &expression.left),
                ] {
                    let used = self.target_dependencies(operand);
                    self.closure_accumulator_found |= !used.is_disjoint(&self.closure_accumulators)
                        && (self.is_reduction_into(operand, contribution, false)
                            || self.retained_index_dependencies(operand).is_empty());
                }
            }
        }
        if self.loop_depth > 0
            && matches!(
                expression.op,
                syn::BinOp::AddAssign(_) | syn::BinOp::SubAssign(_)
            )
            && !self.target_is_current_loop_local(&expression.left)
            && self.is_reduction_into(&expression.left, &expression.right, false)
        {
            self.found = true;
        }
        self.visit_expr(&expression.left);
        self.visit_expr(&expression.right);
        if matches!(
            expression.op,
            syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        ) {
            let mut evidence = self.product_evidence(&expression.left);
            let right_evidence = self.product_evidence(&expression.right);
            if matches!(expression.op, syn::BinOp::MulAssign(_)) {
                evidence.contracted_indices.extend(
                    evidence
                        .indexed_dependencies
                        .intersection(&right_evidence.indexed_dependencies)
                        .cloned(),
                );
                self.mutable_product_found |= !evidence.is_empty();
            }
            evidence.merge(right_evidence);
            self.update_alias(&expression.left, evidence);
        }
    }

    fn visit_expr_assign(&mut self, expression: &'ast ExprAssign) {
        if self.loop_depth > 0
            && !self.target_is_current_loop_local(&expression.left)
            && self.is_reduction_into(&expression.left, &expression.right, true)
        {
            self.found = true;
        }
        self.visit_expr(&expression.left);
        self.visit_expr(&expression.right);
        let evidence = self.product_evidence(&expression.right);
        self.update_alias(&expression.left, evidence);
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast ExprForLoop) {
        self.visit_expr(&expression.expr);
        let mut shadows = HashMap::new();
        let mut identifiers = BTreeSet::new();
        collect_pattern_identifiers(&expression.pat, &mut identifiers);
        let zip_dependency =
            if Self::source_contains_zip(&expression.expr) && identifiers.len() >= 2 {
                let dependency = format!("<zip:{}>", self.next_synthetic_index);
                self.next_synthetic_index += 1;
                Some(dependency)
            } else {
                None
            };
        for identifier in &identifiers {
            let mut evidence = ProductEvidence::default();
            if let Some(dependency) = &zip_dependency {
                evidence.indexed_dependencies.insert(dependency.clone());
            }
            shadows.insert(identifier.clone(), evidence);
        }
        self.aliases.push(shadows);
        self.loop_scope_starts.push(self.aliases.len() - 1);
        self.loop_depth += 1;
        self.visit_block(&expression.body);
        self.loop_depth -= 1;
        self.loop_scope_starts.pop();
        self.aliases.pop();
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        self.visit_expr(&expression.cond);
        let mut shadows = HashMap::new();
        let identifiers = condition_pattern_identifiers(&expression.cond);
        for identifier in &identifiers {
            shadows.insert(identifier.clone(), ProductEvidence::default());
        }
        self.aliases.push(shadows);
        self.loop_scope_starts.push(self.aliases.len() - 1);
        self.loop_depth += 1;
        self.visit_block(&expression.body);
        self.loop_depth -= 1;
        self.loop_scope_starts.pop();
        self.aliases.pop();
    }

    fn visit_expr_loop(&mut self, expression: &'ast syn::ExprLoop) {
        self.loop_scope_starts.push(self.aliases.len());
        self.loop_depth += 1;
        self.visit_block(&expression.body);
        self.loop_depth -= 1;
        self.loop_scope_starts.pop();
    }

    fn visit_expr_closure(&mut self, closure: &'ast ExprClosure) {
        let mut shadows = HashMap::new();
        for pattern in &closure.inputs {
            let mut identifiers = BTreeSet::new();
            collect_pattern_identifiers(pattern, &mut identifiers);
            for identifier in identifiers {
                shadows.insert(identifier, ProductEvidence::default());
            }
        }
        self.aliases.push(shadows);
        self.visit_expr(&closure.body);
        self.aliases.pop();
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        self.visit_expr(&expression.cond);
        let mut shadows = HashMap::new();
        for identifier in condition_pattern_identifiers(&expression.cond) {
            shadows.insert(identifier, ProductEvidence::default());
        }
        self.aliases.push(shadows);
        self.visit_block(&expression.then_branch);
        self.aliases.pop();
        if let Some((_, else_branch)) = &expression.else_branch {
            self.visit_expr(else_branch);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            let mut shadows = HashMap::new();
            let mut identifiers = BTreeSet::new();
            collect_pattern_identifiers(&arm.pat, &mut identifiers);
            for identifier in identifiers {
                shadows.insert(identifier, ProductEvidence::default());
            }
            self.aliases.push(shadows);
            if let Some((_, guard)) = &arm.guard {
                self.visit_expr(guard);
            }
            self.visit_expr(&arm.body);
            self.aliases.pop();
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = identifier_name(&call.method);
        if self.iterator_reduction(&method, &call.receiver, call.args.last()) {
            self.found = true;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        let method = match call.func.as_ref() {
            Expr::Path(path) => {
                let segments = path_segments(&path.path);
                (segments.len() >= 2
                    && segments[..segments.len() - 1]
                        .iter()
                        .any(|segment| segment == "Iterator"))
                .then(|| segments.last().unwrap().clone())
            }
            _ => None,
        };
        if let (Some(method), Some(source)) = (method, call.args.first()) {
            if self.iterator_reduction(&method, source, call.args.last()) {
                self.found = true;
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_item_fn(&mut self, _item: &'ast ItemFn) {}
    fn visit_item_impl(&mut self, _item: &'ast ItemImpl) {}
    fn visit_item_mod(&mut self, _item: &'ast ItemMod) {}
    fn visit_item_trait(&mut self, _item: &'ast syn::ItemTrait) {}
}

struct BoundaryVisitor<'a> {
    relative_path: &'a str,
    model: &'a SourceModel,
    current_module: Vec<String>,
    module_path: Vec<String>,
    item_stack: Vec<ItemIdentity>,
    impl_context: Option<ItemContainer>,
    trait_definition: Option<String>,
    test_context: bool,
    current_cfg: CfgExpr,
    scopes: Vec<ScopeFrame>,
    violations: Vec<String>,
    inventory_seen: BTreeSet<usize>,
}

impl BoundaryVisitor<'_> {
    fn enter_cfg(&mut self, attributes: &[Attribute]) -> CfgExpr {
        let previous = self.current_cfg.clone();
        if let Some(condition) = cfg_condition(attributes) {
            self.current_cfg = CfgExpr::all(vec![previous.clone(), condition]);
        }
        previous
    }

    fn cfg_symbol_is_active(&self, attributes: &[Attribute]) -> bool {
        cfg_condition(attributes).is_some_and(|condition| self.current_cfg.implies(&condition))
    }

    fn current_function(&self) -> &str {
        self.item_stack
            .last()
            .map_or("<module>", |item| item.name.as_str())
    }

    fn exact_item(&self, name: &str, container: &ItemContainer, modules: &[&str]) -> bool {
        self.item_stack.len() == 1
            && self
                .item_stack
                .last()
                .is_some_and(|item| item.name == name && &item.container == container)
            && self
                .module_path
                .iter()
                .map(String::as_str)
                .eq(modules.iter().copied())
    }

    fn explicit_nary_candidate(&self) -> bool {
        self.test_context
            && ((self.relative_path == "src/contraction_contract_candidate.rs"
                && self.exact_item("contract_candidate", &ItemContainer::Free, &[]))
                || (self.relative_path == "src/contraction_boundary_audit.rs"
                    && self.exact_item("audit_one_call_nary_candidate", &ItemContainer::Free, &[])))
    }

    fn direct_pairwise_call(&self, api: RestrictedApi) -> bool {
        match api {
            RestrictedApi::ContractPair => {
                self.relative_path == PAIRWISE_ADAPTER_OWNER
                    && self.exact_item("pairwise", &ItemContainer::Free, &[])
            }
            RestrictedApi::ContractPairWithOperandOptions => {
                self.relative_path == PAIRWISE_ADAPTER_OWNER
                    && self.exact_item("pairwise_with_conjugation", &ItemContainer::Free, &[])
            }
            _ => false,
        }
    }

    fn outer_product_call(&self) -> bool {
        self.relative_path == "src/tensor.rs"
            && self.test_context
            && self.exact_item(
                "relabel_changes_contraction_partner",
                &ItemContainer::Free,
                &["tests"],
            )
    }

    fn approved_outer_helper_caller(&self) -> bool {
        false // Removed TDVP helpers have no approved production caller.
    }

    fn restricted_api_allowed(&self, api: RestrictedApi) -> bool {
        match api {
            RestrictedApi::Contract => self.explicit_nary_candidate(),
            RestrictedApi::ContractPair | RestrictedApi::ContractPairWithOperandOptions => {
                self.direct_pairwise_call(api)
            }
            RestrictedApi::OuterProduct => self.outer_product_call(),
            RestrictedApi::ContractOwned | RestrictedApi::Einsum => false,
        }
    }

    fn unresolved_api(identifier: &str) -> Option<ApiBinding> {
        match identifier {
            "contract" => Some(ApiBinding::Restricted(RestrictedApi::Contract)),
            "contract_owned" => Some(ApiBinding::Restricted(RestrictedApi::ContractOwned)),
            "contract_pair" => Some(ApiBinding::Restricted(RestrictedApi::ContractPair)),
            "contract_pair_with_operand_options" => Some(ApiBinding::Restricted(
                RestrictedApi::ContractPairWithOperandOptions,
            )),
            "outer_product" => Some(ApiBinding::Restricted(RestrictedApi::OuterProduct)),
            "einsum" => Some(ApiBinding::Restricted(RestrictedApi::Einsum)),
            _ => None,
        }
    }

    fn external_api(crate_name: &str, identifier: &str) -> Option<ApiBinding> {
        if crate_name == "tensor4all_core" {
            return Self::unresolved_api(identifier);
        }
        if crate_name.starts_with("tenferro") {
            return match identifier {
                "contract" => Some(ApiBinding::Restricted(RestrictedApi::Contract)),
                "einsum" => Some(ApiBinding::Restricted(RestrictedApi::Einsum)),
                _ => None,
            };
        }
        None
    }

    fn resolve_value(&self, path: &ExprPath) -> Option<ResolvedValue> {
        let alias = AliasPath::from_path(&path.path, &self.current_module);
        self.resolve_value_alias(&alias, self.scopes.len(), &mut BTreeSet::new())
    }

    fn resolve_pending_value_alias(&self, alias: &AliasPath) -> Option<ResolvedValue> {
        if alias.segments.len() > 1 {
            let mut prefix = alias.clone();
            prefix.segments.pop();
            if self.resolve_namespace_alias(&prefix, self.scopes.len(), &mut BTreeSet::new())
                == ResolvedNamespace::Unknown
            {
                return None;
            }
        }
        self.resolve_value_alias(alias, self.scopes.len(), &mut BTreeSet::new())
    }

    fn resolve_value_alias(
        &self,
        path: &AliasPath,
        scope_limit: usize,
        seen: &mut BTreeSet<String>,
    ) -> Option<ResolvedValue> {
        let identifier = path.segments.last()?.as_str();
        if !path.leading_colon && path.segments.len() == 1 {
            for scope in self.scopes[..scope_limit].iter().rev() {
                if let Some(definition) = scope.values.get(identifier) {
                    match definition {
                        ScopedValueDefinition::Local => return None,
                        ScopedValueDefinition::ConditionalLocal(condition) => {
                            if self.current_cfg.implies(condition) {
                                return None;
                            }
                        }
                        ScopedValueDefinition::Api(value) => return Some(*value),
                    }
                }
            }
            if let Some(definition) = self
                .model
                .modules
                .get(&path.origin_module)
                .and_then(|module| module.values.get(identifier))
                .cloned()
            {
                return self.resolve_value_definition(
                    &path.origin_module,
                    identifier,
                    definition,
                    seen,
                );
            }
            match self.resolve_conditional_value(&path.origin_module, identifier, seen) {
                ConditionalValueResolution::Local => return None,
                ConditionalValueResolution::Api(value) => return Some(value),
                ConditionalValueResolution::Absent => {}
            }
            if self
                .model
                .modules
                .get(&path.origin_module)
                .and_then(|module| module.namespaces.get(identifier))
                .is_some()
            {
                return None;
            }
            match self.resolve_glob_value(&path.origin_module, identifier, seen) {
                ConditionalValueResolution::Local => return None,
                ConditionalValueResolution::Api(value) => return Some(value),
                ConditionalValueResolution::Absent => {}
            }
            return Self::unresolved_api(identifier).map(|binding| ResolvedValue {
                binding,
                indirect: false,
            });
        }

        let mut prefix = path.clone();
        prefix.segments.pop();
        match self.resolve_namespace_alias(&prefix, scope_limit, &mut BTreeSet::new()) {
            ResolvedNamespace::External { name, .. } => {
                Self::external_api(&name, identifier).map(|binding| ResolvedValue {
                    binding,
                    indirect: false,
                })
            }
            ResolvedNamespace::Crate(module) => {
                if module == ["tensor"] && identifier == "contract" {
                    return Some(ResolvedValue {
                        binding: ApiBinding::BinaryContract,
                        indirect: false,
                    });
                }
                if module == ["uniform_hamiltonian_kernel"]
                    && identifier == "hamiltonian_action_outer_product"
                {
                    return Some(ResolvedValue {
                        binding: ApiBinding::HamiltonianOuterHelper,
                        indirect: false,
                    });
                }
                if let Some(definition) = self
                    .model
                    .modules
                    .get(&module)
                    .and_then(|symbols| symbols.values.get(identifier))
                    .cloned()
                {
                    return self.resolve_value_definition(&module, identifier, definition, seen);
                }
                if self.model.modules.contains_key(&module) {
                    return None;
                }
                Self::unresolved_api(identifier).map(|binding| ResolvedValue {
                    binding,
                    indirect: false,
                })
            }
            ResolvedNamespace::BlockLocal(scope) => {
                scope.values.get(identifier).and_then(|value| match value {
                    ScopedValueDefinition::Local | ScopedValueDefinition::ConditionalLocal(_) => {
                        None
                    }
                    ScopedValueDefinition::Api(value) => Some(*value),
                })
            }
            ResolvedNamespace::Local => None,
            ResolvedNamespace::Unknown => {
                Self::unresolved_api(identifier).map(|binding| ResolvedValue {
                    binding,
                    indirect: false,
                })
            }
        }
    }

    fn resolve_glob_value(
        &self,
        module: &[String],
        identifier: &str,
        seen: &mut BTreeSet<String>,
    ) -> ConditionalValueResolution {
        let key = format!("glob:{module:?}:{identifier}");
        if !seen.insert(key.clone()) {
            return ConditionalValueResolution::Absent;
        }
        let globs = self
            .model
            .modules
            .get(module)
            .map(|symbols| symbols.glob_imports.clone())
            .unwrap_or_default();
        let mut resolved = ConditionalValueResolution::Absent;
        for glob in globs {
            if !glob.guaranteed_in(&self.current_cfg) {
                continue;
            }
            let candidate = match self.resolve_namespace_alias(&glob, 0, &mut BTreeSet::new()) {
                ResolvedNamespace::External { name, .. } => Self::external_api(&name, identifier)
                    .map(|binding| ResolvedValue {
                        binding,
                        indirect: false,
                    })
                    .map_or(
                        ConditionalValueResolution::Absent,
                        ConditionalValueResolution::Api,
                    ),
                ResolvedNamespace::Crate(target) => {
                    if target == ["tensor"] && identifier == "contract" {
                        ConditionalValueResolution::Api(ResolvedValue {
                            binding: ApiBinding::BinaryContract,
                            indirect: false,
                        })
                    } else if target == ["uniform_hamiltonian_kernel"]
                        && identifier == "hamiltonian_action_outer_product"
                    {
                        ConditionalValueResolution::Api(ResolvedValue {
                            binding: ApiBinding::HamiltonianOuterHelper,
                            indirect: false,
                        })
                    } else if let Some(definition) = self
                        .model
                        .modules
                        .get(&target)
                        .and_then(|symbols| symbols.values.get(identifier))
                        .cloned()
                    {
                        self.resolve_value_definition(&target, identifier, definition, seen)
                            .map_or(
                                ConditionalValueResolution::Local,
                                ConditionalValueResolution::Api,
                            )
                    } else {
                        match self.resolve_conditional_value(&target, identifier, seen) {
                            ConditionalValueResolution::Absent => {
                                self.resolve_glob_value(&target, identifier, seen)
                            }
                            candidate => candidate,
                        }
                    }
                }
                ResolvedNamespace::BlockLocal(scope) => scope.values.get(identifier).map_or(
                    ConditionalValueResolution::Absent,
                    |value| match value {
                        ScopedValueDefinition::Local => ConditionalValueResolution::Local,
                        ScopedValueDefinition::ConditionalLocal(condition) => {
                            if self.current_cfg.implies(condition) {
                                ConditionalValueResolution::Local
                            } else {
                                ConditionalValueResolution::Absent
                            }
                        }
                        ScopedValueDefinition::Api(value) => {
                            ConditionalValueResolution::Api(*value)
                        }
                    },
                ),
                ResolvedNamespace::Local | ResolvedNamespace::Unknown => {
                    ConditionalValueResolution::Absent
                }
            };
            resolved = match (resolved, candidate) {
                (ConditionalValueResolution::Absent, candidate)
                | (candidate, ConditionalValueResolution::Absent) => candidate,
                (ConditionalValueResolution::Local, ConditionalValueResolution::Local) => {
                    ConditionalValueResolution::Local
                }
                (
                    ConditionalValueResolution::Api(known),
                    ConditionalValueResolution::Api(candidate),
                ) if known == candidate => ConditionalValueResolution::Api(known),
                _ => ConditionalValueResolution::Api(ResolvedValue {
                    binding: ApiBinding::Ambiguous,
                    indirect: false,
                }),
            };
        }
        seen.remove(&key);
        resolved
    }

    fn resolve_conditional_value(
        &self,
        module: &[String],
        name: &str,
        seen: &mut BTreeSet<String>,
    ) -> ConditionalValueResolution {
        let definitions = self
            .model
            .modules
            .get(module)
            .and_then(|symbols| symbols.conditional_values.get(name))
            .cloned()
            .unwrap_or_default();
        let mut local = false;
        let mut resolved = None;
        for (condition, definition) in definitions {
            if !self.current_cfg.implies(&condition) {
                continue;
            }
            match self.resolve_value_definition(module, name, definition, seen) {
                None => local = true,
                Some(candidate) => {
                    if resolved.is_some_and(|known| known != candidate) {
                        return ConditionalValueResolution::Api(ResolvedValue {
                            binding: ApiBinding::Ambiguous,
                            indirect: false,
                        });
                    }
                    resolved = Some(candidate);
                }
            }
        }
        match (local, resolved) {
            (false, None) => ConditionalValueResolution::Absent,
            (true, None) => ConditionalValueResolution::Local,
            (false, Some(value)) => ConditionalValueResolution::Api(value),
            (true, Some(_)) => ConditionalValueResolution::Api(ResolvedValue {
                binding: ApiBinding::Ambiguous,
                indirect: false,
            }),
        }
    }

    fn resolve_value_definition(
        &self,
        module: &[String],
        name: &str,
        definition: ValueDefinition,
        seen: &mut BTreeSet<String>,
    ) -> Option<ResolvedValue> {
        let key = format!("value:{module:?}:{name}");
        if !seen.insert(key.clone()) {
            return None;
        }
        let value = match definition {
            ValueDefinition::Local => None,
            ValueDefinition::Api(binding) => Some(ResolvedValue {
                binding,
                indirect: false,
            }),
            ValueDefinition::Import(alias) => {
                let value = self.resolve_value_alias(&alias, 0, seen);
                if !alias.guaranteed_in(&self.current_cfg) {
                    value.or_else(|| {
                        Self::unresolved_api(name).map(|binding| ResolvedValue {
                            binding,
                            indirect: false,
                        })
                    })
                } else {
                    value
                }
            }
            ValueDefinition::FunctionValue(alias) => {
                let value = self.resolve_value_alias(&alias, 0, seen).or_else(|| {
                    (!alias.guaranteed_in(&self.current_cfg)).then(|| {
                        Self::unresolved_api(name).map(|binding| ResolvedValue {
                            binding,
                            indirect: true,
                        })
                    })?
                });
                value.map(|mut value| {
                    value.indirect = true;
                    value
                })
            }
            ValueDefinition::Ambiguous => Some(ResolvedValue {
                binding: ApiBinding::Ambiguous,
                indirect: false,
            }),
        };
        seen.remove(&key);
        value
    }

    fn resolve_namespace_alias(
        &self,
        path: &AliasPath,
        scope_limit: usize,
        seen: &mut BTreeSet<String>,
    ) -> ResolvedNamespace {
        if path.segments.is_empty() {
            return ResolvedNamespace::Unknown;
        }
        let (mut namespace, consumed) = if path.leading_colon {
            (
                ResolvedNamespace::External {
                    name: path.segments[0].clone(),
                    path: Vec::new(),
                },
                1,
            )
        } else if path.segments[0] == "Self"
            && (self.impl_context.is_some() || self.trait_definition.is_some())
        {
            (ResolvedNamespace::Local, 1)
        } else if path.segments[0] == "crate" {
            (ResolvedNamespace::Crate(Vec::new()), 1)
        } else if path.segments[0] == "self" {
            (ResolvedNamespace::Crate(path.origin_module.clone()), 1)
        } else if path.segments[0] == "super" {
            let depth = path
                .segments
                .iter()
                .take_while(|segment| *segment == "super")
                .count();
            if depth > path.origin_module.len() {
                return ResolvedNamespace::Unknown;
            }
            (
                ResolvedNamespace::Crate(
                    path.origin_module[..path.origin_module.len() - depth].to_vec(),
                ),
                depth,
            )
        } else {
            let first = &path.segments[0];
            let mut resolved = None;
            for scope in self.scopes[..scope_limit].iter().rev() {
                if let Some(namespace) = scope.namespaces.get(first) {
                    resolved = Some(namespace.clone());
                    break;
                }
            }
            let resolved = resolved.unwrap_or_else(|| {
                self.resolve_module_namespace_name(&path.origin_module, first, seen)
            });
            (resolved, 1)
        };
        for segment in &path.segments[consumed..] {
            namespace = self.advance_namespace(namespace, segment, seen);
        }
        namespace
    }

    fn resolve_module_namespace_name(
        &self,
        module: &[String],
        name: &str,
        seen: &mut BTreeSet<String>,
    ) -> ResolvedNamespace {
        let key = format!("namespace:{module:?}:{name}");
        if !seen.insert(key.clone()) {
            return ResolvedNamespace::Unknown;
        }
        let definition = self
            .model
            .modules
            .get(module)
            .and_then(|symbols| symbols.namespaces.get(name))
            .cloned();
        let resolved = if let Some(definition) = definition {
            self.resolve_namespace_definition(name, definition, seen)
        } else {
            let definitions = self
                .model
                .modules
                .get(module)
                .and_then(|symbols| symbols.conditional_namespaces.get(name))
                .cloned()
                .unwrap_or_default();
            let mut resolved = None;
            for (condition, definition) in definitions {
                if !self.current_cfg.implies(&condition) {
                    continue;
                }
                let candidate = self.resolve_namespace_definition(name, definition, seen);
                if resolved.as_ref().is_some_and(|known| known != &candidate) {
                    resolved = Some(ResolvedNamespace::Unknown);
                    break;
                }
                resolved = Some(candidate);
            }
            resolved.unwrap_or_else(|| Self::unbound_namespace(name))
        };
        seen.remove(&key);
        resolved
    }

    fn resolve_namespace_definition(
        &self,
        name: &str,
        definition: NamespaceDefinition,
        seen: &mut BTreeSet<String>,
    ) -> ResolvedNamespace {
        match definition {
            NamespaceDefinition::LocalModule(module) => ResolvedNamespace::Crate(module),
            NamespaceDefinition::Local => ResolvedNamespace::Local,
            NamespaceDefinition::Import(alias) => {
                let resolved = self.resolve_namespace_alias(&alias, 0, seen);
                if !alias.guaranteed_in(&self.current_cfg) {
                    let absent = Self::unbound_namespace(name);
                    if resolved == ResolvedNamespace::Unknown {
                        absent
                    } else if absent == ResolvedNamespace::Unknown || resolved == absent {
                        resolved
                    } else {
                        ResolvedNamespace::Unknown
                    }
                } else {
                    resolved
                }
            }
            NamespaceDefinition::CrateRoot => ResolvedNamespace::Crate(Vec::new()),
            NamespaceDefinition::External(name) => ResolvedNamespace::External {
                name,
                path: Vec::new(),
            },
            NamespaceDefinition::Ambiguous => ResolvedNamespace::Unknown,
        }
    }

    fn unbound_namespace(name: &str) -> ResolvedNamespace {
        if name == "tensor4all_core" || name.starts_with("tenferro") {
            ResolvedNamespace::External {
                name: name.to_string(),
                path: Vec::new(),
            }
        } else {
            ResolvedNamespace::Unknown
        }
    }

    fn advance_namespace(
        &self,
        namespace: ResolvedNamespace,
        segment: &str,
        seen: &mut BTreeSet<String>,
    ) -> ResolvedNamespace {
        match namespace {
            ResolvedNamespace::Crate(module) => {
                let resolved = self.resolve_module_namespace_name(&module, segment, seen);
                if resolved == ResolvedNamespace::Unknown {
                    let mut child = module;
                    child.push(segment.to_string());
                    ResolvedNamespace::Crate(child)
                } else {
                    resolved
                }
            }
            ResolvedNamespace::External { name, mut path } => {
                path.push(segment.to_string());
                ResolvedNamespace::External { name, path }
            }
            ResolvedNamespace::BlockLocal(scope) => scope
                .namespaces
                .get(segment)
                .cloned()
                .unwrap_or(ResolvedNamespace::Unknown),
            ResolvedNamespace::Local => ResolvedNamespace::Local,
            ResolvedNamespace::Unknown => ResolvedNamespace::Unknown,
        }
    }

    fn normalized_direct_callee(expression: &Expr) -> Option<&ExprPath> {
        match expression {
            Expr::Path(path) => Some(path),
            Expr::Paren(expression) => Self::normalized_direct_callee(&expression.expr),
            Expr::Group(expression) => Self::normalized_direct_callee(&expression.expr),
            Expr::Cast(expression) => Self::normalized_direct_callee(&expression.expr),
            Expr::Block(ExprBlock { block, .. }) if block.stmts.len() == 1 => {
                let Stmt::Expr(expression, None) = &block.stmts[0] else {
                    return None;
                };
                Self::normalized_direct_callee(expression)
            }
            _ => None,
        }
    }

    fn inspect_call(&mut self, call: &ExprCall, path: &ExprPath) {
        let Some(value) = self.resolve_value(path) else {
            return;
        };
        let function = self.current_function().to_string();
        match value.binding {
            ApiBinding::Ambiguous => self.violations.push(format!(
                "{}::{function} calls an ambiguously cfg-bound contraction API",
                self.relative_path
            )),
            ApiBinding::BinaryContract if call.args.len() != 2 => {
                self.violations.push(format!(
                    "{}::{function} calls the binary crate::tensor::contract adapter with {} arguments",
                    self.relative_path,
                    call.args.len()
                ));
            }
            ApiBinding::Restricted(api) if value.indirect || !self.restricted_api_allowed(api) => {
                self.violations.push(format!(
                    "{}::{function} calls restricted tensor API {api:?} outside an exact direct-call allowlist",
                    self.relative_path
                ));
            }
            ApiBinding::HamiltonianOuterHelper
                if value.indirect
                    || (!self.test_context && !self.approved_outer_helper_caller()) =>
            {
                self.violations.push(format!(
                    "{}::{function} calls hamiltonian_action_outer_product outside AC actions",
                    self.relative_path
                ));
            }
            _ => {}
        }
    }

    fn inspect_function_value(&mut self, path: &ExprPath) {
        let Some(value) = self.resolve_value(path) else {
            return;
        };
        let function = self.current_function().to_string();
        match value.binding {
            ApiBinding::Ambiguous => self.violations.push(format!(
                "{}::{function} references an ambiguously cfg-bound contraction API",
                self.relative_path
            )),
            ApiBinding::Restricted(api) => self.violations.push(format!(
                "{}::{function} references restricted tensor API {api:?} as a function value",
                self.relative_path
            )),
            ApiBinding::HamiltonianOuterHelper if !self.test_context => self.violations.push(
                format!(
                    "{}::{function} passes hamiltonian_action_outer_product as a function value outside AC actions",
                    self.relative_path
                ),
            ),
            ApiBinding::HamiltonianOuterHelper | ApiBinding::BinaryContract => {}
        }
    }

    fn current_inventory_entry(&self) -> Option<usize> {
        HANDWRITTEN_CONTRACTION_INVENTORY
            .iter()
            .enumerate()
            .find_map(|(index, entry)| {
                let container = ItemContainer::Free;
                (self.relative_path == entry.source
                    && self.exact_item(entry.name, &container, entry.modules))
                .then_some(index)
            })
    }

    fn inspect_handwritten_reduction(&mut self, block: &syn::Block) {
        let inventory = self.current_inventory_entry();
        let handwritten_reduction = HandwrittenReductionDetector::detect(block);
        if let Some(index) = inventory.filter(|_| handwritten_reduction) {
            self.inventory_seen.insert(index);
        }
        if self.test_context
            || self.test_only_source()
            || inventory.is_some()
            || !handwritten_reduction
        {
            return;
        }
        let violation = format!(
            "{}::{} uses an unlisted handwritten production contraction reduction",
            self.relative_path,
            self.current_function()
        );
        if !self.violations.contains(&violation) {
            self.violations.push(violation);
        }
    }

    fn test_only_source(&self) -> bool {
        self.relative_path.starts_with("tests/")
            || [
                "src/contraction_boundary_audit.rs",
            ]
            .contains(&self.relative_path)
    }

    fn push_local_scope(&mut self) {
        self.scopes.push(ScopeFrame::default());
    }

    fn bind_pattern(&mut self, pattern: &syn::Pat) {
        let mut identifiers = BTreeSet::new();
        collect_pattern_identifiers(pattern, &mut identifiers);
        self.bind_identifiers(identifiers);
    }

    fn bind_identifiers(&mut self, identifiers: BTreeSet<String>) {
        let scope = self.scopes.last_mut().expect("lexical scope");
        for identifier in identifiers {
            scope
                .values
                .insert(identifier, ScopedValueDefinition::Local);
        }
    }

    fn bind_inputs(&mut self, inputs: &Punctuated<FnArg, Token![,]>) {
        for input in inputs {
            if let FnArg::Typed(argument) = input {
                self.bind_pattern(&argument.pat);
            }
        }
    }

    fn block_module_alias(mut alias: AliasPath) -> AliasPath {
        if !alias.leading_colon && alias.segments.first().is_some_and(|name| name == "self") {
            alias.segments.remove(0);
        }
        alias
    }

    fn block_module_scope(&mut self, item: &ItemMod) -> ScopeFrame {
        let mut scope = ScopeFrame {
            item_scope: true,
            ..ScopeFrame::default()
        };
        let mut pending_values = Vec::new();
        let mut pending_namespaces = Vec::new();
        let mut pending_globs = Vec::new();
        let Some((_, items)) = &item.content else {
            return scope;
        };
        for item in items {
            match item {
                syn::Item::Fn(item) => {
                    let name = identifier_name(&item.sig.ident);
                    if Self::unresolved_api(&name).is_some()
                        || name == "hamiltonian_action_outer_product"
                    {
                        let definition = if is_conditionally_compiled(&item.attrs) {
                            cfg_condition(&item.attrs).map(ScopedValueDefinition::ConditionalLocal)
                        } else {
                            Some(ScopedValueDefinition::Local)
                        };
                        if let Some(definition) = definition {
                            scope.values.insert(name, definition);
                        }
                    }
                }
                syn::Item::Const(item) => {
                    if let Some(path) = transparent_path(&item.expr) {
                        let mut alias = Self::block_module_alias(AliasPath::from_path(
                            &path.path,
                            &self.current_module,
                        ));
                        alias.conditional = is_conditionally_compiled(&item.attrs);
                        alias.condition = alias
                            .conditional
                            .then(|| cfg_condition(&item.attrs))
                            .flatten();
                        pending_values.push((identifier_name(&item.ident), alias, true));
                    }
                }
                syn::Item::Static(item) => {
                    if let Some(path) = transparent_path(&item.expr) {
                        let mut alias = Self::block_module_alias(AliasPath::from_path(
                            &path.path,
                            &self.current_module,
                        ));
                        alias.conditional = is_conditionally_compiled(&item.attrs);
                        alias.condition = alias
                            .conditional
                            .then(|| cfg_condition(&item.attrs))
                            .flatten();
                        pending_values.push((identifier_name(&item.ident), alias, true));
                    }
                }
                syn::Item::Use(item) => {
                    let mut bindings = Vec::new();
                    collect_use_bindings(&item.tree, &mut Vec::new(), &mut bindings);
                    for binding in bindings {
                        let conditional = is_conditionally_compiled(&item.attrs);
                        let alias = Self::block_module_alias(AliasPath {
                            leading_colon: item.leading_colon.is_some(),
                            segments: binding.path,
                            origin_module: self.current_module.clone(),
                            conditional,
                            condition: conditional.then(|| cfg_condition(&item.attrs)).flatten(),
                        });
                        if binding.local_name == "*" {
                            pending_globs.push(alias);
                        } else {
                            pending_values.push((binding.local_name.clone(), alias.clone(), false));
                            pending_namespaces.push((binding.local_name, alias));
                        }
                    }
                }
                syn::Item::Mod(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        let child = self.block_module_scope(item);
                        scope.namespaces.insert(
                            identifier_name(&item.ident),
                            ResolvedNamespace::BlockLocal(Box::new(child)),
                        );
                    }
                }
                syn::Item::Struct(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Enum(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Trait(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Type(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Union(item) if self.cfg_symbol_is_active(&item.attrs) => {
                    scope
                        .namespaces
                        .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                }
                _ => {}
            }
        }
        self.resolve_precollected_scope(scope, pending_values, pending_namespaces, pending_globs)
    }

    fn block_glob_bindings(
        &self,
        namespace: &ResolvedNamespace,
    ) -> Vec<(String, ScopedValueDefinition)> {
        const AUDITED_NAMES: &[&str] = &[
            "contract",
            "contract_owned",
            "contract_pair",
            "contract_pair_with_operand_options",
            "outer_product",
            "einsum",
            "hamiltonian_action_outer_product",
        ];
        let mut bindings = Vec::new();
        for &name in AUDITED_NAMES {
            let value = match namespace {
                ResolvedNamespace::External {
                    name: crate_name, ..
                } => Self::external_api(crate_name, name).map(|binding| ResolvedValue {
                    binding,
                    indirect: false,
                }),
                ResolvedNamespace::Crate(module) => {
                    if module.as_slice() == ["tensor"] && name == "contract" {
                        Some(ResolvedValue {
                            binding: ApiBinding::BinaryContract,
                            indirect: false,
                        })
                    } else if module.as_slice() == ["uniform_hamiltonian_kernel"]
                        && name == "hamiltonian_action_outer_product"
                    {
                        Some(ResolvedValue {
                            binding: ApiBinding::HamiltonianOuterHelper,
                            indirect: false,
                        })
                    } else if let Some(definition) = self
                        .model
                        .modules
                        .get(module)
                        .and_then(|symbols| symbols.values.get(name))
                        .cloned()
                    {
                        match self.resolve_value_definition(
                            module,
                            name,
                            definition,
                            &mut BTreeSet::new(),
                        ) {
                            Some(value) => Some(value),
                            None => {
                                bindings.push((name.to_string(), ScopedValueDefinition::Local));
                                continue;
                            }
                        }
                    } else {
                        match self.resolve_glob_value(module, name, &mut BTreeSet::new()) {
                            ConditionalValueResolution::Local => {
                                bindings.push((name.to_string(), ScopedValueDefinition::Local));
                                continue;
                            }
                            ConditionalValueResolution::Api(value) => Some(value),
                            ConditionalValueResolution::Absent => None,
                        }
                    }
                }
                ResolvedNamespace::BlockLocal(scope) => {
                    if let Some(value) = scope.values.get(name) {
                        bindings.push((name.to_string(), value.clone()));
                    }
                    None
                }
                ResolvedNamespace::Local => None,
                ResolvedNamespace::Unknown => None,
            };
            if let Some(value) = value {
                bindings.push((name.to_string(), ScopedValueDefinition::Api(value)));
            }
        }
        bindings
    }

    fn resolve_precollected_scope(
        &mut self,
        scope: ScopeFrame,
        pending_values: Vec<(String, AliasPath, bool)>,
        pending_namespaces: Vec<(String, AliasPath)>,
        pending_globs: Vec<AliasPath>,
    ) -> ScopeFrame {
        self.scopes.push(scope);
        let resolution_limit =
            pending_values.len() + pending_namespaces.len() + pending_globs.len() + 1;
        let mut resolved_globs = BTreeSet::new();
        for _ in 0..resolution_limit {
            let mut changed = false;
            for (name, alias, indirect) in &pending_values {
                if self
                    .scopes
                    .last()
                    .expect("precollected item scope")
                    .values
                    .contains_key(name)
                {
                    continue;
                }
                if let Some(mut value) = self.resolve_pending_value_alias(alias) {
                    value.indirect |= *indirect;
                    self.scopes
                        .last_mut()
                        .expect("precollected item scope")
                        .values
                        .insert(name.clone(), ScopedValueDefinition::Api(value));
                    changed = true;
                }
            }
            for (name, alias) in &pending_namespaces {
                if self
                    .scopes
                    .last()
                    .expect("precollected item scope")
                    .namespaces
                    .contains_key(name)
                {
                    continue;
                }
                let mut namespace =
                    self.resolve_namespace_alias(alias, self.scopes.len(), &mut BTreeSet::new());
                if !alias.guaranteed_in(&self.current_cfg) {
                    let absent = Self::unbound_namespace(name);
                    if absent != ResolvedNamespace::Unknown && namespace != absent {
                        namespace = ResolvedNamespace::Unknown;
                    }
                }
                if namespace != ResolvedNamespace::Unknown {
                    self.scopes
                        .last_mut()
                        .expect("precollected item scope")
                        .namespaces
                        .insert(name.clone(), namespace);
                    changed = true;
                }
            }
            for (index, alias) in pending_globs.iter().enumerate() {
                if resolved_globs.contains(&index) {
                    continue;
                }
                let namespace =
                    self.resolve_namespace_alias(alias, self.scopes.len(), &mut BTreeSet::new());
                if namespace == ResolvedNamespace::Unknown {
                    continue;
                }
                for (name, value) in self.block_glob_bindings(&namespace) {
                    self.scopes
                        .last_mut()
                        .expect("precollected item scope")
                        .values
                        .entry(name)
                        .or_insert(value);
                }
                resolved_globs.insert(index);
                changed = true;
            }
            if !changed {
                break;
            }
        }

        let mut scope = self.scopes.pop().expect("precollected item scope");
        for (name, alias, _) in pending_values {
            let fallback = if !alias.guaranteed_in(&self.current_cfg) {
                Self::unresolved_api(&name).map(|binding| {
                    ScopedValueDefinition::Api(ResolvedValue {
                        binding,
                        indirect: false,
                    })
                })
            } else {
                None
            };
            scope
                .values
                .entry(name)
                .or_insert_with(|| fallback.unwrap_or(ScopedValueDefinition::Local));
        }
        scope
    }

    fn block_item_scope(&mut self, block: &syn::Block) -> ScopeFrame {
        let mut scope = ScopeFrame {
            item_scope: true,
            ..ScopeFrame::default()
        };
        let mut pending_values = Vec::new();
        let mut pending_namespaces = Vec::new();
        let mut pending_globs = Vec::new();
        for statement in &block.stmts {
            let Stmt::Item(item) = statement else {
                continue;
            };
            match item {
                syn::Item::Fn(item) => {
                    let definition = if is_conditionally_compiled(&item.attrs) {
                        cfg_condition(&item.attrs).map(ScopedValueDefinition::ConditionalLocal)
                    } else {
                        Some(ScopedValueDefinition::Local)
                    };
                    if let Some(definition) = definition {
                        scope
                            .values
                            .insert(identifier_name(&item.sig.ident), definition);
                    }
                }
                syn::Item::Const(item) => {
                    if let Some(path) = transparent_path(&item.expr) {
                        let mut alias = AliasPath::from_path(&path.path, &self.current_module);
                        alias.conditional = is_conditionally_compiled(&item.attrs);
                        alias.condition = alias
                            .conditional
                            .then(|| cfg_condition(&item.attrs))
                            .flatten();
                        pending_values.push((identifier_name(&item.ident), alias, true));
                    } else {
                        let definition = if is_conditionally_compiled(&item.attrs) {
                            cfg_condition(&item.attrs).map(ScopedValueDefinition::ConditionalLocal)
                        } else {
                            Some(ScopedValueDefinition::Local)
                        };
                        if let Some(definition) = definition {
                            scope
                                .values
                                .insert(identifier_name(&item.ident), definition);
                        }
                    }
                }
                syn::Item::Static(item) => {
                    if let Some(path) = transparent_path(&item.expr) {
                        let mut alias = AliasPath::from_path(&path.path, &self.current_module);
                        alias.conditional = is_conditionally_compiled(&item.attrs);
                        alias.condition = alias
                            .conditional
                            .then(|| cfg_condition(&item.attrs))
                            .flatten();
                        pending_values.push((identifier_name(&item.ident), alias, true));
                    } else {
                        let definition = if is_conditionally_compiled(&item.attrs) {
                            cfg_condition(&item.attrs).map(ScopedValueDefinition::ConditionalLocal)
                        } else {
                            Some(ScopedValueDefinition::Local)
                        };
                        if let Some(definition) = definition {
                            scope
                                .values
                                .insert(identifier_name(&item.ident), definition);
                        }
                    }
                }
                syn::Item::Use(item) => {
                    let mut bindings = Vec::new();
                    collect_use_bindings(&item.tree, &mut Vec::new(), &mut bindings);
                    for binding in bindings {
                        let conditional = is_conditionally_compiled(&item.attrs);
                        let alias = AliasPath {
                            leading_colon: item.leading_colon.is_some(),
                            segments: binding.path,
                            origin_module: self.current_module.clone(),
                            conditional,
                            condition: conditional.then(|| cfg_condition(&item.attrs)).flatten(),
                        };
                        if binding.local_name == "*" {
                            pending_globs.push(alias);
                            continue;
                        }
                        pending_values.push((binding.local_name.clone(), alias.clone(), false));
                        pending_namespaces.push((binding.local_name, alias));
                    }
                }
                syn::Item::Mod(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope.namespaces.insert(
                            identifier_name(&item.ident),
                            ResolvedNamespace::BlockLocal(Box::new(self.block_module_scope(item))),
                        );
                    }
                }
                syn::Item::Struct(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Enum(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Trait(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Type(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::Union(item) => {
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope
                            .namespaces
                            .insert(identifier_name(&item.ident), ResolvedNamespace::Local);
                    }
                }
                syn::Item::ExternCrate(item) => {
                    let external_name = identifier_name(&item.ident);
                    let local_name = item.rename.as_ref().map_or_else(
                        || external_name.clone(),
                        |rename| identifier_name(&rename.1),
                    );
                    let namespace = if external_name == "self" {
                        ResolvedNamespace::Crate(Vec::new())
                    } else {
                        ResolvedNamespace::External {
                            name: external_name,
                            path: Vec::new(),
                        }
                    };
                    if self.cfg_symbol_is_active(&item.attrs) {
                        scope.namespaces.insert(local_name, namespace);
                    }
                }
                _ => {}
            }
        }

        self.resolve_precollected_scope(scope, pending_values, pending_namespaces, pending_globs)
    }

    fn retain_item_scopes(&mut self) -> Vec<ScopeFrame> {
        let previous = std::mem::take(&mut self.scopes);
        self.scopes = previous
            .iter()
            .filter(|scope| scope.item_scope)
            .cloned()
            .collect();
        previous
    }

    fn inspect_import(&mut self, item: &ItemUse) {
        let mut bindings = Vec::new();
        collect_use_bindings(&item.tree, &mut Vec::new(), &mut bindings);
        for binding in bindings {
            let conditional = is_conditionally_compiled(&item.attrs);
            let alias = AliasPath {
                leading_colon: item.leading_colon.is_some(),
                segments: binding.path.clone(),
                origin_module: self.current_module.clone(),
                conditional,
                condition: conditional.then(|| cfg_condition(&item.attrs)).flatten(),
            };
            if binding.local_name == "*" {
                if matches!(
                    self.resolve_namespace_alias(
                        &alias,
                        self.scopes.len(),
                        &mut BTreeSet::new()
                    ),
                    ResolvedNamespace::External { ref name, .. }
                        if name == "tensor4all_core" || name.starts_with("tenferro")
                ) {
                    self.violations.push(format!(
                        "{}::<module> glob-imports restricted tensor APIs",
                        self.relative_path
                    ));
                }
                continue;
            }
            let resolved =
                self.resolve_value_alias(&alias, self.scopes.len(), &mut BTreeSet::new());
            let identifier = binding.path.last().map(String::as_str).unwrap_or("");
            let restricted =
                resolved.is_some_and(|value| matches!(value.binding, ApiBinding::Restricted(_)));
            if binding.renamed && restricted {
                self.violations.push(format!(
                    "{}::<module> aliases restricted contraction API {identifier}",
                    self.relative_path
                ));
                continue;
            }
            let external = self.external_import_root(&alias);
            let allowed = match (external.as_deref(), identifier) {
                (Some("tensor4all_core"), "contract") => {
                    false
                }
                (
                    Some("tensor4all_core"),
                    "contract_pair"
                    | "contract_pair_with_operand_options"
                    | "PairwiseContractionOptions",
                ) => self.relative_path == PAIRWISE_ADAPTER_OWNER,
                (Some("tensor4all_core"), "outer_product") => {
                    false
                }
                (Some("tensor4all_core"), "contract_owned" | "einsum") => false,
                (Some(name), "contract" | "einsum") if name.starts_with("tenferro") => false,
                _ => true,
            };
            if !allowed {
                self.violations.push(format!(
                    "{}::<module> imports restricted tensor API {identifier}",
                    self.relative_path
                ));
            }
        }
    }

    fn external_import_root(&self, alias: &AliasPath) -> Option<String> {
        if alias.segments.len() < 2 {
            return None;
        }
        let mut prefix = alias.clone();
        prefix.segments.pop();
        match self.resolve_namespace_alias(&prefix, self.scopes.len(), &mut BTreeSet::new()) {
            ResolvedNamespace::External { name, .. } => Some(name),
            _ => None,
        }
    }

    fn macro_restricted_identifiers(tokens: proc_macro2::TokenStream, output: &mut Vec<String>) {
        for token in tokens {
            match token {
                proc_macro2::TokenTree::Ident(identifier)
                    if matches!(
                        identifier_name(&identifier).as_str(),
                        "contract"
                            | "contract_owned"
                            | "contract_pair"
                            | "contract_pair_with_operand_options"
                            | "outer_product"
                            | "einsum"
                    ) =>
                {
                    output.push(identifier_name(&identifier))
                }
                proc_macro2::TokenTree::Group(group) => {
                    Self::macro_restricted_identifiers(group.stream(), output)
                }
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for BoundaryVisitor<'_> {
    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        let previous_scopes = self.retain_item_scopes();
        let previous_impl = self.impl_context.take();
        let previous_trait = self.trait_definition.take();
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        self.item_stack.push(ItemIdentity {
            name: identifier_name(&item.sig.ident),
            container: ItemContainer::Free,
        });
        self.push_local_scope();
        self.bind_inputs(&item.sig.inputs);
        self.inspect_handwritten_reduction(&item.block);
        self.visit_block(&item.block);
        self.scopes.pop();
        self.item_stack.pop();
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
        self.trait_definition = previous_trait;
        self.impl_context = previous_impl;
        self.scopes = previous_scopes;
    }

    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        let block_scope = self.scopes.iter().rev().find_map(|scope| {
            match scope.namespaces.get(&identifier_name(&item.ident)) {
                Some(ResolvedNamespace::BlockLocal(block_scope)) => Some((**block_scope).clone()),
                _ => None,
            }
        });
        let previous_scopes = std::mem::take(&mut self.scopes);
        if let Some(block_scope) = block_scope {
            self.scopes.push(block_scope);
        }
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        self.current_module.push(identifier_name(&item.ident));
        self.module_path.push(identifier_name(&item.ident));
        if let Some((_, items)) = &item.content {
            for item in items {
                self.visit_item(item);
            }
        }
        self.module_path.pop();
        self.current_module.pop();
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
        self.scopes = previous_scopes;
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        let previous_scopes = std::mem::take(&mut self.scopes);
        let previous_impl = self.impl_context.take();
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        let self_path = match item.self_ty.as_ref() {
            syn::Type::Path(path) => path_segments(&path.path),
            _ => Vec::new(),
        };
        self.impl_context = Some(if let Some((_, trait_path, _)) = &item.trait_ {
            ItemContainer::TraitImpl {
                trait_path: path_segments(trait_path),
                self_path,
            }
        } else {
            ItemContainer::Inherent { self_path }
        });
        visit::visit_item_impl(self, item);
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
        self.impl_context = previous_impl;
        self.scopes = previous_scopes;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        self.item_stack.push(ItemIdentity {
            name: identifier_name(&item.sig.ident),
            container: self.impl_context.clone().unwrap_or(ItemContainer::Free),
        });
        self.push_local_scope();
        self.bind_inputs(&item.sig.inputs);
        self.inspect_handwritten_reduction(&item.block);
        self.visit_block(&item.block);
        self.scopes.pop();
        self.item_stack.pop();
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        let previous = self.trait_definition.replace(identifier_name(&item.ident));
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        visit::visit_item_trait(self, item);
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
        self.trait_definition = previous;
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        let Some(block) = &item.default else {
            return;
        };
        let previous_test = self.test_context;
        let previous_cfg = self.enter_cfg(&item.attrs);
        self.test_context |= has_cfg_test(&item.attrs);
        self.item_stack.push(ItemIdentity {
            name: identifier_name(&item.sig.ident),
            container: ItemContainer::TraitDefinition {
                trait_name: self
                    .trait_definition
                    .clone()
                    .unwrap_or_else(|| "<trait>".to_string()),
            },
        });
        self.push_local_scope();
        self.bind_inputs(&item.sig.inputs);
        self.inspect_handwritten_reduction(block);
        self.visit_block(block);
        self.scopes.pop();
        self.item_stack.pop();
        self.test_context = previous_test;
        self.current_cfg = previous_cfg;
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        let item_scope = self.block_item_scope(block);
        self.scopes.push(item_scope);
        self.push_local_scope();
        visit::visit_block(self, block);
        self.scopes.pop();
        self.scopes.pop();
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if let Some(initializer) = &local.init {
            self.visit_expr(&initializer.expr);
            if let Some((_, diverge)) = &initializer.diverge {
                self.visit_expr(diverge);
            }
        }
        let alias = local
            .init
            .as_ref()
            .and_then(|initializer| transparent_path(&initializer.expr))
            .and_then(|path| self.resolve_value(path))
            .map(|mut value| {
                value.indirect = true;
                value
            });
        let mut identifiers = BTreeSet::new();
        collect_pattern_identifiers(&local.pat, &mut identifiers);
        let scope = self.scopes.last_mut().expect("local scope");
        for identifier in identifiers {
            scope.values.insert(
                identifier,
                alias.map_or(ScopedValueDefinition::Local, ScopedValueDefinition::Api),
            );
        }
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Some(path) = Self::normalized_direct_callee(&call.func) {
            self.inspect_call(call, path);
        } else {
            self.visit_expr(&call.func);
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast ExprPath) {
        self.inspect_function_value(path);
    }

    fn visit_expr_closure(&mut self, closure: &'ast ExprClosure) {
        self.push_local_scope();
        for input in &closure.inputs {
            self.bind_pattern(input);
        }
        self.visit_expr(&closure.body);
        self.scopes.pop();
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast ExprForLoop) {
        self.visit_expr(&expression.expr);
        self.push_local_scope();
        self.bind_pattern(&expression.pat);
        self.visit_block(&expression.body);
        self.scopes.pop();
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        self.visit_expr(&expression.cond);
        self.push_local_scope();
        self.bind_identifiers(condition_pattern_identifiers(&expression.cond));
        self.visit_block(&expression.then_branch);
        self.scopes.pop();
        if let Some((_, else_branch)) = &expression.else_branch {
            self.visit_expr(else_branch);
        }
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        self.visit_expr(&expression.cond);
        self.push_local_scope();
        self.bind_identifiers(condition_pattern_identifiers(&expression.cond));
        self.visit_block(&expression.body);
        self.scopes.pop();
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            self.push_local_scope();
            self.bind_pattern(&arm.pat);
            if let Some((_, guard)) = &arm.guard {
                self.visit_expr(guard);
            }
            self.visit_expr(&arm.body);
            self.scopes.pop();
        }
    }

    fn visit_item_use(&mut self, item: &'ast ItemUse) {
        self.inspect_import(item);
    }

    fn visit_macro(&mut self, expression: &'ast Macro) {
        let mut identifiers = Vec::new();
        Self::macro_restricted_identifiers(expression.tokens.clone(), &mut identifiers);
        if let Some(identifier) = expression.path.segments.last() {
            let identifier = identifier_name(&identifier.ident);
            if Self::unresolved_api(&identifier).is_some() {
                identifiers.push(identifier);
            }
        }
        let exact_candidate = self.explicit_nary_candidate()
            && !identifiers.is_empty()
            && identifiers
                .iter()
                .all(|identifier| identifier == "contract");
        if !identifiers.is_empty() && !exact_candidate {
            self.violations.push(format!(
                "{}::{} contains restricted contraction tokens inside a macro",
                self.relative_path,
                self.current_function()
            ));
        }
        visit::visit_macro(self, expression);
    }
}

fn source_root_module(relative_path: &str) -> Vec<String> {
    let path = Path::new(relative_path);
    if relative_path == "src/lib.rs" || relative_path == "src/main.rs" {
        return Vec::new();
    }
    if path
        .components()
        .next()
        .is_some_and(|part| part.as_os_str() == "tests")
    {
        return Vec::new();
    }
    let Ok(path) = path.strip_prefix("src") else {
        return Vec::new();
    };
    let mut components = path
        .parent()
        .into_iter()
        .flat_map(|parent| parent.components())
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let stem = path.file_stem().unwrap().to_string_lossy();
    if stem != "mod" {
        components.push(stem.into_owned());
    }
    components
}

pub(super) struct AuditOutcome {
    pub(super) violations: Vec<String>,
    pub(super) inventory_seen: BTreeSet<usize>,
}

fn audit_with_root(relative_path: &str, file: &File, root_module: Vec<String>) -> AuditOutcome {
    let model = SourceModel::collect(relative_path, file, root_module.clone());
    let mut visitor = BoundaryVisitor {
        relative_path,
        model: &model,
        current_module: root_module,
        module_path: Vec::new(),
        item_stack: Vec::new(),
        impl_context: None,
        trait_definition: None,
        test_context: false,
        current_cfg: CfgExpr::True,
        scopes: Vec::new(),
        violations: Vec::new(),
        inventory_seen: BTreeSet::new(),
    };
    visitor.visit_file(file);
    AuditOutcome {
        violations: visitor.violations,
        inventory_seen: visitor.inventory_seen,
    }
}

pub(super) fn audit_repository_file(relative_path: &str, file: &File) -> AuditOutcome {
    audit_with_root(relative_path, file, source_root_module(relative_path))
}

pub(super) fn audit_fixture(relative_path: &str, source: &str) -> Vec<String> {
    let file = syn::parse_file(source).unwrap();
    audit_with_root(relative_path, &file, Vec::new()).violations
}

pub(super) fn inventory_len() -> usize {
    HANDWRITTEN_CONTRACTION_INVENTORY.len()
}
