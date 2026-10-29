//! Bounded, failure-only real gate SVD evidence. Never an iMPS checkpoint.
use crate::tensor::{Idx, Tensor};
use anyhow::{ensure, Context, Result};
use hdf5_metno::{File, H5Type};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};

#[derive(Clone)]
struct Capture {
    path: PathBuf,
    source: String,
    worker: String,
    step: u64,
    beta: f64,
    call: u64,
    attempted: bool,
}
thread_local! {static ACTIVE:RefCell<Option<Capture>>=const{RefCell::new(None)};}
pub(crate) struct CaptureGuard;
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        ACTIVE.with(|v| *v.borrow_mut() = None);
    }
}
pub(crate) fn activate(directory: &Path, source: &str, worker: &str) -> Result<CaptureGuard> {
    let directory = std::path::absolute(directory)?;
    let path = directory
        .parent()
        .context("missing failure-record parent")?
        .canonicalize()?
        .join("svd-failure.h5");
    ensure!(
        !path.exists() && !path.is_symlink(),
        "SVD failure destination already exists"
    );
    ACTIVE.with(|slot| -> Result<()> {
        let mut slot = slot.borrow_mut();
        ensure!(slot.is_none(), "nested SVD failure capture unsupported");
        *slot = Some(Capture {
            path,
            source: source.into(),
            worker: worker.into(),
            step: 0,
            beta: 0.,
            call: 0,
            attempted: false,
        });
        Ok(())
    })?;
    Ok(CaptureGuard)
}
pub(crate) fn begin_step(step: u64, beta: f64) {
    ACTIVE.with(|v| {
        if let Some(c) = v.borrow_mut().as_mut() {
            c.step = step;
            c.beta = beta;
            c.call = 0;
        }
    });
}
pub(crate) fn enter_svd() {
    ACTIVE.with(|v| {
        if let Some(c) = v.borrow_mut().as_mut() {
            c.call += 1;
        }
    });
}
pub(crate) fn record_error(t: &Tensor, left: &[Idx], error: &str) {
    let context = ACTIVE.with(|v| {
        let mut v = v.borrow_mut();
        let c = v.as_mut()?;
        if c.attempted {
            return None;
        }
        c.attempted = true;
        Some(c.clone())
    });
    if let Some(c) = context {
        if let Err(e) = write(&c, t, left, error) {
            eprintln!("SVD failure capture failed: {e:#}; preserving original SVD error");
        }
    }
}
fn scalar<T: H5Type>(f: &File, name: &str, value: &T) -> Result<()> {
    f.new_dataset::<T>()
        .shape(())
        .create(name)?
        .write_scalar(value)?;
    Ok(())
}
fn array<T: H5Type>(f: &File, name: &str, value: &[T]) -> Result<()> {
    f.new_dataset::<T>()
        .shape(value.len())
        .create(name)?
        .write_raw(value)?;
    Ok(())
}
fn digest(data: &[f64]) -> String {
    let mut h = Sha256::new();
    for x in data {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn write(c: &Capture, t: &Tensor, left: &[Idx], error: &str) -> Result<()> {
    let mut order = left.to_vec();
    order.extend(t.indices.iter().filter(|v| !left.contains(v)).cloned());
    let m = left
        .iter()
        .try_fold(1usize, |v, i| v.checked_mul(i.dim))
        .context("failure matrix dimension overflow")?;
    let n = t
        .indices
        .iter()
        .filter(|i| !left.contains(i))
        .try_fold(1usize, |v, i| v.checked_mul(i.dim))
        .context("failure matrix dimension overflow")?;
    ensure!(
        (1..=2304).contains(&m) && (1..=2304).contains(&n),
        "failure matrix exceeds bounded cap256 dimensions"
    );
    let data = t.permute_indices(&order)?.to_vec::<f64>()?;
    ensure!(data.len() == m * n, "failure matrix storage mismatch");
    let positions = left
        .iter()
        .map(|i| {
            t.indices
                .iter()
                .position(|v| v == i)
                .map(|v| v as u64)
                .context("invalid failure split")
        })
        .collect::<Result<Vec<_>>>()?;
    let f = File::create_excl(&c.path).context("reserve exclusive failure HDF5")?;
    scalar(&f, "complete", &0u8)?;
    scalar(&f, "schema_version", &1u32)?;
    scalar(&f, "step", &c.step)?;
    scalar(&f, "beta_before_step", &c.beta)?;
    scalar(&f, "svd_call", &c.call)?;
    array(&f, "shape", &[m as u64, n as u64])?;
    array(&f, "left_positions", &positions)?;
    array(
        &f,
        "tensor_dimensions",
        &t.dims().into_iter().map(|v| v as u64).collect::<Vec<_>>(),
    )?;
    array(&f, "matrix_col_major", &data)?;
    array(&f, "matrix_sha256", digest(&data).as_bytes())?;
    array(&f, "kind", b"real-bond-svd-failure-v1")?;
    array(&f, "source_revision", c.source.as_bytes())?;
    array(&f, "worker_sha256", c.worker.as_bytes())?;
    array(&f, "error", error.as_bytes())?;
    f.flush()?;
    f.dataset("complete")?.write_scalar(&1u8)?;
    f.flush()?;
    Ok(())
}
fn text(f: &File, name: &str) -> Result<String> {
    let d = f.dataset(name)?;
    ensure!(d.size() <= 65536, "oversized diagnostic text");
    Ok(String::from_utf8(d.read_raw::<u8>()?)?)
}
pub(crate) fn inspect(path: &Path) -> Result<Value> {
    ensure!(
        std::fs::symlink_metadata(path)?.file_type().is_file(),
        "diagnostic must be a regular file"
    );
    let f = File::open(path)?;
    ensure!(
        f.dataset("complete")?.read_scalar::<u8>()? == 1
            && f.dataset("schema_version")?.read_scalar::<u32>()? == 1
            && text(&f, "kind")? == "real-bond-svd-failure-v1",
        "incomplete/unsupported SVD diagnostic"
    );
    let shape = f.dataset("shape")?;
    ensure!(shape.size() == 2, "invalid diagnostic matrix rank");
    let shape = shape.read_raw::<u64>()?;
    ensure!(
        shape.iter().all(|v| (1..=2304).contains(v)),
        "diagnostic exceeds cap256 matrix bound"
    );
    let d = f.dataset("matrix_col_major")?;
    ensure!(
        d.size() == (shape[0] * shape[1]) as usize,
        "diagnostic matrix length mismatch"
    );
    let data = d.read_raw::<f64>()?;
    let sha = digest(&data);
    ensure!(
        text(&f, "matrix_sha256")? == sha,
        "diagnostic matrix hash mismatch"
    );
    let dims = f.dataset("tensor_dimensions")?;
    let axes = f.dataset("left_positions")?;
    ensure!(
        (2..=16).contains(&dims.size()) && (1..dims.size()).contains(&axes.size()),
        "invalid diagnostic tensor split"
    );
    let dims = dims.read_raw::<u64>()?;
    let axes = axes.read_raw::<u64>()?;
    let selected = axes
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        selected.len() == axes.len()
            && selected.iter().all(|v| (*v as usize) < dims.len())
            && dims.iter().all(|v| (1..=2304).contains(v)),
        "invalid diagnostic axes/dimensions"
    );
    let mut split = [1u64, 1];
    for (i, d) in dims.iter().enumerate() {
        let side = usize::from(!selected.contains(&(i as u64)));
        split[side] = split[side]
            .checked_mul(*d)
            .context("diagnostic dimension overflow")?;
    }
    ensure!(
        split.as_slice() == shape.as_slice(),
        "diagnostic tensor/matrix shape mismatch"
    );
    let step = f.dataset("step")?.read_scalar::<u64>()?;
    let beta = f.dataset("beta_before_step")?.read_scalar::<f64>()?;
    let call = f.dataset("svd_call")?.read_scalar::<u64>()?;
    ensure!(
        (155..=1014).contains(&step)
            && beta.is_finite()
            && (10. / 7. - 1e-12..=10.).contains(&beta)
            && (1..=3).contains(&call),
        "invalid diagnostic step/substep"
    );
    let source = text(&f, "source_revision")?;
    let worker = text(&f, "worker_sha256")?;
    ensure!(
        [(source.as_str(), 40), (worker.as_str(), 64)]
            .iter()
            .all(|(s, n)| s.len() == *n
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))),
        "invalid diagnostic producer identity"
    );
    let error = text(&f, "error")?;
    ensure!(!error.trim().is_empty(), "missing diagnostic source error");
    Ok(
        json!({"kind":"real-bond-svd-failure-v1","diagnostic_only":true,"step":step,"beta_before_step":beta,"svd_call":call,
        "shape":shape,"matrix_sha256":sha,"nonfinite_count":data.iter().filter(|v|!v.is_finite()).count(),
        "source_revision":source,"worker_sha256":worker,"error":error,"payload_verified":true}),
    )
}
