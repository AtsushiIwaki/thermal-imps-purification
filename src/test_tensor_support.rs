use crate::tensor::Tensor;
use num_complex::Complex64;

pub(crate) fn dense_c64(stage: &'static str, tensor: &Tensor) -> Result<Vec<Complex64>, String> {
    let data = if tensor.is_complex() {
        tensor.to_vec::<Complex64>().map_err(|e| format!("{stage}: {e}"))?
    } else {
        tensor.to_vec::<f64>().map_err(|e| format!("{stage}: {e}"))?
            .into_iter().map(|re| Complex64::new(re, 0.0)).collect()
    };
    if data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(format!("{stage}: tensor contains a non-finite value"));
    }
    Ok(data)
}

pub(crate) fn frobenius_norm(data: &[Complex64]) -> f64 {
    let mut scale = 0.0;
    let mut scaled_sum_squares = 1.0;
    for component in data
        .iter()
        .flat_map(|value| [value.re.abs(), value.im.abs()])
        .filter(|component| *component != 0.0)
    {
        if component.is_nan() {
            return f64::NAN;
        }
        if component.is_infinite() {
            return f64::INFINITY;
        }
        if scale < component {
            scaled_sum_squares = 1.0 + scaled_sum_squares * (scale / component).powi(2);
            scale = component;
        } else {
            scaled_sum_squares += (component / scale).powi(2);
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * scaled_sum_squares.sqrt()
    }
}

pub(crate) fn relative_distance(left: &[Complex64], right: &[Complex64]) -> f64 {
    assert_eq!(
        left.len(),
        right.len(),
        "relative_distance requires equal lengths"
    );
    let difference: Vec<Complex64> = left
        .iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect();
    frobenius_norm(&difference) / frobenius_norm(left).max(f64::EPSILON)
}
