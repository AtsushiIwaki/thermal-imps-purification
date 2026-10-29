//! Fixed-matrix SVD timing control, compiled directly with rustc against either Faer build.
//! Input/generation, validation and process startup are outside each timed thin_svd call.
use std::{fs, hint::black_box, time::Instant};
fn json_float(x: f64) -> String {
    if x.is_finite() {
        x.to_string()
    } else {
        "null".into()
    }
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        3,
        "usage: captured-f64-column-major-file|healthy threads samples"
    );
    let n = 963;
    let threads: usize = args[1].parse().unwrap();
    let count: usize = args[2].parse().unwrap();
    assert!(matches!(threads, 1 | 24) && (1..=5).contains(&count));
    faer::set_global_parallelism(if threads == 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(threads)
    });
    let data = if args[0] == "healthy" {
        let mut seed = 0x4b4c_545f_5356_4431_u64;
        // Deterministic dense, diagonally dominated comparison; initialization only.
        (0..n * n)
            .map(|k| {
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                let bits = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
                let noise = ((bits as f64) * (1.0 / ((1_u64 << 53) as f64)) - 0.5) * 0.2 / n as f64;
                noise
                    + if k % n == k / n {
                        0.5 + (k % n) as f64 / n as f64
                    } else {
                        0.0
                    }
            })
            .collect::<Vec<_>>()
    } else {
        let bytes = fs::read(&args[0]).unwrap();
        assert_eq!(bytes.len(), 8 * n * n);
        bytes
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert!(data.iter().all(|x| x.is_finite()));
    let a = faer::MatRef::from_column_major_slice(&data, n, n);
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for x in &data {
        for b in x.to_bits().to_le_bytes() {
            checksum ^= u64::from(b);
            checksum = checksum.wrapping_mul(0x100_0000_01b3);
        }
    }
    let _warmup = black_box(a.thin_svd());
    drop(_warmup);
    let mut last = None;
    let mut times = Vec::with_capacity(count);
    let mut nonfinite = Vec::with_capacity(count);
    for _ in 0..count {
        let start = Instant::now();
        let result = black_box(a.thin_svd());
        times.push(start.elapsed().as_secs_f64());
        match result {
            Ok(r) => {
                let mut bad = (0..n).filter(|&i| !r.S()[i].is_finite()).count();
                for j in 0..n {
                    for i in 0..n {
                        bad += usize::from(!r.U()[(i, j)].is_finite())
                            + usize::from(!r.V()[(i, j)].is_finite());
                    }
                }
                nonfinite.push(bad);
                last = Some(r);
            }
            Err(_) => {
                nonfinite.push(usize::MAX);
                last = None;
            }
        }
    }
    let mut reconstruction = f64::NAN;
    let mut u_orthogonality = f64::NAN;
    let mut v_orthogonality = f64::NAN;
    if nonfinite.last() == Some(&0) {
        let r = last.as_ref().unwrap();
        let us = faer::Mat::from_fn(n, n, |i, j| r.U()[(i, j)] * r.S()[j]);
        let reconstructed = &us * r.V().transpose();
        reconstruction = (&reconstructed - a).norm_l2() / a.norm_l2();
        let eye = faer::Mat::<f64>::identity(n, n);
        u_orthogonality = (r.U().transpose() * r.U() - &eye).norm_l2();
        v_orthogonality = (r.V().transpose() * r.V() - &eye).norm_l2();
    }
    println!("{{\"n\":{n},\"threads\":{threads},\"warmup\":1,\"input_fnv1a64\":\"{checksum:016x}\",\"seconds\":{times:?},\"nonfinite_counts\":{nonfinite:?},\"relative_reconstruction\":{},\"u_orthogonality\":{},\"v_orthogonality\":{}}}",json_float(reconstruction),json_float(u_orthogonality),json_float(v_orthogonality));
}
