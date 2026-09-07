use std::hint::black_box;
use std::time::{Duration, Instant};

pub(crate) const WARMUPS: usize = 3;
pub(crate) const SAMPLE_COUNT: usize = 9;

pub(crate) fn measure_release<T>(mut operation: impl FnMut() -> T) -> Vec<Duration> {
    for _ in 0..WARMUPS {
        black_box(operation());
    }
    (0..SAMPLE_COUNT)
        .map(|_| {
            let started = Instant::now();
            black_box(operation());
            started.elapsed()
        })
        .collect()
}

pub(crate) fn measure_release_pairwise_and_frozen<T, U>(
    mut pairwise_operation: impl FnMut() -> T,
    mut frozen_operation: impl FnMut() -> U,
) -> (Vec<Duration>, Vec<Duration>) {
    for _ in 0..WARMUPS {
        black_box(pairwise_operation());
        black_box(frozen_operation());
    }

    let mut pairwise_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut frozen_samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let started = Instant::now();
        black_box(pairwise_operation());
        pairwise_samples.push(started.elapsed());

        let started = Instant::now();
        black_box(frozen_operation());
        frozen_samples.push(started.elapsed());
    }
    (pairwise_samples, frozen_samples)
}

pub(crate) fn sorted_median(samples: &[Duration]) -> Duration {
    assert_eq!(samples.len(), SAMPLE_COUNT);
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[SAMPLE_COUNT / 2]
}

#[allow(dead_code)]
pub(crate) fn sample_nanos(samples: &[Duration]) -> Vec<u128> {
    samples.iter().map(Duration::as_nanos).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn sampler_uses_three_warmups_nine_samples_and_sorted_median() {
        let calls = Cell::new(0usize);
        let samples = measure_release(|| {
            calls.set(calls.get() + 1);
            calls.get()
        });
        assert_eq!(calls.get(), 12);
        assert_eq!(samples.len(), 9);
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        assert_eq!(sorted_median(&samples), sorted[4]);
    }

    #[test]
    fn paired_sampler_alternates_paths_for_every_warmup_and_sample() {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let pairwise_calls = std::rc::Rc::clone(&calls);
        let frozen_calls = std::rc::Rc::clone(&calls);

        let (pairwise_samples, frozen_samples) = measure_release_pairwise_and_frozen(
            move || pairwise_calls.borrow_mut().push("pairwise"),
            move || frozen_calls.borrow_mut().push("frozen"),
        );

        assert_eq!(pairwise_samples.len(), SAMPLE_COUNT);
        assert_eq!(frozen_samples.len(), SAMPLE_COUNT);
        let calls = calls.borrow();
        assert_eq!(calls.len(), 2 * (WARMUPS + SAMPLE_COUNT));
        for pair in calls.chunks_exact(2) {
            assert_eq!(pair, ["pairwise", "frozen"]);
        }
    }
}
