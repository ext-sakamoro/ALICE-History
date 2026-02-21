// ALICE-History — 1D inverse entropy restoration solver
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use rayon::prelude::*;

use crate::core::{compute_confidence, fnv1a, hash_f64_slice, measure_entropy, result_content_hash};
use crate::{ConfidenceMap, Fragment, InversionConfig, RestorationField, RestorationResult};

// ---------------------------------------------------------------------------
// Core 1D solver
// ---------------------------------------------------------------------------

/// 1D inverse entropy restoration.
///
/// Uses iterative regularised least-squares to fill in missing values while
/// minimising entropy of the result.
///
/// ## Algorithm
///
/// 1. Initialise missing values with the mean of known values.
/// 2. For each iteration:
///    a. Compute a local smoothness gradient (minimise discontinuities at
///       boundaries between known and unknown regions).
///    b. Apply Tikhonov regularisation (prevent overfitting).
///    c. Update missing values: `new = old - lr * gradient`.
///    d. Check convergence (`max_change < threshold`).
/// 3. Compute confidence: higher for values near known data, lower for values
///    far from known data.
pub fn restore_1d(fragment: &Fragment, config: &InversionConfig) -> RestorationResult {
    let start = std::time::Instant::now();
    let n = fragment.data.len();

    if n == 0 {
        let field = RestorationField {
            values: Vec::new(),
            confidence: ConfidenceMap {
                scores: Vec::new(),
                mean_confidence: 0.0,
                min_confidence: 0.0,
                restoration_boundary: config.confidence_floor,
            },
            entropy_before: 0.0,
            entropy_after: 0.0,
            iterations: 0,
            content_hash: fnv1a(&[]),
        };
        return RestorationResult {
            fragment_id: fragment.id,
            field,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            content_hash: fnv1a(&[]),
        };
    }

    let entropy_before = measure_entropy(&fragment.data, 64).shannon_entropy;

    // Step 1: initialise restored array -- copy known, fill missing with mean.
    let known_sum: f64 = fragment
        .data
        .iter()
        .zip(fragment.mask.iter())
        .filter(|(_, &m)| m == 1.0)
        .map(|(&d, _)| d)
        .sum();
    let known_count = fragment.mask.iter().filter(|&&m| m == 1.0).count();
    let mean_known = if known_count > 0 {
        known_sum * (known_count as f64).recip() // precomputed reciprocal
    } else {
        0.0
    };

    let mut restored: Vec<f64> = fragment
        .data
        .iter()
        .zip(fragment.mask.iter())
        .map(|(&d, &m)| if m == 1.0 { d } else { mean_known })
        .collect();

    // Step 2: iterative solver.
    let learning_rate = 0.5;
    let mut iterations: u32 = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let mut max_change: f64 = 0.0;

        for i in 0..n {
            if fragment.mask[i] == 1.0 {
                continue;
            }

            let left = if i > 0 { restored[i - 1] } else { restored[i] };
            let right = if i + 1 < n { restored[i + 1] } else { restored[i] };
            let neighbour_avg = (left + right) * 0.5;
            let smooth_grad = restored[i] - neighbour_avg;

            let reg_grad = config.regularization * (restored[i] - mean_known);

            let grad = smooth_grad + reg_grad;
            let new_val = restored[i] - learning_rate * grad;
            let change = (new_val - restored[i]).abs();
            if change > max_change {
                max_change = change;
            }
            restored[i] = new_val;
        }

        if max_change < config.convergence_threshold {
            break;
        }
    }

    // Step 3: confidence and entropy.
    let confidence =
        compute_confidence(&fragment.data, &fragment.mask, &restored, config.confidence_floor);
    let entropy_after = measure_entropy(&restored, 64).shannon_entropy;
    let content_hash = hash_f64_slice(&restored);

    let field = RestorationField {
        values: restored,
        confidence,
        entropy_before,
        entropy_after,
        iterations,
        content_hash,
    };

    let rh = result_content_hash(fragment.id, field.content_hash);

    RestorationResult {
        fragment_id: fragment.id,
        field,
        elapsed_ns: start.elapsed().as_nanos() as u64,
        content_hash: rh,
    }
}

/// Batch restore multiple fragments using the 1D solver in parallel (Rayon).
pub fn restore_1d_batch(
    fragments: &[Fragment],
    config: &InversionConfig,
) -> Vec<RestorationResult> {
    fragments.par_iter().map(|f| restore_1d(f, config)).collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FragmentKind;

    #[test]
    fn test_restore_1d_all_known() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mask = vec![1.0; 5];
        let f = Fragment::new(1, FragmentKind::Text, data.clone(), mask, 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        for (orig, restored) in data.iter().zip(r.field.values.iter()) {
            assert!(
                (orig - restored).abs() < 1e-12,
                "known value changed: {} -> {}",
                orig,
                restored
            );
        }
    }

    #[test]
    fn test_restore_1d_single_gap() {
        let data = vec![10.0, 0.0, 30.0];
        let mask = vec![1.0, 0.0, 1.0];
        let f = Fragment::new(2, FragmentKind::Inscription, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        let mid = r.field.values[1];
        assert!((mid - 20.0).abs() < 1.0, "expected ~20, got {}", mid);
    }

    #[test]
    fn test_restore_1d_multiple_gaps() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 40.0];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let f = Fragment::new(3, FragmentKind::Audio, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        for i in 0..4 {
            assert!(
                r.field.values[i] <= r.field.values[i + 1] + 1.0,
                "expected roughly increasing at index {}: {} > {}",
                i,
                r.field.values[i],
                r.field.values[i + 1]
            );
        }
    }

    #[test]
    fn test_restore_1d_empty() {
        let f = Fragment::new(0, FragmentKind::Text, vec![], vec![], 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        assert!(r.field.values.is_empty());
        assert_eq!(r.field.iterations, 0);
    }

    #[test]
    fn test_restore_1d_all_missing() {
        let f = Fragment::new(0, FragmentKind::Text, vec![0.0; 10], vec![0.0; 10], 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        for &v in &r.field.values {
            assert!((v - 0.0).abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_restore_1d_batch_multiple() {
        let f1 =
            Fragment::new(10, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let f2 = Fragment::new(11, FragmentKind::Image, vec![5.0, 5.0], vec![1.0, 1.0], 0);
        let config = InversionConfig::default();
        let results = restore_1d_batch(&[f1, f2], &config);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].fragment_id, 10);
        assert_eq!(results[1].fragment_id, 11);
    }

    #[test]
    fn test_restore_1d_hash_deterministic() {
        let f =
            Fragment::new(99, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let config = InversionConfig::default();
        let r1 = restore_1d(&f, &config);
        let r2 = restore_1d(&f, &config);
        assert_eq!(r1.field.content_hash, r2.field.content_hash);
        assert_eq!(r1.content_hash, r2.content_hash);
    }

    #[test]
    fn test_restore_1d_heavily_degraded() {
        let n = 50;
        let mut data = vec![0.0; n];
        let mut mask = vec![0.0; n];
        data[0] = 100.0;
        mask[0] = 1.0;
        data[n - 1] = 100.0;
        mask[n - 1] = 1.0;
        let f = Fragment::new(4, FragmentKind::Image, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_1d(&f, &config);
        for (i, &v) in r.field.values.iter().enumerate() {
            assert!((v - 100.0).abs() < 5.0, "index {} too far from 100: {}", i, v);
        }
    }
}
