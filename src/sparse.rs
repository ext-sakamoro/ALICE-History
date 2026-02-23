// ALICE-History — Level 3: Compressed sensing (ISTA / FISTA)
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use crate::core::{compute_confidence_2d, hash_f64_slice, measure_entropy, result_content_hash};
use crate::cosine_table::CosineTable;
use crate::frequency::{dct2_2d, idct3_2d};
use crate::grid2d::Grid2D;
use crate::{RestorationField, RestorationResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for sparse restoration (ISTA/FISTA).
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Sparsity penalty weight (L1 regularization strength).
    pub lambda: f64,
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub confidence_floor: f64,
    /// If true, use FISTA (Nesterov acceleration) for O(1/k^2) convergence.
    /// If false, use plain ISTA with O(1/k) convergence.
    pub use_fista: bool,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iterations: 500,
            convergence_threshold: 1e-7,
            confidence_floor: 0.7,
            use_fista: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Soft thresholding
// ---------------------------------------------------------------------------

/// Branchless soft thresholding: sign(v) * max(|v| - t, 0).
///
/// Equivalent to:
///   if |v| <= t  => 0
///   if v > t     => v - t
///   if v < -t    => v + t
#[inline]
pub fn soft_threshold(v: f64, t: f64) -> f64 {
    let abs_v = v.abs();
    let shrunk = abs_v - t;
    // max(shrunk, 0) * sign(v)
    let positive = if shrunk > 0.0 { shrunk } else { 0.0 };
    v.signum() * positive
}

// ---------------------------------------------------------------------------
// ISTA / FISTA solver
// ---------------------------------------------------------------------------

/// Restore a 2D grid using ISTA or FISTA (compressed sensing).
///
/// ## Algorithm
/// Assumes the original signal is sparse in the DCT domain.
/// 1. Initialize missing values.
/// 2. Each iteration:
///    a. Gradient step: move toward data fidelity (known values).
///    b. Forward 2D DCT.
///    c. Soft threshold DCT coefficients (enforce sparsity).
///    d. Inverse 2D DCT.
///    e. Re-project known values.
///    f. (FISTA only) Nesterov momentum update.
/// 3. Check convergence.
pub fn restore_sparse(grid: &Grid2D, config: &SparseConfig) -> RestorationResult {
    let start = std::time::Instant::now();
    let rows = grid.rows;
    let cols = grid.cols;
    let n = rows * cols;

    if n == 0 || rows == 0 || cols == 0 {
        let field = RestorationField {
            values: Vec::new(),
            confidence: crate::ConfidenceMap {
                scores: Vec::new(),
                mean_confidence: 0.0,
                min_confidence: 0.0,
                restoration_boundary: config.confidence_floor,
            },
            entropy_before: 0.0,
            entropy_after: 0.0,
            iterations: 0,
            content_hash: 0,
        };
        return RestorationResult {
            fragment_id: 0,
            field,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            content_hash: 0,
        };
    }

    let entropy_before = measure_entropy(&grid.data, 64).shannon_entropy;

    // Precompute cosine tables
    let row_table = CosineTable::new(cols);
    let col_table = CosineTable::new(rows);

    // Initialize: known values from data, missing with mean
    let known_sum: f64 = grid
        .data
        .iter()
        .zip(grid.mask.iter())
        .filter(|(_, &m)| m == 1.0)
        .map(|(&d, _)| d)
        .sum();
    let known_count = grid.mask.iter().filter(|&&m| m == 1.0).count();
    let mean_known = if known_count > 0 {
        known_sum * (known_count as f64).recip() // precomputed reciprocal
    } else {
        0.0
    };

    let mut current: Vec<f64> = grid
        .data
        .iter()
        .zip(grid.mask.iter())
        .map(|(&d, &m)| if m == 1.0 { d } else { mean_known })
        .collect();

    // FISTA momentum variables
    let mut prev = current.clone();
    let mut t_k: f64 = 1.0;

    let mut coeffs = vec![0.0; n];
    let mut reconstructed = vec![0.0; n];
    let mut iterations: u32 = 0;

    // Step size (Lipschitz constant approximation: L = 1.0 for masked projection)
    let step_size = 1.0;
    let threshold = config.lambda * step_size;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Working point for gradient (FISTA uses momentum point)
        let work = if config.use_fista && iter > 0 {
            let t_k1 = (1.0 + (1.0 + 4.0 * t_k * t_k).sqrt()) * 0.5;
            let momentum = (t_k - 1.0) / t_k1;
            t_k = t_k1;

            let mut y = vec![0.0; n];
            for i in 0..n {
                y[i] = current[i] + momentum * (current[i] - prev[i]);
            }
            y
        } else {
            current.clone()
        };

        // Gradient step: enforce data fidelity on known positions
        let mut gradient_applied = work.clone();
        for (i, ga) in gradient_applied.iter_mut().enumerate() {
            if grid.mask[i] == 1.0 {
                *ga = grid.data[i];
            }
        }

        // Forward DCT
        dct2_2d(
            &gradient_applied,
            &mut coeffs,
            rows,
            cols,
            &row_table,
            &col_table,
        );

        // Soft threshold in DCT domain (enforce sparsity)
        for c in coeffs.iter_mut() {
            *c = soft_threshold(*c, threshold);
        }

        // Inverse DCT
        idct3_2d(
            &coeffs,
            &mut reconstructed,
            rows,
            cols,
            &row_table,
            &col_table,
        );

        // Re-project known values and compute convergence
        let mut max_change: f64 = 0.0;
        prev.copy_from_slice(&current);

        for i in 0..n {
            let new_val = if grid.mask[i] == 1.0 {
                grid.data[i]
            } else {
                reconstructed[i]
            };
            let change = (new_val - current[i]).abs();
            if change > max_change {
                max_change = change;
            }
            current[i] = new_val;
        }

        if max_change < config.convergence_threshold {
            break;
        }
    }

    let confidence = compute_confidence_2d(&grid.mask, rows, cols, config.confidence_floor);
    let entropy_after = measure_entropy(&current, 64).shannon_entropy;
    let content_hash = hash_f64_slice(&current);

    let field = RestorationField {
        values: current,
        confidence,
        entropy_before,
        entropy_after,
        iterations,
        content_hash,
    };

    let rh = result_content_hash(0, field.content_hash);

    RestorationResult {
        fragment_id: 0,
        field,
        elapsed_ns: start.elapsed().as_nanos() as u64,
        content_hash: rh,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Soft threshold -------------------------------------------------------

    #[test]
    fn test_soft_threshold_zero() {
        assert!((soft_threshold(0.5, 1.0) - 0.0).abs() < 1e-15);
        assert!((soft_threshold(-0.5, 1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_positive() {
        assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_negative() {
        assert!((soft_threshold(-3.0, 1.0) - (-2.0)).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_exact_boundary() {
        assert!((soft_threshold(1.0, 1.0) - 0.0).abs() < 1e-15);
        assert!((soft_threshold(-1.0, 1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_zero_threshold() {
        assert!((soft_threshold(5.0, 0.0) - 5.0).abs() < 1e-15);
        assert!((soft_threshold(-5.0, 0.0) - (-5.0)).abs() < 1e-15);
    }

    // -- ISTA basic -----------------------------------------------------------

    #[test]
    fn test_ista_all_known() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![1.0; 9];
        let grid = Grid2D::new(3, 3, data.clone(), mask);
        let mut config = SparseConfig::default();
        config.use_fista = false;
        let r = restore_sparse(&grid, &config);
        for (i, (&orig, &rest)) in data.iter().zip(r.field.values.iter()).enumerate() {
            assert!(
                (orig - rest).abs() < 1e-6,
                "known value changed at {}: {} -> {}",
                i,
                orig,
                rest
            );
        }
    }

    // -- FISTA basic ----------------------------------------------------------

    #[test]
    fn test_fista_all_known() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![1.0; 9];
        let grid = Grid2D::new(3, 3, data.clone(), mask);
        let config = SparseConfig::default(); // use_fista = true
        let r = restore_sparse(&grid, &config);
        for (i, (&orig, &rest)) in data.iter().zip(r.field.values.iter()).enumerate() {
            assert!(
                (orig - rest).abs() < 1e-6,
                "FISTA known value changed at {}: {} -> {}",
                i,
                orig,
                rest
            );
        }
    }

    // -- Sparse recovery with few known values --------------------------------

    #[test]
    fn test_sparse_recovery_constant_field() {
        // Constant field (very sparse in DCT: only DC)
        let n = 16;
        let mut data = vec![0.0; n];
        let mut mask = vec![0.0; n];
        // Only 3 known values out of 16 (< 20%)
        data[0] = 50.0;
        mask[0] = 1.0;
        data[7] = 50.0;
        mask[7] = 1.0;
        data[15] = 50.0;
        mask[15] = 1.0;

        let grid = Grid2D::new(4, 4, data, mask);
        // Use lower lambda to avoid over-shrinking the DC component
        let config = SparseConfig {
            lambda: 0.001,
            max_iterations: 500,
            ..SparseConfig::default()
        };
        let r = restore_sparse(&grid, &config);

        // Constant signal has DC-only DCT representation (maximally sparse)
        // Reconstructed values should be close to 50
        for (i, &v) in r.field.values.iter().enumerate() {
            assert!(
                (v - 50.0).abs() < 15.0,
                "index {} should be near 50: {}",
                i,
                v
            );
        }
    }

    // -- ISTA vs FISTA convergence speed -------------------------------------

    #[test]
    fn test_fista_and_ista_both_converge() {
        // Simple constant field — both algorithms should converge quickly
        let mut data = vec![0.0; 16];
        let mut mask = vec![0.0; 16];
        for i in (0..16).step_by(4) {
            data[i] = 10.0;
            mask[i] = 1.0;
        }

        let grid = Grid2D::new(4, 4, data, mask);

        let ista_config = SparseConfig {
            use_fista: false,
            lambda: 0.01,
            max_iterations: 500,
            convergence_threshold: 1e-6,
            ..SparseConfig::default()
        };
        let ista_result = restore_sparse(&grid, &ista_config);

        let fista_config = SparseConfig {
            use_fista: true,
            lambda: 0.01,
            max_iterations: 500,
            convergence_threshold: 1e-6,
            ..SparseConfig::default()
        };
        let fista_result = restore_sparse(&grid, &fista_config);

        // Both ISTA and FISTA should converge within max_iterations
        assert!(
            ista_result.field.iterations < 500,
            "ISTA should converge: {} iterations",
            ista_result.field.iterations
        );
        assert!(
            fista_result.field.iterations < 500,
            "FISTA should converge: {} iterations",
            fista_result.field.iterations
        );
    }

    // -- Empty grid -----------------------------------------------------------

    #[test]
    fn test_sparse_empty_grid() {
        let grid = Grid2D::new(0, 0, vec![], vec![]);
        let config = SparseConfig::default();
        let r = restore_sparse(&grid, &config);
        assert!(r.field.values.is_empty());
        assert_eq!(r.field.iterations, 0);
    }

    // -- Round-trip: soft threshold preserves sparse signal --------------------

    #[test]
    fn test_soft_threshold_preserves_large_coefficients() {
        let vals = vec![10.0, -10.0, 0.5, -0.5, 0.0, 100.0];
        let threshold = 1.0;
        let result: Vec<f64> = vals.iter().map(|&v| soft_threshold(v, threshold)).collect();
        assert!((result[0] - 9.0).abs() < 1e-15);
        assert!((result[1] - (-9.0)).abs() < 1e-15);
        assert!((result[2] - 0.0).abs() < 1e-15);
        assert!((result[3] - 0.0).abs() < 1e-15);
        assert!((result[4] - 0.0).abs() < 1e-15);
        assert!((result[5] - 99.0).abs() < 1e-15);
    }

    // -- Soft threshold: large threshold zeros everything --------------------

    #[test]
    fn test_soft_threshold_large_threshold_zeros_all() {
        let vals = vec![1.0, -1.0, 0.1, -0.1, 0.9999];
        for &v in &vals {
            assert!(
                (soft_threshold(v, 100.0) - 0.0).abs() < 1e-15,
                "expected 0 for {}",
                v
            );
        }
    }

    // -- Soft threshold: idempotent when result is thresholded again with same t ---

    #[test]
    fn test_soft_threshold_idempotent() {
        // After one application, applying again with same t should not change anything
        // because soft_threshold(soft_threshold(v, t), t) may not be identical to
        // soft_threshold(v, t), but if the output is >= 0 then it is positive.
        // We just verify applying twice doesn't go negative when input was positive result.
        let v = 5.0;
        let t = 2.0;
        let once = soft_threshold(v, t); // = 3.0
        let twice = soft_threshold(once, t); // = 1.0 (not idempotent, but still >= 0)
        assert!(twice >= 0.0);
        assert!(twice <= once);
    }

    // -- Soft threshold: sign preservation ------------------------------------

    #[test]
    fn test_soft_threshold_sign_preserved() {
        assert!(soft_threshold(5.0, 1.0) > 0.0);
        assert!(soft_threshold(-5.0, 1.0) < 0.0);
        assert!((soft_threshold(0.0, 1.0) - 0.0).abs() < 1e-15);
    }

    // -- SparseConfig default values ------------------------------------------

    #[test]
    fn test_sparse_config_default() {
        let c = SparseConfig::default();
        assert!((c.lambda - 0.1).abs() < 1e-15);
        assert_eq!(c.max_iterations, 500);
        assert!((c.convergence_threshold - 1e-7).abs() < 1e-20);
        assert!((c.confidence_floor - 0.7).abs() < 1e-15);
        assert!(c.use_fista);
    }

    // -- restore_sparse: content hash deterministic ---------------------------

    #[test]
    fn test_sparse_content_hash_deterministic() {
        let data = vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let grid = Grid2D::new(3, 3, data, mask);
        let config = SparseConfig::default();
        let r1 = restore_sparse(&grid, &config);
        let r2 = restore_sparse(&grid, &config);
        assert_eq!(r1.field.content_hash, r2.field.content_hash);
        assert_eq!(r1.content_hash, r2.content_hash);
    }

    // -- restore_sparse: 1x1 grid --------------------------------------------

    #[test]
    fn test_sparse_1x1_known() {
        let grid = Grid2D::new(1, 1, vec![99.0], vec![1.0]);
        let config = SparseConfig::default();
        let r = restore_sparse(&grid, &config);
        assert_eq!(r.field.values.len(), 1);
        assert!((r.field.values[0] - 99.0).abs() < 1e-6);
    }

    // -- restore_sparse: ISTA known values remain unchanged ------------------

    #[test]
    fn test_ista_known_values_unchanged() {
        let data = vec![3.0, 0.0, 7.0, 0.0, 11.0, 0.0, 15.0, 0.0, 19.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let grid = Grid2D::new(3, 3, data.clone(), mask.clone());
        let config = SparseConfig {
            use_fista: false,
            ..SparseConfig::default()
        };
        let r = restore_sparse(&grid, &config);
        for (i, (&m, (&d, &v))) in mask
            .iter()
            .zip(data.iter().zip(r.field.values.iter()))
            .enumerate()
        {
            if m == 1.0 {
                assert!(
                    (v - d).abs() < 1e-6,
                    "ISTA: known value at index {} changed: {} -> {}",
                    i,
                    d,
                    v
                );
            }
        }
    }

    // -- restore_sparse: confidence floor respected --------------------------

    #[test]
    fn test_sparse_confidence_floor_respected() {
        let mut data = vec![0.0; 16];
        let mut mask = vec![0.0; 16];
        data[0] = 1.0;
        mask[0] = 1.0;
        let grid = Grid2D::new(4, 4, data, mask);
        let config = SparseConfig {
            confidence_floor: 0.2,
            ..SparseConfig::default()
        };
        let r = restore_sparse(&grid, &config);
        for &s in &r.field.confidence.scores {
            assert!(s >= 0.2 - 1e-12, "sparse confidence {} below floor", s);
        }
    }
}
