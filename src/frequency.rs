// ALICE-History — Level 2: Frequency-domain restoration (DCT + POCS)
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use crate::core::{compute_confidence_2d, hash_f64_slice, measure_entropy, result_content_hash};
use crate::cosine_table::CosineTable;
use crate::grid2d::Grid2D;
use crate::{RestorationField, RestorationResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for frequency-domain restoration.
#[derive(Debug, Clone)]
pub struct FrequencyConfig {
    pub max_pocs_iterations: u32,
    pub spectral_decay: f64,
    pub convergence_threshold: f64,
    pub confidence_floor: f64,
}

impl Default for FrequencyConfig {
    fn default() -> Self {
        Self {
            max_pocs_iterations: 50,
            spectral_decay: 2.0,
            convergence_threshold: 1e-6,
            confidence_floor: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// 1D DCT-II (forward) and DCT-III (inverse) using CosineTable
// ---------------------------------------------------------------------------

/// Forward DCT-II of a 1D signal using the precomputed cosine table.
///
/// X[k] = sum_n x[n] * cos(pi * (2n+1) * k / (2N))
pub(crate) fn dct2_1d(input: &[f64], output: &mut [f64], table: &CosineTable) {
    let n = table.size;
    debug_assert_eq!(input.len(), n);
    debug_assert_eq!(output.len(), n);
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, &inp) in input.iter().enumerate() {
            sum += inp * table.get(i, k);
        }
        *out = sum;
    }
}

/// Inverse DCT-III (synthesis) of a 1D signal using the precomputed cosine table.
///
/// x[n] = (1/N) * (X[0] + 2 * sum_{k=1}^{N-1} X[k] * cos(pi * (2n+1) * k / (2N)))
pub(crate) fn idct3_1d(input: &[f64], output: &mut [f64], table: &CosineTable) {
    let n = table.size;
    debug_assert_eq!(input.len(), n);
    debug_assert_eq!(output.len(), n);
    let rcp_n = 1.0 / n as f64;
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = input[0]; // full DC term
        for (k, &inp) in input.iter().enumerate().skip(1) {
            sum += 2.0 * inp * table.get(i, k);
        }
        *out = sum * rcp_n;
    }
}

// ---------------------------------------------------------------------------
// 2D Separable DCT-II / DCT-III
// ---------------------------------------------------------------------------

/// Forward 2D DCT-II via separable 1D transforms (rows then columns).
///
/// Uses row_table for the column dimension and col_table for the row dimension.
pub(crate) fn dct2_2d(
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    cols: usize,
    row_table: &CosineTable,
    col_table: &CosineTable,
) {
    let n = rows * cols;
    debug_assert_eq!(input.len(), n);
    debug_assert_eq!(output.len(), n);

    // Temporary buffer for row-transformed data
    let mut temp = vec![0.0; n];
    let mut row_in = vec![0.0; cols];
    let mut row_out = vec![0.0; cols];

    // Step 1: DCT along rows (each row of length cols)
    for r in 0..rows {
        let base = r * cols;
        row_in.copy_from_slice(&input[base..base + cols]);
        dct2_1d(&row_in, &mut row_out, row_table);
        temp[base..base + cols].copy_from_slice(&row_out);
    }

    // Step 2: DCT along columns (each column of length rows)
    let mut col_in = vec![0.0; rows];
    let mut col_out = vec![0.0; rows];

    for c in 0..cols {
        for r in 0..rows {
            col_in[r] = temp[r * cols + c];
        }
        dct2_1d(&col_in, &mut col_out, col_table);
        for r in 0..rows {
            output[r * cols + c] = col_out[r];
        }
    }
}

/// Inverse 2D DCT-III via separable 1D transforms (columns then rows).
pub(crate) fn idct3_2d(
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    cols: usize,
    row_table: &CosineTable,
    col_table: &CosineTable,
) {
    let n = rows * cols;
    debug_assert_eq!(input.len(), n);
    debug_assert_eq!(output.len(), n);

    // Step 1: IDCT along columns
    let mut temp = vec![0.0; n];
    let mut col_in = vec![0.0; rows];
    let mut col_out = vec![0.0; rows];

    for c in 0..cols {
        for r in 0..rows {
            col_in[r] = input[r * cols + c];
        }
        idct3_1d(&col_in, &mut col_out, col_table);
        for r in 0..rows {
            temp[r * cols + c] = col_out[r];
        }
    }

    // Step 2: IDCT along rows
    let mut row_in = vec![0.0; cols];
    let mut row_out = vec![0.0; cols];

    for r in 0..rows {
        let base = r * cols;
        row_in.copy_from_slice(&temp[base..base + cols]);
        idct3_1d(&row_in, &mut row_out, row_table);
        output[base..base + cols].copy_from_slice(&row_out);
    }
}

// ---------------------------------------------------------------------------
// Spectral filter
// ---------------------------------------------------------------------------

/// Apply Gaussian spectral filter: weight(k) = exp(-decay * (k/N)^2).
///
/// Applied separably: row frequency kr and column frequency kc are
/// filtered independently and combined multiplicatively.
fn spectral_filter(coeffs: &mut [f64], rows: usize, cols: usize, decay: f64) {
    // Precompute row weights
    let row_weights: Vec<f64> = (0..cols)
        .map(|k| {
            let frac = k as f64 / cols as f64;
            (-decay * frac * frac).exp()
        })
        .collect();

    // Precompute col weights
    let col_weights: Vec<f64> = (0..rows)
        .map(|k| {
            let frac = k as f64 / rows as f64;
            (-decay * frac * frac).exp()
        })
        .collect();

    for r in 0..rows {
        for c in 0..cols {
            coeffs[r * cols + c] *= col_weights[r] * row_weights[c];
        }
    }
}

// ---------------------------------------------------------------------------
// POCS (Projection Onto Convex Sets)
// ---------------------------------------------------------------------------

/// Restore a 2D grid using DCT-based POCS iteration.
///
/// ## Algorithm
/// 1. Initialize with Level-1 spatial interpolation (or raw data for known cells).
/// 2. Forward 2D DCT-II.
/// 3. Apply spectral filter (suppress high frequencies).
/// 4. Inverse 2D DCT-III.
/// 5. Re-project known values: `restored[i] = mask[i]*original[i] + (1-mask[i])*reconstructed[i]`.
/// 6. Repeat 2-5 until convergence.
pub fn restore_frequency(grid: &Grid2D, config: &FrequencyConfig) -> RestorationResult {
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

    // Initialize: known values from data, missing values with mean of known
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

    let mut coeffs = vec![0.0; n];
    let mut reconstructed = vec![0.0; n];
    let mut iterations: u32 = 0;

    for iter in 0..config.max_pocs_iterations {
        iterations = iter + 1;

        // Forward DCT
        dct2_2d(&current, &mut coeffs, rows, cols, &row_table, &col_table);

        // Spectral filter
        spectral_filter(&mut coeffs, rows, cols, config.spectral_decay);

        // Inverse DCT
        idct3_2d(
            &coeffs,
            &mut reconstructed,
            rows,
            cols,
            &row_table,
            &col_table,
        );

        // Re-project known values and compute max change
        let mut max_change: f64 = 0.0;
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

    // -- DCT round-trip (1D) --------------------------------------------------

    #[test]
    fn test_dct_roundtrip_1d_identity() {
        let n = 8;
        let table = CosineTable::new(n);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut forward = vec![0.0; n];
        let mut inverse = vec![0.0; n];

        dct2_1d(&input, &mut forward, &table);
        idct3_1d(&forward, &mut inverse, &table);

        for i in 0..n {
            assert!(
                (input[i] - inverse[i]).abs() < 1e-10,
                "round-trip failed at {}: {} vs {}",
                i,
                input[i],
                inverse[i]
            );
        }
    }

    #[test]
    fn test_dct_roundtrip_1d_constant() {
        let n = 16;
        let table = CosineTable::new(n);
        let input = vec![42.0; n];
        let mut forward = vec![0.0; n];
        let mut inverse = vec![0.0; n];

        dct2_1d(&input, &mut forward, &table);
        idct3_1d(&forward, &mut inverse, &table);

        for i in 0..n {
            assert!(
                (42.0 - inverse[i]).abs() < 1e-10,
                "constant round-trip failed at {}: {}",
                i,
                inverse[i]
            );
        }
    }

    // -- DCT round-trip (2D) --------------------------------------------------

    #[test]
    fn test_dct_roundtrip_2d() {
        let rows = 4;
        let cols = 4;
        let n = rows * cols;
        let row_table = CosineTable::new(cols);
        let col_table = CosineTable::new(rows);

        let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.7 + 1.0).collect();
        let mut forward = vec![0.0; n];
        let mut inverse = vec![0.0; n];

        dct2_2d(&input, &mut forward, rows, cols, &row_table, &col_table);
        idct3_2d(&forward, &mut inverse, rows, cols, &row_table, &col_table);

        for i in 0..n {
            assert!(
                (input[i] - inverse[i]).abs() < 1e-9,
                "2D round-trip failed at {}: {} vs {}",
                i,
                input[i],
                inverse[i]
            );
        }
    }

    // -- Spectral filter preserves low frequency --------------------------------

    #[test]
    fn test_spectral_filter_preserves_dc() {
        let rows = 4;
        let cols = 4;
        let n = rows * cols;
        let mut coeffs = vec![0.0; n];
        coeffs[0] = 100.0; // DC component

        spectral_filter(&mut coeffs, rows, cols, 2.0);

        // DC (k=0 for both) should be preserved (weight = exp(0) = 1)
        assert!((coeffs[0] - 100.0).abs() < 1e-12, "DC should be preserved");
    }

    #[test]
    fn test_spectral_filter_attenuates_high_frequency() {
        let rows = 8;
        let cols = 8;
        let n = rows * cols;
        let mut coeffs = vec![0.0; n];
        // Set a high-frequency component
        let high_idx = (rows - 1) * cols + (cols - 1);
        coeffs[high_idx] = 100.0;

        spectral_filter(&mut coeffs, rows, cols, 5.0);

        // High-frequency should be significantly attenuated
        assert!(
            coeffs[high_idx].abs() < 10.0,
            "high frequency should be attenuated: {}",
            coeffs[high_idx]
        );
    }

    // -- POCS convergence test -------------------------------------------------

    #[test]
    fn test_pocs_converges() {
        let mut data = vec![10.0; 16];
        let mut mask = vec![1.0; 16];
        data[5] = 0.0;
        mask[5] = 0.0;
        data[6] = 0.0;
        mask[6] = 0.0;

        let grid = Grid2D::new(4, 4, data, mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);

        assert!(
            r.field.iterations <= config.max_pocs_iterations,
            "POCS should converge"
        );
        // Missing values should be close to 10
        assert!(
            (r.field.values[5] - 10.0).abs() < 3.0,
            "value at 5 should be ~10: {}",
            r.field.values[5]
        );
    }

    // -- Level 1 + Level 2 pipeline -------------------------------------------

    #[test]
    fn test_frequency_restores_smooth_field() {
        // 4x4 grid with checkerboard pattern, half missing
        let data = vec![
            10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 10.0,
        ];
        let mask = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let grid = Grid2D::new(4, 4, data, mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);

        // All values should be close to 10 (constant field)
        for (i, &v) in r.field.values.iter().enumerate() {
            assert!((v - 10.0).abs() < 3.0, "index {} should be ~10: {}", i, v);
        }
    }

    // -- Empty grid -----------------------------------------------------------

    #[test]
    fn test_frequency_empty_grid() {
        let grid = Grid2D::new(0, 0, vec![], vec![]);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);
        assert!(r.field.values.is_empty());
        assert_eq!(r.field.iterations, 0);
    }

    // -- Known values preserved -----------------------------------------------

    #[test]
    fn test_frequency_known_values_preserved() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![1.0; 9];
        let grid = Grid2D::new(3, 3, data.clone(), mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);

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

    // -- 1D DCT: constant signal has energy in DC only -----------------------

    #[test]
    fn test_dct_1d_constant_signal_energy_in_dc() {
        let n = 8;
        let table = CosineTable::new(n);
        let c = 7.0;
        let input = vec![c; n];
        let mut forward = vec![0.0; n];
        dct2_1d(&input, &mut forward, &table);
        // DC component should be c*N (sum of all cosines at k=0 = N)
        assert!(
            (forward[0] - c * n as f64).abs() < 1e-8,
            "DC not correct: {}",
            forward[0]
        );
        // All higher-frequency components should be ~0
        for k in 1..n {
            assert!(
                forward[k].abs() < 1e-8,
                "non-DC component at k={} should be 0: {}",
                k,
                forward[k]
            );
        }
    }

    // -- 1D DCT size-1 --------------------------------------------------------

    #[test]
    fn test_dct_1d_size_one() {
        let table = CosineTable::new(1);
        let input = vec![5.0];
        let mut forward = vec![0.0; 1];
        let mut inverse = vec![0.0; 1];
        dct2_1d(&input, &mut forward, &table);
        idct3_1d(&forward, &mut inverse, &table);
        assert!((inverse[0] - 5.0).abs() < 1e-12);
    }

    // -- 2D DCT size 1x1 -------------------------------------------------------

    #[test]
    fn test_dct_2d_1x1() {
        let row_table = CosineTable::new(1);
        let col_table = CosineTable::new(1);
        let input = vec![42.0];
        let mut forward = vec![0.0; 1];
        let mut inverse = vec![0.0; 1];
        dct2_2d(&input, &mut forward, 1, 1, &row_table, &col_table);
        idct3_2d(&forward, &mut inverse, 1, 1, &row_table, &col_table);
        assert!((inverse[0] - 42.0).abs() < 1e-10);
    }

    // -- FrequencyConfig default values ---------------------------------------

    #[test]
    fn test_frequency_config_default() {
        let c = FrequencyConfig::default();
        assert_eq!(c.max_pocs_iterations, 50);
        assert!((c.spectral_decay - 2.0).abs() < 1e-15);
        assert!((c.convergence_threshold - 1e-6).abs() < 1e-15);
        assert!((c.confidence_floor - 0.7).abs() < 1e-15);
    }

    // -- restore_frequency: single-row grid -----------------------------------

    #[test]
    fn test_frequency_single_row_grid() {
        let data = vec![10.0, 0.0, 10.0, 0.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let grid = Grid2D::new(1, 4, data, mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);
        assert_eq!(r.field.values.len(), 4);
        // Known values should be preserved
        assert!((r.field.values[0] - 10.0).abs() < 1e-6);
        assert!((r.field.values[2] - 10.0).abs() < 1e-6);
    }

    // -- restore_frequency: all-missing initializes with mean -----------------

    #[test]
    fn test_frequency_all_missing_does_not_panic() {
        let data = vec![0.0; 9];
        let mask = vec![0.0; 9];
        let grid = Grid2D::new(3, 3, data, mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);
        assert_eq!(r.field.values.len(), 9);
    }

    // -- restore_frequency: content hash is non-zero for non-trivial input ---

    #[test]
    fn test_frequency_content_hash_nonzero() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0; 4];
        let grid = Grid2D::new(2, 2, data, mask);
        let config = FrequencyConfig::default();
        let r = restore_frequency(&grid, &config);
        assert_ne!(r.content_hash, 0);
        assert_ne!(r.field.content_hash, 0);
    }
}
