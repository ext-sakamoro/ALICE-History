// ALICE-History — Level 1: 2D grid solver (Gauss-Seidel)
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use rayon::prelude::*;

use crate::core::{compute_confidence_2d, hash_f64_slice, measure_entropy, result_content_hash};
use crate::{RestorationField, RestorationResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A 2D grid of historical data with missing values.
#[derive(Debug, Clone)]
pub struct Grid2D {
    pub rows: usize,
    pub cols: usize,
    /// Row-major data: `data[r * cols + c]`.
    pub data: Vec<f64>,
    /// Mask: 1.0 = known, 0.0 = missing.
    pub mask: Vec<f64>,
}

impl Grid2D {
    /// Create a new Grid2D.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols` or `mask.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>, mask: Vec<f64>) -> Self {
        let n = rows * cols;
        assert_eq!(data.len(), n, "data length must equal rows * cols");
        assert_eq!(mask.len(), n, "mask length must equal rows * cols");
        Self {
            rows,
            cols,
            data,
            mask,
        }
    }

    /// Fraction of cells that are known.
    pub fn known_fraction(&self) -> f64 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f64 = self.mask.iter().sum();
        sum / n as f64
    }
}

/// Neighbor mode for 2D Gauss-Seidel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborMode {
    /// 4-connected: N, S, E, W.
    Four,
    /// 8-connected: N, S, E, W, NE, NW, SE, SW.
    Eight,
}

/// Configuration for the 2D grid solver.
#[derive(Debug, Clone)]
pub struct Grid2DConfig {
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub regularization: f64,
    pub confidence_floor: f64,
    pub neighbor_mode: NeighborMode,
}

impl Default for Grid2DConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            convergence_threshold: 1e-8,
            regularization: 0.01,
            confidence_floor: 0.7,
            neighbor_mode: NeighborMode::Four,
        }
    }
}

// ---------------------------------------------------------------------------
// 2D Gauss-Seidel solver
// ---------------------------------------------------------------------------

/// Restore a 2D grid using iterative Gauss-Seidel with Tikhonov regularization.
///
/// ## Algorithm
/// 1. Initialize missing cells with the mean of known cells.
/// 2. For each iteration, update each missing cell to the weighted average of
///    its neighbors, with Tikhonov regularization pulling toward the global mean.
/// 3. Check convergence (max_change < threshold).
pub fn restore_2d(grid: &Grid2D, config: &Grid2DConfig) -> RestorationResult {
    let start = std::time::Instant::now();
    let n = grid.rows * grid.cols;

    if n == 0 {
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

    // Compute mean of known values
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

    // Initialize: copy known, fill missing with mean
    let mut restored: Vec<f64> = grid
        .data
        .iter()
        .zip(grid.mask.iter())
        .map(|(&d, &m)| if m == 1.0 { d } else { mean_known })
        .collect();

    let rows = grid.rows;
    let cols = grid.cols;
    let mut iterations: u32 = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let mut max_change: f64 = 0.0;

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if grid.mask[idx] == 1.0 {
                    continue;
                }

                // Gather neighbors
                let (sum, count) = match config.neighbor_mode {
                    NeighborMode::Four => gather_4(r, c, rows, cols, &restored),
                    NeighborMode::Eight => gather_8(r, c, rows, cols, &restored),
                };

                if count == 0 {
                    continue;
                }

                // precomputed reciprocal — replaces division per cell per iteration
                let neighbor_avg = sum * (count as f64).recip();
                let smooth_grad = restored[idx] - neighbor_avg;
                let reg_grad = config.regularization * (restored[idx] - mean_known);
                let grad = smooth_grad + reg_grad;
                let new_val = restored[idx] - 0.5 * grad;
                let change = (new_val - restored[idx]).abs();
                if change > max_change {
                    max_change = change;
                }
                restored[idx] = new_val;
            }
        }

        if max_change < config.convergence_threshold {
            break;
        }
    }

    let confidence = compute_confidence_2d(&grid.mask, rows, cols, config.confidence_floor);
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

    let rh = result_content_hash(0, field.content_hash);

    RestorationResult {
        fragment_id: 0,
        field,
        elapsed_ns: start.elapsed().as_nanos() as u64,
        content_hash: rh,
    }
}

/// Batch restore multiple 2D grids in parallel (Rayon).
pub fn restore_2d_batch(grids: &[Grid2D], config: &Grid2DConfig) -> Vec<RestorationResult> {
    grids.par_iter().map(|g| restore_2d(g, config)).collect()
}

// ---------------------------------------------------------------------------
// Neighbor gathering
// ---------------------------------------------------------------------------

#[inline]
fn gather_4(r: usize, c: usize, rows: usize, cols: usize, data: &[f64]) -> (f64, u32) {
    let mut sum = 0.0;
    let mut count = 0u32;
    if r > 0 {
        sum += data[(r - 1) * cols + c];
        count += 1;
    }
    if r + 1 < rows {
        sum += data[(r + 1) * cols + c];
        count += 1;
    }
    if c > 0 {
        sum += data[r * cols + (c - 1)];
        count += 1;
    }
    if c + 1 < cols {
        sum += data[r * cols + (c + 1)];
        count += 1;
    }
    (sum, count)
}

#[inline]
fn gather_8(r: usize, c: usize, rows: usize, cols: usize, data: &[f64]) -> (f64, u32) {
    let mut sum = 0.0;
    let mut count = 0u32;

    let r_lo = r > 0;
    let r_hi = r + 1 < rows;
    let c_lo = c > 0;
    let c_hi = c + 1 < cols;

    // Cardinal
    if r_lo {
        sum += data[(r - 1) * cols + c];
        count += 1;
    }
    if r_hi {
        sum += data[(r + 1) * cols + c];
        count += 1;
    }
    if c_lo {
        sum += data[r * cols + (c - 1)];
        count += 1;
    }
    if c_hi {
        sum += data[r * cols + (c + 1)];
        count += 1;
    }

    // Diagonal
    if r_lo && c_lo {
        sum += data[(r - 1) * cols + (c - 1)];
        count += 1;
    }
    if r_lo && c_hi {
        sum += data[(r - 1) * cols + (c + 1)];
        count += 1;
    }
    if r_hi && c_lo {
        sum += data[(r + 1) * cols + (c - 1)];
        count += 1;
    }
    if r_hi && c_hi {
        sum += data[(r + 1) * cols + (c + 1)];
        count += 1;
    }

    (sum, count)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(rows: usize, cols: usize, data: Vec<f64>, mask: Vec<f64>) -> Grid2D {
        Grid2D::new(rows, cols, data, mask)
    }

    // -- All known: values unchanged ------------------------------------------

    #[test]
    fn test_2d_all_known_unchanged() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![1.0; 9];
        let grid = make_grid(3, 3, data.clone(), mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        for (i, (&orig, &rest)) in data.iter().zip(r.field.values.iter()).enumerate() {
            assert!(
                (orig - rest).abs() < 1e-12,
                "index {} changed: {} -> {}",
                i,
                orig,
                rest
            );
        }
    }

    // -- Single pixel missing in center ---------------------------------------

    #[test]
    fn test_2d_single_center_missing() {
        // 3x3 grid, center missing, surrounded by 10.0
        let mut data = vec![10.0; 9];
        let mut mask = vec![1.0; 9];
        data[4] = 0.0;
        mask[4] = 0.0;
        let grid = make_grid(3, 3, data, mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        assert!(
            (r.field.values[4] - 10.0).abs() < 0.5,
            "center should be ~10, got {}",
            r.field.values[4]
        );
    }

    // -- Row missing ----------------------------------------------------------

    #[test]
    fn test_2d_row_missing() {
        // 3x3 grid, middle row missing
        let data = vec![10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0];
        let mask = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let grid = make_grid(3, 3, data, mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        // Middle row should interpolate between top and bottom
        // Regularization pulls toward the global mean (20.0), so col-0 value
        // (between 10 and 10) will be pulled slightly above 10.
        assert!(
            (r.field.values[3] - 10.0).abs() < 5.0,
            "row interpolation failed at col 0: {}",
            r.field.values[3]
        );
        assert!(
            (r.field.values[4] - 20.0).abs() < 5.0,
            "row interpolation failed at col 1: {}",
            r.field.values[4]
        );
    }

    // -- Corner missing -------------------------------------------------------

    #[test]
    fn test_2d_corner_missing() {
        let mut data = vec![5.0; 9];
        let mut mask = vec![1.0; 9];
        // Remove top-left corner
        data[0] = 0.0;
        mask[0] = 0.0;
        let grid = make_grid(3, 3, data, mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        assert!(
            (r.field.values[0] - 5.0).abs() < 1.0,
            "corner should be ~5, got {}",
            r.field.values[0]
        );
    }

    // -- 8-neighbor vs 4-neighbor ---------------------------------------------

    #[test]
    fn test_2d_eight_neighbor_converges() {
        let mut data = vec![10.0; 9];
        let mut mask = vec![1.0; 9];
        data[4] = 0.0;
        mask[4] = 0.0;
        let grid = make_grid(3, 3, data, mask);
        let mut config = Grid2DConfig::default();
        config.neighbor_mode = NeighborMode::Eight;
        let r = restore_2d(&grid, &config);
        assert!(
            (r.field.values[4] - 10.0).abs() < 0.5,
            "8-neighbor center should be ~10, got {}",
            r.field.values[4]
        );
    }

    #[test]
    fn test_2d_four_vs_eight_both_converge() {
        let mut data = vec![0.0; 25];
        let mut mask = vec![0.0; 25];
        // Set corners known
        data[0] = 0.0;
        mask[0] = 1.0;
        data[4] = 100.0;
        mask[4] = 1.0;
        data[20] = 100.0;
        mask[20] = 1.0;
        data[24] = 0.0;
        mask[24] = 1.0;
        let grid = make_grid(5, 5, data, mask);

        let mut config4 = Grid2DConfig::default();
        config4.neighbor_mode = NeighborMode::Four;
        let r4 = restore_2d(&grid, &config4);

        let mut config8 = Grid2DConfig::default();
        config8.neighbor_mode = NeighborMode::Eight;
        let r8 = restore_2d(&grid, &config8);

        // Both should converge (iterations < max)
        assert!(r4.field.iterations < config4.max_iterations);
        assert!(r8.field.iterations < config8.max_iterations);
    }

    // -- Convergence ----------------------------------------------------------

    #[test]
    fn test_2d_convergence_iterations() {
        let mut data = vec![50.0; 16];
        let mut mask = vec![1.0; 16];
        data[5] = 0.0;
        mask[5] = 0.0;
        data[6] = 0.0;
        mask[6] = 0.0;
        let grid = make_grid(4, 4, data, mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        // Should converge well before max_iterations
        assert!(
            r.field.iterations < config.max_iterations,
            "did not converge: {} iterations",
            r.field.iterations
        );
    }

    // -- Confidence: known = 1.0 -----------------------------------------------

    #[test]
    fn test_2d_confidence_known_is_one() {
        let data = vec![1.0, 0.0, 1.0, 0.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let grid = make_grid(2, 2, data, mask);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        assert!((r.field.confidence.scores[0] - 1.0).abs() < 1e-12);
        assert!((r.field.confidence.scores[2] - 1.0).abs() < 1e-12);
    }

    // -- Confidence: distance decay --------------------------------------------

    #[test]
    fn test_2d_confidence_distance_decay() {
        // 5x5 grid, only center known
        let mut data = vec![0.0; 25];
        let mut mask = vec![0.0; 25];
        data[12] = 100.0; // center
        mask[12] = 1.0;
        let grid = make_grid(5, 5, data, mask);
        let mut config = Grid2DConfig::default();
        config.confidence_floor = 0.05;
        let r = restore_2d(&grid, &config);
        // Center = 1.0
        assert!((r.field.confidence.scores[12] - 1.0).abs() < 1e-12);
        // Adjacent (dist=1) = 0.5
        assert!(r.field.confidence.scores[7] > r.field.confidence.scores[0]);
    }

    // -- Confidence: floor respected -------------------------------------------

    #[test]
    fn test_2d_confidence_floor_respected() {
        let mut data = vec![0.0; 25];
        let mut mask = vec![0.0; 25];
        data[0] = 1.0;
        mask[0] = 1.0;
        let grid = make_grid(5, 5, data, mask);
        let mut config = Grid2DConfig::default();
        config.confidence_floor = 0.3;
        let r = restore_2d(&grid, &config);
        for &s in &r.field.confidence.scores {
            assert!(s >= 0.3 - 1e-12, "confidence below floor: {}", s);
        }
    }

    // -- Empty grid ------------------------------------------------------------

    #[test]
    fn test_2d_empty_grid() {
        let grid = make_grid(0, 0, vec![], vec![]);
        let config = Grid2DConfig::default();
        let r = restore_2d(&grid, &config);
        assert!(r.field.values.is_empty());
        assert_eq!(r.field.iterations, 0);
    }

    // -- Batch -----------------------------------------------------------------

    #[test]
    fn test_2d_batch() {
        let g1 = make_grid(2, 2, vec![1.0; 4], vec![1.0; 4]);
        let g2 = make_grid(2, 2, vec![2.0; 4], vec![1.0; 4]);
        let config = Grid2DConfig::default();
        let results = restore_2d_batch(&[g1, g2], &config);
        assert_eq!(results.len(), 2);
    }

    // -- Known fraction -------------------------------------------------------

    #[test]
    fn test_2d_known_fraction() {
        let grid = make_grid(2, 2, vec![0.0; 4], vec![1.0, 0.0, 1.0, 0.0]);
        assert!((grid.known_fraction() - 0.5).abs() < 1e-12);
    }
}
