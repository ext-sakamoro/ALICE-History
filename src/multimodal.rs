// ALICE-History — Level 4: Multi-modal Bayesian fusion
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use crate::core::{compute_confidence_2d, hash_f64_slice, measure_entropy, result_content_hash};
use crate::grid2d::{Grid2D, Grid2DConfig};
use crate::{RestorationField, RestorationResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of modality contributing to the observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityKind {
    Image,
    Text,
    Spatial,
    Spectral,
    Temporal,
}

/// A single modality's observation at one position.
#[derive(Debug, Clone)]
pub struct ModalObservation {
    pub mean: f64,
    pub variance: f64,
    pub modality: ModalityKind,
}

/// Configuration for Bayesian fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub confidence_floor: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            prior_mean: 0.0,
            prior_variance: 1e6, // very weak prior
            max_iterations: 100,
            convergence_threshold: 1e-8,
            confidence_floor: 0.7,
        }
    }
}

/// Result of fusing multiple modalities at a single position.
#[derive(Debug, Clone)]
pub struct FusedEstimate {
    pub mean: f64,
    pub variance: f64,
    pub precision: f64,
    pub modality_weights: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Bayesian precision-weighted fusion
// ---------------------------------------------------------------------------

/// Fuse multiple observations at a single position using Bayesian precision-weighted average.
///
/// ## Algorithm
/// - Each modality i provides (mu_i, sigma_i^2).
/// - Precision_i = 1 / sigma_i^2.
/// - Fused precision = prior_precision + sum(precision_i).
/// - Fused mean = fused_variance * (prior_precision * prior_mean + sum(precision_i * mu_i)).
pub fn bayesian_fuse(observations: &[ModalObservation], config: &FusionConfig) -> FusedEstimate {
    if observations.is_empty() {
        return FusedEstimate {
            mean: config.prior_mean,
            variance: config.prior_variance,
            precision: 1.0 / config.prior_variance,
            modality_weights: Vec::new(),
        };
    }

    let prior_precision = 1.0 / config.prior_variance;
    let mut total_precision = prior_precision;
    let mut weighted_sum = prior_precision * config.prior_mean;

    let mut precisions = Vec::with_capacity(observations.len());

    for obs in observations {
        let p = if obs.variance > 0.0 {
            1.0 / obs.variance
        } else {
            1e12 // near-infinite precision for zero variance
        };
        precisions.push(p);
        total_precision += p;
        weighted_sum += p * obs.mean;
    }

    let fused_variance = 1.0 / total_precision;
    let fused_mean = fused_variance * weighted_sum;

    // Compute modality weights (fraction of total precision)
    let modality_weights: Vec<f64> = precisions.iter().map(|&p| p / total_precision).collect();

    FusedEstimate {
        mean: fused_mean,
        variance: fused_variance,
        precision: total_precision,
        modality_weights,
    }
}

// ---------------------------------------------------------------------------
// Grid-level fusion
// ---------------------------------------------------------------------------

/// Fuse multiple modality observations over a 2D grid, then fill remaining
/// gaps using the Level 1 (Gauss-Seidel) solver.
///
/// `modal_obs[i]` contains the observations for grid position i.
/// Positions with no observations remain as gaps to be filled by spatial interpolation.
pub fn fuse_grid(
    grid: &Grid2D,
    modal_obs: &[Vec<ModalObservation>],
    config: &FusionConfig,
) -> RestorationResult {
    let start = std::time::Instant::now();
    let rows = grid.rows;
    let cols = grid.cols;
    let n = rows * cols;

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

    assert_eq!(
        modal_obs.len(),
        n,
        "modal_obs length must match grid size"
    );

    let entropy_before = measure_entropy(&grid.data, 64).shannon_entropy;

    // Phase 1: Bayesian fusion at each position
    let mut fused_data = vec![0.0; n];
    let mut fused_mask = vec![0.0; n];

    for i in 0..n {
        if grid.mask[i] == 1.0 {
            // Known from original data
            fused_data[i] = grid.data[i];
            fused_mask[i] = 1.0;
        } else if !modal_obs[i].is_empty() {
            // Fuse available modalities
            let estimate = bayesian_fuse(&modal_obs[i], config);
            fused_data[i] = estimate.mean;
            fused_mask[i] = 1.0; // now "known" via fusion
        }
        // else: remains 0.0 / 0.0 (gap)
    }

    // Phase 2: Fill remaining gaps with Gauss-Seidel
    let fused_grid = Grid2D::new(rows, cols, fused_data.clone(), fused_mask.clone());
    let grid_config = Grid2DConfig {
        max_iterations: config.max_iterations,
        convergence_threshold: config.convergence_threshold,
        regularization: 0.01,
        confidence_floor: config.confidence_floor,
        ..Grid2DConfig::default()
    };

    let spatial_result = crate::grid2d::restore_2d(&fused_grid, &grid_config);
    let mut final_values = spatial_result.field.values;

    // Re-project original known values
    for i in 0..n {
        if grid.mask[i] == 1.0 {
            final_values[i] = grid.data[i];
        }
    }

    let confidence = compute_confidence_2d(&grid.mask, rows, cols, config.confidence_floor);
    let entropy_after = measure_entropy(&final_values, 64).shannon_entropy;
    let content_hash = hash_f64_slice(&final_values);

    let field = RestorationField {
        values: final_values,
        confidence,
        entropy_before,
        entropy_after,
        iterations: spatial_result.field.iterations,
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

    // -- bayesian_fuse: single modality passthrough ----------------------------

    #[test]
    fn test_fuse_single_modality_passthrough() {
        let obs = vec![ModalObservation {
            mean: 42.0,
            variance: 1.0,
            modality: ModalityKind::Image,
        }];
        let config = FusionConfig {
            prior_variance: 1e12, // effectively no prior
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        assert!(
            (result.mean - 42.0).abs() < 0.01,
            "single modality should pass through: {}",
            result.mean
        );
    }

    // -- bayesian_fuse: high precision dominates ------------------------------

    #[test]
    fn test_fuse_high_precision_dominates() {
        let obs = vec![
            ModalObservation {
                mean: 100.0,
                variance: 0.01, // high precision
                modality: ModalityKind::Image,
            },
            ModalObservation {
                mean: 0.0,
                variance: 100.0, // low precision
                modality: ModalityKind::Text,
            },
        ];
        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        // Should be much closer to 100 than 0
        assert!(
            result.mean > 90.0,
            "high precision should dominate: {}",
            result.mean
        );
        assert!(result.modality_weights[0] > result.modality_weights[1]);
    }

    // -- bayesian_fuse: equal precision -> average ----------------------------

    #[test]
    fn test_fuse_equal_precision_averages() {
        let obs = vec![
            ModalObservation {
                mean: 10.0,
                variance: 1.0,
                modality: ModalityKind::Spatial,
            },
            ModalObservation {
                mean: 20.0,
                variance: 1.0,
                modality: ModalityKind::Temporal,
            },
        ];
        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        assert!(
            (result.mean - 15.0).abs() < 0.1,
            "equal precision should average: {}",
            result.mean
        );
    }

    // -- bayesian_fuse: three modalities --------------------------------------

    #[test]
    fn test_fuse_three_modalities() {
        let obs = vec![
            ModalObservation {
                mean: 10.0,
                variance: 1.0,
                modality: ModalityKind::Image,
            },
            ModalObservation {
                mean: 20.0,
                variance: 1.0,
                modality: ModalityKind::Text,
            },
            ModalObservation {
                mean: 30.0,
                variance: 1.0,
                modality: ModalityKind::Spectral,
            },
        ];
        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        assert!(
            (result.mean - 20.0).abs() < 0.5,
            "three equal modalities should average: {}",
            result.mean
        );
        assert_eq!(result.modality_weights.len(), 3);
    }

    // -- bayesian_fuse: prior influence ----------------------------------------

    #[test]
    fn test_fuse_prior_influence() {
        let obs = vec![ModalObservation {
            mean: 100.0,
            variance: 10.0,
            modality: ModalityKind::Image,
        }];
        let config = FusionConfig {
            prior_mean: 0.0,
            prior_variance: 10.0, // strong prior
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        // With equal variance, fused should be midpoint
        assert!(
            (result.mean - 50.0).abs() < 1.0,
            "prior should pull toward 0: {}",
            result.mean
        );
    }

    // -- bayesian_fuse: empty observations -> prior ---------------------------

    #[test]
    fn test_fuse_empty_returns_prior() {
        let config = FusionConfig {
            prior_mean: 42.0,
            prior_variance: 1.0,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&[], &config);
        assert!((result.mean - 42.0).abs() < 1e-12);
        assert!(result.modality_weights.is_empty());
    }

    // -- fuse_grid: basic grid fusion -----------------------------------------

    #[test]
    fn test_fuse_grid_basic() {
        let data = vec![0.0; 9];
        let mask = vec![0.0; 9];
        let grid = Grid2D::new(3, 3, data, mask);

        // All positions get one observation: 50.0
        let obs: Vec<Vec<ModalObservation>> = (0..9)
            .map(|_| {
                vec![ModalObservation {
                    mean: 50.0,
                    variance: 1.0,
                    modality: ModalityKind::Image,
                }]
            })
            .collect();

        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let r = fuse_grid(&grid, &obs, &config);

        for (i, &v) in r.field.values.iter().enumerate() {
            assert!(
                (v - 50.0).abs() < 2.0,
                "index {} should be ~50: {}",
                i,
                v
            );
        }
    }

    // -- fuse_grid: with gaps filled by spatial interpolation -----------------

    #[test]
    fn test_fuse_grid_with_gaps() {
        let mut data = vec![0.0; 9];
        let mut mask = vec![0.0; 9];
        data[0] = 10.0;
        mask[0] = 1.0;
        data[8] = 10.0;
        mask[8] = 1.0;

        let grid = Grid2D::new(3, 3, data, mask);

        // Only corners have observations, center positions have none
        let mut obs: Vec<Vec<ModalObservation>> = vec![Vec::new(); 9];
        obs[4] = vec![ModalObservation {
            mean: 10.0,
            variance: 1.0,
            modality: ModalityKind::Spatial,
        }];

        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let r = fuse_grid(&grid, &obs, &config);

        // Known values should be preserved
        assert!((r.field.values[0] - 10.0).abs() < 1e-6);
        assert!((r.field.values[8] - 10.0).abs() < 1e-6);
    }

    // -- fuse_grid: empty grid ------------------------------------------------

    #[test]
    fn test_fuse_grid_empty() {
        let grid = Grid2D::new(0, 0, vec![], vec![]);
        let config = FusionConfig::default();
        let r = fuse_grid(&grid, &[], &config);
        assert!(r.field.values.is_empty());
    }

    // -- FusedEstimate precision consistency -----------------------------------

    #[test]
    fn test_fused_precision_consistency() {
        let obs = vec![
            ModalObservation {
                mean: 10.0,
                variance: 2.0,
                modality: ModalityKind::Image,
            },
            ModalObservation {
                mean: 20.0,
                variance: 3.0,
                modality: ModalityKind::Text,
            },
        ];
        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);

        // precision = 1/variance
        assert!(
            (result.precision - 1.0 / result.variance).abs() < 1e-6,
            "precision should equal 1/variance"
        );

        // weights should sum to ~1 (minus tiny prior weight)
        let weight_sum: f64 = result.modality_weights.iter().sum();
        assert!(weight_sum > 0.99 && weight_sum <= 1.0 + 1e-6);
    }

    // -- Zero variance observation -------------------------------------------

    #[test]
    fn test_fuse_zero_variance_dominates() {
        let obs = vec![
            ModalObservation {
                mean: 42.0,
                variance: 0.0, // "perfect" observation
                modality: ModalityKind::Image,
            },
            ModalObservation {
                mean: 0.0,
                variance: 1.0,
                modality: ModalityKind::Text,
            },
        ];
        let config = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };
        let result = bayesian_fuse(&obs, &config);
        assert!(
            (result.mean - 42.0).abs() < 0.01,
            "zero variance should dominate: {}",
            result.mean
        );
    }
}
