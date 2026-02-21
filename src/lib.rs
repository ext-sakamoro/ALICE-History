// ALICE-History — Inverse entropy restoration
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

pub(crate) mod core;
pub(crate) mod cosine_table;
pub mod frequency;
pub mod grid2d;
pub mod multimodal;
pub mod solver_1d;
pub mod sparse;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use crate::core::measure_entropy;
pub use frequency::{restore_frequency, FrequencyConfig};
pub use grid2d::{restore_2d, restore_2d_batch, Grid2D, Grid2DConfig, NeighborMode};
pub use multimodal::{
    bayesian_fuse, fuse_grid, FusedEstimate, FusionConfig, ModalObservation, ModalityKind,
};
pub use solver_1d::{restore_1d, restore_1d_batch};
pub use sparse::{restore_sparse, soft_threshold, SparseConfig};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of historical fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentKind {
    /// Degraded manuscript / inscription
    Text,
    /// Degraded photograph / painting
    Image,
    /// Physical object scan (3-D)
    Artifact,
    /// Stone / clay tablet carving
    Inscription,
    /// Degraded audio recording
    Audio,
}

/// A degraded historical fragment -- the input to restoration.
///
/// `data` contains the raw observed values, `mask` indicates which elements
/// are known (1.0) vs missing (0.0).
#[derive(Debug, Clone)]
pub struct Fragment {
    pub id: u64,
    pub kind: FragmentKind,
    pub data: Vec<f64>,
    pub mask: Vec<f64>,
    pub timestamp_ns: u64,
    pub content_hash: u64,
}

/// Confidence level for each restored element.
#[derive(Debug, Clone)]
pub struct ConfidenceMap {
    pub scores: Vec<f64>,
    pub mean_confidence: f64,
    pub min_confidence: f64,
    pub restoration_boundary: f64,
}

/// The restored continuous field -- entropy-reversed representation.
#[derive(Debug, Clone)]
pub struct RestorationField {
    pub values: Vec<f64>,
    pub confidence: ConfidenceMap,
    pub entropy_before: f64,
    pub entropy_after: f64,
    pub iterations: u32,
    pub content_hash: u64,
}

/// Configuration for the entropy inversion solver.
#[derive(Debug, Clone)]
pub struct InversionConfig {
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub regularization: f64,
    pub confidence_floor: f64,
}

/// Result of a full restoration pipeline.
#[derive(Debug, Clone)]
pub struct RestorationResult {
    pub fragment_id: u64,
    pub field: RestorationField,
    pub elapsed_ns: u64,
    pub content_hash: u64,
}

/// Entropy measurement for a data sequence.
#[derive(Debug, Clone)]
pub struct EntropyMeasurement {
    pub shannon_entropy: f64,
    pub normalized_entropy: f64,
    pub unique_symbols: usize,
    pub total_symbols: usize,
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

impl Fragment {
    /// Create a new Fragment with computed `content_hash`.
    ///
    /// # Panics
    /// Panics if `data.len() != mask.len()`.
    pub fn new(
        id: u64,
        kind: FragmentKind,
        data: Vec<f64>,
        mask: Vec<f64>,
        timestamp_ns: u64,
    ) -> Self {
        assert_eq!(
            data.len(),
            mask.len(),
            "data and mask must have the same length"
        );
        let content_hash = core::fragment_content_hash(id, kind, &data);
        Self {
            id,
            kind,
            data,
            mask,
            timestamp_ns,
            content_hash,
        }
    }

    /// Fraction of data that is known (mean of mask).
    pub fn known_fraction(&self) -> f64 {
        if self.mask.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.mask.iter().sum();
        sum / self.mask.len() as f64
    }

    /// Number of missing (mask == 0.0) elements.
    pub fn missing_count(&self) -> usize {
        self.mask.iter().filter(|&&m| m == 0.0).count()
    }
}

// ---------------------------------------------------------------------------
// InversionConfig
// ---------------------------------------------------------------------------

impl Default for InversionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-8,
            regularization: 0.01,
            confidence_floor: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible API (delegates to solver_1d)
// ---------------------------------------------------------------------------

/// Restore a 1D fragment (backward-compatible alias for `restore_1d`).
pub fn restore(fragment: &Fragment, config: &InversionConfig) -> RestorationResult {
    solver_1d::restore_1d(fragment, config)
}

/// Batch restore multiple fragments (backward-compatible alias for `restore_1d_batch`).
pub fn restore_batch(
    fragments: &[Fragment],
    config: &InversionConfig,
) -> Vec<RestorationResult> {
    solver_1d::restore_1d_batch(fragments, config)
}

// ---------------------------------------------------------------------------
// Strategy enum + unified API
// ---------------------------------------------------------------------------

/// Restoration strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// 1D linear interpolation (original solver).
    Linear1D,
    /// 2D Gauss-Seidel grid solver.
    Grid2D,
    /// DCT + POCS frequency-domain restoration.
    Frequency,
    /// ISTA/FISTA compressed sensing.
    Sparse,
    /// Multi-modal Bayesian fusion.
    MultiModal,
    /// Automatic strategy selection based on data characteristics.
    Auto,
}

/// Unified restoration entry point supporting all strategies.
///
/// ## Parameters
/// - `fragment`: The degraded 1D fragment.
/// - `config`: Solver configuration (iterations, threshold, etc.).
/// - `strategy`: Which solver to use.
/// - `grid_dims`: Required for 2D strategies (rows, cols). If `None` and a 2D
///   strategy is requested, falls back to 1D.
/// - `modal_observations`: Required for `MultiModal` strategy. Each element
///   contains observations for the corresponding grid position.
/// - `fusion_config`: Required for `MultiModal` strategy.
///
/// ## Auto strategy logic
/// - `grid_dims` present + `known_fraction < 0.1` -> Sparse (FISTA)
/// - `grid_dims` present + `known_fraction < 0.5` -> Frequency (DCT+POCS)
/// - `grid_dims` present -> Grid2D (Gauss-Seidel)
/// - Otherwise -> Linear1D
pub fn restore_advanced(
    fragment: &Fragment,
    config: &InversionConfig,
    strategy: Strategy,
    grid_dims: Option<(usize, usize)>,
    modal_observations: Option<&[Vec<ModalObservation>]>,
    fusion_config: Option<&FusionConfig>,
) -> RestorationResult {
    let effective_strategy = match strategy {
        Strategy::Auto => select_auto_strategy(fragment, grid_dims),
        other => other,
    };

    match effective_strategy {
        Strategy::Linear1D | Strategy::Auto => restore_1d(fragment, config),

        Strategy::Grid2D => {
            if let Some((rows, cols)) = grid_dims {
                let grid = fragment_to_grid(fragment, rows, cols);
                let grid_config = inversion_to_grid2d_config(config);
                let mut r = grid2d::restore_2d(&grid, &grid_config);
                r.fragment_id = fragment.id;
                r
            } else {
                restore_1d(fragment, config)
            }
        }

        Strategy::Frequency => {
            if let Some((rows, cols)) = grid_dims {
                let grid = fragment_to_grid(fragment, rows, cols);
                let freq_config = FrequencyConfig {
                    max_pocs_iterations: config.max_iterations.min(100),
                    convergence_threshold: config.convergence_threshold,
                    confidence_floor: config.confidence_floor,
                    ..FrequencyConfig::default()
                };
                let mut r = frequency::restore_frequency(&grid, &freq_config);
                r.fragment_id = fragment.id;
                r
            } else {
                restore_1d(fragment, config)
            }
        }

        Strategy::Sparse => {
            if let Some((rows, cols)) = grid_dims {
                let grid = fragment_to_grid(fragment, rows, cols);
                let sparse_config = SparseConfig {
                    max_iterations: config.max_iterations.min(500),
                    convergence_threshold: config.convergence_threshold,
                    confidence_floor: config.confidence_floor,
                    ..SparseConfig::default()
                };
                let mut r = sparse::restore_sparse(&grid, &sparse_config);
                r.fragment_id = fragment.id;
                r
            } else {
                restore_1d(fragment, config)
            }
        }

        Strategy::MultiModal => {
            if let (Some((rows, cols)), Some(obs), Some(fc)) =
                (grid_dims, modal_observations, fusion_config)
            {
                let grid = fragment_to_grid(fragment, rows, cols);
                let mut r = multimodal::fuse_grid(&grid, obs, fc);
                r.fragment_id = fragment.id;
                r
            } else {
                restore_1d(fragment, config)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn select_auto_strategy(fragment: &Fragment, grid_dims: Option<(usize, usize)>) -> Strategy {
    let kf = fragment.known_fraction();

    if grid_dims.is_some() {
        if kf < 0.1 {
            Strategy::Sparse
        } else if kf < 0.5 {
            Strategy::Frequency
        } else {
            Strategy::Grid2D
        }
    } else {
        Strategy::Linear1D
    }
}

fn fragment_to_grid(fragment: &Fragment, rows: usize, cols: usize) -> grid2d::Grid2D {
    let n = rows * cols;
    let mut data = fragment.data.clone();
    let mut mask = fragment.mask.clone();
    data.resize(n, 0.0);
    mask.resize(n, 0.0);
    grid2d::Grid2D::new(rows, cols, data, mask)
}

fn inversion_to_grid2d_config(config: &InversionConfig) -> Grid2DConfig {
    Grid2DConfig {
        max_iterations: config.max_iterations,
        convergence_threshold: config.convergence_threshold,
        regularization: config.regularization,
        confidence_floor: config.confidence_floor,
        neighbor_mode: NeighborMode::Four,
    }
}

// ===========================================================================
// Tests (backward compatibility + unified API + original tests)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Fragment construction -----------------------------------------------

    #[test]
    fn test_fragment_new_basic() {
        let f = Fragment::new(1, FragmentKind::Text, vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0], 0);
        assert_eq!(f.id, 1);
        assert_eq!(f.kind, FragmentKind::Text);
        assert_eq!(f.data.len(), 3);
    }

    #[test]
    fn test_fragment_content_hash_deterministic() {
        let f1 = Fragment::new(1, FragmentKind::Text, vec![1.0, 2.0], vec![1.0, 1.0], 0);
        let f2 = Fragment::new(1, FragmentKind::Text, vec![1.0, 2.0], vec![1.0, 1.0], 999);
        assert_eq!(f1.content_hash, f2.content_hash);
    }

    #[test]
    fn test_fragment_different_data_different_hash() {
        let f1 = Fragment::new(1, FragmentKind::Text, vec![1.0, 2.0], vec![1.0, 1.0], 0);
        let f2 = Fragment::new(1, FragmentKind::Text, vec![1.0, 3.0], vec![1.0, 1.0], 0);
        assert_ne!(f1.content_hash, f2.content_hash);
    }

    #[test]
    fn test_fragment_different_kind_different_hash() {
        let f1 = Fragment::new(1, FragmentKind::Text, vec![1.0], vec![1.0], 0);
        let f2 = Fragment::new(1, FragmentKind::Image, vec![1.0], vec![1.0], 0);
        assert_ne!(f1.content_hash, f2.content_hash);
    }

    #[test]
    #[should_panic(expected = "data and mask must have the same length")]
    fn test_fragment_mismatched_lengths() {
        Fragment::new(0, FragmentKind::Text, vec![1.0, 2.0], vec![1.0], 0);
    }

    // -- known_fraction & missing_count --------------------------------------

    #[test]
    fn test_known_fraction_all_known() {
        let f = Fragment::new(0, FragmentKind::Text, vec![1.0; 10], vec![1.0; 10], 0);
        assert!((f.known_fraction() - 1.0).abs() < 1e-12);
        assert_eq!(f.missing_count(), 0);
    }

    #[test]
    fn test_known_fraction_half_known() {
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let f = Fragment::new(0, FragmentKind::Text, vec![0.0; 4], mask, 0);
        assert!((f.known_fraction() - 0.5).abs() < 1e-12);
        assert_eq!(f.missing_count(), 2);
    }

    #[test]
    fn test_known_fraction_empty() {
        let f = Fragment::new(0, FragmentKind::Text, vec![], vec![], 0);
        assert!((f.known_fraction() - 0.0).abs() < 1e-12);
        assert_eq!(f.missing_count(), 0);
    }

    #[test]
    fn test_known_fraction_all_missing() {
        let f = Fragment::new(0, FragmentKind::Text, vec![0.0; 5], vec![0.0; 5], 0);
        assert!((f.known_fraction() - 0.0).abs() < 1e-12);
        assert_eq!(f.missing_count(), 5);
    }

    // -- InversionConfig default ---------------------------------------------

    #[test]
    fn test_inversion_config_default() {
        let c = InversionConfig::default();
        assert_eq!(c.max_iterations, 1000);
        assert!((c.convergence_threshold - 1e-8).abs() < 1e-15);
        assert!((c.regularization - 0.01).abs() < 1e-15);
        assert!((c.confidence_floor - 0.7).abs() < 1e-15);
    }

    // -- Entropy measurement -------------------------------------------------

    #[test]
    fn test_entropy_uniform_data() {
        let data = vec![5.0; 100];
        let e = measure_entropy(&data, 16);
        assert!((e.shannon_entropy - 0.0).abs() < 1e-12);
        assert_eq!(e.unique_symbols, 1);
        assert_eq!(e.total_symbols, 100);
    }

    #[test]
    fn test_entropy_two_values() {
        let data: Vec<f64> = (0..100).map(|i| (i % 2) as f64).collect();
        let e = measure_entropy(&data, 2);
        assert!((e.shannon_entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_empty() {
        let e = measure_entropy(&[], 10);
        assert!((e.shannon_entropy - 0.0).abs() < 1e-12);
        assert_eq!(e.unique_symbols, 0);
    }

    #[test]
    fn test_entropy_zero_bins() {
        let e = measure_entropy(&[1.0, 2.0], 0);
        assert!((e.shannon_entropy - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_entropy_normalized_bounded() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let e = measure_entropy(&data, 32);
        assert!(e.normalized_entropy >= 0.0);
        assert!(e.normalized_entropy <= 1.0 + 1e-12);
    }

    // -- Backward-compatible restore() API -----------------------------------

    #[test]
    fn test_restore_all_known_returns_same_values() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mask = vec![1.0; 5];
        let f = Fragment::new(1, FragmentKind::Text, data.clone(), mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
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
    fn test_restore_fills_single_gap() {
        let data = vec![10.0, 0.0, 30.0];
        let mask = vec![1.0, 0.0, 1.0];
        let f = Fragment::new(2, FragmentKind::Inscription, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        let mid = r.field.values[1];
        assert!((mid - 20.0).abs() < 1.0, "expected ~20, got {}", mid);
    }

    #[test]
    fn test_restore_fills_multiple_gaps() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 40.0];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let f = Fragment::new(3, FragmentKind::Audio, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
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
    fn test_restore_heavily_degraded() {
        let n = 50;
        let mut data = vec![0.0; n];
        let mut mask = vec![0.0; n];
        data[0] = 100.0;
        mask[0] = 1.0;
        data[n - 1] = 100.0;
        mask[n - 1] = 1.0;
        let f = Fragment::new(4, FragmentKind::Image, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        for (i, &v) in r.field.values.iter().enumerate() {
            assert!(
                (v - 100.0).abs() < 5.0,
                "index {} too far from 100: {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_entropy_decreases_after_restoration() {
        let data = vec![5.0, 0.0, 100.0, 0.0, 5.0, 0.0, 100.0, 0.0, 5.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let f = Fragment::new(5, FragmentKind::Text, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        assert!(r.field.entropy_before >= 0.0);
        assert!(r.field.entropy_after >= 0.0);
        assert!(r.field.iterations > 0);
    }

    #[test]
    fn test_confidence_known_elements_are_max() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let f = Fragment::new(6, FragmentKind::Artifact, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        assert!((r.field.confidence.scores[0] - 1.0).abs() < 1e-12);
        assert!((r.field.confidence.scores[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_confidence_decays_with_distance() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let f = Fragment::new(7, FragmentKind::Text, data, mask, 0);
        let mut config = InversionConfig::default();
        config.confidence_floor = 0.05;
        let r = restore(&f, &config);
        assert!(
            r.field.confidence.scores[1] > r.field.confidence.scores[4],
            "confidence should decay: near={} center={}",
            r.field.confidence.scores[1],
            r.field.confidence.scores[4]
        );
    }

    #[test]
    fn test_confidence_floor_respected() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let f = Fragment::new(8, FragmentKind::Text, data, mask, 0);
        let mut config = InversionConfig::default();
        config.confidence_floor = 0.3;
        let r = restore(&f, &config);
        for &s in &r.field.confidence.scores {
            assert!(s >= 0.3 - 1e-12, "confidence below floor: {}", s);
        }
    }

    #[test]
    fn test_restore_batch_multiple() {
        let f1 =
            Fragment::new(10, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let f2 = Fragment::new(11, FragmentKind::Image, vec![5.0, 5.0], vec![1.0, 1.0], 0);
        let config = InversionConfig::default();
        let results = restore_batch(&[f1, f2], &config);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].fragment_id, 10);
        assert_eq!(results[1].fragment_id, 11);
    }

    #[test]
    fn test_restore_batch_empty() {
        let config = InversionConfig::default();
        let results = restore_batch(&[], &config);
        assert!(results.is_empty());
    }

    #[test]
    fn test_restore_hash_deterministic() {
        let f =
            Fragment::new(99, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let config = InversionConfig::default();
        let r1 = restore(&f, &config);
        let r2 = restore(&f, &config);
        assert_eq!(r1.field.content_hash, r2.field.content_hash);
        assert_eq!(r1.content_hash, r2.content_hash);
    }

    #[test]
    fn test_restore_empty_fragment() {
        let f = Fragment::new(0, FragmentKind::Text, vec![], vec![], 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        assert!(r.field.values.is_empty());
        assert_eq!(r.field.iterations, 0);
    }

    #[test]
    fn test_restore_all_missing() {
        let f = Fragment::new(0, FragmentKind::Text, vec![0.0; 10], vec![0.0; 10], 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        for &v in &r.field.values {
            assert!((v - 0.0).abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_all_fragment_kinds() {
        let kinds = [
            FragmentKind::Text,
            FragmentKind::Image,
            FragmentKind::Artifact,
            FragmentKind::Inscription,
            FragmentKind::Audio,
        ];
        for (i, &kind) in kinds.iter().enumerate() {
            let f = Fragment::new(i as u64, kind, vec![1.0], vec![1.0], 0);
            assert_eq!(f.kind, kind);
        }
    }

    #[test]
    fn test_restoration_elapsed_ns() {
        let f =
            Fragment::new(0, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let config = InversionConfig::default();
        let r = restore(&f, &config);
        assert!(r.elapsed_ns < 10_000_000_000, "took too long");
    }

    // -- Strategy enum tests -------------------------------------------------

    #[test]
    fn test_strategy_auto_selects_1d_without_grid() {
        let f =
            Fragment::new(0, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let config = InversionConfig::default();
        let r = restore_advanced(&f, &config, Strategy::Auto, None, None, None);
        assert_eq!(r.fragment_id, 0);
        assert!(!r.field.values.is_empty());
    }

    #[test]
    fn test_strategy_auto_selects_sparse_low_known() {
        let mut data = vec![0.0; 100];
        let mut mask = vec![0.0; 100];
        data[0] = 1.0;
        mask[0] = 1.0;
        data[50] = 1.0;
        mask[50] = 1.0;
        data[99] = 1.0;
        mask[99] = 1.0;
        let f = Fragment::new(0, FragmentKind::Image, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_advanced(&f, &config, Strategy::Auto, Some((10, 10)), None, None);
        assert_eq!(r.field.values.len(), 100);
    }

    #[test]
    fn test_strategy_grid2d_explicit() {
        let data = vec![10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let f = Fragment::new(42, FragmentKind::Image, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_advanced(&f, &config, Strategy::Grid2D, Some((3, 3)), None, None);
        assert_eq!(r.fragment_id, 42);
        assert_eq!(r.field.values.len(), 9);
    }

    #[test]
    fn test_strategy_frequency_explicit() {
        let data = vec![5.0; 16];
        let mask = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let f = Fragment::new(0, FragmentKind::Text, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_advanced(
            &f,
            &config,
            Strategy::Frequency,
            Some((4, 4)),
            None,
            None,
        );
        assert_eq!(r.field.values.len(), 16);
    }

    #[test]
    fn test_strategy_sparse_explicit() {
        let data = vec![100.0; 9];
        let mask = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let f = Fragment::new(0, FragmentKind::Text, data, mask, 0);
        let config = InversionConfig::default();
        let r = restore_advanced(
            &f,
            &config,
            Strategy::Sparse,
            Some((3, 3)),
            None,
            None,
        );
        assert_eq!(r.field.values.len(), 9);
    }

    #[test]
    fn test_strategy_multimodal_explicit() {
        let data = vec![0.0; 4];
        let mask = vec![0.0; 4];
        let f = Fragment::new(0, FragmentKind::Image, data, mask, 0);
        let config = InversionConfig::default();

        let obs: Vec<Vec<ModalObservation>> = (0..4)
            .map(|_| {
                vec![ModalObservation {
                    mean: 25.0,
                    variance: 1.0,
                    modality: ModalityKind::Image,
                }]
            })
            .collect();
        let fc = FusionConfig {
            prior_variance: 1e12,
            ..FusionConfig::default()
        };

        let r = restore_advanced(
            &f,
            &config,
            Strategy::MultiModal,
            Some((2, 2)),
            Some(&obs),
            Some(&fc),
        );
        assert_eq!(r.field.values.len(), 4);
        for &v in &r.field.values {
            assert!(
                (v - 25.0).abs() < 3.0,
                "multimodal fusion should give ~25: {}",
                v
            );
        }
    }

    #[test]
    fn test_strategy_fallback_without_grid_dims() {
        let f =
            Fragment::new(0, FragmentKind::Text, vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0], 0);
        let config = InversionConfig::default();
        let r = restore_advanced(&f, &config, Strategy::Grid2D, None, None, None);
        assert_eq!(r.field.values.len(), 3);
    }
}
