// ALICE-History — Precomputed cosine table for DCT-II / DCT-III
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use std::f64::consts::PI;

/// Precomputed cosine table for DCT.
///
/// Stores `cos(PI * (2*n + 1) * k / (2*N))` in row-major layout:
/// `table[n * size + k]` where n is the spatial index, k is the frequency index.
///
/// Row-major layout ensures cache-friendly access during row-direction DCT,
/// where we iterate over k for a fixed n.
pub(crate) struct CosineTable {
    /// Table size (N).
    pub size: usize,
    /// Precomputed reciprocal: 1.0 / (2 * N), computed once at construction.
    #[allow(dead_code)]
    pub rcp_2n: f64,
    /// Flat row-major storage: table[n * size + k].
    pub table: Vec<f64>,
}

impl CosineTable {
    /// Create a new CosineTable of given size.
    ///
    /// Precomputes all cos(PI * (2*n + 1) * k / (2*N)) values.
    pub fn new(size: usize) -> Self {
        if size == 0 {
            return Self {
                size: 0,
                rcp_2n: 0.0,
                table: Vec::new(),
            };
        }

        // Reciprocal pre-computation to avoid division in hot loop
        let rcp_2n = 1.0 / (2.0 * size as f64);
        let mut table = Vec::with_capacity(size * size);

        for n in 0..size {
            let base = PI * (2 * n + 1) as f64 * rcp_2n;
            for k in 0..size {
                table.push((base * k as f64).cos());
            }
        }

        Self {
            size,
            rcp_2n,
            table,
        }
    }

    /// Lookup cos(PI * (2*n + 1) * k / (2*N)).
    #[inline]
    pub fn get(&self, n: usize, k: usize) -> f64 {
        self.table[n * self.size + k]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_cosine_table_size_zero() {
        let ct = CosineTable::new(0);
        assert_eq!(ct.size, 0);
        assert!(ct.table.is_empty());
    }

    #[test]
    fn test_cosine_table_size_one() {
        let ct = CosineTable::new(1);
        assert_eq!(ct.size, 1);
        assert_eq!(ct.table.len(), 1);
        // cos(PI * 1 * 0 / 2) = cos(0) = 1.0
        assert!((ct.get(0, 0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_cosine_table_matches_direct_computation() {
        let n_size = 16;
        let ct = CosineTable::new(n_size);
        for n in 0..n_size {
            for k in 0..n_size {
                let expected = (PI * (2 * n + 1) as f64 * k as f64 / (2.0 * n_size as f64)).cos();
                let got = ct.get(n, k);
                assert!(
                    (got - expected).abs() < 1e-12,
                    "mismatch at n={}, k={}: {} vs {}",
                    n,
                    k,
                    got,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_cosine_table_k0_is_always_one() {
        // cos(PI * (2n+1) * 0 / (2N)) = cos(0) = 1.0
        let ct = CosineTable::new(32);
        for n in 0..32 {
            assert!((ct.get(n, 0) - 1.0).abs() < 1e-15, "k=0 should always be 1.0");
        }
    }

    #[test]
    fn test_cosine_table_symmetry() {
        // cos is even: cos(x) = cos(-x), but we check orthogonality property:
        // sum_n cos(pi*(2n+1)*k/(2N)) * cos(pi*(2n+1)*j/(2N)) = 0 for k != j
        let n_size = 8;
        let ct = CosineTable::new(n_size);
        for k in 0..n_size {
            for j in 0..n_size {
                if k == j {
                    continue;
                }
                let mut dot = 0.0;
                for n in 0..n_size {
                    dot += ct.get(n, k) * ct.get(n, j);
                }
                assert!(
                    dot.abs() < 1e-10,
                    "orthogonality failed for k={}, j={}: dot={}",
                    k,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_cosine_table_reciprocal() {
        let ct = CosineTable::new(64);
        assert!((ct.rcp_2n - 1.0 / 128.0).abs() < 1e-15);
    }

    #[test]
    fn test_cosine_table_layout_is_row_major() {
        let ct = CosineTable::new(4);
        assert_eq!(ct.table.len(), 16);
        // table[n * size + k]
        // n=0, k=0 -> index 0
        // n=0, k=3 -> index 3
        // n=1, k=0 -> index 4
        assert!((ct.table[0] - ct.get(0, 0)).abs() < 1e-15);
        assert!((ct.table[3] - ct.get(0, 3)).abs() < 1e-15);
        assert!((ct.table[4] - ct.get(1, 0)).abs() < 1e-15);
    }
}
