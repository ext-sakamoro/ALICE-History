// ALICE-History — Shared utilities
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto

use crate::{ConfidenceMap, EntropyMeasurement, FragmentKind};

// ---------------------------------------------------------------------------
// FNV-1a hash
// ---------------------------------------------------------------------------

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub(crate) const FNV_PRIME: u64 = 0x0100_0000_01b3;

pub(crate) fn fnv1a(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash helper: convert an f64 slice to bytes and hash.
pub(crate) fn hash_f64_slice(slice: &[f64]) -> u64 {
    let mut buf = Vec::with_capacity(slice.len() * 8);
    for &v in slice {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a(&buf)
}

/// Build a content hash for a Fragment from id + kind discriminant + data.
pub(crate) fn fragment_content_hash(id: u64, kind: FragmentKind, data: &[f64]) -> u64 {
    let mut buf: Vec<u8> = Vec::with_capacity(8 + 1 + data.len() * 8);
    buf.extend_from_slice(&id.to_le_bytes());
    buf.push(kind as u8);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a(&buf)
}

// ---------------------------------------------------------------------------
// Entropy measurement
// ---------------------------------------------------------------------------

/// Measure Shannon entropy of a data sequence by binning values.
///
/// Values are linearly mapped into `bins` buckets between the data min and
/// max.  Shannon entropy is computed as H = -sum(p * log2(p)) over non-zero
/// bins.  `normalized_entropy` divides H by log2(bins).
pub fn measure_entropy(data: &[f64], bins: usize) -> EntropyMeasurement {
    if data.is_empty() || bins == 0 {
        return EntropyMeasurement {
            shannon_entropy: 0.0,
            normalized_entropy: 0.0,
            unique_symbols: 0,
            total_symbols: 0,
        };
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    let mut counts = vec![0usize; bins];

    if range == 0.0 {
        counts[0] = data.len();
    } else {
        for &v in data {
            let normalized = (v - min_val) / range;
            let idx = ((normalized * (bins as f64 - 1.0)).round() as usize).min(bins - 1);
            counts[idx] += 1;
        }
    }

    let n = data.len();
    let rcp_n = 1.0 / n as f64; // precomputed reciprocal — avoids repeated division
    let mut entropy = 0.0f64;
    let mut unique = 0usize;

    for &c in &counts {
        if c > 0 {
            unique += 1;
            let p = c as f64 * rcp_n;
            entropy -= p * p.log2();
        }
    }

    let max_entropy = if bins > 1 { (bins as f64).log2() } else { 1.0 };
    let normalized = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    };

    EntropyMeasurement {
        shannon_entropy: entropy,
        normalized_entropy: normalized,
        unique_symbols: unique,
        total_symbols: n,
    }
}

// ---------------------------------------------------------------------------
// Confidence computation (1D)
// ---------------------------------------------------------------------------

/// Compute a 1D ConfidenceMap based on distance from known values.
///
/// Known elements receive confidence 1.0; missing elements receive a value
/// that decays with distance (1 / (1 + distance)), clamped above `floor`.
pub(crate) fn compute_confidence(
    _data: &[f64],
    mask: &[f64],
    _restored: &[f64],
    floor: f64,
) -> ConfidenceMap {
    let n = mask.len();
    if n == 0 {
        return ConfidenceMap {
            scores: Vec::new(),
            mean_confidence: 0.0,
            min_confidence: 0.0,
            restoration_boundary: floor,
        };
    }

    // Forward pass: distance to nearest known element on the left.
    let mut left_dist = vec![n as f64; n];
    for i in 0..n {
        if mask[i] == 1.0 {
            left_dist[i] = 0.0;
        } else if i > 0 {
            left_dist[i] = left_dist[i - 1] + 1.0;
        }
    }

    // Backward pass: distance to nearest known element on the right.
    let mut right_dist = vec![n as f64; n];
    for i in (0..n).rev() {
        if mask[i] == 1.0 {
            right_dist[i] = 0.0;
        } else if i + 1 < n {
            right_dist[i] = right_dist[i + 1] + 1.0;
        }
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let dist = left_dist[i].min(right_dist[i]);
        // reciprocal: 1.0 / (1.0 + dist) — avoids division in hot path
        let raw = (1.0 + dist).recip();
        scores.push(raw.max(floor));
    }

    let rcp_n = 1.0 / n as f64; // precomputed reciprocal for mean
    let mean = scores.iter().sum::<f64>() * rcp_n;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);

    ConfidenceMap {
        scores,
        mean_confidence: mean,
        min_confidence: min,
        restoration_boundary: floor,
    }
}

// ---------------------------------------------------------------------------
// 2D Confidence computation (Manhattan distance approximation)
// ---------------------------------------------------------------------------

/// Compute a 2D ConfidenceMap using two-pass Manhattan distance approximation.
///
/// Pass 1 (top-left → bottom-right): d(r,c) = min(d(r,c), d(r-1,c)+1, d(r,c-1)+1)
/// Pass 2 (bottom-right → top-left): d(r,c) = min(d(r,c), d(r+1,c)+1, d(r,c+1)+1)
/// confidence = max(floor, 1.0 / (1.0 + distance))
pub(crate) fn compute_confidence_2d(
    mask: &[f64],
    rows: usize,
    cols: usize,
    floor: f64,
) -> ConfidenceMap {
    let n = rows * cols;
    if n == 0 {
        return ConfidenceMap {
            scores: Vec::new(),
            mean_confidence: 0.0,
            min_confidence: 0.0,
            restoration_boundary: floor,
        };
    }

    let big = (rows + cols) as f64;
    let mut dist = vec![big; n];

    // Known cells have distance 0
    for i in 0..n {
        if mask[i] == 1.0 {
            dist[i] = 0.0;
        }
    }

    // Pass 1: top-left to bottom-right
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if r > 0 {
                let up = dist[(r - 1) * cols + c] + 1.0;
                if up < dist[idx] {
                    dist[idx] = up;
                }
            }
            if c > 0 {
                let left = dist[r * cols + (c - 1)] + 1.0;
                if left < dist[idx] {
                    dist[idx] = left;
                }
            }
        }
    }

    // Pass 2: bottom-right to top-left
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            let idx = r * cols + c;
            if r + 1 < rows {
                let down = dist[(r + 1) * cols + c] + 1.0;
                if down < dist[idx] {
                    dist[idx] = down;
                }
            }
            if c + 1 < cols {
                let right = dist[r * cols + (c + 1)] + 1.0;
                if right < dist[idx] {
                    dist[idx] = right;
                }
            }
        }
    }

    let mut scores = Vec::with_capacity(n);
    for d in &dist {
        // reciprocal: 1.0 / (1.0 + d) — avoids division in hot path
        let raw = (1.0 + d).recip();
        scores.push(raw.max(floor));
    }

    let rcp_n = 1.0 / n as f64; // precomputed reciprocal for mean
    let mean = scores.iter().sum::<f64>() * rcp_n;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);

    ConfidenceMap {
        scores,
        mean_confidence: mean,
        min_confidence: min,
        restoration_boundary: floor,
    }
}

// ---------------------------------------------------------------------------
// Result hash helper
// ---------------------------------------------------------------------------

/// Compute a result-level content hash from fragment_id + field content_hash.
pub(crate) fn result_content_hash(fragment_id: u64, field_hash: u64) -> u64 {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&fragment_id.to_le_bytes());
    buf.extend_from_slice(&field_hash.to_le_bytes());
    fnv1a(&buf)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FragmentKind;

    // -- fnv1a -----------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty_returns_offset_basis() {
        assert_eq!(fnv1a(&[]), FNV_OFFSET);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"ALICE-History";
        assert_eq!(fnv1a(data), fnv1a(data));
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(fnv1a(b"abc"), fnv1a(b"xyz"));
    }

    #[test]
    fn test_fnv1a_single_byte_differs() {
        assert_ne!(fnv1a(&[0u8]), fnv1a(&[1u8]));
    }

    // -- hash_f64_slice -------------------------------------------------------

    #[test]
    fn test_hash_f64_slice_empty() {
        // Empty slice hashes like empty byte slice
        assert_eq!(hash_f64_slice(&[]), fnv1a(&[]));
    }

    #[test]
    fn test_hash_f64_slice_deterministic() {
        let s = &[1.0f64, 2.0, 3.0];
        assert_eq!(hash_f64_slice(s), hash_f64_slice(s));
    }

    #[test]
    fn test_hash_f64_slice_order_sensitive() {
        assert_ne!(hash_f64_slice(&[1.0, 2.0]), hash_f64_slice(&[2.0, 1.0]));
    }

    #[test]
    fn test_hash_f64_slice_nan_not_equal_zero() {
        let nan_slice = &[f64::NAN];
        let zero_slice = &[0.0f64];
        // NaN has different bit pattern from 0.0 so hashes must differ
        assert_ne!(hash_f64_slice(nan_slice), hash_f64_slice(zero_slice));
    }

    // -- fragment_content_hash ------------------------------------------------

    #[test]
    fn test_fragment_content_hash_same_inputs_same_output() {
        let h1 = fragment_content_hash(42, FragmentKind::Text, &[1.0, 2.0]);
        let h2 = fragment_content_hash(42, FragmentKind::Text, &[1.0, 2.0]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fragment_content_hash_different_ids_differ() {
        let h1 = fragment_content_hash(1, FragmentKind::Text, &[1.0]);
        let h2 = fragment_content_hash(2, FragmentKind::Text, &[1.0]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fragment_content_hash_different_kinds_differ() {
        let h1 = fragment_content_hash(0, FragmentKind::Text, &[1.0]);
        let h2 = fragment_content_hash(0, FragmentKind::Image, &[1.0]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fragment_content_hash_empty_data_nonzero() {
        // Even empty data should produce a non-trivial hash (id + kind bytes present)
        let h = fragment_content_hash(0, FragmentKind::Audio, &[]);
        // Must not be the FNV offset (i.e., id and kind bytes were included)
        // id=0 bytes are all zero, kind=Audio byte is non-zero, so the hash will differ
        assert_ne!(h, 0);
    }

    // -- result_content_hash --------------------------------------------------

    #[test]
    fn test_result_content_hash_deterministic() {
        let h1 = result_content_hash(10, 99999);
        let h2 = result_content_hash(10, 99999);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_result_content_hash_different_ids_differ() {
        let h1 = result_content_hash(1, 100);
        let h2 = result_content_hash(2, 100);
        assert_ne!(h1, h2);
    }

    // -- compute_confidence (1D) ----------------------------------------------

    #[test]
    fn test_compute_confidence_empty_mask() {
        let cm = compute_confidence(&[], &[], &[], 0.5);
        assert!(cm.scores.is_empty());
        assert!((cm.mean_confidence - 0.0).abs() < 1e-12);
        assert!((cm.restoration_boundary - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_confidence_all_known() {
        let mask = vec![1.0; 5];
        let cm = compute_confidence(&[0.0; 5], &mask, &[0.0; 5], 0.5);
        for &s in &cm.scores {
            assert!(
                (s - 1.0).abs() < 1e-12,
                "all-known should score 1.0, got {}",
                s
            );
        }
        assert!((cm.mean_confidence - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_confidence_single_known_at_start() {
        // [1, 0, 0, 0, 0]: confidence should decrease away from position 0
        let mask = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let cm = compute_confidence(&[0.0; 5], &mask, &[0.0; 5], 0.0);
        // score[0] = 1.0, score[1] = 1/(1+1)=0.5, score[2] = 1/3, ...
        assert!((cm.scores[0] - 1.0).abs() < 1e-12);
        assert!(cm.scores[0] > cm.scores[1]);
        assert!(cm.scores[1] > cm.scores[2]);
        assert!(cm.scores[2] > cm.scores[3]);
    }

    #[test]
    fn test_compute_confidence_known_at_both_ends() {
        // [1, 0, 0, 0, 1]: center distance = 2, so score[2] = 1/3
        let mask = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let cm = compute_confidence(&[0.0; 5], &mask, &[0.0; 5], 0.0);
        assert!((cm.scores[0] - 1.0).abs() < 1e-12);
        assert!((cm.scores[4] - 1.0).abs() < 1e-12);
        // Center (index 2): dist = min(2,2) = 2 -> 1/3
        let expected_center = 1.0 / 3.0;
        assert!(
            (cm.scores[2] - expected_center).abs() < 1e-10,
            "center score {} != expected {}",
            cm.scores[2],
            expected_center
        );
    }

    #[test]
    fn test_compute_confidence_floor_applied() {
        let mask = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let floor = 0.25;
        let cm = compute_confidence(&[0.0; 10], &mask, &[0.0; 10], floor);
        for &s in &cm.scores {
            assert!(s >= floor - 1e-12, "score {} below floor {}", s, floor);
        }
    }

    #[test]
    fn test_compute_confidence_min_confidence_correct() {
        let mask = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let cm = compute_confidence(&[0.0; 10], &mask, &[0.0; 10], 0.0);
        let actual_min = cm.scores.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (cm.min_confidence - actual_min).abs() < 1e-12,
            "reported min {} != actual min {}",
            cm.min_confidence,
            actual_min
        );
    }

    // -- compute_confidence_2d ------------------------------------------------

    #[test]
    fn test_compute_confidence_2d_empty() {
        let cm = compute_confidence_2d(&[], 0, 0, 0.5);
        assert!(cm.scores.is_empty());
        assert!((cm.restoration_boundary - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_confidence_2d_all_known() {
        let mask = vec![1.0; 9];
        let cm = compute_confidence_2d(&mask, 3, 3, 0.5);
        for &s in &cm.scores {
            assert!((s - 1.0).abs() < 1e-12, "all-known 2D score should be 1.0");
        }
    }

    #[test]
    fn test_compute_confidence_2d_center_known() {
        // 3x3 grid, only center (index 4) known
        let mut mask = vec![0.0; 9];
        mask[4] = 1.0;
        let cm = compute_confidence_2d(&mask, 3, 3, 0.0);
        // Center = 1.0
        assert!((cm.scores[4] - 1.0).abs() < 1e-12);
        // Adjacent cells (dist=1): score = 1/(1+1) = 0.5
        assert!((cm.scores[1] - 0.5).abs() < 1e-12); // top-center
        assert!((cm.scores[3] - 0.5).abs() < 1e-12); // left
        assert!((cm.scores[5] - 0.5).abs() < 1e-12); // right
        assert!((cm.scores[7] - 0.5).abs() < 1e-12); // bottom-center
                                                     // Corner (dist=2): score = 1/(1+2) = 1/3
        let expected_corner = 1.0 / 3.0;
        assert!(
            (cm.scores[0] - expected_corner).abs() < 1e-10,
            "corner score {} != {}",
            cm.scores[0],
            expected_corner
        );
    }

    #[test]
    fn test_compute_confidence_2d_floor_applied() {
        let mask = vec![0.0; 25]; // 5x5, all unknown
        let floor = 0.3;
        let cm = compute_confidence_2d(&mask, 5, 5, floor);
        for &s in &cm.scores {
            assert!(s >= floor - 1e-12, "2D score {} below floor", s);
        }
    }

    #[test]
    fn test_compute_confidence_2d_1x1_known() {
        let mask = vec![1.0];
        let cm = compute_confidence_2d(&mask, 1, 1, 0.5);
        assert_eq!(cm.scores.len(), 1);
        assert!((cm.scores[0] - 1.0).abs() < 1e-12);
    }
}
