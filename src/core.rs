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
    for i in 0..n {
        // reciprocal: 1.0 / (1.0 + dist[i]) — avoids division in hot path
        let raw = (1.0 + dist[i]).recip();
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
