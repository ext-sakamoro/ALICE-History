// ALICE-History — Temporal coherence solver
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Temporal Coherence Solver — 時間次元付き体積復元。
//!
//! 時間軸方向のスムージングと空間補間を組み合わせた
//! 3D + time の復元フレームワーク。

use crate::{ConfidenceMap, RestorationField, RestorationResult};

/// 時間フレーム: ある時刻におけるデータスナップショット。
#[derive(Debug, Clone)]
pub struct TemporalFrame {
    /// フレーム内データ。
    pub data: Vec<f64>,
    /// 既知マスク (1.0 = 既知, 0.0 = 欠損)。
    pub mask: Vec<f64>,
    /// タイムスタンプ (任意単位)。
    pub timestamp: f64,
}

/// Temporal coherence 設定。
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// 空間方向の重み。
    pub spatial_weight: f64,
    /// 時間方向の重み。
    pub temporal_weight: f64,
    /// 最大反復数。
    pub max_iterations: u32,
    /// 収束閾値。
    pub convergence_threshold: f64,
    /// 信頼度下限。
    pub confidence_floor: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            spatial_weight: 1.0,
            temporal_weight: 0.5,
            max_iterations: 200,
            convergence_threshold: 1e-6,
            confidence_floor: 0.3,
        }
    }
}

/// 時系列復元結果。
#[derive(Debug, Clone)]
pub struct TemporalResult {
    /// 各フレームの復元値。
    pub frames: Vec<Vec<f64>>,
    /// 各フレームの信頼度。
    pub confidences: Vec<Vec<f64>>,
    /// 反復回数。
    pub iterations: u32,
    /// 最終残差。
    pub final_residual: f64,
}

/// 時間的整合性を考慮した復元。
///
/// 隣接フレーム間の時間スムージングと空間補間を交互に適用し、
/// 全フレームを同時に復元する。
#[must_use]
pub fn restore_temporal(frames: &[TemporalFrame], config: &TemporalConfig) -> TemporalResult {
    if frames.is_empty() {
        return TemporalResult {
            frames: Vec::new(),
            confidences: Vec::new(),
            iterations: 0,
            final_residual: 0.0,
        };
    }

    let n_frames = frames.len();
    let n_elements = frames[0].data.len();

    // 初期値: 既知データをコピー
    let mut restored: Vec<Vec<f64>> = frames.iter().map(|f| f.data.clone()).collect();

    let mut iterations = 0;
    let mut residual = f64::MAX;

    for iter in 0..config.max_iterations {
        let prev = restored.clone();
        iterations = iter + 1;

        // 各フレーム、各要素を更新
        for t in 0..n_frames {
            for i in 0..n_elements {
                if frames[t].mask.get(i).copied().unwrap_or(0.0) > 0.5 {
                    continue; // 既知データは変更しない
                }

                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;

                // 空間: 同フレーム内の隣接要素
                if i > 0 {
                    weighted_sum += config.spatial_weight * restored[t][i - 1];
                    weight_total += config.spatial_weight;
                }
                if i + 1 < n_elements {
                    weighted_sum += config.spatial_weight * restored[t][i + 1];
                    weight_total += config.spatial_weight;
                }

                // 時間: 前後フレームの同位置
                if t > 0 {
                    weighted_sum += config.temporal_weight * restored[t - 1][i];
                    weight_total += config.temporal_weight;
                }
                if t + 1 < n_frames {
                    weighted_sum += config.temporal_weight * restored[t + 1][i];
                    weight_total += config.temporal_weight;
                }

                if weight_total > 0.0 {
                    restored[t][i] = weighted_sum / weight_total;
                }
            }
        }

        // 残差計算
        residual = 0.0;
        for t in 0..n_frames {
            for i in 0..n_elements {
                let diff = restored[t][i] - prev[t][i];
                residual += diff * diff;
            }
        }
        residual = residual.sqrt();

        if residual < config.convergence_threshold {
            break;
        }
    }

    // 信頼度計算
    let confidences: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| compute_temporal_confidence(f, config.confidence_floor))
        .collect();

    TemporalResult {
        frames: restored,
        confidences,
        iterations,
        final_residual: residual,
    }
}

/// 単一フレームを `RestorationResult` に変換するユーティリティ。
#[must_use]
pub fn temporal_to_restoration(
    result: &TemporalResult,
    frame_index: usize,
    fragment_id: u64,
) -> Option<RestorationResult> {
    if frame_index >= result.frames.len() {
        return None;
    }

    let values = result.frames[frame_index].clone();
    let scores = result.confidences[frame_index].clone();
    let mean_confidence = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };
    let min_confidence = scores.iter().copied().fold(f64::INFINITY, f64::min);

    Some(RestorationResult {
        fragment_id,
        field: RestorationField {
            values,
            confidence: ConfidenceMap {
                scores,
                mean_confidence,
                min_confidence: if min_confidence.is_infinite() {
                    0.0
                } else {
                    min_confidence
                },
                restoration_boundary: 0.0,
            },
            entropy_before: 0.0,
            entropy_after: 0.0,
            iterations: result.iterations,
            content_hash: 0,
        },
        elapsed_ns: 0,
        content_hash: 0,
    })
}

/// 時間フレームの信頼度を計算。
fn compute_temporal_confidence(frame: &TemporalFrame, floor: f64) -> Vec<f64> {
    let n = frame.data.len();
    let mut conf = vec![floor; n];

    for (c, m) in conf.iter_mut().zip(frame.mask.iter()) {
        if *m > 0.5 {
            *c = 1.0;
        }
    }

    // 既知点からの距離に基づく減衰
    // 前方スキャン
    let mut last_known: Option<usize> = None;
    for (i, c) in conf.iter_mut().enumerate() {
        if *c >= 1.0 {
            last_known = Some(i);
        } else if let Some(k) = last_known {
            let dist = (i - k) as f64;
            let decay = (1.0 - floor).mul_add((-0.3 * dist).exp(), floor);
            *c = c.max(decay);
        }
    }

    // 後方スキャン
    last_known = None;
    for (i, c) in conf.iter_mut().enumerate().rev() {
        if *c >= 1.0 {
            last_known = Some(i);
        } else if let Some(k) = last_known {
            let dist = (k - i) as f64;
            let decay = (1.0 - floor).mul_add((-0.3 * dist).exp(), floor);
            *c = c.max(decay);
        }
    }

    conf
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(data: Vec<f64>, mask: Vec<f64>, ts: f64) -> TemporalFrame {
        TemporalFrame {
            data,
            mask,
            timestamp: ts,
        }
    }

    #[test]
    fn empty_frames() {
        let r = restore_temporal(&[], &TemporalConfig::default());
        assert!(r.frames.is_empty());
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn single_frame_all_known() {
        let frames = vec![make_frame(vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0], 0.0)];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        assert_eq!(r.frames.len(), 1);
        assert!((r.frames[0][0] - 1.0).abs() < 1e-12);
        assert!((r.frames[0][1] - 2.0).abs() < 1e-12);
        assert!((r.frames[0][2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn single_frame_gap_filled() {
        let frames = vec![make_frame(vec![10.0, 0.0, 30.0], vec![1.0, 0.0, 1.0], 0.0)];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        let mid = r.frames[0][1];
        assert!((mid - 20.0).abs() < 2.0, "expected ~20, got {mid}");
    }

    #[test]
    fn temporal_smoothing_across_frames() {
        let frames = vec![
            make_frame(vec![10.0, 0.0], vec![1.0, 0.0], 0.0),
            make_frame(vec![0.0, 0.0], vec![0.0, 0.0], 1.0),
            make_frame(vec![0.0, 20.0], vec![0.0, 1.0], 2.0),
        ];
        let config = TemporalConfig {
            temporal_weight: 1.0,
            ..TemporalConfig::default()
        };
        let r = restore_temporal(&frames, &config);
        // 中間フレームは前後から値を受け取る
        assert!(r.frames[1][0] > 0.0, "should receive value from frame 0");
        assert!(r.frames[1][1] > 0.0, "should receive value from frame 2");
    }

    #[test]
    fn convergence_reached() {
        let frames = vec![
            make_frame(vec![5.0, 0.0, 5.0], vec![1.0, 0.0, 1.0], 0.0),
            make_frame(vec![5.0, 0.0, 5.0], vec![1.0, 0.0, 1.0], 1.0),
        ];
        let config = TemporalConfig {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            ..TemporalConfig::default()
        };
        let r = restore_temporal(&frames, &config);
        assert!(r.final_residual < 1e-6);
    }

    #[test]
    fn known_values_preserved() {
        let frames = vec![
            make_frame(vec![42.0, 0.0], vec![1.0, 0.0], 0.0),
            make_frame(vec![0.0, 99.0], vec![0.0, 1.0], 1.0),
        ];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        assert!((r.frames[0][0] - 42.0).abs() < 1e-12);
        assert!((r.frames[1][1] - 99.0).abs() < 1e-12);
    }

    #[test]
    fn confidence_known_is_max() {
        let frames = vec![make_frame(vec![1.0, 0.0, 1.0], vec![1.0, 0.0, 1.0], 0.0)];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        assert!((r.confidences[0][0] - 1.0).abs() < 1e-12);
        assert!((r.confidences[0][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn confidence_floor_respected() {
        let frames = vec![make_frame(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
        )];
        let config = TemporalConfig {
            confidence_floor: 0.2,
            ..TemporalConfig::default()
        };
        let r = restore_temporal(&frames, &config);
        for &c in &r.confidences[0] {
            assert!(c >= 0.2 - 1e-12, "confidence below floor: {c}");
        }
    }

    #[test]
    fn to_restoration_result() {
        let frames = vec![make_frame(vec![1.0, 2.0], vec![1.0, 1.0], 0.0)];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        let res = temporal_to_restoration(&r, 0, 42).unwrap();
        assert_eq!(res.fragment_id, 42);
        assert_eq!(res.field.values.len(), 2);
    }

    #[test]
    fn to_restoration_out_of_bounds() {
        let r = restore_temporal(&[], &TemporalConfig::default());
        assert!(temporal_to_restoration(&r, 0, 0).is_none());
    }

    #[test]
    fn default_config() {
        let c = TemporalConfig::default();
        assert!((c.spatial_weight - 1.0).abs() < 1e-12);
        assert!((c.temporal_weight - 0.5).abs() < 1e-12);
        assert_eq!(c.max_iterations, 200);
    }

    #[test]
    fn multi_frame_all_known() {
        let frames = vec![
            make_frame(vec![1.0, 2.0], vec![1.0, 1.0], 0.0),
            make_frame(vec![3.0, 4.0], vec![1.0, 1.0], 1.0),
            make_frame(vec![5.0, 6.0], vec![1.0, 1.0], 2.0),
        ];
        let r = restore_temporal(&frames, &TemporalConfig::default());
        assert_eq!(r.frames.len(), 3);
        assert!((r.frames[0][0] - 1.0).abs() < 1e-12);
        assert!((r.frames[2][1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn confidence_decays_with_distance() {
        let frames = vec![make_frame(
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
        )];
        let config = TemporalConfig {
            confidence_floor: 0.1,
            ..TemporalConfig::default()
        };
        let r = restore_temporal(&frames, &config);
        assert!(
            r.confidences[0][1] > r.confidences[0][3],
            "confidence should decay: near={}, far={}",
            r.confidences[0][1],
            r.confidences[0][3]
        );
    }
}
