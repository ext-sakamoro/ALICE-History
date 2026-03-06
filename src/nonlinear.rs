// ALICE-History — Non-linear restoration
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-linear Restoration — Newton法 + Total Variation 正則化。
//!
//! 線形ソルバーでは対応できない非線形劣化パターンに対応する復元手法。

use crate::{ConfidenceMap, Fragment, InversionConfig, RestorationField, RestorationResult};

/// 非線形復元の設定。
#[derive(Debug, Clone)]
pub struct NonlinearConfig {
    /// 最大反復数。
    pub max_iterations: u32,
    /// 収束閾値。
    pub convergence_threshold: f64,
    /// Total Variation 正則化の重み。
    pub tv_weight: f64,
    /// Newton法のステップサイズ減衰。
    pub step_damping: f64,
    /// 信頼度下限。
    pub confidence_floor: f64,
}

impl Default for NonlinearConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            convergence_threshold: 1e-7,
            tv_weight: 0.1,
            step_damping: 0.5,
            confidence_floor: 0.3,
        }
    }
}

impl From<&InversionConfig> for NonlinearConfig {
    fn from(ic: &InversionConfig) -> Self {
        Self {
            max_iterations: ic.max_iterations,
            convergence_threshold: ic.convergence_threshold,
            tv_weight: ic.regularization,
            confidence_floor: ic.confidence_floor,
            ..Self::default()
        }
    }
}

/// 非線形復元結果。
#[derive(Debug, Clone)]
pub struct NonlinearResult {
    /// 復元値。
    pub values: Vec<f64>,
    /// 反復回数。
    pub iterations: u32,
    /// 最終残差。
    pub final_residual: f64,
    /// TV正則化項の値。
    pub tv_norm: f64,
}

/// Total Variation ノルム (1D)。
#[must_use]
pub fn total_variation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    data.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Total Variation 近接演算子 (proximal operator)。
///
/// Condat (2013) アルゴリズムの簡易版。
#[must_use]
pub fn tv_proximal(data: &[f64], lambda: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    if data.len() == 1 {
        return data.to_vec();
    }

    // 反復射影法による TV denoising
    let n = data.len();
    let mut result = data.to_vec();

    for _ in 0..50 {
        let prev = result.clone();

        // 勾配降下ステップ
        for i in 1..n - 1 {
            let grad = 2.0 * (result[i] - data[i]);
            let tv_grad = tv_gradient_at(&result, i, lambda);
            result[i] -= 0.1 * (grad + tv_grad);
        }

        // 収束チェック
        let diff: f64 = result
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if diff.sqrt() < 1e-8 {
            break;
        }
    }

    result
}

/// TV 正則化の勾配 (位置 i)。
fn tv_gradient_at(data: &[f64], i: usize, lambda: f64) -> f64 {
    let eps = 1e-8;
    let n = data.len();
    let mut grad = 0.0;

    if i > 0 {
        let diff = data[i] - data[i - 1];
        grad += lambda * diff / (diff.abs() + eps);
    }
    if i + 1 < n {
        let diff = data[i] - data[i + 1];
        grad += lambda * diff / (diff.abs() + eps);
    }

    grad
}

/// Newton法ベースの非線形復元。
///
/// データフィッティング + Total Variation 正則化の最小化。
#[must_use]
pub fn restore_nonlinear(fragment: &Fragment, config: &NonlinearConfig) -> NonlinearResult {
    let n = fragment.data.len();
    if n == 0 {
        return NonlinearResult {
            values: Vec::new(),
            iterations: 0,
            final_residual: 0.0,
            tv_norm: 0.0,
        };
    }

    // 初期値: 既知データをコピー、欠損は隣接平均で初期化
    let mut x = fragment.data.clone();
    initialize_missing(&mut x, &fragment.mask);

    let mut iterations = 0;
    let mut residual = f64::MAX;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let prev = x.clone();

        // Damped Newton ステップ
        for i in 0..n {
            if fragment.mask[i] > 0.5 {
                continue; // 既知データは変更しない
            }

            // データフィッティング勾配
            let data_grad = 2.0 * (x[i] - fragment.data[i]) * fragment.mask[i];

            // TV 正則化勾配
            let tv_grad = tv_gradient_at(&x, i, config.tv_weight);

            // ヘシアン近似 (対角)
            let hessian_approx = 2.0f64.mul_add(fragment.mask[i], config.tv_weight * 2.0) + 1e-8;

            // Newton ステップ
            let step = (data_grad + tv_grad) / hessian_approx;
            x[i] -= config.step_damping * step;
        }

        // 残差計算
        residual = x
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        if residual < config.convergence_threshold {
            break;
        }
    }

    let tv_norm = total_variation(&x);

    NonlinearResult {
        values: x,
        iterations,
        final_residual: residual,
        tv_norm,
    }
}

/// Fragment から `RestorationResult` を生成する非線形復元。
#[must_use]
pub fn restore_nonlinear_full(fragment: &Fragment, config: &NonlinearConfig) -> RestorationResult {
    let result = restore_nonlinear(fragment, config);

    let scores = compute_confidence(&fragment.mask, config.confidence_floor);
    let mean_confidence = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };
    let min_confidence = scores.iter().copied().fold(f64::INFINITY, f64::min);

    RestorationResult {
        fragment_id: fragment.id,
        field: RestorationField {
            values: result.values,
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
            content_hash: fragment.content_hash,
        },
        elapsed_ns: 0,
        content_hash: fragment.content_hash,
    }
}

/// 欠損位置を隣接平均で初期化。
fn initialize_missing(data: &mut [f64], mask: &[f64]) {
    let n = data.len();
    for i in 0..n {
        if mask[i] < 0.5 {
            let mut sum = 0.0;
            let mut count = 0;
            if i > 0 && mask[i - 1] > 0.5 {
                sum += data[i - 1];
                count += 1;
            }
            if i + 1 < n && mask[i + 1] > 0.5 {
                sum += data[i + 1];
                count += 1;
            }
            if count > 0 {
                data[i] = sum / count as f64;
            }
        }
    }
}

/// 信頼度マップ生成。
fn compute_confidence(mask: &[f64], floor: f64) -> Vec<f64> {
    let n = mask.len();
    let mut conf = vec![floor; n];

    for (c, &m) in conf.iter_mut().zip(mask.iter()) {
        if m > 0.5 {
            *c = 1.0;
        }
    }

    // 距離減衰
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
    use crate::FragmentKind;

    fn make_fragment(data: Vec<f64>, mask: Vec<f64>) -> Fragment {
        Fragment::new(0, FragmentKind::Text, data, mask, 0)
    }

    #[test]
    fn empty_fragment() {
        let f = make_fragment(vec![], vec![]);
        let r = restore_nonlinear(&f, &NonlinearConfig::default());
        assert!(r.values.is_empty());
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn all_known_preserved() {
        let f = make_fragment(vec![10.0, 20.0, 30.0], vec![1.0, 1.0, 1.0]);
        let r = restore_nonlinear(&f, &NonlinearConfig::default());
        assert!((r.values[0] - 10.0).abs() < 1e-12);
        assert!((r.values[1] - 20.0).abs() < 1e-12);
        assert!((r.values[2] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn single_gap_filled() {
        let f = make_fragment(vec![10.0, 0.0, 30.0], vec![1.0, 0.0, 1.0]);
        let r = restore_nonlinear(&f, &NonlinearConfig::default());
        assert!(
            (r.values[1] - 20.0).abs() < 5.0,
            "expected ~20, got {}",
            r.values[1]
        );
    }

    #[test]
    fn tv_reduces_noise() {
        // 高TV正則化でスムーズに
        let f = make_fragment(
            vec![10.0, 0.0, 0.0, 0.0, 10.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let config = NonlinearConfig {
            tv_weight: 1.0,
            ..NonlinearConfig::default()
        };
        let r = restore_nonlinear(&f, &config);
        // TV正則化により平滑
        let variation: f64 = r.values.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(variation < 50.0, "TV should smooth: tv_norm={variation}");
    }

    #[test]
    fn total_variation_empty() {
        assert!((total_variation(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn total_variation_single() {
        assert!((total_variation(&[5.0]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn total_variation_ramp() {
        let tv = total_variation(&[0.0, 1.0, 2.0, 3.0]);
        assert!((tv - 3.0).abs() < 1e-12);
    }

    #[test]
    fn total_variation_zigzag() {
        let tv = total_variation(&[0.0, 10.0, 0.0, 10.0]);
        assert!((tv - 30.0).abs() < 1e-12);
    }

    #[test]
    fn tv_proximal_empty() {
        assert!(tv_proximal(&[], 1.0).is_empty());
    }

    #[test]
    fn tv_proximal_single() {
        let r = tv_proximal(&[5.0], 1.0);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn tv_proximal_preserves_constant() {
        let data = vec![5.0; 5];
        let r = tv_proximal(&data, 0.5);
        for &v in &r {
            assert!((v - 5.0).abs() < 0.1, "constant should be preserved: {v}");
        }
    }

    #[test]
    fn restore_full_returns_result() {
        let f = make_fragment(vec![1.0, 0.0, 3.0], vec![1.0, 0.0, 1.0]);
        let r = restore_nonlinear_full(&f, &NonlinearConfig::default());
        assert_eq!(r.fragment_id, 0);
        assert_eq!(r.field.values.len(), 3);
        assert!(r.field.confidence.mean_confidence > 0.0);
    }

    #[test]
    fn from_inversion_config() {
        let ic = InversionConfig {
            max_iterations: 200,
            convergence_threshold: 1e-5,
            regularization: 0.05,
            confidence_floor: 0.4,
        };
        let nc = NonlinearConfig::from(&ic);
        assert_eq!(nc.max_iterations, 200);
        assert!((nc.convergence_threshold - 1e-5).abs() < 1e-15);
        assert!((nc.tv_weight - 0.05).abs() < 1e-15);
    }

    #[test]
    fn default_config() {
        let c = NonlinearConfig::default();
        assert_eq!(c.max_iterations, 500);
        assert!((c.step_damping - 0.5).abs() < 1e-12);
    }
}
