// ALICE-History — GPU acceleration abstraction
// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Acceleration — GPU バックエンド抽象化レイヤー。
//!
//! CPU fallback 付きの計算バックエンド選択。
//! 将来の CUDA/Metal 統合を見据えた抽象化。

/// GPU バックエンド種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// CPU fallback (常に利用可能)。
    Cpu,
    /// CUDA (NVIDIA)。
    Cuda,
    /// Metal (Apple)。
    Metal,
    /// Vulkan compute。
    Vulkan,
}

/// GPU デバイス情報。
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// バックエンド種別。
    pub backend: GpuBackend,
    /// デバイス名。
    pub name: String,
    /// メモリ容量 (バイト)。
    pub memory_bytes: u64,
    /// 最大ワークグループサイズ。
    pub max_workgroup_size: u32,
    /// 利用可能フラグ。
    pub available: bool,
}

/// 計算カーネル種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// 1D 線形補間。
    LinearInterpolation,
    /// 2D Gauss-Seidel。
    GaussSeidel,
    /// DCT 変換。
    Dct,
    /// FISTA 圧縮センシング。
    Fista,
    /// TV 正則化。
    TotalVariation,
}

/// GPU 計算コンテキスト。
#[derive(Debug)]
pub struct ComputeContext {
    /// 使用バックエンド。
    pub backend: GpuBackend,
    /// ワークグループサイズ。
    pub workgroup_size: u32,
    /// 利用可能なカーネル。
    available_kernels: Vec<KernelType>,
}

impl ComputeContext {
    /// CPU fallback コンテキストを作成。
    #[must_use]
    pub fn cpu_fallback() -> Self {
        Self {
            backend: GpuBackend::Cpu,
            workgroup_size: 1,
            available_kernels: vec![
                KernelType::LinearInterpolation,
                KernelType::GaussSeidel,
                KernelType::Dct,
                KernelType::Fista,
                KernelType::TotalVariation,
            ],
        }
    }

    /// 指定バックエンドのコンテキストを作成 (現在は CPU のみ)。
    #[must_use]
    pub fn new(backend: GpuBackend) -> Self {
        match backend {
            GpuBackend::Cpu => Self::cpu_fallback(),
            // GPU バックエンドは未実装、CPU にフォールバック
            _ => Self {
                backend: GpuBackend::Cpu,
                workgroup_size: 1,
                available_kernels: vec![
                    KernelType::LinearInterpolation,
                    KernelType::GaussSeidel,
                    KernelType::Dct,
                    KernelType::Fista,
                    KernelType::TotalVariation,
                ],
            },
        }
    }

    /// カーネルが利用可能か確認。
    #[must_use]
    pub fn supports_kernel(&self, kernel: KernelType) -> bool {
        self.available_kernels.contains(&kernel)
    }

    /// 利用可能なカーネル数。
    #[must_use]
    pub const fn kernel_count(&self) -> usize {
        self.available_kernels.len()
    }

    /// データの並列 map 演算 (CPU 実装)。
    #[must_use]
    pub fn parallel_map(&self, data: &[f64], f: impl Fn(f64) -> f64 + Send + Sync) -> Vec<f64> {
        // CPU fallback: 単純な map
        data.iter().copied().map(f).collect()
    }

    /// データの並列 reduce 演算 (CPU 実装)。
    #[must_use]
    pub fn parallel_reduce(&self, data: &[f64], init: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
        data.iter().copied().fold(init, &f)
    }

    /// 行列ベクトル積 (CPU 実装)。
    ///
    /// `matrix` は行優先、`rows x cols` 行列。
    ///
    /// # Panics
    /// `matrix.len() != rows * cols` または `vector.len() != cols` の場合。
    #[must_use]
    pub fn matvec(&self, matrix: &[f64], vector: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        assert_eq!(matrix.len(), rows * cols, "matrix size mismatch");
        assert_eq!(vector.len(), cols, "vector size mismatch");

        (0..rows)
            .map(|r| {
                let row_start = r * cols;
                matrix[row_start..row_start + cols]
                    .iter()
                    .zip(vector.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }
}

/// 最適な利用可能バックエンドを検出。
///
/// 現在は常に CPU を返す。
#[must_use]
pub const fn detect_best_backend() -> GpuBackend {
    // GPU 検出は将来実装
    GpuBackend::Cpu
}

/// GPU デバイス列挙 (CPU fallback のみ)。
#[must_use]
pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
    vec![GpuDeviceInfo {
        backend: GpuBackend::Cpu,
        name: String::from("CPU Fallback"),
        memory_bytes: 0,
        max_workgroup_size: 1,
        available: true,
    }]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_context() {
        let ctx = ComputeContext::cpu_fallback();
        assert_eq!(ctx.backend, GpuBackend::Cpu);
        assert_eq!(ctx.workgroup_size, 1);
    }

    #[test]
    fn new_cpu() {
        let ctx = ComputeContext::new(GpuBackend::Cpu);
        assert_eq!(ctx.backend, GpuBackend::Cpu);
    }

    #[test]
    fn new_cuda_falls_back() {
        let ctx = ComputeContext::new(GpuBackend::Cuda);
        assert_eq!(ctx.backend, GpuBackend::Cpu);
    }

    #[test]
    fn new_metal_falls_back() {
        let ctx = ComputeContext::new(GpuBackend::Metal);
        assert_eq!(ctx.backend, GpuBackend::Cpu);
    }

    #[test]
    fn supports_all_kernels() {
        let ctx = ComputeContext::cpu_fallback();
        assert!(ctx.supports_kernel(KernelType::LinearInterpolation));
        assert!(ctx.supports_kernel(KernelType::GaussSeidel));
        assert!(ctx.supports_kernel(KernelType::Dct));
        assert!(ctx.supports_kernel(KernelType::Fista));
        assert!(ctx.supports_kernel(KernelType::TotalVariation));
    }

    #[test]
    fn kernel_count() {
        let ctx = ComputeContext::cpu_fallback();
        assert_eq!(ctx.kernel_count(), 5);
    }

    #[test]
    fn parallel_map_double() {
        let ctx = ComputeContext::cpu_fallback();
        let data = vec![1.0, 2.0, 3.0];
        let result = ctx.parallel_map(&data, |x| x * 2.0);
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - 4.0).abs() < 1e-12);
        assert!((result[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_map_empty() {
        let ctx = ComputeContext::cpu_fallback();
        let result = ctx.parallel_map(&[], |x| x);
        assert!(result.is_empty());
    }

    #[test]
    fn parallel_reduce_sum() {
        let ctx = ComputeContext::cpu_fallback();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sum = ctx.parallel_reduce(&data, 0.0, |a, b| a + b);
        assert!((sum - 10.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_reduce_empty() {
        let ctx = ComputeContext::cpu_fallback();
        let sum = ctx.parallel_reduce(&[], 42.0, |a, b| a + b);
        assert!((sum - 42.0).abs() < 1e-12);
    }

    #[test]
    fn matvec_identity() {
        let ctx = ComputeContext::cpu_fallback();
        let matrix = vec![1.0, 0.0, 0.0, 1.0];
        let vector = vec![3.0, 7.0];
        let result = ctx.matvec(&matrix, &vector, 2, 2);
        assert!((result[0] - 3.0).abs() < 1e-12);
        assert!((result[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn matvec_general() {
        let ctx = ComputeContext::cpu_fallback();
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![1.0, 1.0, 1.0];
        let result = ctx.matvec(&matrix, &vector, 2, 3);
        assert!((result[0] - 6.0).abs() < 1e-12);
        assert!((result[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn detect_backend_cpu() {
        assert_eq!(detect_best_backend(), GpuBackend::Cpu);
    }

    #[test]
    fn enumerate_has_cpu() {
        let devices = enumerate_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].backend, GpuBackend::Cpu);
        assert!(devices[0].available);
    }

    #[test]
    fn gpu_backend_eq() {
        assert_eq!(GpuBackend::Cpu, GpuBackend::Cpu);
        assert_ne!(GpuBackend::Cpu, GpuBackend::Cuda);
        assert_ne!(GpuBackend::Metal, GpuBackend::Vulkan);
    }

    #[test]
    fn kernel_type_eq() {
        assert_eq!(KernelType::Dct, KernelType::Dct);
        assert_ne!(KernelType::Dct, KernelType::Fista);
    }
}
