//! C FFI for ALICE-History
//!
//! Provides 10 `extern "C"` functions for Unity / UE5 / native integration.
//!
//! License: AGPL-3.0-or-later
//! Author: Moroya Sakamoto

use std::slice;

use crate::{Fragment, FragmentKind, InversionConfig, RestorationResult};

// ── ヘルパー ────────────────────────────────────────────────────────

fn fragment_kind_from_u8(v: u8) -> Option<FragmentKind> {
    match v {
        0 => Some(FragmentKind::Text),
        1 => Some(FragmentKind::Image),
        2 => Some(FragmentKind::Artifact),
        3 => Some(FragmentKind::Inscription),
        4 => Some(FragmentKind::Audio),
        _ => None,
    }
}

// ── Fragment (3) ────────────────────────────────────────────────────

/// `Fragment` を作成する。
///
/// # Safety
///
/// `data_ptr` は `len` 個の `f64` の有効なメモリであること。
/// `mask_ptr` は `len` 個の `f64` の有効なメモリであること。
/// 戻り値は `alice_fragment_destroy` で解放すること。
/// `kind`: Text=0, Image=1, Artifact=2, Inscription=3, Audio=4
#[no_mangle]
pub unsafe extern "C" fn alice_fragment_new(
    id: u64,
    kind: u8,
    data_ptr: *const f64,
    mask_ptr: *const f64,
    len: u32,
    timestamp_ns: u64,
) -> *mut Fragment {
    let kind = match fragment_kind_from_u8(kind) {
        Some(k) => k,
        None => return std::ptr::null_mut(),
    };
    let data = if data_ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(data_ptr, len as usize).to_vec()
    };
    let mask = if mask_ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(mask_ptr, len as usize).to_vec()
    };
    Box::into_raw(Box::new(Fragment::new(id, kind, data, mask, timestamp_ns)))
}

/// 既知データの割合を返す。
///
/// # Safety
///
/// `fragment` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_fragment_known_fraction(fragment: *const Fragment) -> f64 {
    if fragment.is_null() {
        return 0.0;
    }
    (*fragment).known_fraction()
}

/// `Fragment` を解放する。
///
/// # Safety
///
/// `fragment` は `alice_fragment_new` で取得したポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_fragment_destroy(fragment: *mut Fragment) {
    if !fragment.is_null() {
        drop(Box::from_raw(fragment));
    }
}

// ── InversionConfig (2) ─────────────────────────────────────────────

/// デフォルトの `InversionConfig` を作成する。
///
/// # Safety
///
/// 戻り値は `alice_config_destroy` で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_config_default() -> *mut InversionConfig {
    Box::into_raw(Box::new(InversionConfig::default()))
}

/// `InversionConfig` を解放する。
///
/// # Safety
///
/// `config` は `alice_config_default` で取得したポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_config_destroy(config: *mut InversionConfig) {
    if !config.is_null() {
        drop(Box::from_raw(config));
    }
}

// ── restore (1) ─────────────────────────────────────────────────────

/// 1Dフラグメントを復元する。
///
/// # Safety
///
/// `fragment` と `config` は有効なポインタであること。
/// 戻り値は `alice_result_destroy` で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_restore(
    fragment: *const Fragment,
    config: *const InversionConfig,
) -> *mut RestorationResult {
    if fragment.is_null() || config.is_null() {
        return std::ptr::null_mut();
    }
    let result = crate::restore(&*fragment, &*config);
    Box::into_raw(Box::new(result))
}

// ── RestorationResult (4) ───────────────────────────────────────────

/// 復元結果の値配列へのポインタを返す。
///
/// # Safety
///
/// `result` は有効なポインタであること。
/// 戻り値のポインタは `result` が有効な間のみ使用可能。
#[no_mangle]
pub unsafe extern "C" fn alice_result_values_ptr(result: *const RestorationResult) -> *const f64 {
    if result.is_null() {
        return std::ptr::null();
    }
    (*result).field.values.as_ptr()
}

/// 復元結果の値配列の長さを返す。
///
/// # Safety
///
/// `result` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_result_values_len(result: *const RestorationResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    (*result).field.values.len() as u32
}

/// 復元前のエントロピーを返す。
///
/// # Safety
///
/// `result` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_result_entropy_before(result: *const RestorationResult) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    (*result).field.entropy_before
}

/// 反復回数を返す。
///
/// # Safety
///
/// `result` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_result_iterations(result: *const RestorationResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    (*result).field.iterations
}

/// `RestorationResult` を解放する。
///
/// # Safety
///
/// `result` は `alice_restore` で取得したポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_result_destroy(result: *mut RestorationResult) {
    if !result.is_null() {
        drop(Box::from_raw(result));
    }
}

// ── テスト ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_fragment_lifecycle() {
        unsafe {
            let data = [10.0_f64, 0.0, 30.0];
            let mask = [1.0_f64, 0.0, 1.0];
            let frag = alice_fragment_new(1, 0, data.as_ptr(), mask.as_ptr(), 3, 0);
            assert!(!frag.is_null());

            let kf = alice_fragment_known_fraction(frag);
            assert!((kf - 2.0 / 3.0).abs() < 1e-12);

            alice_fragment_destroy(frag);
        }
    }

    #[test]
    fn test_restore_lifecycle() {
        unsafe {
            let data = [10.0_f64, 0.0, 30.0];
            let mask = [1.0_f64, 0.0, 1.0];
            let frag = alice_fragment_new(1, 0, data.as_ptr(), mask.as_ptr(), 3, 0);
            let config = alice_config_default();

            let result = alice_restore(frag, config);
            assert!(!result.is_null());

            let len = alice_result_values_len(result);
            assert_eq!(len, 3);

            let ptr = alice_result_values_ptr(result);
            assert!(!ptr.is_null());

            let iters = alice_result_iterations(result);
            assert!(iters > 0);

            alice_result_destroy(result);
            alice_config_destroy(config);
            alice_fragment_destroy(frag);
        }
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            assert_eq!(alice_fragment_known_fraction(ptr::null()), 0.0);
            alice_fragment_destroy(ptr::null_mut());

            alice_config_destroy(ptr::null_mut());

            assert!(alice_restore(ptr::null(), ptr::null()).is_null());

            assert!(alice_result_values_ptr(ptr::null()).is_null());
            assert_eq!(alice_result_values_len(ptr::null()), 0);
            assert_eq!(alice_result_entropy_before(ptr::null()), 0.0);
            assert_eq!(alice_result_iterations(ptr::null()), 0);
            alice_result_destroy(ptr::null_mut());
        }
    }

    #[test]
    fn test_invalid_kind() {
        unsafe {
            let data = [1.0_f64];
            let mask = [1.0_f64];
            let frag = alice_fragment_new(1, 255, data.as_ptr(), mask.as_ptr(), 1, 0);
            assert!(frag.is_null());
        }
    }
}
