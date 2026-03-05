// ALICE-History — UE5 C++ バインディング
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto
//
// 10 FFI 関数 (Fragment 3 + InversionConfig 2 + restore 1 + RestorationResult 4)

#pragma once
#include <cstdint>
#include <cstddef>
#include <utility>

// ── extern "C" ──────────────────────────────────────────────────────

struct AliceFragment;
struct AliceInversionConfig;
struct AliceRestorationResult;

extern "C"
{
    // Fragment (3)
    AliceFragment* alice_fragment_new(
        uint64_t id, uint8_t kind,
        const double* data_ptr, const double* mask_ptr,
        uint32_t len, uint64_t timestamp_ns);
    double alice_fragment_known_fraction(const AliceFragment* fragment);
    void   alice_fragment_destroy(AliceFragment* fragment);

    // InversionConfig (2)
    AliceInversionConfig* alice_config_default();
    void alice_config_destroy(AliceInversionConfig* config);

    // restore (1)
    AliceRestorationResult* alice_restore(
        const AliceFragment* fragment, const AliceInversionConfig* config);

    // RestorationResult (4)
    const double* alice_result_values_ptr(const AliceRestorationResult* result);
    uint32_t alice_result_values_len(const AliceRestorationResult* result);
    double   alice_result_entropy_before(const AliceRestorationResult* result);
    uint32_t alice_result_iterations(const AliceRestorationResult* result);
    void     alice_result_destroy(AliceRestorationResult* result);
}

// ── RAII ラッパー ────────────────────────────────────────────────────

namespace Alice { namespace History {

/// フラグメント種別。
enum class FragmentKind : uint8_t
{
    Text        = 0,
    Image       = 1,
    Artifact    = 2,
    Inscription = 3,
    Audio       = 4,
};

/// 歴史フラグメント RAII。
class Fragment final
{
    AliceFragment* p_ = nullptr;
public:
    Fragment(uint64_t id, FragmentKind kind,
             const double* data, const double* mask,
             uint32_t len, uint64_t timestamp_ns)
        : p_(alice_fragment_new(id, static_cast<uint8_t>(kind), data, mask, len, timestamp_ns)) {}
    ~Fragment() { if (p_) alice_fragment_destroy(p_); }

    Fragment(const Fragment&) = delete;
    Fragment& operator=(const Fragment&) = delete;
    Fragment(Fragment&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Fragment& operator=(Fragment&& o) noexcept
    { if (this != &o) { if (p_) alice_fragment_destroy(p_); p_ = std::exchange(o.p_, nullptr); } return *this; }

    double KnownFraction() const { return p_ ? alice_fragment_known_fraction(p_) : 0.0; }
    const AliceFragment* Ptr() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }
};

/// 復元設定 RAII。
class InversionConfig final
{
    AliceInversionConfig* p_ = nullptr;
public:
    InversionConfig() : p_(alice_config_default()) {}
    ~InversionConfig() { if (p_) alice_config_destroy(p_); }

    InversionConfig(const InversionConfig&) = delete;
    InversionConfig& operator=(const InversionConfig&) = delete;
    InversionConfig(InversionConfig&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    InversionConfig& operator=(InversionConfig&& o) noexcept
    { if (this != &o) { if (p_) alice_config_destroy(p_); p_ = std::exchange(o.p_, nullptr); } return *this; }

    const AliceInversionConfig* Ptr() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }
};

/// 復元結果 RAII。
class RestorationResult final
{
    AliceRestorationResult* p_ = nullptr;
public:
    explicit RestorationResult(AliceRestorationResult* ptr) : p_(ptr) {}
    ~RestorationResult() { if (p_) alice_result_destroy(p_); }

    RestorationResult(const RestorationResult&) = delete;
    RestorationResult& operator=(const RestorationResult&) = delete;
    RestorationResult(RestorationResult&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    RestorationResult& operator=(RestorationResult&& o) noexcept
    { if (this != &o) { if (p_) alice_result_destroy(p_); p_ = std::exchange(o.p_, nullptr); } return *this; }

    const double* ValuesPtr() const { return p_ ? alice_result_values_ptr(p_) : nullptr; }
    uint32_t ValuesLen()      const { return p_ ? alice_result_values_len(p_) : 0; }
    double EntropyBefore()    const { return p_ ? alice_result_entropy_before(p_) : 0.0; }
    uint32_t Iterations()     const { return p_ ? alice_result_iterations(p_) : 0; }

    explicit operator bool() const { return p_ != nullptr; }
};

/// フラグメントを復元する。
inline RestorationResult Restore(const Fragment& frag, const InversionConfig& cfg)
{
    return RestorationResult(alice_restore(frag.Ptr(), cfg.Ptr()));
}

}} // namespace Alice::History
