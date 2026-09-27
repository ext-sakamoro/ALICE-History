# Changelog

All notable changes to ALICE-History will be documented in this file.

## [Unreleased]

### Changed
- **License: `AGPL-3.0-or-later` → `AGPL-3.0-or-later OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ / ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`

## [0.1.0] - 2026-02-23

### Added
- `solver_1d` — 1D inverse entropy restoration (gradient descent + Tikhonov regularisation)
- `grid2d` — 2D Gauss-Seidel grid solver with 4/8-neighbour modes
- `frequency` — DCT-based POCS frequency-domain restoration
- `sparse` — ISTA/FISTA compressed sensing with soft thresholding
- `multimodal` — Multi-modal Bayesian fusion (Image, Text, Spatial, Spectral, Temporal)
- `core` — FNV-1a hashing, Shannon entropy, confidence maps (1D/2D)
- `Fragment`, `RestorationField`, `ConfidenceMap`, `InversionConfig` types
- `Strategy` enum with `Auto` mode for automatic solver selection
- `restore_advanced` unified entry point
- Rayon-parallel batch restoration
- 164 unit tests

### Fixed
- Loop variables only used as indices → iterator style (clippy)
- Doc list item indentation in solver_1d (clippy)
- `[i]` in doc comments interpreted as intra-doc links (frequency.rs)
