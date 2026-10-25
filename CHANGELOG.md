# Changelog

All notable changes to ALICE-History will be documented in this file.

## [Unreleased]

### Changed
- License: `AGPL-3.0-or-later` → `LicenseRef-Proprietary`; `LICENSE-Proprietary` holds the text
- CI: `ci/**` の branch への送信でも CI を走らせる
- preflight: `.github/preflight-exact` を置いた `scripts/preflight.sh` が生成器の出力 (見出しの `generator rev` の版) と byte で一致しなければ送る前の確認が通らない
- 設定 file の comment を中身の説明だけにし、.gitignore を project の build 出力だけにした
- TRADEMARK_NOTICE が指す licence の file を LICENSE-* の file にした

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
