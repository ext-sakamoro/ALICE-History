# Changelog

All notable changes to ALICE-History will be documented in this file.

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
