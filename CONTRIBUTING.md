# Contributing to ALICE-History

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **Inverse entropy restoration**: all solvers aim to reduce Shannon entropy of degraded data while preserving known values.
- **Confidence-aware**: every restoration produces a `ConfidenceMap` scoring each element's reliability.
- **Multi-strategy**: 1D linear, 2D Gauss-Seidel, DCT-POCS, ISTA/FISTA, and Bayesian fusion.
- **Deterministic**: same input always produces the same restored output (FNV-1a content hashing).
- **Rayon parallel**: batch operations use work-stealing parallelism.
- **Minimal dependencies**: only `rayon` for parallelism; all math is self-contained.
