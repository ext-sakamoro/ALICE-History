# ALICE-History

Inverse entropy restoration -- mathematically reversing information degradation to restore historical data (texts, images, artifacts) to their original state.

## Overview

ALICE-History implements a regularized iterative solver that fills in missing or degraded elements of historical fragments while minimizing the Shannon entropy of the restored result. Confidence scores indicate which restored values are well-supported by surrounding known data and which are speculative.

## Tests

The crate includes 30 tests covering fragment construction, entropy measurement, restoration correctness, confidence scoring, batch processing, hash determinism, and edge cases.

```bash
cargo test
```

## License

AGPL-3.0-or-later
