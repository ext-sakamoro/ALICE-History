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

`AGPL-3.0-or-later OR LicenseRef-Commercial` — dual-licensed. Pick either.

| Option | Terms | Use it when |
|--------|-------|-------------|
| **AGPL-3.0-or-later** | [LICENSE-AGPL](LICENSE-AGPL) — free, no reporting obligation | Your project is itself AGPL-compatible open source, or you are only using it internally |
| **Commercial License** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) — paid, removes the copyleft | Closed-source product, proprietary SaaS, edge / firmware distribution, plugin redistribution, or a platform NDA that forbids source disclosure |

AGPL is a strong copyleft: a product, firmware image, or service that links
`alice-history` and is distributed or served to users must be released under the AGPL
as well. That is intentional for the open ecosystem, and the Commercial
License exists for the cases where it is not something you are able to do.

Commercial licence enquiries: <contact@extoria.co.jp>
