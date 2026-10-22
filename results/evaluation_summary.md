# Baseline vs. Adapter Evaluation

## Dataset alignment
- Baseline rows: **337**
- Adapter rows: **337**
- Matched unique IDs: **337**
- IDs only in baseline: **0**
- IDs only in adapter: **0**

## Metrics on matched IDs

| Metric | Baseline | Adapter (train prompt) |
|---|---:|---:|
| Exact match | 0.00% (0/337) | 49.85% (168/337) |
| Mean token F1 | 0.0618 | 0.5692 |
| Abstention rate | 0.00% | 0.00% |
| Mean reported confidence | 0.8729 | 0.7852 |
| Mean generation time (s) | 6.3528 | 3.3491 |

## Paired comparison
- Adapter minus baseline exact-match rate: **49.85%**
- Baseline not exact-match, adapter exact-match: **168**
- Baseline exact-match, adapter not exact-match: **0**

## Interpretation caveat
Exact match and token F1 are lexical reference-answer metrics, not clinically validated accuracy or a hallucination rate. Abstention is estimated from saved flags/text. Reported confidence is not necessarily a calibrated probability. This script analyses saved predictions and does not run inference.

Metrics are computed from saved JSONL predictions; no model inference is performed.
