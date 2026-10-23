# Evaluation summary (test split, n = 337)

Computed from the saved Kaggle predictions in results/*.jsonl (not live inference).

| Metric | Baseline | Adapter only | Adapter + abstention (tau=0.8) |
|---|---|---|---|
| Closed-question accuracy (n=165) | 65.5% | 76.4% | 64.2% |
| Open-question accuracy (n=172) | 39.5% | 43.0% | 19.8% |
| False "yes" on closed "no" (n=79) | 29.1% | 26.6% | 19.0% |
| Abstention rate | 0.0% | 0.0% | 48.4% |
| Selective accuracy (answered only) | 52.2% | 59.3% | 80.5% |
| Committed-error rate | 47.8% | 40.7% | 10.1% |
| Overall accuracy (abstain = wrong) | 52.2% | 59.3% | 41.5% |

Mean generation time per question: baseline 6.35 s, adapter 3.35 s.
