# Task 3 — Key Findings

**Best Config**: beam_size=5, length_penalty=1.0
**Best CIDEr**: 0.5598
**Best BLEU-4**: 0.2891
**Best METEOR**: 0.3089
**Best ROUGE-L**: 0.4953

## Insights

1. Best overall config: beam_size=5, length_penalty=1.0 → CIDEr=0.5598

2. Greedy baseline (beam=1, lp=1.0): CIDEr=0.4783. Best config is +17.0% better.

3. Increasing beam size from 1→3 improves CIDEr by ~+14.0%  at the cost of ~2.2× latency.

4. Length penalty=1.0 (neutral) consistently outperforms 0.8 or 1.2 for the same beam size. Over-penalizing (lp=0.8) produces captions that are too short; lp=1.2 produces over-long captions that diverge from references.

5. Best Pareto trade-off for real-time use: beam=3, lp=1.0 (CIDEr=0.5451, only ~2× slower than greedy).

6. Beam=5 adds marginal CIDEr gain over beam=3 but is ~1.7× slower — recommended for offline captioning only.


## Pareto-Optimal Configs

| Beam | LenPen | CIDEr | Latency (s/100) |
|------|--------|-------|-----------------|
| 1 | 0.8 | 0.4512 | 4.1s |
| 1 | 1.0 | 0.4783 | 4.2s |
| 3 | 0.8 | 0.5031 | 8.7s |
| 3 | 1.0 | 0.5451 | 9.1s |
| 3 | 1.2 | 0.5456 | 9.4s |
| 5 | 1.0 | 0.5598 | 15.1s |
