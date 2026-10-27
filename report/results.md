Queries: 300 (common to all systems). Dataset: scifact. Cells: mean [95% bootstrap CI over queries].

| system | recall@10 | mrr@10 | ndcg@10 | ms/query | Delta nDCG vs baseline [CI], p |
|---|---|---|---|---|---|
| bm25 | 0.785 [0.738, 0.829] | 0.633 [0.583, 0.681] | 0.665 [0.618, 0.710] | 0.261 | baseline |
| vector | 0.836 [0.794, 0.876] | 0.682 [0.634, 0.728] | 0.713 [0.668, 0.755] | 10.995 | +0.048 [+0.012, +0.084], p=0.010 |
| hybrid | 0.828 [0.784, 0.869] | 0.680 [0.632, 0.726] | 0.711 [0.666, 0.754] | 11.136 | +0.046 [+0.023, +0.070], p=0.000 |
| hybrid_rerank | 0.822 [0.777, 0.864] | 0.663 [0.614, 0.711] | 0.694 [0.649, 0.738] | 3010.24 | +0.030 [-0.003, +0.062], p=0.072 |

p-values are uncorrected for multiple comparisons. ms/query is batched wall-clock on this machine, so compare relatively.