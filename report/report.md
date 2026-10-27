# Hybrid retrieval and reranking for RAG: a controlled study

## 1. Question and hypotheses
H1: hybrid > best single retriever on nDCG@10. H2: reranking adds a further gain but costs latency. (State what would falsify each.)

## 2. Setup
Dataset, split, #queries/#docs, systems and configs (link `configs/`), hardware, commit hash from `run.json`.

## 3. Retrieval results
Paste `report/results.md`. Discuss effect sizes and CIs, not just rank order.

## 4. Error analysis
Categorise the 50 failures from `analysis/errors_*.csv` (vocabulary mismatch, needs reasoning, label noise, wrong granularity). Table of counts per system.

## 5. Generation and faithfulness
Judge validation (n, accuracy, kappa) first, then faithfulness / citation metrics per system.

## 6. Limitations
Single dataset, small model sizes, unjudged relevant documents, judge bias, no stemming in BM25.

## 7. Reproduce
`make all`
