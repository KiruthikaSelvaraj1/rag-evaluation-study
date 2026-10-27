"""Retrieval metrics, implemented from scratch (graded relevance, linear gain)."""
from __future__ import annotations

import math


def _pos(rel: dict) -> set:
    return {d for d, s in rel.items() if s > 0}


def recall_at_k(ranked: list, rel: dict, k: int) -> float:
    pos = _pos(rel)
    return len(pos & set(ranked[:k])) / len(pos) if pos else 0.0


def mrr_at_k(ranked: list, rel: dict, k: int) -> float:
    pos = _pos(rel)
    for i, d in enumerate(ranked[:k], 1):
        if d in pos:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list, rel: dict, k: int) -> float:
    dcg = sum(max(rel.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((s for s in rel.values() if s > 0), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_run(run: dict, qrels: dict, k: int) -> dict:
    """run: qid -> ranked [(doc_id, score)] or [doc_id]. Returns {metric: {qid: value}}."""
    out = {f"recall@{k}": {}, f"mrr@{k}": {}, f"ndcg@{k}": {}}
    for qid, rel in qrels.items():
        if qid not in run or not _pos(rel):
            continue
        ranked = [x[0] if isinstance(x, (tuple, list)) else x for x in run[qid]]
        out[f"recall@{k}"][qid] = recall_at_k(ranked, rel, k)
        out[f"mrr@{k}"][qid] = mrr_at_k(ranked, rel, k)
        out[f"ndcg@{k}"][qid] = ndcg_at_k(ranked, rel, k)
    return out


def summarize(per_query: dict) -> dict:
    return {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in per_query.items()}
