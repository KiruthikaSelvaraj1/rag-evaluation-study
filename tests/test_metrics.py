import math

import numpy as np
import pytest

from rag_study.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from rag_study.stats import bootstrap_ci, paired_bootstrap

REL = {"a": 1, "b": 1}


def test_recall():
    assert recall_at_k(["x", "a", "y"], REL, 3) == 0.5
    assert recall_at_k(["x", "a", "b"], REL, 2) == 0.5
    assert recall_at_k(["a", "b"], REL, 2) == 1.0


def test_mrr():
    assert mrr_at_k(["x", "y", "b"], REL, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "b"], REL, 2) == 0.0


def test_ndcg_perfect_and_hand_computed():
    assert ndcg_at_k(["a", "b", "x"], REL, 3) == pytest.approx(1.0)
    # only 'a' found at rank 2: dcg=1/log2(3); idcg=1+1/log2(3)
    expect = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "y"], REL, 3) == pytest.approx(expect)


def test_ndcg_graded():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["b", "a"], rel, 2) < ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)


def test_evaluate_run_skips_queries_without_positives():
    out = evaluate_run({"q1": [("a", 1.0)], "q2": [("a", 1.0)]}, {"q1": {"a": 1}, "q2": {"a": 0}}, 5)
    assert set(out["ndcg@5"]) == {"q1"}


def test_bootstrap():
    mean, lo, hi = bootstrap_ci(np.linspace(0, 1, 100), n_boot=2000)
    assert lo < mean < hi
    d, lo, hi, p = paired_bootstrap(np.ones(50), np.zeros(50), n_boot=2000)
    assert d == 1 and lo > 0 and p < 0.01
    _, _, _, p = paired_bootstrap(np.arange(50) % 2, np.arange(50) % 2, n_boot=2000)
    assert p == 1.0
