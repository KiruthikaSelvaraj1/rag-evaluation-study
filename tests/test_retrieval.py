from rag_study.data import make_toy
from rag_study.metrics import evaluate_run, summarize
from rag_study.retrieval import BM25, Hybrid, Rerank, Retriever, Vector, build_retriever

CORPUS = {
    "d1": {"title": "cats", "text": "cats purr and sleep all day"},
    "d2": {"title": "dogs", "text": "dogs bark and fetch sticks"},
    "d3": {"title": "birds", "text": "birds sing and fly south"},
}


def test_bm25_ranks_matching_doc_first():
    r = BM25()
    r.fit(CORPUS)
    out = r.search({"q": "dogs fetch"}, 3)["q"]
    assert out[0][0] == "d2" and len(out) == 3


class Fixed(Retriever):
    def __init__(self, order): self.order = order
    def fit(self, corpus): pass
    def search(self, queries, depth): return {q: [(d, 1.0) for d in self.order][:depth] for q in queries}


def test_rrf_math():
    h = Hybrid([Fixed(["a", "b", "c"]), Fixed(["b", "a", "c"])], rrf_k=60)
    out = dict(h.search({"q": "x"}, 3)["q"])
    assert out["a"] == out["b"] == 1 / 61 + 1 / 62
    assert out["c"] == 2 / 63


def test_rrf_weights_break_symmetry():
    h = Hybrid([Fixed(["a", "b"]), Fixed(["b", "a"])], weights=[1.0, 3.0])
    assert h.search({"q": "x"}, 2)["q"][0][0] == "b"


def test_rerank_reorders_head_keeps_tail():
    class Scorer:
        def predict(self, pairs): return [1.0 if "d3" in d else 0.0 for _, d in pairs]
    corpus = {k: {"title": "", "text": k} for k in ["d1", "d2", "d3", "d4"]}
    rr = Rerank(Fixed(["d1", "d2", "d3", "d4"]), Scorer(), candidates=3)
    rr.fit(corpus)
    assert [d for d, _ in rr.search({"q": "x"}, 4)["q"]] == ["d3", "d1", "d2", "d4"]


def test_vector_with_hashing_encoder_finds_exact_text():
    v = Vector(model="hashing", cache_dir=".cache/test_emb")
    v.fit(CORPUS)
    assert v.search({"q": "birds sing and fly south"}, 1)["q"][0][0] == "d3"


def test_toy_pipeline_quality_sanity():
    ds = make_toy()
    for cfg in [{"type": "bm25"}, {"type": "hybrid", "members": [{"type": "bm25"}, {"type": "vector", "model": "hashing"}]}]:
        r = build_retriever(cfg)
        r.fit(ds.corpus)
        s = summarize(evaluate_run(r.search(ds.queries, 20), ds.qrels, 10))
        assert s["recall@10"] > 0.9
