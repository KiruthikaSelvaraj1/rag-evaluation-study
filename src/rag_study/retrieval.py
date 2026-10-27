"""Retrievers: BM25, dense vector, hybrid (RRF), cross-encoder reranker.

Every retriever: fit(corpus) then search(queries, depth) -> {qid: [(doc_id, score), ...]}.
Encoders / scorers are injectable so tests and smoke runs need no model downloads
(model="hashing" and model="overlap" are offline stand-ins, not real baselines).
"""
from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def doc_text(d: dict) -> str:
    return f"{d.get('title', '')} {d.get('text', '')}".strip()


def _topk(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(depth, len(scores))
    idx = np.argpartition(-scores, depth - 1)[:depth]
    return idx[np.argsort(-scores[idx], kind="stable")]


class Retriever:
    def fit(self, corpus: dict) -> None:
        raise NotImplementedError

    def search(self, queries: dict, depth: int) -> dict:
        raise NotImplementedError


class BM25(Retriever):
    """Okapi BM25 with Lucene-style IDF; regex tokenizer, no stemming or stopwords."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, corpus):
        self.ids = list(corpus)
        docs = [tokenize(doc_text(corpus[i])) for i in self.ids]
        self.dl = np.array([len(d) for d in docs], float)
        self.avg = max(self.dl.mean(), 1e-9)
        post = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                post[t].append((i, c))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in post.items()}
        self.post = {t: (np.array([i for i, _ in p]), np.array([c for _, c in p], float))
                     for t, p in post.items()}

    def _scores(self, q: str) -> np.ndarray:
        s = np.zeros(len(self.ids))
        for t in set(tokenize(q)):
            if t in self.post:
                idx, tf = self.post[t]
                s[idx] += self.idf[t] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / self.avg))
        return s

    def search(self, queries, depth):
        out = {}
        for qid, q in queries.items():
            s = self._scores(q)
            out[qid] = [(self.ids[i], float(s[i])) for i in _topk(s, depth)]
        return out


class HashingEncoder:
    """Offline test double: signed feature hashing of tokens. Not a real embedding model."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts, is_query=False):
        m = np.zeros((len(texts), self.dim), np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                h = zlib.crc32(tok.encode())
                m[r, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class SentenceTransformerEncoder:
    BGE_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str, query_prefix: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
        if query_prefix is None and "bge" in model.lower() and "-en" in model.lower():
            query_prefix = self.BGE_PREFIX
        self.prefix, self.bs = query_prefix or "", batch_size

    def encode(self, texts, is_query=False):
        if is_query and self.prefix:
            texts = [self.prefix + t for t in texts]
        return np.asarray(self.m.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                                        show_progress_bar=False), np.float32)


class Vector(Retriever):
    """Exact inner-product search over L2-normalised embeddings (= cosine)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", encoder=None,
                 query_prefix: str | None = None, cache_dir: str = ".cache/embeddings"):
        self.model, self.cache_dir = model, Path(cache_dir)
        self.encoder = encoder or (HashingEncoder() if model == "hashing"
                                   else SentenceTransformerEncoder(model, query_prefix))

    def fit(self, corpus):
        self.ids = list(corpus)
        texts = [doc_text(corpus[i]) for i in self.ids]
        key = hashlib.sha1((self.model + "\x00".join(self.ids) + "\x01".join(texts)).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            self.doc_emb = np.load(path)
        else:
            self.doc_emb = self.encoder.encode(texts, is_query=False)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, self.doc_emb)

    def search(self, queries, depth):
        qids = list(queries)
        q = self.encoder.encode([queries[i] for i in qids], is_query=True)
        out = {}
        for start in range(0, len(qids), 256):
            sims = q[start:start + 256] @ self.doc_emb.T
            for row, qid in zip(sims, qids[start:start + 256]):
                out[qid] = [(self.ids[i], float(row[i])) for i in _topk(row, depth)]
        return out


class Hybrid(Retriever):
    """Reciprocal Rank Fusion: score(d) = sum_m w_m / (rrf_k + rank_m(d))."""

    def __init__(self, members, rrf_k: int = 60, weights=None, candidate_depth: int = 100):
        self.members, self.rrf_k, self.candidate_depth = members, rrf_k, candidate_depth
        self.weights = weights or [1.0] * len(members)

    def fit(self, corpus):
        for m in self.members:
            m.fit(corpus)

    def search(self, queries, depth):
        runs = [m.search(queries, max(depth, self.candidate_depth)) for m in self.members]
        out = {}
        for qid in queries:
            fused = defaultdict(float)
            for run, w in zip(runs, self.weights):
                for rank, (d, _) in enumerate(run[qid], 1):
                    fused[d] += w / (self.rrf_k + rank)
            out[qid] = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:depth]
        return out


class OverlapScorer:
    """Offline test double for a cross-encoder: fraction of query tokens present in the doc."""

    def predict(self, pairs):
        return [len(set(tokenize(q)) & set(tokenize(d))) / (len(set(tokenize(q))) or 1) for q, d in pairs]


class CrossEncoderScorer:
    def __init__(self, model: str, batch_size: int = 32, max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.m, self.bs = CrossEncoder(model, max_length=max_length), batch_size

    def predict(self, pairs):
        return self.m.predict(pairs, batch_size=self.bs, show_progress_bar=False)


class Rerank(Retriever):
    """Re-scores the top `candidates` of a first-stage retriever; the tail keeps first-stage order."""

    def __init__(self, first_stage: Retriever, scorer, candidates: int = 50):
        self.first, self.scorer, self.candidates = first_stage, scorer, candidates

    def fit(self, corpus):
        self.corpus = corpus
        self.first.fit(corpus)

    def search(self, queries, depth):
        base = self.first.search(queries, max(depth, self.candidates))
        out = {}
        for qid, ranked in base.items():
            head, tail = ranked[:self.candidates], ranked[self.candidates:]
            scores = self.scorer.predict([(queries[qid], doc_text(self.corpus[d])) for d, _ in head])
            reranked = sorted(((d, float(s)) for (d, _), s in zip(head, scores)), key=lambda x: -x[1])
            out[qid] = (reranked + tail)[:depth]
        return out


def build_retriever(cfg: dict) -> Retriever:
    t = cfg["type"]
    if t == "bm25":
        return BM25(cfg.get("k1", 1.5), cfg.get("b", 0.75))
    if t == "vector":
        return Vector(cfg.get("model", "BAAI/bge-small-en-v1.5"), query_prefix=cfg.get("query_prefix"))
    if t == "hybrid":
        return Hybrid([build_retriever(m) for m in cfg["members"]], cfg.get("rrf_k", 60),
                      cfg.get("weights"), cfg.get("candidate_depth", 100))
    if t == "rerank":
        model = cfg["model"]
        scorer = OverlapScorer() if model == "overlap" else CrossEncoderScorer(model)
        return Rerank(build_retriever(cfg["first_stage"]), scorer, cfg.get("candidates", 50))
    raise ValueError(f"unknown retriever type: {t}")
