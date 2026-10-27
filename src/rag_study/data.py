"""Dataset loading: BEIR-format datasets plus a synthetic 'toy' set for offline tests."""
from __future__ import annotations

import csv
import json
import random
import zipfile

import requests
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    name: str
    split: str
    corpus: dict   # doc_id -> {"title", "text"}
    queries: dict  # qid -> text
    qrels: dict    # qid -> {doc_id: graded relevance}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _ensure_beir(name: str, root: Path) -> Path:
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / f"{name}.zip"
    try:
        with requests.get(BEIR_URL.format(name=name), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    except Exception as e:  # network / hosting changes
        raise RuntimeError(
            f"Could not download BEIR dataset '{name}' ({e}). Download it manually "
            f"and unzip so that {target}/corpus.jsonl exists."
        ) from e
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    return target


def load_beir(name: str, split: str, root: str) -> Dataset:
    d = _ensure_beir(name, Path(root))
    corpus = {r["_id"]: {"title": r.get("title", ""), "text": r.get("text", "")}
              for r in _read_jsonl(d / "corpus.jsonl")}
    all_q = {r["_id"]: r["text"] for r in _read_jsonl(d / "queries.jsonl")}
    qrels: dict = {}
    with open(d / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {q: all_q[q] for q in qrels if q in all_q}
    return Dataset(name, split, corpus, queries, qrels)


def make_toy(n_docs: int = 300, n_queries: int = 60, seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(600)]
    corpus, own = {}, {}
    for i in range(n_docs):
        words = rng.sample(vocab, 12)
        noise = rng.choices(vocab, k=20)
        corpus[f"d{i}"] = {"title": " ".join(words[:2]), "text": " ".join(words[2:] + noise)}
        own[f"d{i}"] = words
    queries, qrels = {}, {}
    for j, did in enumerate(rng.sample(sorted(corpus), n_queries)):
        queries[f"q{j}"] = " ".join(rng.sample(own[did], 5) + rng.choices(vocab, k=2))
        qrels[f"q{j}"] = {did: 1}
    return Dataset("toy", "test", corpus, queries, qrels)


def load_dataset(name: str, split: str = "test", root: str = "data/raw",
                 max_queries: int | None = None, seed: int = 0) -> Dataset:
    ds = make_toy(seed=seed) if name == "toy" else load_beir(name, split, root)
    if max_queries and len(ds.queries) > max_queries:
        keep = random.Random(seed).sample(sorted(ds.queries), max_queries)
        ds.queries = {q: ds.queries[q] for q in sorted(keep)}
        ds.qrels = {q: ds.qrels[q] for q in ds.queries}
    return ds
