"""Run one experiment config end to end and write results/<name>/{run.json,ranking.json}."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

from .data import load_dataset
from .metrics import evaluate_run, summarize
from .retrieval import build_retriever


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(cfg: dict, out_root: str = "results", dataset=None) -> dict:
    k, depth = cfg.get("k", 10), max(cfg.get("depth", 100), cfg.get("k", 10))
    ds = dataset or load_dataset(cfg["dataset"], cfg.get("split", "test"),
                                 cfg.get("data_root", "data/raw"), cfg.get("max_queries"), cfg.get("seed", 0))
    retriever = build_retriever(cfg["retriever"])

    t = time.perf_counter()
    retriever.fit(ds.corpus)
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    ranking = retriever.search(ds.queries, depth)
    search_s = time.perf_counter() - t

    per_query = evaluate_run(ranking, ds.qrels, k)
    result = {
        "name": cfg["name"],
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10],
        "git_commit": _git_commit(),
        "dataset": {"name": ds.name, "split": ds.split, "max_queries": cfg.get("max_queries"),
                    "seed": cfg.get("seed", 0), "n_docs": len(ds.corpus), "n_queries": len(per_query[f"ndcg@{k}"])},
        "k": k,
        "summary": summarize(per_query),
        "timing": {"fit_s": round(fit_s, 3), "search_s": round(search_s, 3),
                   "ms_per_query_batched": round(1000 * search_s / max(len(ds.queries), 1), 3)},
        "per_query": per_query,
    }
    out = Path(out_root) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")
    return result
