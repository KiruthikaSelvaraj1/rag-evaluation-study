"""Dump retrieval failures to CSV for manual categorisation (the error-analysis step)."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def dump_failures(run_dir: str, ds, n: int, seed: int, out: str) -> int:
    run = json.loads((Path(run_dir) / "run.json").read_text())
    ranking = json.loads((Path(run_dir) / "ranking.json").read_text())
    k = run["k"]
    missed = sorted(q for q, v in run["per_query"][f"recall@{k}"].items() if v == 0)
    pick = sorted(random.Random(seed).sample(missed, min(n, len(missed))))
    title = lambda d: ds.corpus[d]["title"][:90] or ds.corpus[d]["text"][:90]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", "gold", "gold_rank", "top3_retrieved", "category", "notes"])
        for q in pick:
            gold = [d for d, s in ds.qrels[q].items() if s > 0]
            ids = [d for d, _ in ranking[q]]
            ranks = [ids.index(g) + 1 if g in ids else f">{len(ids)}" for g in gold]
            w.writerow([q, ds.queries[q], " | ".join(f"{g}: {title(g)}" for g in gold), ranks,
                        " | ".join(title(d) for d in ids[:3]), "", ""])
    return len(pick)
