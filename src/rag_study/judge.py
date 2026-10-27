"""LLM faithfulness judge + validation against human labels (accuracy, Cohen's kappa)."""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from .generation import format_passages

SYSTEM = "You are a strict fact-checking judge."
PROMPT = """Decide whether EVERY factual statement in the ANSWER is supported by the PASSAGES.
Ignore the passage-id citations themselves; judge only whether the content is supported.

PASSAGES:
{passages}

ANSWER:
{answer}

Return ONLY JSON: {{"supported": true or false, "unsupported_claims": ["..."]}}"""


def parse_judgement(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        return bool(j["supported"]), list(j.get("unsupported_claims", []))
    except Exception:
        return None


def judge_one(llm, corpus, rec):
    return parse_judgement(llm.complete(
        PROMPT.format(passages=format_passages(corpus, rec["context"]), answer=rec["answer"]),
        system=SYSTEM, max_tokens=300))


def judge_records(llm, ds, records: dict) -> dict:
    """Adds 'faithful' / 'unsupported_claims' to non-abstained answers (abstentions are excluded)."""
    for rec in records.values():
        if rec["abstained"]:
            continue
        res = judge_one(llm, ds.corpus, rec)
        if res is not None:
            rec["faithful"], rec["unsupported_claims"] = res
    return records


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in (0, 1))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def make_label_sheet(results_root, systems, n, seed, out):
    pool = []
    for s in systems:
        g = json.loads((Path(results_root) / s / "generations.json").read_text())
        pool += [(s, q) for q, r in sorted(g.items()) if not r["abstained"]]
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cache = {s: json.loads((Path(results_root) / s / "generations.json").read_text()) for s in systems}
    from .data import load_dataset  # corpus text for the human to read
    run = json.loads((Path(results_root) / systems[0] / "run.json").read_text())["dataset"]
    ds = load_dataset(run["name"], run["split"], max_queries=run["max_queries"], seed=run["seed"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "qid", "query", "answer", "passages", "human_faithful"])
        for s, q in pick:
            r = cache[s][q]
            w.writerow([s, q, r["query"], r["answer"], format_passages(ds.corpus, r["context"]), ""])
    return len(pick)


def validate_judge(llm, results_root, labels_csv, ds) -> dict:
    gens, human, machine = {}, [], []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["human_faithful"].strip() not in ("0", "1"):
                continue
            s = row["system"]
            gens.setdefault(s, json.loads((Path(results_root) / s / "generations.json").read_text()))
            res = judge_one(llm, ds.corpus, gens[s][row["qid"]])
            if res is None:
                continue
            human.append(int(row["human_faithful"]))
            machine.append(int(res[0]))
    if not human:
        return {"n": 0}
    return {"n": len(human), "accuracy": sum(h == m for h, m in zip(human, machine)) / len(human),
            "kappa": cohens_kappa(human, machine), "human_positive_rate": sum(human) / len(human)}
