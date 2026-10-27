"""Grounded answer generation + deterministic citation metrics."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .retrieval import doc_text

SYSTEM = "You answer strictly from the provided passages and cite them."
PROMPT = """Answer the question or assess the claim using ONLY the passages below.
After every statement, cite supporting passage ids in square brackets, e.g. [123].
If the passages do not contain enough evidence, reply exactly: INSUFFICIENT EVIDENCE.
Keep the answer under 120 words.

Passages:
{passages}

Question/claim: {query}
Answer:"""
_CITE = re.compile(r"\[([^\[\]]+)\]")


def parse_citations(answer: str) -> list[str]:
    ids = []
    for m in _CITE.findall(answer):
        ids += [x.strip() for x in m.split(",") if x.strip()]
    return ids


def format_passages(corpus, doc_ids, max_chars=1200) -> str:
    return "\n".join(f"[{d}] {doc_text(corpus[d])[:max_chars]}" for d in doc_ids)


def sample_qids(qids, n: int, seed: int = 0) -> list[str]:
    qids = sorted(qids)
    return sorted(random.Random(seed).sample(qids, min(n, len(qids))))


def generate(llm, ds, ranking: dict, qids, k: int = 5, tag: str = "") -> dict:
    out = {}
    for qid in qids:
        ctx = [d for d, _ in ranking[qid][:k]]
        answer = llm.complete(PROMPT.format(passages=format_passages(ds.corpus, ctx), query=ds.queries[qid]),
                              system=SYSTEM, tag=tag).strip()
        cited = parse_citations(answer)
        valid = [c for c in cited if c in ctx]
        gold = {d for d, s in ds.qrels.get(qid, {}).items() if s > 0}
        out[qid] = {"query": ds.queries[qid], "answer": answer, "context": ctx, "cited": cited,
                    "valid_cited": valid, "abstained": "INSUFFICIENT EVIDENCE" in answer.upper(),
                    "cites_gold": any(c in gold for c in valid), "gold_in_context": any(d in gold for d in ctx)}
    return out


def summarize_generation(records: dict) -> dict:
    r = list(records.values())
    n = len(r) or 1
    answered = [x for x in r if not x["abstained"]]
    with_cites = [x for x in answered if x["cited"]]
    judged = [x for x in answered if "faithful" in x]
    return {
        "n": len(r),
        "abstain_rate": sum(x["abstained"] for x in r) / n,
        "valid_citation_rate": (sum(len(x["valid_cited"]) / len(x["cited"]) for x in with_cites) / len(with_cites)
                                if with_cites else 0.0),
        "cites_gold_rate": sum(x["cites_gold"] for x in r) / n,
        "gold_in_context_rate": sum(x["gold_in_context"] for x in r) / n,
        "faithful_rate": (sum(x["faithful"] for x in judged) / len(judged)) if judged else None,
        "n_judged": len(judged),
    }


def save(run_dir, records):
    d = Path(run_dir)
    (d / "generations.json").write_text(json.dumps(records, indent=1))
    (d / "generation_summary.json").write_text(json.dumps(summarize_generation(records), indent=1))
