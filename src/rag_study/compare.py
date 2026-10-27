"""Compare runs with bootstrap CIs and paired tests against a baseline -> Markdown table."""
from __future__ import annotations

import json
from pathlib import Path

from .stats import bootstrap_ci, paired_bootstrap


def _load(root, name):
    return json.loads((Path(root) / name / "run.json").read_text())


def compare(results_root: str, baseline: str, names: list[str]) -> str:
    runs = {n: _load(results_root, n) for n in names}
    base = runs[baseline]
    k = base["k"]
    metrics = [f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    qids = sorted(set.intersection(*[set(r["per_query"][metrics[0]]) for r in runs.values()]))
    vec = lambda r, m: [r["per_query"][m][q] for q in qids]

    lines = [f"Queries: {len(qids)} (common to all systems). Dataset: {base['dataset']['name']}. "
             f"Cells: mean [95% bootstrap CI over queries].", ""]
    head = "| system | " + " | ".join(metrics) + " | ms/query | Delta nDCG vs baseline [CI], p |"
    lines += [head, "|" + "---|" * (len(metrics) + 3)]
    for n, r in runs.items():
        cells = []
        for m in metrics:
            mean, lo, hi = bootstrap_ci(vec(r, m))
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        if n == baseline:
            delta = "baseline"
        else:
            d, lo, hi, p = paired_bootstrap(vec(r, metrics[2]), vec(base, metrics[2]))
            delta = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.3f}"
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {r['timing']['ms_per_query_batched']} | {delta} |")
    lines += ["", "p-values are uncorrected for multiple comparisons. ms/query is batched wall-clock "
                  "on this machine, so compare relatively."]

    gens = [(n, Path(results_root) / n / "generation_summary.json") for n in names]
    gens = [(n, json.loads(p.read_text())) for n, p in gens if p.exists()]
    if gens:
        lines += ["", "### Generation", "", "| system | n | faithful (judge) | abstained | valid citations | cites gold | gold in context |",
                  "|---|---|---|---|---|---|---|"]
        for n, g in gens:
            f = g.get("faithful_rate")
            lines.append(f"| {n} | {g['n']} | {'-' if f is None else f'{f:.3f}'} | {g['abstain_rate']:.3f} | "
                         f"{g['valid_citation_rate']:.3f} | {g['cites_gold_rate']:.3f} | {g['gold_in_context_rate']:.3f} |")
    return "\n".join(lines)
