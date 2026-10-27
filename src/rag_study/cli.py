from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare
from .data import load_dataset
from .errors import dump_failures
from .generation import generate, sample_qids, save, summarize_generation
from .judge import judge_records, make_label_sheet, validate_judge
from .llm import LLM
from .runner import load_config, run_experiment


def _ds_from_run(run_dir):
    d = json.loads((Path(run_dir) / "run.json").read_text())["dataset"]
    return load_dataset(d["name"], d["split"], max_queries=d["max_queries"], seed=d["seed"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag_study")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run a retrieval experiment from a YAML config")
    s.add_argument("config"); s.add_argument("--out", default="results")

    s = sub.add_parser("compare", help="results table with CIs and paired tests")
    s.add_argument("--results", default="results"); s.add_argument("--baseline", required=True)
    s.add_argument("--names", required=True, help="comma-separated run names")
    s.add_argument("--out", default="report/results.md")

    s = sub.add_parser("generate", help="grounded answers from a run's top-k passages")
    s.add_argument("name"); s.add_argument("--results", default="results")
    s.add_argument("--llm", default="configs/llm.yaml"); s.add_argument("--n", type=int, default=100)
    s.add_argument("--gen-k", type=int, default=5); s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("judge", help="LLM-judge faithfulness of generated answers")
    s.add_argument("name"); s.add_argument("--results", default="results"); s.add_argument("--llm", default="configs/llm.yaml")

    s = sub.add_parser("label-sheet", help="CSV for hand-labelling faithfulness")
    s.add_argument("--results", default="results"); s.add_argument("--systems", required=True)
    s.add_argument("--n", type=int, default=100); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/human_labels/to_label.csv")

    s = sub.add_parser("validate", help="judge vs human labels: accuracy + Cohen's kappa")
    s.add_argument("--results", default="results"); s.add_argument("--llm", default="configs/llm.yaml")
    s.add_argument("--labels", default="data/human_labels/labels.csv"); s.add_argument("--system", required=True)

    s = sub.add_parser("errors", help="dump retrieval failures for manual categorisation")
    s.add_argument("name"); s.add_argument("--results", default="results")
    s.add_argument("--n", type=int, default=50); s.add_argument("--seed", type=int, default=0)

    a = p.parse_args(argv)
    rd = lambda name: Path(a.results) / name if hasattr(a, "results") else None

    if a.cmd == "run":
        r = run_experiment(load_config(a.config), a.out)
        print(r["name"], {k: round(v, 4) for k, v in r["summary"].items()}, r["timing"])
    elif a.cmd == "compare":
        md = compare(a.results, a.baseline, a.names.split(","))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
        print(md)
    elif a.cmd == "generate":
        run_dir = rd(a.name)
        ds = _ds_from_run(run_dir)
        ranking = json.loads((run_dir / "ranking.json").read_text())
        qids = sample_qids(ds.queries, a.n, a.seed)  # same sample for every system
        recs = generate(LLM.from_config(a.llm), ds, ranking, qids, k=a.gen_k)
        save(run_dir, recs)
        print(summarize_generation(recs))
    elif a.cmd == "judge":
        run_dir = rd(a.name)
        recs = json.loads((run_dir / "generations.json").read_text())
        recs = judge_records(LLM.from_config(a.llm), _ds_from_run(run_dir), recs)
        save(run_dir, recs)
        print(summarize_generation(recs))
    elif a.cmd == "label-sheet":
        print("rows:", make_label_sheet(a.results, a.systems.split(","), a.n, a.seed, a.out))
    elif a.cmd == "validate":
        print(validate_judge(LLM.from_config(a.llm), a.results, a.labels, _ds_from_run(Path(a.results) / a.system)))
    elif a.cmd == "errors":
        run_dir = rd(a.name)
        out = f"analysis/errors_{a.name}.csv"
        print("rows:", dump_failures(str(run_dir), _ds_from_run(run_dir), a.n, a.seed, out), "->", out)
