"""Diagnose judge vs human labels. Judge calls are cached, so this is free to rerun."""
import csv, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter, defaultdict
from pathlib import Path

from rag_study.cli import _ds_from_run
from rag_study.judge import judge_one
from rag_study.llm import LLM

def diagnose(llm, results="results", labels="data/human_labels/labels.csv", show=4):
    ds, gens, rows = None, {}, []
    for r in csv.DictReader(open(labels, encoding="utf-8-sig")):
        if r["human_faithful"] not in ("0", "1"):
            continue
        s = r["system"]
        if s not in gens:
            gens[s] = json.loads((Path(results) / s / "generations.json").read_text(encoding="utf-8"))
        ds = ds or _ds_from_run(Path(results) / s)
        rec = gens[s][r["qid"]]
        res = judge_one(llm, ds.corpus, rec)
        rows.append({"system": s, "qid": r["qid"], "human": int(r["human_faithful"]),
                     "judge": None if res is None else int(res[0]), "claims": [] if res is None else res[1],
                     "query": rec["query"], "answer": rec["answer"]})
    c = Counter((x["human"], x["judge"]) for x in rows)
    print("\nConfusion matrix (rows = human, columns = judge)")
    print("              judge=0  judge=1  unparsed")
    for h in (0, 1):
        print(f"  human={h}     {c[(h, 0)]:5d}    {c[(h, 1)]:5d}    {c[(h, None)]:5d}")
    print(f"\nJudge says 'supported' for {sum(x['judge'] == 1 for x in rows)}/{len(rows)} answers; "
          f"humans say faithful for {sum(x['human'] for x in rows)}/{len(rows)}.")
    by = defaultdict(list)
    for x in rows:
        by[x["system"]].append(x["human"])
    print("\nHUMAN faithful rate by system (your labels):")
    for s, v in by.items():
        print(f"  {s}: {sum(v)}/{len(v)} = {sum(v) / len(v):.2f}")
    bad = [x for x in rows if x["human"] == 0 and x["judge"] == 1][:show]
    print(f"\nExamples where YOU said unfaithful but the judge said supported ({len(bad)} shown):")
    for x in bad:
        print(f"\n- [{x['system']} q{x['qid']}] {x['query']}\n  ANSWER: {x['answer'][:350]}")
    return rows


if __name__ == "__main__":
    diagnose(LLM.from_config("configs/llm.yaml"))
