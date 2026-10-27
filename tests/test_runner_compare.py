import json

from rag_study.compare import compare
from rag_study.runner import load_config, run_experiment


def test_run_and_compare(tmp_path):
    for c in ("smoke_bm25", "smoke_hybrid"):
        run_experiment(load_config(f"configs/{c}.yaml"), str(tmp_path))
    run = json.loads((tmp_path / "smoke_bm25" / "run.json").read_text())
    assert run["dataset"]["n_queries"] == 60 and run["summary"]["ndcg@10"] > 0.8
    assert (tmp_path / "smoke_bm25" / "ranking.json").exists()
    md = compare(str(tmp_path), "smoke_bm25", ["smoke_bm25", "smoke_hybrid"])
    assert "baseline" in md and "smoke_hybrid" in md and "p=" in md
