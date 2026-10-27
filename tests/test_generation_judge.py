import json

from rag_study.data import make_toy
from rag_study.generation import generate, parse_citations, summarize_generation
from rag_study.judge import cohens_kappa, judge_records, parse_judgement, validate_judge
from rag_study.llm import LLM


def test_parse_citations():
    assert parse_citations("A [d1]. B [d2, d3]. C [ d4 ]") == ["d1", "d2", "d3", "d4"]


def test_parse_judgement():
    assert parse_judgement('ok {"supported": true, "unsupported_claims": []}') == (True, [])
    assert parse_judgement("no json") is None


def test_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0


def _setup():
    ds = make_toy()
    qids = sorted(ds.queries)[:6]
    ranking = {q: [(d, 1.0) for d in list(ds.corpus)[:5]] for q in qids}
    return ds, qids, ranking


def test_generate_and_judge_with_mock_llm():
    ds, qids, ranking = _setup()

    def responder(prompt, system):
        if "fact-checking" in system:
            return '{"supported": false, "unsupported_claims": ["x"]}'
        return "INSUFFICIENT EVIDENCE" if "q0" in prompt else "Claim holds [d0] and [zzz]."

    llm = LLM("mock", cache_path=":memory:", responder=responder)
    recs = generate(llm, ds, ranking, qids, k=5)
    s = summarize_generation(recs)
    assert s["n"] == 6 and s["valid_citation_rate"] == 0.5
    recs = judge_records(llm, ds, recs)
    s = summarize_generation(recs)
    assert s["faithful_rate"] == 0.0 and s["n_judged"] == sum(not r["abstained"] for r in recs.values())


def test_llm_cache_avoids_second_call():
    calls = []
    llm = LLM("mock", cache_path=":memory:", responder=lambda p, s: calls.append(1) or "x")
    llm.complete("hi"); llm.complete("hi")
    assert len(calls) == 1


def test_validate_judge(tmp_path):
    ds, qids, ranking = _setup()
    llm = LLM("mock", cache_path=":memory:", responder=lambda p, s: "Claim [d0]." if "fact" not in s else '{"supported": true}')
    recs = generate(llm, ds, ranking, qids, k=5)
    (tmp_path / "sys").mkdir()
    (tmp_path / "sys" / "generations.json").write_text(json.dumps(recs))
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("system,qid,query,answer,passages,human_faithful\n" +
                        "".join(f"sys,{q},x,x,x,{i % 2}\n" for i, q in enumerate(qids)))
    out = validate_judge(llm, str(tmp_path), str(csv_path), ds)
    assert out["n"] == 6 and out["accuracy"] == 0.5
