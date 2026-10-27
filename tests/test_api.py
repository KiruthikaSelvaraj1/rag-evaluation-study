from fastapi.testclient import TestClient

from rag_study import api


def test_submit_and_poll(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    c = TestClient(api.app)
    r = c.post("/experiments", json={"name": "t1", "dataset": "toy", "retriever": {"type": "bm25"}})
    assert r.status_code == 202
    j = c.get(f"/experiments/{r.json()['id']}").json()
    assert j["status"] == "done" and j["summary"]["ndcg@10"] > 0.5
    assert c.get("/results").json()[0]["name"] == "t1"
    assert c.get("/experiments/nope").status_code == 404


def test_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    r = TestClient(api.app).post("/experiments", json={"name": "../evil", "dataset": "toy", "retriever": {"type": "bm25"}})
    assert r.status_code == 422
