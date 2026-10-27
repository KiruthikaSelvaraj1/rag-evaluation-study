"""Thin developer API: submit an experiment, poll for results.

Run: uvicorn rag_study.api:app --reload
Local dev tool: the request chooses which models get loaded, so don't expose it publicly.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runner import run_experiment

app = FastAPI(title="rag-retrieval-study")
JOBS: dict[str, dict] = {}


def results_dir() -> str:
    return os.environ.get("RESULTS_DIR", "results")


class ExperimentRequest(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    dataset: str = Field(default="scifact", pattern=r"^[A-Za-z0-9_-]+$")
    split: str = Field(default="test", pattern=r"^[A-Za-z0-9_-]+$")
    k: int = Field(default=10, ge=1, le=100)
    depth: int = Field(default=100, ge=1, le=1000)
    max_queries: int | None = Field(default=None, ge=1)
    retriever: dict


def _job(job_id: str, cfg: dict) -> None:
    JOBS[job_id] = {"status": "running", "name": cfg["name"]}
    try:
        res = run_experiment(cfg, out_root=results_dir())
        JOBS[job_id] = {"status": "done", "name": cfg["name"], "summary": res["summary"],
                        "timing": res["timing"], "dataset": res["dataset"]}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "name": cfg["name"], "error": str(e)}


@app.post("/experiments", status_code=202)
def submit(req: ExperimentRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "queued", "name": req.name}
    bg.add_task(_job, job_id, req.model_dump())
    return {"id": job_id, "status": "queued"}


@app.get("/experiments/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown experiment id")
    return JOBS[job_id]


@app.get("/results")
def saved_results():
    out = []
    for p in sorted(Path(results_dir()).glob("*/run.json")):
        r = json.loads(p.read_text())
        out.append({"name": r["name"], "dataset": r["dataset"]["name"], "summary": r["summary"]})
    return out
