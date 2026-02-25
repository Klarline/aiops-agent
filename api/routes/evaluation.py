"""Evaluation API routes for benchmark execution."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from evaluation.benchmark_runner import run_benchmark
from evaluation.report_generator import generate_report

router = APIRouter()


class BenchmarkRequest(BaseModel):
    seed: int = 42
    episodes: int = 3


@router.post("/run")
async def run_evaluation(req: BenchmarkRequest):
    """Trigger a benchmark run."""
    results = run_benchmark(
        seed=req.seed,
        episodes_per_scenario=req.episodes,
        max_steps=150,
    )
    generate_report(results)
    return {
        "aggregate": results["aggregate"],
        "per_scenario": results["per_scenario"],
        "total_episodes": results["total_episodes"],
    }


@router.get("/results")
async def get_results():
    """Get latest benchmark results."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "evaluation", "results", "benchmark_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": "No results found. Run benchmark first."}
