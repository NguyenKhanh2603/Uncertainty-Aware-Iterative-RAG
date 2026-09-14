"""Run multiple Qwen2-VL probing-gate extraction jobs with one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from research.internal_state_rag.run_qwen2vl_probing_gate_states import (
    ResearchTextClient,
    main as run_gate_job,
)


DEFAULTS: dict[str, Any] = {
    "start": 0,
    "n": 100,
    "layers": "3,7,11,15,19,23,27",
    "projection_dim": 32,
    "max_new_tokens": 12,
    "max_answer_tokens": 12,
    "max_image_pixels": None,
    "compute_dtype": "bfloat16",
    "seed": 20260914,
    "feature_npz": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    return parser.parse_args()


def load_jobs(path: Path, model: str) -> list[argparse.Namespace]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("jobs", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No jobs found in {path}")
    required = {
        "feature_manifest",
        "questions",
        "corpus",
        "retrieval",
        "output",
    }
    jobs: list[argparse.Namespace] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Job {index} must be a JSON object")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Job {index} missing fields: {sorted(missing)}")
        values = {**DEFAULTS, **row, "model": model}
        for key in required | {"feature_npz"}:
            if values.get(key) is not None:
                values[key] = Path(values[key])
        jobs.append(argparse.Namespace(**values))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.jobs, args.model)
    compute_dtypes = {job.compute_dtype for job in jobs}
    if len(compute_dtypes) != 1:
        raise ValueError("All suite jobs must use the same compute dtype")
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[compute_dtypes.pop()]

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    if next(client.model.parameters()).dtype != compute_dtype:
        client.model.to(dtype=compute_dtype)

    completed = []
    for index, job in enumerate(jobs, start=1):
        print(
            f"[suite {index}/{len(jobs)}] {job.output} n={job.n} start={job.start}",
            flush=True,
        )
        output = run_gate_job(job, client=client)
        completed.append(
            {
                "output": str(job.output),
                "requested_queries": output["requested_queries"],
                "completed_queries": output["completed_queries"],
                "status": output["status"],
            }
        )
        torch.cuda.empty_cache()
    print(json.dumps({"status": "complete", "jobs": completed}, indent=2))


if __name__ == "__main__":
    main()
