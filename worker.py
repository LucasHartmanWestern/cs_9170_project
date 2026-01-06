# worker.py
import json
import torch
from spec_runner import run_one_experiment_from_spec

def worker(jobs_file: str, gpu_id: int):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[worker] device={device} jobs_file={jobs_file}")

    with open(jobs_file, "r") as f:
        jobs = json.load(f)

    for spec_path, seed in jobs:
        run_one_experiment_from_spec(spec_path, int(seed), device)
