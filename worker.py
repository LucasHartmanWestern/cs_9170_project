# worker.py
import torch
from training import run_one_experiment  # define below

def worker(exp_specs, gpu_id):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    for spec, seed in exp_specs:
        run_one_experiment(spec, seed, device)
