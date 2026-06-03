# Main v2 script including parallelization for multiple runs on one GPU. Intended to be used on DRAC.

import os
import sys
import torch
import gc
import argparse
import itertools
import torch.multiprocessing as mp
from pathlib import Path

_project_root = Path(__file__).parent
for _p in [str(_project_root), str(_project_root / 'utilities'), str(_project_root / 'FORGE')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training import Training
from spec_helpers import _load_spec, build_exp_group

# Workaround: Python 3.12 + certain PyTorch versions have an ABCMeta incompatibility
# in torch.autograd.profiler.record_function, which the DataLoader wraps every
# __next__ call with. Replacing it with a no-op is safe — it only disables profiling,
# not training. Runs in both parent and spawned worker processes (spawn re-imports).
class _NoOpRecordFunction:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
torch.autograd.profiler.record_function = _NoOpRecordFunction

def worker(exp_group, spec_name, spec, output_dir, seed, process_label, device):
    trainer = Training(
        exp_group=exp_group,
        spec_name=spec_name,
        spec=spec,
        output_dir=output_dir,
        process_label=process_label,
        seed=seed,
        device=device
    )
    done = trainer()
    del trainer
    if torch.cuda.is_available() and "cuda" in device:
        gc.collect()
        torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()
    return done

def _run_sequential(seeds, exp_group, spec_name, spec, args, device):
    done_flags = [False] * len(seeds)
    for process_count, seed in enumerate(seeds):
        process_label = f"Process {process_count}"
        trainer = Training(
            exp_group=exp_group,
            spec_name=spec_name,
            spec=spec,
            output_dir=args.output_dir,
            process_label=process_label,
            seed=seed,
            device=device
        )
        done_flags[process_count] = trainer()
        del trainer
        if torch.cuda.is_available() and "cuda" in device:
            gc.collect()
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()
    return done_flags

def run_specs(args):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device
    spec = _load_spec(args.spec)
    if args.output_dir is None:
        args.output_dir = spec.get("output_dir", "training_runs")

    exp_group = build_exp_group(args.spec, spec)
    spec_base = os.path.basename(args.spec)
    spec_name = os.path.splitext(spec_base)[0]

    print(f"[main] spec={spec_base} device={device}")
    print(f"[main] exp_group={exp_group}")

    # --- Backward compat: old flat format (no permutations block) ---
    if "permutations" not in spec:
        seeds = spec["seeds"]
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds}")
        print(f"[main] seeds={seeds}")
        done_flags = _run_sequential(seeds, exp_group, spec_name, spec, args, device)
        if done_flags.count(False) > 0:
            print(f"[main] WARNING: only {done_flags.count(True)}/{len(seeds)} seeds completed")
        return

    # --- New format: permutations block ---
    principal_vars = spec["permutations"]["principals"].get("variables", [])
    principal_lists = [spec["permutations"]["principals"][var] for var in principal_vars]
    principal_permutations = list(itertools.product(*principal_lists))
    max_parallel = int(spec.get("max_parallel", len(principal_permutations)))

    print(f"[main] permutations={len(principal_permutations)} max_parallel={max_parallel}")

    done_flags = [False] * len(principal_permutations)

    seed_idx = principal_vars.index('seed') if 'seed' in principal_vars else 0

    if args.parallel:
        for batch_start in range(0, len(principal_permutations), max_parallel):
            batch = principal_permutations[batch_start:batch_start + max_parallel]
            jobs = []
            for process_count, permutation in enumerate(batch, start=batch_start):
                seed = permutation[seed_idx]
                process_label = f"Process {process_count}"
                process_spec = spec.copy()
                for idx, var in enumerate(principal_vars):
                    process_spec[var] = permutation[idx]
                p = mp.Process(
                    target=worker,
                    args=(exp_group, spec_name, process_spec, args.output_dir, seed, process_label, device)
                )
                jobs.append((process_count, p))
                p.start()
            for process_count, p in jobs:
                p.join()
                done_flags[process_count] = True
    else:
        for process_count, permutation in enumerate(principal_permutations):
            seed = permutation[seed_idx]
            process_label = f"Process {process_count}"
            process_spec = spec.copy()
            for idx, var in enumerate(principal_vars):
                process_spec[var] = permutation[idx]
            trainer = Training(
                exp_group=exp_group,
                spec_name=spec_name,
                spec=process_spec,
                output_dir=args.output_dir,
                process_label=process_label,
                seed=seed,
                device=device
            )
            done_flags[process_count] = trainer()
            del trainer
            if torch.cuda.is_available() and "cuda" in device:
                gc.collect()
                torch.cuda.synchronize(torch.device(device))
                torch.cuda.empty_cache()

    if done_flags.count(False) > 0:
        print(f"[main] WARNING: only {done_flags.count(True)}/{len(principal_permutations)} permutations completed")


def main():
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Path to a YAML experiment spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu if no CUDA)")
    p.add_argument("--output_dir", default=None, help="Directory to save results (overrides spec output_dir if set)")
    p.add_argument("--parallel", action="store_true", help="Run in parallel mode")
    args = p.parse_args()
    run_specs(args)

if __name__ == "__main__":
    main()
