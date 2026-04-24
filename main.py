# Main v2 script including parallelization for multiple runs on one GPU. Intended to be used on DRAC.

import os
import numpy as np
import torch
import gc
import argparse
from training import Training
from helpers_main import _load_spec, build_exp_group, _seed_everything

def run_specs(args):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device
    spec = _load_spec(args.spec) #loads and validates spec

    # Seeds come from JSON
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds}")

    eo_guard_threshold = float(spec.get("eo_guard_threshold", 0.0))
    n_seeds_needed = len(seeds)

    # Build fallback pool: first 20 non-primary seeds (used when EO guard skips a seed)
    primary_set = set(int(s) for s in seeds)
    fallback_pool = [s for s in range(200) if s not in primary_set][:20]
    seed_queue = [int(s) for s in seeds] + fallback_pool

    # One exp_group for all seeds
    exp_group = build_exp_group(args.spec, spec)
    spec_base = os.path.basename(args.spec)
    spec_name = os.path.splitext(spec_base)[0]

    print(f"[main] spec={spec_base} device={device}")
    print(f"[main] exp_group={exp_group}")
    print(f"[main] seeds={seeds}")
    if eo_guard_threshold > 0.0:
        print(f"[main] eo_guard_threshold={eo_guard_threshold} (fallback pool: {fallback_pool[:5]}...)")

    #Parallelization or sequential execution
    if args.parallel:
        run_parallel_specs(args)
    else:
        # Run seeds sequentially; skip via EO guard and pull from fallback pool as needed
        completed = 0
        for seed in seed_queue:
            if completed >= n_seeds_needed:
                break
            # seed = int(seed) # already an int
            print(f"[main] ---- running seed={seed} ----")
            _seed_everything(seed)


            trainer = Training(
                exp_group=exp_group,
                spec_name=spec_name,
                spec=spec,
                output_dir=args.output_dir,
                seed=seed,
                device=device
            )

            result = trainer()

            if result == "eo_guard_skip":
                print(f"[main] seed={seed} skipped by EO guard — pulling from fallback pool")
                continue

            completed += 1

            #Clean up
            del trainer
            if torch.cuda.is_available() and "cuda" in device:
                gc.collect()
                torch.cuda.synchronize(torch.device(device))
                torch.cuda.empty_cache()

    if completed < n_seeds_needed:
        print(f"[main] WARNING: only {completed}/{n_seeds_needed} seeds completed — fallback pool exhausted")

    # def clean_up(trainer, device):
    #     del trainer
    #     if torch.cuda.is_available() and "cuda" in device:
    #         gc.collect()
    #         torch.cuda.synchronize(torch.device(device))
    #         torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Path to a JSON experiment spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu if no CUDA)")
    p.add_argument("--output_dir", default="training_runs", help="Directory to save results")
    p.add_argument("--parallel", default=False, action="store_true", help="Run in parallel mode")
    args = p.parse_args()


    run_specs(args)

if __name__ == "__main__":
    main()
