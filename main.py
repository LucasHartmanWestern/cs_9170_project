# Main v2 script including parallelization for multiple runs on one GPU. Intended to be used on DRAC.

import os
import torch
import gc
import argparse
import numpy as np
import itertools
from training import Training
from helpers_main import _load_spec, build_exp_group

def run_specs(args):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device
    spec = _load_spec(args.spec) #loads and validates spec

    # # Seeds come from spec
    # seeds = spec["seeds"]
    # if not isinstance(seeds, list) or len(seeds) == 0:
    #     raise ValueError(f"spec['seeds'] must be a non-empty list. Got: {seeds}")


    # One exp_group for all seeds
    exp_group = build_exp_group(args.spec, spec)
    spec_base = os.path.basename(args.spec)
    spec_name = os.path.splitext(spec_base)[0]

    print(f"[main] spec={spec_base} device={device}")
    print(f"[main] exp_group={exp_group}")
    # print(f"[main] seeds={seeds}")
   
    
    # Prepare the principal variable lists in order
    principal_vars =  spec["permutations"]["principals"].get("variables", [])
    principal_lists = [spec["permutations"]["principals"][var] for var in principal_vars]
    principal_permutations = list(itertools.product(*principal_lists))
    done_flags = [False] * len(principal_permutations)
    
    #Parallelization or sequential execution
    if args.parallel:
        run_parallel_specs(args)
    else:
        # Run permutations sequentially
        for process_count, permutation in enumerate(principal_permutations):
            seed = permutation[0]
            process_label = f"Training process {process_count}"
            # For each principal variable, set spec[var] to the current value from the permutation
            procces_spec = spec.copy()
            for idx, var in enumerate(principal_vars):
                procces_spec[var] = permutation[idx]
     
            trainer = Training(
                exp_group=exp_group,
                spec_name=spec_name,
                spec=procces_spec,
                output_dir=args.output_dir,
                process_label=process_label,
                seed=seed,
                device=device
            )
            done = trainer()
            done_flags[process_count] = done

            #Clean up
            del trainer
            if torch.cuda.is_available() and "cuda" in device:
                gc.collect()
                torch.cuda.synchronize(torch.device(device))
                torch.cuda.empty_cache()

    if done_flags.count(False) > 0:
        print(f"[main] WARNING: only {done_flags.count(True)}/{len(seeds)} seeds completed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Path to a YAML experiment spec file")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (auto-falls back to cpu if no CUDA)")
    p.add_argument("--output_dir", default="training_runs", help="Directory to save results")
    p.add_argument("--parallel", default=False, action="store_true", help="Run in parallel mode")
    args = p.parse_args()


    run_specs(args)

if __name__ == "__main__":
    main()
