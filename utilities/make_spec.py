#!/usr/bin/env python3
"""
make_spec.py  —  Create experiment spec variants from a base spec.

Usage:
    python make_spec.py --base v7_census_hard_anchors --name v8_census_rc5 \
        --patch radius_clip=5.0 local_weights.sigma_calibration_factor=4.0

    python make_spec.py --base v7_census_hard_anchors --name v8_census_smoke \
        --smoke --patch radius_clip=5.0

    # Sweep: generate N variants over a single param
    python make_spec.py --base v7_census_hard_anchors --sweep radius_clip 2.0 3.0 5.0

Flags:
    --base      Base spec name (in experiment_specs/) or path to YAML
    --name      Output name (default: base + patch summary)
    --patch     key=value pairs (dot-notation for nested keys)
    --smoke     Set total_episodes=500, seeds=[42], SLURM time=1:00:00
    --medium    Set total_episodes=1500, seeds=[42], SLURM time=4:00:00
    --sweep     Generate one spec per value for a single key
    --time      Override SLURM time limit (e.g. 8:00:00)
    --dry-run   Print what would be created without writing files
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path
import yaml

SPEC_DIR = Path("experiment_specs")
SLURM_ACCOUNT = "def-mcapretz"


def load_base_spec(name_or_path: str) -> tuple[dict, str]:
    """Return (spec_dict, base_name)."""
    p = Path(name_or_path)
    if not p.exists():
        p = SPEC_DIR / name_or_path
        if not p.exists():
            p = SPEC_DIR / (name_or_path + ".yaml")
    if not p.exists():
        print(f"ERROR: base spec not found: {name_or_path}")
        sys.exit(1)
    spec = yaml.safe_load(p.read_text())
    base_name = p.stem
    return spec, base_name


def set_nested(d: dict, dotkey: str, value):
    """Set a nested key using dot-notation, coercing value type."""
    parts = dotkey.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur:
            cur[part] = {}
        cur = cur[part]
    leaf = parts[-1]

    # Type coercion: try to match existing type, then fall back to JSON parse
    existing = cur.get(leaf)
    if existing is not None:
        try:
            if isinstance(existing, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                value = int(float(value))
            elif isinstance(existing, float):
                value = float(value)
            elif isinstance(existing, list):
                value = json.loads(value)
            else:
                # Try JSON parse for null, lists, etc.
                try:
                    value = json.loads(value)
                except Exception:
                    pass  # keep as string
        except (ValueError, TypeError):
            pass
    else:
        # No existing value: try JSON parse
        try:
            value = json.loads(value)
        except Exception:
            pass

    cur[leaf] = value


def apply_patches(spec: dict, patches: list[str]) -> dict:
    s = copy.deepcopy(spec)
    for patch in patches:
        if "=" not in patch:
            print(f"WARNING: ignoring malformed patch '{patch}' (expected key=value)")
            continue
        key, val = patch.split("=", 1)
        set_nested(s, key.strip(), val.strip())
    return s


def patch_summary(patches: list[str]) -> str:
    """Short string summarising the patches, safe for filenames."""
    parts = []
    for p in patches:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        short_k = k.split(".")[-1]  # last segment only
        safe_v  = re.sub(r"[^a-zA-Z0-9._-]", "", v)
        parts.append(f"{short_k}{safe_v}")
    return "_".join(parts) if parts else "patched"


def apply_scale(spec: dict, scale: str) -> dict:
    """Apply smoke/medium episode count and seed reduction."""
    s = copy.deepcopy(spec)
    if scale == "smoke":
        s["total_episodes"] = 500
        s["seeds"] = [42]
    elif scale == "medium":
        s["total_episodes"] = 1500
        s["seeds"] = [42]
    return s


def slurm_time_for(spec: dict, override: str | None) -> str:
    if override:
        return override
    eps = spec.get("total_episodes", 6000)
    seeds_val = spec.get("seeds", [1])
    seeds = len(seeds_val) if seeds_val is not None else 1
    # In k-fold mode one spec = one fold = 1 run
    if spec.get("fold_idx") is not None:
        seeds = 1
    # Rough estimate: ~8s/ep on GPU × seeds, +20% buffer
    est_hours = (eps * seeds * 9) / 3600 * 1.2
    if est_hours <= 1.5:
        return "1:30:00"
    elif est_hours <= 4:
        return "4:00:00"
    elif est_hours <= 8:
        return "8:00:00"
    else:
        return "16:00:00"


def write_sh(name: str, slurm_time: str, dry_run: bool) -> Path:
    sh_path = SPEC_DIR / f"{name}.sh"
    content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --time={slurm_time}
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/{name}.out
#SBATCH --error=experiment_specs/logs/{name}.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate

mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/{name}.yaml --device cuda:0
"""
    if not dry_run:
        sh_path.write_text(content)
    return sh_path


def create_spec(name: str, spec: dict, slurm_time: str, dry_run: bool) -> None:
    yaml_path = SPEC_DIR / f"{name}.yaml"
    sh_path   = SPEC_DIR / f"{name}.sh"

    if not dry_run:
        yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        write_sh(name, slurm_time, dry_run=False)
        print(f"  Created: {yaml_path}")
        print(f"  Created: {sh_path}")
        print(f"  Submit:  sbatch {sh_path}")
    else:
        print(f"  [dry-run] Would create: {yaml_path}")
        print(f"  [dry-run] Would create: {sh_path}")
        print(f"  [dry-run] Spec preview:")
        print(yaml.safe_dump(spec, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description="Create experiment spec variants")
    ap.add_argument("--base",    required=True, help="Base spec name or path")
    ap.add_argument("--name",    default=None,  help="Output name (default: auto)")
    ap.add_argument("--patch",   nargs="*",     default=[], metavar="KEY=VAL",
                    help="Patches in dot-notation: key=value ...")
    ap.add_argument("--smoke",   action="store_true", help="500 eps, 1 seed, short time")
    ap.add_argument("--medium",  action="store_true", help="1500 eps, 1 seed")
    ap.add_argument("--sweep",      nargs="+",     default=None, metavar="KEY VAL ...",
                    help="KEY val1 val2 ... — generate one spec per value")
    ap.add_argument("--fold-sweep", type=int,      default=None, metavar="N_FOLDS",
                    help="Generate one spec per fold (fold_idx 0..N-1), patching fold_idx and n_folds")
    ap.add_argument("--time",    default=None,  help="Override SLURM time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_spec, base_name = load_base_spec(args.base)
    scale = "smoke" if args.smoke else ("medium" if args.medium else None)

    # ── Fold sweep mode ─────────────────────────────────────────────────────
    if args.fold_sweep:
        n_folds = args.fold_sweep
        for fold_idx in range(n_folds):
            patches = list(args.patch) + [
                f"fold_idx={fold_idx}",
                f"n_folds={n_folds}",
            ]
            spec = apply_patches(base_spec, patches)
            if scale:
                spec = apply_scale(spec, scale)
            suffix = "_smoke" if args.smoke else ("_medium" if args.medium else "")
            name = args.name or f"{base_name}_fold{fold_idx}of{n_folds}{suffix}"
            if args.name:
                name = f"{args.name}_fold{fold_idx}of{n_folds}{suffix}"
            slurm_time = slurm_time_for(spec, args.time)
            print(f"\n── {name}  [fold_idx={fold_idx}, n_folds={n_folds}]  time={slurm_time}")
            create_spec(name, spec, slurm_time, args.dry_run)
        return

    # ── Sweep mode ──────────────────────────────────────────────────────────
    if args.sweep:
        if len(args.sweep) < 2:
            print("--sweep requires KEY val1 [val2 ...]")
            sys.exit(1)
        sweep_key = args.sweep[0]
        sweep_vals = args.sweep[1:]
        short_key = sweep_key.split(".")[-1]

        for val in sweep_vals:
            patches = list(args.patch) + [f"{sweep_key}={val}"]
            spec = apply_patches(base_spec, patches)
            if scale:
                spec = apply_scale(spec, scale)
            safe_val = re.sub(r"[^a-zA-Z0-9._-]", "", val)
            suffix = f"_smoke" if args.smoke else ("_medium" if args.medium else "")
            name = args.name or f"{base_name}_{short_key}{safe_val}{suffix}"
            slurm_time = slurm_time_for(spec, args.time)
            print(f"\n── {name}  [{sweep_key}={val}]  time={slurm_time}")
            create_spec(name, spec, slurm_time, args.dry_run)
        return

    # ── Single spec mode ────────────────────────────────────────────────────
    spec = apply_patches(base_spec, args.patch)
    if scale:
        spec = apply_scale(spec, scale)

    suffix = "_smoke" if args.smoke else ("_medium" if args.medium else "")
    if args.name:
        name = args.name
    elif args.patch:
        name = f"{base_name}_{patch_summary(args.patch)}{suffix}"
    else:
        name = f"{base_name}{suffix}"

    slurm_time = slurm_time_for(spec, args.time)
    print(f"\n── {name}  time={slurm_time}  "
          f"eps={spec.get('total_episodes')}  seeds={spec.get('seeds')}")
    create_spec(name, spec, slurm_time, args.dry_run)


if __name__ == "__main__":
    main()
