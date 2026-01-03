import argparse, json, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctab_repo", required=True)
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--label_col", required=True)
    ap.add_argument("--n_rows", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--categorical_cols_json", required=True)
    ap.add_argument("--mixed_cols_json", default="{}")
    ap.add_argument("--general_cols_json", default="[]")
    ap.add_argument("--integer_cols_json", default="[]")

    ap.add_argument("--filter_to_label", default=None)  # e.g. "1"
    ap.add_argument("--oversample_factor", type=int, default=5)
    args = ap.parse_args()

    seed_all(args.seed)

    # Import CTAB from repo
    import sys
    sys.path.insert(0, os.path.abspath(args.ctab_repo))
    from model.ctabgan import CTABGAN

    categorical_cols = json.loads(args.categorical_cols_json)
    mixed_cols = json.loads(args.mixed_cols_json)
    general_cols = json.loads(args.general_cols_json)
    integer_cols = json.loads(args.integer_cols_json)

    synth = CTABGAN(
        raw_csv_path=args.real_csv,
        test_ratio=0.20,
        categorical_columns=categorical_cols,
        log_columns=[],
        mixed_columns=mixed_cols,
        general_columns=general_cols,
        non_categorical_columns=[],
        integer_columns=integer_cols,
        problem_type={"Classification": args.label_col},
    )

    synth.fit()

    # CTAB often generates a fixed number; we oversample by looping if needed
    target_n = int(args.n_rows)
    want_filter = args.filter_to_label is not None
    oversample = max(1, int(args.oversample_factor))

    out_chunks = []
    while sum(len(c) for c in out_chunks) < target_n:
        df = synth.generate_samples()
        if args.label_col not in df.columns:
            raise RuntimeError(f"CTAB output missing label column: {args.label_col}. cols={list(df.columns)}")

        if want_filter:
            df = df[df[args.label_col].astype(str) == str(args.filter_to_label)]

        out_chunks.append(df)

        # if filtering yields nothing, avoid infinite loop
        if want_filter and len(df) == 0:
            # try again a few times; CTAB isn't conditional so this can happen
            if len(out_chunks) >= 3:
                break
            else:
                continue


        # prevent runaway if generate_samples is huge
        if not want_filter and sum(len(c) for c in out_chunks) > target_n * oversample:
            break

    if len(out_chunks) == 0:
        df_out = pd.DataFrame()
    else:
        df_out = pd.concat(out_chunks, ignore_index=True)

    if len(df_out) >= target_n:
        df_out = df_out.sample(n=target_n, random_state=args.seed).reset_index(drop=True)
    else:
        # pad if too small
        if len(df_out) > 0:
            need = target_n - len(df_out)
            df_out = pd.concat([df_out, df_out.sample(n=need, replace=True, random_state=args.seed)],
                               ignore_index=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
