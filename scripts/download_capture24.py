#!/usr/bin/env python3
"""
Download CAPTURE-24 subset and build feature cache for the fairness RL experiment.

Strategy:
  1. Download the full capture24.zip (~6.9 GB) from Oxford ORA.
  2. Extract only metadata + annotation dictionary + N selected participants.
  3. Compute 5-second sliding-window features (32-dim) for each participant.
  4. Save datasets/capture24/capture24_features_cache.npz (X, y, a, subject_ids).
  5. Delete zip + raw participant files to recover disk space.

Usage:
    python scripts/download_capture24.py --data-dir datasets/capture24

The resulting cache (~100-300 MB) is all that split_capture24() needs.

Default subset: 27 female + 13 male participants (40 total) out of 151.
Maintaining the ~2:1 F:M ratio of the full dataset while fitting comfortably on disk.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CAPTURE24_URL = (
    "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001"
    "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
)

# Walmsley2020 label → binary  (MVPA = positive class)
# NOTE: the actual label value is "moderate-vigorous", not "mvpa"
LABEL_MAP = {
    "sedentary":         0,
    "light":             0,
    "sleep":             0,   # sleep treated as inactive (maximises MVPA rarity)
    "moderate-vigorous": 1,
}

WINDOW_SAMPLES = 500   # 5 s at 100 Hz
STEP_SAMPLES   = 250   # 2.5 s step → 50% overlap

FEATURE_NAMES = [
    f"{ax}_{stat}"
    for ax in ("x", "y", "z", "vm")
    for stat in ("mean", "std", "min", "max", "rms", "p25", "p75", "iqr")
]  # 32 features


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _window_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute 32 statistical features from one 500-sample triplet."""
    vm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    feats: list[float] = []
    for sig in (x, y, z, vm):
        p25, p75 = float(np.percentile(sig, 25)), float(np.percentile(sig, 75))
        feats += [
            float(np.mean(sig)),
            float(np.std(sig)),
            float(np.min(sig)),
            float(np.max(sig)),
            float(np.sqrt(np.mean(sig ** 2))),  # rms
            p25,
            p75,
            p75 - p25,                           # iqr
        ]
    return np.array(feats, dtype=np.float32)


def process_participant(
    csv_gz_path: Path,
    label_dict: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read one participant CSV, slide windows, return (X, y).
    Returns None if the file is unreadable or too short.
    """
    try:
        df = pd.read_csv(csv_gz_path, compression="gzip", index_col="time",
                         parse_dates=True, low_memory=False)
    except Exception as exc:
        print(f"  [warn] Could not read {csv_gz_path.name}: {exc}")
        return None

    # Map raw annotation → Walmsley2020 label
    df = df.join(label_dict[["label:Walmsley2020"]], on="annotation")
    df["y_bin"] = df["label:Walmsley2020"].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["y_bin"])

    if len(df) < WINDOW_SAMPLES:
        print(f"  [warn] {csv_gz_path.name}: too few annotated rows ({len(df)})")
        return None

    x_arr   = df["x"].to_numpy(dtype=np.float32)
    y_arr   = df["y"].to_numpy(dtype=np.float32)
    z_arr   = df["z"].to_numpy(dtype=np.float32)
    lbl_arr = df["y_bin"].to_numpy(dtype=np.int32)

    starts = np.arange(0, len(x_arr) - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    if len(starts) == 0:
        return None

    X_windows = np.zeros((len(starts), 32), dtype=np.float32)
    y_windows = np.zeros(len(starts),       dtype=np.int8)

    for i, s in enumerate(starts):
        e = s + WINDOW_SAMPLES
        X_windows[i] = _window_features(x_arr[s:e], y_arr[s:e], z_arr[s:e])
        # Majority vote determines window label
        counts = np.bincount(lbl_arr[s:e])
        y_windows[i] = int(counts.argmax())

    return X_windows, y_windows


# ---------------------------------------------------------------------------
# Participant selection
# ---------------------------------------------------------------------------

def select_participants(
    meta: pd.DataFrame,
    n_female: int,
    n_male: int,
    seed: int,
) -> tuple[list, list]:
    """
    Select n_female female + n_male male participant IDs from metadata.
    Stratifies by age group to maintain demographic diversity.
    Returns (sel_female_ids, sel_male_ids).
    """
    sex_col = next(
        (c for c in meta.columns if c.lower() in ("sex", "gender")), None
    )
    id_col = next(
        (c for c in meta.columns
         if c.lower() in ("id", "participant", "subject", "pid", "participant_id")),
        meta.columns[0],
    )
    age_col = next(
        (c for c in meta.columns if "age" in c.lower()), None
    )

    if sex_col is None:
        raise ValueError(
            f"Could not find sex column in metadata. Columns: {list(meta.columns)}"
        )
    print(f"  Using id_col='{id_col}', sex_col='{sex_col}', age_col='{age_col}'")

    meta = meta.copy()
    meta["_sex"] = meta[sex_col].str.strip().str.lower()
    female_rows = meta[meta["_sex"].isin(["female", "f", "woman"])]
    male_rows   = meta[meta["_sex"].isin(["male",   "m", "man"])]

    print(f"  Metadata: {len(female_rows)} female, {len(male_rows)} male")

    rng = np.random.default_rng(seed)

    def _stratified_sample(rows: pd.DataFrame, n: int) -> list:
        if n >= len(rows):
            return rows[id_col].tolist()
        if age_col and age_col in rows.columns:
            # Sample proportionally from each age group
            groups = rows.groupby(age_col)[id_col].apply(list).to_dict()
            weights = {g: len(ids) / len(rows) for g, ids in groups.items()}
            selected = []
            remaining = n
            for g, ids in groups.items():
                quota = max(1, round(weights[g] * n))
                quota = min(quota, len(ids), remaining)
                chosen = rng.choice(ids, size=quota, replace=False).tolist()
                selected.extend(chosen)
                remaining -= quota
                if remaining <= 0:
                    break
            # Top up if rounding left us short
            already = set(selected)
            pool = [i for i in rows[id_col].tolist() if i not in already]
            if remaining > 0 and pool:
                extra = rng.choice(pool, size=min(remaining, len(pool)),
                                   replace=False).tolist()
                selected.extend(extra)
            return selected[:n]
        else:
            return rng.choice(rows[id_col].tolist(), size=n, replace=False).tolist()

    sel_female = _stratified_sample(female_rows, n_female)
    sel_male   = _stratified_sample(male_rows,   n_male)
    return sel_female, sel_male


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & preprocess CAPTURE-24 subset"
    )
    parser.add_argument("--data-dir",  default="datasets/capture24",
                        help="Target directory (default: datasets/capture24)")
    parser.add_argument("--n-female",  type=int, default=20,
                        help="Female participants to include (default: 20)")
    parser.add_argument("--n-male",    type=int, default=20,
                        help="Male participants to include (default: 20)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="RNG seed for participant selection (default: 42)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if capture24.zip already exists")
    parser.add_argument("--keep-raw",  action="store_true",
                        help="Keep per-participant CSV files + zip after processing")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path   = data_dir / "capture24.zip"
    cache_path = data_dir / "capture24_features_cache.npz"

    if cache_path.exists():
        print(f"[capture24] Feature cache already exists at {cache_path}.")
        print("  Delete it and re-run to rebuild.")
        return

    # ---- 1) Download --------------------------------------------------------
    if zip_path.exists():
        print(f"[capture24] Found existing {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB)")
    elif args.skip_download:
        print("[capture24] ERROR: zip not found and --skip-download set.")
        sys.exit(1)
    else:
        print(f"[capture24] Downloading capture24.zip (~6.9 GB) ...")
        print(f"  URL: {CAPTURE24_URL[:80]}...")
        # Use curl with a browser user-agent (ORA returns 403 for urllib's default UA)
        cmd = [
            "curl", "-L", "--progress-bar",
            "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "-o", str(zip_path),
            CAPTURE24_URL,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            zip_path.unlink(missing_ok=True)
            print("[capture24] ERROR: curl download failed.")
            sys.exit(1)
        print(f"\n[capture24] Downloaded → {zip_path} "
              f"({zip_path.stat().st_size/1e9:.2f} GB)")

    # ---- 2) Extract metadata + annotation dictionary -----------------------
    print("[capture24] Extracting metadata and annotation dictionary ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        meta_name = next(
            (n for n in all_names if "metadata" in n.lower() and n.endswith(".csv")),
            None,
        )
        dict_name = next(
            (n for n in all_names if "annotation-label-dictionary" in n.lower()),
            None,
        )

        if meta_name is None:
            print("[capture24] ERROR: metadata.csv not found in zip. "
                  "Archive contents (first 20):", all_names[:20])
            sys.exit(1)
        if dict_name is None:
            print("[capture24] ERROR: annotation-label-dictionary not found in zip.")
            sys.exit(1)

        zf.extract(meta_name, data_dir)
        zf.extract(dict_name, data_dir)

    # Resolve actual paths (may be nested, e.g. capture24/metadata.csv)
    meta_path = next(data_dir.rglob("metadata.csv"), None)
    dict_path = next(data_dir.rglob("annotation-label-dictionary.csv"), None)

    if meta_path is None or dict_path is None:
        print("[capture24] ERROR: Could not locate extracted metadata/dictionary.")
        sys.exit(1)

    # ---- 3) Select participants ---------------------------------------------
    meta = pd.read_csv(meta_path)
    print(f"[capture24] metadata.csv columns: {list(meta.columns)}")
    sel_female, sel_male = select_participants(meta, args.n_female, args.n_male, args.seed)
    selected = sel_female + sel_male
    print(f"[capture24] Selected {len(sel_female)}F + {len(sel_male)}M = {len(selected)} participants")

    # ---- 4) Extract selected participant files ------------------------------
    print("[capture24] Extracting selected participant files ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        for pid in selected:
            pid_str = str(pid)
            matches = [n for n in all_names if pid_str in n and n.endswith(".csv.gz")]
            if not matches:
                print(f"  [warn] No .csv.gz found for participant '{pid}'")
                continue
            zf.extract(matches[0], data_dir)

    # ---- 5) Load annotation dictionary -------------------------------------
    label_dict = pd.read_csv(dict_path, index_col=0)
    if "label:Walmsley2020" not in label_dict.columns:
        avail = [c for c in label_dict.columns if "walmsley" in c.lower()
                 or "label" in c.lower()]
        if avail:
            label_dict = label_dict.rename(columns={avail[0]: "label:Walmsley2020"})
        else:
            print(f"[capture24] ERROR: 'label:Walmsley2020' column not found. "
                  f"Available: {list(label_dict.columns)}")
            sys.exit(1)

    # ---- 6) Process features -----------------------------------------------
    print("[capture24] Extracting sliding-window features (5 s / 2.5 s step) ...")
    sel_female_set = set(sel_female)

    all_X:   list[np.ndarray] = []
    all_y:   list[np.ndarray] = []
    all_a:   list[np.ndarray] = []
    all_sid: list[np.ndarray] = []
    processed_ids: list       = []

    for pid_idx, pid in enumerate(selected):
        pid_str = str(pid)
        candidates = list(data_dir.rglob(f"*{pid_str}*.csv.gz"))
        if not candidates:
            print(f"  [warn] No extracted file for participant '{pid}'")
            continue

        result = process_participant(candidates[0], label_dict)

        if not args.keep_raw:
            candidates[0].unlink(missing_ok=True)

        if result is None:
            continue

        X_p, y_p = result
        is_female = int(pid in sel_female_set)
        mvpa_pct  = 100.0 * float(y_p.sum()) / len(y_p)

        all_X.append(X_p)
        all_y.append(y_p)
        all_a.append(np.full(len(y_p), is_female,  dtype=np.int8))
        all_sid.append(np.full(len(y_p), pid_idx,  dtype=np.int32))
        processed_ids.append(pid)

        print(f"  [{pid_idx+1:3d}/{len(selected)}] {pid_str}: "
              f"{len(y_p):>7,} windows, MVPA={mvpa_pct:5.1f}%, "
              f"sex={'F' if is_female else 'M'}")

    if not all_X:
        print("[capture24] ERROR: No participants processed successfully.")
        sys.exit(1)

    X   = np.concatenate(all_X,   axis=0)
    y   = np.concatenate(all_y,   axis=0).astype(np.int8)
    a   = np.concatenate(all_a,   axis=0).astype(np.int8)
    sid = np.concatenate(all_sid, axis=0).astype(np.int32)

    print(f"\n[capture24] Total windows : {len(X):,}")
    print(f"  MVPA (y=1)             : {int(y.sum()):,}  ({100.*y.mean():.1f}%)")
    print(f"  Female windows (a=1)   : {int((a==1).sum()):,}  ({100.*(a==1).mean():.1f}%)")
    print(f"  Female MVPA            : {int(((y==1)&(a==1)).sum()):,}")
    print(f"  Male MVPA              : {int(((y==1)&(a==0)).sum()):,}")
    female_mvpa_rate = ((y==1)&(a==1)).sum() / max(1, (a==1).sum())
    male_mvpa_rate   = ((y==1)&(a==0)).sum() / max(1, (a==0).sum())
    print(f"  Female MVPA rate       : {100.*female_mvpa_rate:.1f}%")
    print(f"  Male MVPA rate         : {100.*male_mvpa_rate:.1f}%")

    if female_mvpa_rate >= male_mvpa_rate:
        print(
            "\n[capture24] NOTE: Female MVPA rate >= Male MVPA rate in this subset.\n"
            "  Consider setting minority_id=0 (male) in experiment specs,\n"
            "  or using group-specific bias injection.\n"
            "  Check the smoke test results before running full experiments."
        )

    # ---- 7) Save cache ------------------------------------------------------
    np.savez_compressed(
        cache_path,
        X=X, y=y, a=a, subject_ids=sid,
        feature_names=np.array(FEATURE_NAMES),
    )
    sz_mb = cache_path.stat().st_size / 1e6
    print(f"\n[capture24] Cache saved → {cache_path} ({sz_mb:.1f} MB)")

    # ---- 8) Cleanup ----------------------------------------------------------
    if not args.keep_raw:
        print(f"[capture24] Deleting {zip_path} ...")
        zip_path.unlink(missing_ok=True)
        # Remove the nested capture24/ directory (contains metadata + dictionary)
        # Keep them so the user can inspect; only raw accel files are deleted above.

    print("[capture24] Done. Run experiments with  dataset_name='capture24'.")


if __name__ == "__main__":
    main()
