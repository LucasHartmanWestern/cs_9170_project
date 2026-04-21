"""
check_run.py — Standard post-run analysis script.

Usage:
    python check_run.py <run_dir> [--interval 150] [--device cpu] [--no-gen-curve]

<run_dir> is the experiment directory containing seed_0/, seed_1/, etc.

Outputs (saved alongside the run directory):
    check_<run_name>/
        summary.txt          — printed summary table
        fig_learning.png     — episode return + EO per seed + mean band
        fig_gen_curve.png    — test-set EO / F1w / AUC vs episode (generalizability curve)

Requires the project virtualenv: source ~/envs/rl/bin/activate
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Standard post-run analysis")
    p.add_argument("run_dir", type=str, help="Experiment directory (contains seed_N/)")
    p.add_argument("--interval", type=int, default=150,
                   help="Episode interval for generalizability curve (default 150)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-gen-curve", action="store_true",
                   help="Skip generalizability curve (faster, skips FFNN retraining)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_seed_dirs(run_dir: Path):
    dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not dirs:
        sys.exit(f"No seed_* directories found in {run_dir}")
    return dirs


def load_meta(seed_dir: Path) -> dict:
    return load_json(seed_dir / "meta.json")


def load_best_checkpoint_info(seed_dir: Path) -> dict:
    """Return best checkpoint episode for phase1 (and phase2 if present)."""
    info = {}
    p1 = seed_dir / "best_beta_meta_phase1_class1.json"
    p2 = seed_dir / "best_beta_meta_phase2_class0.json"
    if p1.exists():
        d = load_json(p1)
        info["best_ep_phase1"] = d.get("episode", float("nan"))
        info["best_val_phase1"] = d.get("metric_value", float("nan"))
    else:
        info["best_ep_phase1"] = float("nan")
        info["best_val_phase1"] = float("nan")
    if p2.exists():
        d = load_json(p2)
        info["best_ep_phase2"] = d.get("episode", float("nan"))
    else:
        info["best_ep_phase2"] = float("nan")
    return info


def load_final_metrics(seed_dir: Path) -> pd.Series | None:
    p = seed_dir / "final_test_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return df.iloc[0]


def load_metrics_csv(seed_dir: Path) -> pd.DataFrame | None:
    p = seed_dir / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Use row index as global episode number (avoids phase-reset issue)
    df["global_ep"] = range(len(df))
    return df


def deadzone_pct(metrics_df: pd.DataFrame) -> float:
    """Fraction of phase-1 episodes where global_obj < 0.5."""
    phase_col = metrics_df.get("meta.phase", pd.Series(dtype=str))
    phase1 = metrics_df[phase_col.astype(str).str.startswith("phase1")]
    if len(phase1) == 0:
        # fallback: all rows
        phase1 = metrics_df
    col = "global.global_obj"
    if col not in phase1.columns:
        return float("nan")
    return float((phase1[col] < 0.5).mean() * 100)


# ---------------------------------------------------------------------------
# Dataset reconstruction
# ---------------------------------------------------------------------------

def reconstruct_dataset(meta: dict, device: str):
    """Re-instantiate Dataset from meta.json and call get_data_splits.
    Returns (dataset, x_train, x_val, x_test, y_train, y_val, y_test).
    dataset.a_test and dataset.pca_transform are available after this call.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import Dataset

    ds_name = meta["dataset_name"]
    seed    = int(meta["seed"])
    pca_k   = int(meta.get("pca_components", 10))
    min_id  = int(meta.get("minority_id", 1))
    maj_id  = int(meta.get("majority_id", 0))
    third   = meta.get("third_id", None)
    bias    = meta.get("BIAS_PCT", None)
    da_pct  = meta.get("DA_PCT", None)
    real_sz = meta.get("REAL_DATA_SIZE", None)
    win_s   = meta.get("win_seconds", 1.0)   # capture24 default
    step_s  = meta.get("step_seconds", 0.5)  # capture24 default
    dp_col  = meta.get("dp_protected_col", None)
    pool_pf = meta.get("pool_pos_fraction", None)

    dataset = Dataset(
        dataset_name=ds_name,
        multiclass=bool(meta.get("multiclass", False)),
        minority_id=min_id,
        majority_id=maj_id,
        third_id=third,
        pca_components=pca_k,
        seed=seed,
        device=device,
        use_pca=True,
    )

    kwargs = dict(
        train_size=real_sz,
        bias_pct=bias,
        da_pct=da_pct,
        pca_components=pca_k,
        drop_protected=False,
        protected_cols=dataset.protected_attributes,
        win_seconds=win_s,
        step_seconds=step_s,
    )
    if dp_col is not None:
        kwargs["dp_protected_col"] = dp_col
    if pool_pf is not None:
        kwargs["pool_pos_fraction"] = pool_pf

    splits = dataset.get_data_splits(**kwargs)
    x_train, x_val, x_test, y_train, y_val, y_test = splits
    return dataset, x_train, x_val, x_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# FFNN retraining for generalizability curve
# ---------------------------------------------------------------------------

def train_beta_on_synthetic(x_real, y_real, x_syn, y_syn, ffnn_cfg: dict, device: str):
    """Combine real + synthetic, train a fresh FFNN, return trained agent."""
    from agents.ffnn_agent2 import FFNNAgent

    x_comb = torch.cat([x_real, x_syn], dim=0)
    y_comb = torch.cat([y_real, y_syn], dim=0)

    ds = TensorDataset(x_comb, y_comb)
    loader = DataLoader(ds, batch_size=ffnn_cfg["batch_size"], shuffle=True)

    agent = FFNNAgent(
        input_size=x_comb.shape[1],
        hidden_sizes=ffnn_cfg["hidden_sizes"],
        output_size=1,
        learning_rate=ffnn_cfg["lr"],
        batch_size=ffnn_cfg["batch_size"],
        epochs=ffnn_cfg["epochs"],
        device=device,
        seed=ffnn_cfg["seed"],
    )
    agent.train(loader)
    return agent


def eval_on_test(agent, x_test, y_test, a_test, device: str) -> dict:
    """Evaluate agent on test set. Returns EO, F1w, AUC."""
    from sklearn.metrics import roc_auc_score, f1_score
    agent.model.eval()
    with torch.no_grad():
        logits = agent.model(x_test.to(device)).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
    y_np = y_test.cpu().numpy()
    a_np = a_test.cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    # AUC
    try:
        auc = roc_auc_score(y_np, probs)
    except Exception:
        auc = float("nan")

    # F1 weighted
    f1w = f1_score(y_np, preds, average="weighted", zero_division=0)

    # EO (TPR gap)
    eo = float("nan")
    try:
        groups = np.unique(a_np)
        if len(groups) >= 2:
            tprs = []
            for g in groups:
                mask = (a_np == g) & (y_np == 1)
                if mask.sum() > 0:
                    tprs.append(preds[mask].mean())
            if len(tprs) == 2:
                eo = abs(tprs[0] - tprs[1])
    except Exception:
        pass

    return {"eo": eo, "f1w": f1w, "auc": auc}


def snapshot_episodes(snap_dir: Path, phase: str) -> list[int]:
    """Return sorted list of episode numbers for which snapshots exist."""
    suffix = f"_{phase}.npz"
    eps = []
    for f in snap_dir.glob(f"synthetic_ep*{suffix}"):
        try:
            ep = int(f.stem.split("_ep")[1].split("_")[0])
            eps.append(ep)
        except Exception:
            pass
    return sorted(eps)


def load_snapshot(snap_dir: Path, ep: int, phase: str, device: str):
    """Load synthetic snapshot, return (x_tensor, y_tensor)."""
    fname = snap_dir / f"synthetic_ep{ep:04d}_{phase}.npz"
    if not fname.exists():
        return None, None
    d = np.load(fname)
    x = torch.tensor(d["x"], dtype=torch.float32, device=device)
    y = torch.tensor(d["y"], dtype=torch.float32, device=device)
    return x, y


# ---------------------------------------------------------------------------
# Generalizability curve for one seed
# ---------------------------------------------------------------------------

def gen_curve_one_seed(seed_dir: Path, meta: dict, interval: int, device: str) -> pd.DataFrame:
    """
    For each episode at [interval, 2*interval, ...], load the phase1 snapshot,
    retrain beta on real+synthetic, evaluate on test set.
    Returns DataFrame with columns [episode, eo, f1w, auc].
    """
    snap_dir = seed_dir / "synthetic_snapshots"
    if not snap_dir.exists():
        return pd.DataFrame()

    available = snapshot_episodes(snap_dir, "phase1_class1")
    if not available:
        return pd.DataFrame()

    max_ep = max(available)
    target_eps = list(range(interval, max_ep + 1, interval))
    # snap every 5 eps; find closest available
    eval_eps = []
    for t in target_eps:
        closest = min(available, key=lambda e: abs(e - t))
        if abs(closest - t) <= interval // 2:
            eval_eps.append(closest)
    eval_eps = sorted(set(eval_eps))

    if not eval_eps:
        return pd.DataFrame()

    # Reconstruct dataset once per seed
    print(f"    Reconstructing dataset for seed {meta['seed']}...")
    try:
        dataset, x_train, x_val, x_test, y_train, y_val, y_test = reconstruct_dataset(meta, device)
    except Exception as e:
        print(f"    WARNING: dataset reconstruction failed: {e}")
        return pd.DataFrame()

    a_test = getattr(dataset, "a_test", None)
    if a_test is None:
        print("    WARNING: a_test not available, EO will be NaN")

    ffnn_cfg = {
        "hidden_sizes": [32, 16],
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "seed": int(meta["seed"]),
    }

    rows = []
    for ep in eval_eps:
        x_syn, y_syn = load_snapshot(snap_dir, ep, "phase1_class1", device)
        if x_syn is None:
            continue
        try:
            agent = train_beta_on_synthetic(x_train, y_train.float(), x_syn, y_syn.float(), ffnn_cfg, device)
            metrics = eval_on_test(agent, x_test, y_test, a_test, device)
            rows.append({"episode": ep, **metrics})
        except Exception as e:
            print(f"    WARNING: ep {ep} failed: {e}")
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(seed_dfs: list[tuple[int, pd.DataFrame]], out_path: Path):
    """Learning curves: episode return and validation EO per seed + mean band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cols = {
        "meta.episode_return": ("Episode Return", axes[0]),
        "fairness.eo_tpr_diff": ("Val EO Gap", axes[1]),
    }

    for col, (ylabel, ax) in cols.items():
        all_curves = []
        for seed, df in seed_dfs:
            if col not in df.columns:
                continue
            phase_col = df.get("meta.phase", pd.Series("phase1_class1", index=df.index))
            phase1 = df[phase_col.astype(str).str.startswith("phase1")]
            x = phase1["global_ep"].values
            y = phase1[col].values
            all_curves.append((x, y))

        if all_curves:
            min_len = min(len(c[1]) for c in all_curves)
            ys = np.array([c[1][:min_len] for c in all_curves])
            xs = all_curves[0][0][:min_len]
            mean = ys.mean(axis=0)
            ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), alpha=0.2, color="steelblue", label="seed range")
            ax.plot(xs, mean, color="steelblue", linewidth=1.8, label="mean")

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Learning Curves (Validation)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gen_curves(seed_curves: list[tuple[int, pd.DataFrame]], out_path: Path):
    """Generalizability curves: test-set EO / F1w / AUC vs episode."""
    metrics = [("eo", "EO Gap (test) ↓"), ("f1w", "F1-Weighted (test) ↑"), ("auc", "AUC (test) ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for (col, ylabel), ax in zip(metrics, axes):
        all_curves = []
        for seed, df in seed_curves:
            if df.empty or col not in df.columns:
                continue
            x = df["episode"].values
            y = df[col].values
            all_curves.append((x, y))

        if all_curves:
            min_len = min(len(c[1]) for c in all_curves)
            ys = np.array([c[1][:min_len] for c in all_curves])
            xs = all_curves[0][0][:min_len]
            mean = ys.mean(axis=0)
            ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), alpha=0.2, color="steelblue", label="seed range")
            ax.plot(xs, mean, color="steelblue", linewidth=2.0, marker="o", markersize=4, label="mean")

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7)

    fig.suptitle("Generalizability Curve (Test Set)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(seed_rows: list[dict]) -> str:
    lines = []
    header = (
        f"{'Seed':>5}  {'α-EO':>7}  {'β-EO':>7}  {'EO-Δ':>7}  "
        f"{'α-F1w':>7}  {'β-F1w':>7}  {'F1w-Δ':>7}  "
        f"{'α-AUC':>7}  {'β-AUC':>7}  {'AUC-Δ':>7}  "
        f"{'Dead%':>6}  {'BestEp-P1':>10}  {'BestEp-P2':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in seed_rows:
        def fmt(v):
            return f"{v:7.4f}" if isinstance(v, float) and not np.isnan(v) else "    nan"
        def fmti(v):
            return f"{int(v):10d}" if isinstance(v, (int, float)) and not np.isnan(float(v)) else "       nan"

        lines.append(
            f"{r['seed']:>5}  {fmt(r['a_eo'])}  {fmt(r['b_eo'])}  {fmt(r['delta_eo'])}  "
            f"{fmt(r['a_f1w'])}  {fmt(r['b_f1w'])}  {fmt(r['delta_f1w'])}  "
            f"{fmt(r['a_auc'])}  {fmt(r['b_auc'])}  {fmt(r['delta_auc'])}  "
            f"{r['dead_pct']:6.1f}  {fmti(r['best_ep_p1'])}  {fmti(r['best_ep_p2'])}"
        )

    lines.append("-" * len(header))

    # Aggregate
    def agg(key):
        vals = [r[key] for r in seed_rows if not np.isnan(float(r[key]))]
        if not vals:
            return float("nan"), float("nan")
        return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    improved = sum(1 for r in seed_rows if r["delta_eo"] < 0)
    total    = len(seed_rows)

    def fmtm(v, s):
        if np.isnan(v):
            return "    nan"
        return f"{v:6.4f}±{s:.4f}"

    a_eo_m, a_eo_s   = agg("a_eo")
    b_eo_m, b_eo_s   = agg("b_eo")
    d_eo_m, d_eo_s   = agg("delta_eo")
    a_f1_m, a_f1_s   = agg("a_f1w")
    b_f1_m, b_f1_s   = agg("b_f1w")
    d_f1_m, d_f1_s   = agg("delta_f1w")
    a_auc_m, a_auc_s = agg("a_auc")
    b_auc_m, b_auc_s = agg("b_auc")
    d_auc_m, d_auc_s = agg("delta_auc")
    dead_m, dead_s   = agg("dead_pct")

    lines.append(
        f"\nMean±std:  α-EO={fmtm(a_eo_m, a_eo_s)}  β-EO={fmtm(b_eo_m, b_eo_s)}  "
        f"EO-Δ={fmtm(d_eo_m, d_eo_s)}"
    )
    lines.append(
        f"           α-F1w={fmtm(a_f1_m, a_f1_s)}  β-F1w={fmtm(b_f1_m, b_f1_s)}  "
        f"F1w-Δ={fmtm(d_f1_m, d_f1_s)}"
    )
    lines.append(
        f"           α-AUC={fmtm(a_auc_m, a_auc_s)}  β-AUC={fmtm(b_auc_m, b_auc_s)}  "
        f"AUC-Δ={fmtm(d_auc_m, d_auc_s)}"
    )
    lines.append(f"           Deadzone={dead_m:.1f}%±{dead_s:.1f}%")
    lines.append(f"           Seeds improved: {improved}/{total}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    seed_dirs = find_seed_dirs(run_dir)
    print(f"Run:   {run_dir.name}")
    print(f"Seeds: {[d.name for d in seed_dirs]}")
    print(f"Out:   {out_dir}")

    # Print key config from first seed's meta.json
    try:
        first_meta = load_meta(seed_dirs[0])
        k    = first_meta.get("global_sigmoid_k", "?")
        bias = first_meta.get("BIAS_PCT", None)
        da   = first_meta.get("DA_PCT", None)
        eps  = first_meta.get("EPISODES", "?")
        scarcity = f"da_pct={da}" if da is not None else f"bias_pct={bias}"
        print(f"Config: global_sigmoid_k={k}  {scarcity}  episodes={eps}")
    except Exception:
        pass
    print()

    seed_rows   = []
    metrics_dfs = []   # (seed_int, df) for learning curves
    gen_curves  = []   # (seed_int, df) for gen curve

    for sd in seed_dirs:
        seed_int = int(sd.name.replace("seed_", ""))
        print(f"  Processing {sd.name}...")

        meta    = load_meta(sd)
        ckpt    = load_best_checkpoint_info(sd)
        final   = load_final_metrics(sd)
        mdf     = load_metrics_csv(sd)

        if final is not None:
            a_eo  = float(final.get("alpha_eo_tpr_diff", float("nan")))
            b_eo  = float(final.get("beta_eo_tpr_diff",  float("nan")))
            a_f1w = float(final.get("alpha_f1_weighted", float("nan")))
            b_f1w = float(final.get("beta_f1_weighted",  float("nan")))
            a_auc = float(final.get("alpha_roc_auc",     float("nan")))
            b_auc = float(final.get("beta_roc_auc",      float("nan")))
        else:
            print(f"    WARNING: no final_test_metrics.csv found")
            a_eo = b_eo = a_f1w = b_f1w = a_auc = b_auc = float("nan")

        dead = deadzone_pct(mdf) if mdf is not None else float("nan")

        seed_rows.append({
            "seed":       seed_int,
            "a_eo":       a_eo,
            "b_eo":       b_eo,
            "delta_eo":   b_eo  - a_eo,
            "a_f1w":      a_f1w,
            "b_f1w":      b_f1w,
            "delta_f1w":  b_f1w - a_f1w,
            "a_auc":      a_auc,
            "b_auc":      b_auc,
            "delta_auc":  b_auc - a_auc,
            "dead_pct":   dead,
            "best_ep_p1": ckpt["best_ep_phase1"],
            "best_ep_p2": ckpt["best_ep_phase2"],
        })

        if mdf is not None:
            metrics_dfs.append((seed_int, mdf))

        if not args.no_gen_curve:
            print(f"    Building generalizability curve (interval={args.interval})...")
            gc = gen_curve_one_seed(sd, meta, args.interval, args.device)
            gen_curves.append((seed_int, gc))

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = build_summary(seed_rows)
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Run: {run_dir}\n\n{summary}\n")
    print(f"\n  Saved: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if metrics_dfs:
        plot_learning_curves(metrics_dfs, out_dir / "fig_learning.png")

    if gen_curves and any(not gc.empty for _, gc in gen_curves):
        plot_gen_curves(gen_curves, out_dir / "fig_gen_curve.png")
    elif not args.no_gen_curve:
        print("  No generalizability curve data available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
