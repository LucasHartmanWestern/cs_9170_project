import os, json, time, sys, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
import hashlib
from copy import deepcopy


def _fingerprint_run_stats(run_stats: dict, exclude_keys=("seed", "started_at", "device", "notes")) -> tuple[str, dict]:
    """Return (experiment_id, filtered_stats) for a config, excluding volatile keys.
    experiment_id is human-readable slug + short hash for uniqueness."""
    rs = deepcopy(run_stats) if run_stats else {}
    for k in exclude_keys:
        rs.pop(k, None)

    # Build a compact, human-friendly slug from common keys if present
    parts = []
    def add(tag, key, fmt=None):
        if key in rs and rs[key] is not None:
            v = rs[key]
            if fmt: v = fmt(v)
            parts.append(f"{tag}{v}")

    add("EP", "EPISODES", int)
    add("TRJ", "TRAJ_LENGTH", int)
    add("REAL", "REAL_DATA_SIZE", int)
    add("BIAS", "BIAS_PCT", lambda x: str(x).rstrip("0").rstrip(".") if isinstance(x, float) else x)
    add("L", "lambda_", lambda x: ("None" if x is None else str(x)))
    add("NR", "use_new_reward", lambda b: int(bool(b)))
    add("TAU", "ema_tau", lambda x: str(x).rstrip("0").rstrip(".") if x is not None else "None")

    slug = "_".join(parts) if parts else "exp"
    # Stable short hash from the filtered stats (sorted JSON)
    j = json.dumps(rs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(j).hexdigest()[:8]
    experiment_id = f"{slug}_{h}"
    return experiment_id, rs

def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


class _TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data: str):
        for s in self.streams:
            s.write(data)
            if data.endswith("\n"):
                s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def _flatten_with_prefix(prefix: str, d: dict):
    # Return dict with "prefix.key" -> value. Safe on None.
    d = d or {}
    return {f"{prefix}.{k}": v for k, v in d.items()}

class EpisodeTracker:
    """
    Per-run folder with:
      meta.json, metrics.csv (dynamic headers), console.log,
      synthetic snapshots, best synthetic, and best β weights.
    """
    def __init__(self, run_stats: dict, save_dir: str = "runs",
                 capture_console: bool = True,
                 ckpt_every: int = 5,
                 compare_metric: str = "reward.avg_reward",
                 flush_every_episodes: int = 10,
                 snapshot_csv: bool = False,
                 beta_factory=None):

        self.capture_console = capture_console
        self.flush_every_episodes = max(1, int(flush_every_episodes))
        self._since_last_flush = 0
        self.episode_rewards = []
        self.t0 = time.time()
        self.compare_metric = compare_metric
        self.ckpt_every = max(1, int(ckpt_every))
        self.snapshot_csv = bool(snapshot_csv)
        self.beta_factory = beta_factory

        # --- Identify seed & experiment ---
        self.seed = run_stats.get("seed", None)

        self.start_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.uid = f"{os.getpid()}_{str(uuid.uuid4())[:8]}"

        # Build experiment id (same for same config regardless of seed or time)
        self.experiment_id, filtered_stats = _fingerprint_run_stats(run_stats)

        # Folders
        self.experiment_dir = Path(save_dir) / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.seed_dir = self.experiment_dir / f"seed_{self.seed}"
        self.seed_dir.mkdir(parents=True, exist_ok=True)

        # Optional convenience: a short per-run id string for logs
        self.run_id = f"{self.start_ts}_seed{self.seed}_{self.uid}"

        # --- Experiment-level meta & seeds registry (merge/append) ---
        meta_path = self.experiment_dir / "meta.json"
        existing_meta = _read_json(meta_path)
        merged_meta = dict(existing_meta)
        merged_meta.update({
            "experiment_id": self.experiment_id,
            "last_started_at": self.start_ts,
            "config": filtered_stats,     # seedless config snapshot
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged_meta, f, indent=2)

        seeds_path = self.experiment_dir / "seeds.json"
        seeds_info = _read_json(seeds_path)
        seeds_list = set(seeds_info.get("seeds", []))
        seeds_list.add(str(self.seed))
        with open(seeds_path, "w", encoding="utf-8") as f:
            json.dump({"seeds": sorted(seeds_list)}, f, indent=2)

        # --- Per-seed meta (keeps the full run_stats incl. seed) ---
        with open(self.seed_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "seed": self.seed,
                "started_at": self.start_ts,
                **run_stats
            }, f, indent=2)

        # --- CSV in the seed folder ---
        self.csv_path = self.seed_dir / "metrics.csv"
        self._csv_fh = None            # no persistent handle
        self._csv_columns = None
        self._header_written = False


        # --- console mirroring per seed ---
        self._orig_stdout = self._orig_stderr = self._log_file = None
        if self.capture_console:
            self._log_file = open(self.seed_dir / "console.log", "a", buffering=1, encoding="utf-8", errors="replace")
            self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _TeeLogger(sys.__stdout__, self._log_file)
            sys.stderr = _TeeLogger(sys.__stderr__, self._log_file)
            print(f"[Tracker] Experiment: {self.experiment_id}")
            print(f"[Tracker] Seed folder: {self.seed_dir}")
            print(f"[Tracker] Run: {self.run_id}")
            print(f"[Tracker] Metrics CSV: {self.csv_path}")
            print(f"[Tracker] Mirroring console to: {self.seed_dir / 'console.log'}")

        # --- Per-seed synthetic checkpoint paths ---
        self.best_metric = -float("inf")
        self.snap_dir = self.seed_dir / "synthetic_snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.best_meta_path = self.seed_dir / "best_synthetic.json"
        self.best_csv_path  = self.seed_dir / "best_synthetic.csv"
        self.best_npz_path  = self.seed_dir / "best_synthetic.npz"
        self.best_beta_path = self.seed_dir / "best_beta_state_dict.pt"
        self.best_beta_meta = self.seed_dir / "best_beta_meta.json"

    # Context manager support
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _ensure_header(self, flat_row: dict):
        """
        Decide and write CSV header on first call.
        Columns: episode, wall_seconds, then all keys in flat_row in stable order.
        """
        if self._header_written:
            return

        # stable order: episode, wall_seconds, grouped by prefix
        keys = list(flat_row.keys())
        def sort_key(k):
            if   k.startswith("reward."): g = 0
            elif k.startswith("ema."):    g = 1
            elif k.startswith("newr."):   g = 2
            elif k.startswith("align."):  g = 3
            else:                         g = 9
            return (g, k)
        keys_sorted = sorted(keys, key=sort_key)

        self._csv_columns = ["episode", "wall_seconds"] + keys_sorted

        # write header with a fresh open; no persistent handle
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(self._csv_columns) + "\n")

        self._header_written = True

        # save chosen header for reference
        with open(self.seed_dir / "metrics_header.json", "w", encoding="utf-8") as f:
            json.dump({"columns": self._csv_columns}, f, indent=2)

        print(f"[Tracker] CSV header written with {len(self._csv_columns)} columns.")

    def _row_for_csv(self, episode_num: int, flat_row: dict):
        wall_seconds = time.time() - self.t0
        row = {"episode": episode_num, "wall_seconds": round(wall_seconds, 2)}
        for k in self._csv_columns[2:]:
            v = flat_row.get(k, np.nan)
            # cast tensors
            if hasattr(v, "item") and callable(getattr(v, "item")):
                try: v = v.item()
                except Exception: pass
            row[k] = v
        return row

    def log_episode(self, episode_num, reward_metrics, ema_metrics, new_reward_metrics, alignment_metrics):
        """
        All metric dicts are flattened into:
        reward.*, ema.*, newr.*, align.* columns.
        Header is decided on the first call from the keys present.
        """
        flat = {}
        flat.update(_flatten_with_prefix("reward", reward_metrics))
        flat.update(_flatten_with_prefix("ema", ema_metrics))
        flat.update(_flatten_with_prefix("newr", new_reward_metrics))
        flat.update(_flatten_with_prefix("align", alignment_metrics))

        # first-use header
        self._ensure_header(flat)

        # write row
        csv_row = self._row_for_csv(episode_num, flat)
        self.episode_rewards.append(float(csv_row.get("reward.avg_reward", np.nan)))

        # console one-liner
        avg_r   = csv_row.get("reward.avg_reward", np.nan)
        obj1    = csv_row.get("reward.obj1_f1_minority_beta", np.nan)
        obj2    = csv_row.get("reward.obj2_local_useful_mean", np.nan)
        macro_f = csv_row.get("reward.macro_f1_beta", np.nan)
        print(f"[Tracker] Ep {episode_num:4d} | AvgR {avg_r:.4f} | Global {obj1:.4f} | Local_Avg {obj2:.4f} | F1_macro {macro_f:.4f}")

        # append using the file PATH (not a handle)
        pd.DataFrame([csv_row]).to_csv(self.csv_path, mode="a", header=False, index=False)

        # no flush needed since we don't hold a handle
        self._since_last_flush = 0

        # stash last flat row for compare_metric & maybe_save_synthetic convenience
        self._last_flat_row = flat


    def _metric_from_flat_row(self, flat_row: dict):
        """
        Compare by any column name. Back-compat:
        - if compare_metric matches a key in flat_row, use it
        - else if legacy names are used, map them
        """
        key = self.compare_metric
        if key in flat_row:
            v = flat_row[key]
        else:
            # legacy fallbacks
            legacy = {
                "average_reward": "reward.avg_reward",
                "obj1": "reward.obj1_f1_minority_beta",
                "obj2_mean": "reward.obj2_local_useful_mean",
                "global_f1": "reward.macro_f1_beta",
            }
            mapped = legacy.get(key, "reward.avg_reward")
            v = flat_row.get(mapped, np.nan)
        try:
            return float(v)
        except Exception:
            return -float("inf")

    def maybe_save_synthetic(self, episode_num, x_syn, y_syn,
                             avg_reward=None, obj1=None, obj2_mean=None, global_f1=None,
                             feature_names=None, beta_model=None):
        """
        Save snapshot every ckpt_every episodes and update 'best' if metric improves.

        Backward compatible:
          - If avg_reward/obj1/obj2_mean/global_f1 are provided, we build a tiny flat-row
            under reward.* keys.
          - If not provided and log_episode has been called, we use the last flat row.
        """
        # Decide if we need to save anything first (avoid conversions when not saving)
        save_snap = (episode_num % self.ckpt_every == 0)

        # Build a metric row to evaluate compare_metric
        if avg_reward is not None or obj1 is not None or obj2_mean is not None or global_f1 is not None:
            flat_row = {
                "reward.avg_reward": avg_reward,
                "reward.obj1_f1_minority_beta": obj1,
                "reward.obj2_local_useful_mean": obj2_mean,
                "reward.macro_f1_beta": global_f1,
            }
        else:
            flat_row = getattr(self, "_last_flat_row", {})

        metric_val = self._metric_from_flat_row(flat_row)
        is_best = metric_val > self.best_metric

        if not (save_snap or is_best):
            return  # nothing to do

        # Convert to numpy once
        if hasattr(x_syn, "detach"): x_syn = x_syn.detach().cpu().numpy()
        if hasattr(y_syn, "detach"): y_syn = y_syn.detach().cpu().numpy()
        x_syn = np.asarray(x_syn)
        y_syn = np.asarray(y_syn).reshape(-1)

        # Periodic snapshot (NPZ always; CSV optionally)
        if save_snap:
            snap_npz = self.snap_dir / f"synthetic_ep{episode_num:04d}.npz"
            np.savez_compressed(snap_npz, x=x_syn, y=y_syn)
            if self.snapshot_csv:
                if feature_names is None:
                    feature_names = [f"pca_{i}" for i in range(x_syn.shape[1])]
                df = pd.DataFrame(x_syn, columns=feature_names)
                df["target"] = y_syn
                snap_csv = self.snap_dir / f"synthetic_ep{episode_num:04d}.csv"
                df.to_csv(snap_csv, index=False)
            print(f"[Tracker] Saved snapshot: ep{episode_num:04d}")

        # Best-so-far checkpoint (synthetic + optional model weights)
        if is_best:
            self.best_metric = metric_val

            # Save best synthetic (NPZ + CSV + meta)
            np.savez_compressed(self.best_npz_path, x=x_syn, y=y_syn)
            if feature_names is None:
                feature_names = [f"pca_{i}" for i in range(x_syn.shape[1])]
            df_best = pd.DataFrame(x_syn, columns=feature_names)
            df_best["target"] = y_syn
            df_best.to_csv(self.best_csv_path, index=False)
            with open(self.best_meta_path, "w") as f:
                json.dump({
                    "episode": episode_num,
                    "metric": self.compare_metric,
                    "metric_value": metric_val,
                    "updated_at": time.strftime("%Y-%m-%d_%H-%M-%S")
                }, f, indent=2)
            print(f"[Tracker] New BEST synthetic (by {self.compare_metric}: {metric_val:.6f}) saved.")

            # Also save BEST beta model weights if provided
            if beta_model is not None:
                torch.save(beta_model.model.state_dict(), self.best_beta_path)
                with open(self.best_beta_meta, "w") as f:
                    json.dump({
                        "episode": episode_num,
                        "metric": self.compare_metric,
                        "metric_value": metric_val,
                        "updated_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
                        "checkpoint": str(self.best_beta_path.name)
                    }, f, indent=2)
                print(f"[Tracker] BEST β weights saved -> {self.best_beta_path}")

    def summary_path(self): return str(self.seed_dir)

    def close(self):
        # restore std streams
        if self.capture_console and self._log_file:
            sys.stdout = self._orig_stdout or sys.__stdout__
            sys.stderr = self._orig_stderr or sys.__stderr__
            try:
                self._log_file.flush()
                os.fsync(self._log_file.fileno())
            except Exception:
                pass
            self._log_file.close()
            self._log_file = None
        # close CSV handle
        try:
            if getattr(self, "_csv_fh", None):
                self._csv_fh.flush()
                self._csv_fh.close()
        except Exception:
            pass

    def log_final_test(self, alpha_model, x_test, y_test, f1_thresh: float = 0.5,
                    prefer_best_beta: bool = True, beta_model=None,
                    x_train=None, y_train=None, jitter_n=None, jitter_scale: float = 0.20):
        """
        End-of-run θ_test evaluation (+ minority jitter baseline).
        Appends to final_test_metrics.csv as before.
        """
        beta_for_eval = None
        if prefer_best_beta and self.beta_factory is not None and self.best_beta_path.exists():
            beta_for_eval = self.beta_factory()  # fresh β
            state = torch.load(self.best_beta_path, map_location=x_test.device)
            beta_for_eval.model.load_state_dict(state)
            print(f"[Tracker] Loaded best β weights from: {self.best_beta_path}")
        elif beta_model is not None:
            beta_for_eval = beta_model
            print("[Tracker] Using provided β (no best checkpoint found or factory missing).")
        else:
            print("[Tracker] No β available for TEST evaluation (skipping β metrics).")

        eps = 1e-8

        def _p1_from_agent(agent, x):
            agent.model.eval()
            with torch.no_grad():
                logits = agent.model(x)
                probs  = torch.softmax(logits, -1)
                return probs[..., 1]

        def _all_f1_from_probs(y_true, p1, threshold=0.5):
            y_true = y_true.to(p1.device).long()
            y_pred_pos = (p1 >= threshold).long()
            tp1 = ((y_pred_pos == 1) & (y_true == 1)).sum().float()
            fp1 = ((y_pred_pos == 1) & (y_true == 0)).sum().float()
            fn1 = ((y_pred_pos == 0) & (y_true == 1)).sum().float()
            prec1 = tp1 / (tp1 + fp1 + eps)
            rec1  = tp1 / (tp1 + fn1 + eps)
            f1_1  = (2 * prec1 * rec1) / (prec1 + rec1 + eps)

            tp0 = ((y_pred_pos == 0) & (y_true == 0)).sum().float()
            fp0 = ((y_pred_pos == 0) & (y_true == 1)).sum().float()
            fn0 = ((y_pred_pos == 1) & (y_true == 0)).sum().float()
            prec0 = tp0 / (tp0 + fn0 + eps)
            rec0  = tp0 / (tp0 + fp0 + eps)
            f1_0  = (2 * prec0 * rec0) / (prec0 + rec0 + eps)

            n1 = (y_true == 1).sum().float()
            n0 = (y_true == 0).sum().float()
            n  = n0 + n1 + eps

            f1_macro    = 0.5 * (f1_1 + f1_0)
            f1_weighted = (f1_1 * (n1 / n)) + (f1_0 * (n0 / n))
            return float(f1_1), float(f1_0), float(f1_weighted), float(f1_macro)

        def _brier_mean(y_true, p1):
            y_true = y_true.to(p1.device).float()
            return float(((p1 - y_true) ** 2).mean())

        with torch.no_grad():
            p1_alpha = _p1_from_agent(alpha_model, x_test)
            a_f1_min, a_f1_maj, a_f1_w, a_f1_macro = _all_f1_from_probs(y_test, p1_alpha, f1_thresh)
            a_brier = _brier_mean(y_test, p1_alpha)

            if beta_for_eval is not None:
                p1_beta = _p1_from_agent(beta_for_eval, x_test)
                b_f1_min, b_f1_maj, b_f1_w, b_f1_macro = _all_f1_from_probs(y_test, p1_beta, f1_thresh)
                b_brier = _brier_mean(y_test, p1_beta)
            else:
                b_f1_min = b_f1_maj = b_f1_w = b_f1_macro = b_brier = float('nan')

        # ---------------- Minority-jitter baseline ----------------
        j_f1_min = j_f1_maj = j_f1_w = j_f1_macro = j_brier = float('nan')
        if (x_train is not None) and (y_train is not None) and (self.beta_factory is not None):
            with torch.no_grad():
                Xmin = x_train[y_train == 1]
                if Xmin.numel() == 0:
                    print("[Tracker] Jitter baseline skipped: no minority samples in x_train.")
                else:
                    if jitter_n is None:
                        jitter_n = Xmin.shape[0]
                    idx   = torch.randint(0, Xmin.shape[0], (int(jitter_n),), device=Xmin.device)
                    base  = Xmin[idx]
                    std   = x_train.std(dim=0, unbiased=False)
                    noise = torch.randn_like(base) * (jitter_scale * (std + 1e-6))
                    Xj    = base + noise
                    yj    = torch.ones(int(jitter_n), dtype=y_train.dtype, device=y_train.device)

            jitter_beta = self.beta_factory()
            if hasattr(jitter_beta.model, "to"):
                jitter_beta.model.to(x_train.device)

            x_hybrid = torch.cat([x_train, Xj], dim=0)
            y_hybrid = torch.cat([y_train, yj], dim=0)
            ds = TensorDataset(x_hybrid, y_hybrid)
            dl = DataLoader(ds, batch_size=32, shuffle=True)
            jitter_beta.train(dl)

            with torch.no_grad():
                p1_j = _p1_from_agent(jitter_beta, x_test)
                j_f1_min, j_f1_maj, j_f1_w, j_f1_macro = _all_f1_from_probs(y_test, p1_j, f1_thresh)
                j_brier = _brier_mean(y_test, p1_j)
            print(f"[TEST] Jitterβ -> F1(min)={j_f1_min:.4f} | F1(maj)={j_f1_maj:.4f} | "
                  f"F1(weighted)={j_f1_w:.4f} | F1(macro)={j_f1_macro:.4f} | Brier={j_brier:.4f}")
        else:
            print("[Tracker] Jitter baseline not run (missing x_train/y_train or beta_factory).")

        # ---------------- Console summary ----------------
        print("\n[TEST] Alpha -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
              % (a_f1_min, a_f1_maj, a_f1_w, a_f1_macro, a_brier))
        if not np.isnan(b_f1_min):
            print("[TEST] Beta  -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
                  % (b_f1_min, b_f1_maj, b_f1_w, b_f1_macro, b_brier))
            winner = "beta" if b_f1_min > a_f1_min else "alpha"
            print(f"[TEST] Winner (by F1 minority): {winner}")
        else:
            print("[TEST] Beta  -> (no checkpoint/factory)")

        # ---------------- Append to CSV ----------------
        seed_csv = self.seed_dir / "final_test_metrics.csv"
        row = {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "seed": str(self.seed),
            "run_id": self.run_id,
            "threshold": float(f1_thresh),

            "alpha_f1_minority": a_f1_min,
            "alpha_f1_majority": a_f1_maj,
            "alpha_f1_weighted": a_f1_w,
            "alpha_f1_macro":    a_f1_macro,
            "alpha_brier":       a_brier,

            "beta_f1_minority":  b_f1_min,
            "beta_f1_majority":  b_f1_maj,
            "beta_f1_weighted":  b_f1_w,
            "beta_f1_macro":     b_f1_macro,
            "beta_brier":        b_brier,

            "jitter_n":          int(jitter_n) if jitter_n is not None else 0,
            "jitter_scale":      float(jitter_scale),
            "jitter_f1_minority": j_f1_min,
            "jitter_f1_majority": j_f1_maj,
            "jitter_f1_weighted": j_f1_w,
            "jitter_f1_macro":    j_f1_macro,
            "jitter_brier":       j_brier,

            "winner_by_f1_minority": ("beta" if (not np.isnan(b_f1_min) and b_f1_min > a_f1_min) else "alpha")
        }
        pd.DataFrame([row]).to_csv(seed_csv, mode="a", header=not seed_csv.exists(), index=False)
        print(f"[Tracker] Final test metrics appended to (seed): {seed_csv}")

        # 2) experiment-level rollup with the same row
        exp_csv = self.experiment_dir / "final_test_metrics.csv"
        pd.DataFrame([row]).to_csv(exp_csv, mode="a", header=not exp_csv.exists(), index=False)
        print(f"[Tracker] Final test metrics appended to (experiment): {exp_csv}")
        return row

