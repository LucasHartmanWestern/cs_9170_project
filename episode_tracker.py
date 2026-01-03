import os, json, time, sys, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
from test_suite import TestSuite
import hashlib
from copy import deepcopy
from datetime import datetime

def _read_json(path: Path) -> dict:
    """
    Safe JSON reader. Returns {} on any error or if the file doesn't exist.
    Accepts Path or str.
    """
    try:
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


#For naming the parent folder
def _fingerprint_run_stats(
    run_stats: dict,
    exclude_keys=("seed", "device", "notes", "started_at", "ts", "timestamp", "time", "date")
) -> tuple[str, dict]:
    rs = deepcopy(run_stats) if run_stats else {}
    for k in exclude_keys:
        rs.pop(k, None)

    parts = []

    def add_key(tag, key, fmt=None):
        if key in run_stats and run_stats[key] is not None:
            v = run_stats[key]
            if fmt:
                v = fmt(v)
            parts.append(f"{tag}{v}")

    def add_literal(tag, value):
        parts.append(f"{tag}{value}")

    add_key("EP",    "EPISODES",       int)
    add_key("PCA",   "pca_components", int)
    add_key("REW",   "reward_mode",    lambda s: str(s).lower())
    add_key("minID", "minority_id",    int)
    add_key("majID", "majority_id",    int)
    add_key("thirdID","third_id",      int)
    add_key("TRJ",   "TRAJ_LENGTH",    int)
    add_key("REAL",  "REAL_DATA_SIZE", int)
    add_key("BIAS",  "BIAS_PCT",       lambda x: str(x).rstrip("0").rstrip(".") if isinstance(x, float) else x)

    add_key("G",     "EXP_GROUP",      lambda s: str(s)[:12])

    slug = "_".join(parts) if parts else "exp"

    j = json.dumps(rs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(j).hexdigest()[:8]
    experiment_id = f"{slug}_{h}"
    return experiment_id, rs

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
    def __init__(self, run_stats: dict, dataset, save_dir: str = "runs",
                 capture_console: bool = True,
                 ckpt_every: int = 5,
                 compare_metric: str = "reward.avg_reward",
                 flush_every_episodes: int = 10,
                 snapshot_csv: bool = False,
                 beta_factory=None, seed=42):

        self.seed=seed
        self.dataset=dataset
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
        # use run_stats as-is; no timestamp in the experiment fingerprint
        stats_for_fingerprint = dict(run_stats)
        self.experiment_id, filtered_stats = _fingerprint_run_stats(stats_for_fingerprint)

        
        self.uid = f"{os.getpid()}_{str(uuid.uuid4())[:8]}"

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
        self.alpha_state_dict_path = self.seed_dir / "alpha_state_dict.pt"

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

    def log_episode(self, episode_num, reward_metrics, new_reward_metrics, alignment_metrics):
        """
        All metric dicts are flattened into:
        reward.*, ema.*, newr.*, align.* columns.
        Header is decided on the first call from the keys present.
        """
        flat = {}
        flat.update(_flatten_with_prefix("reward", reward_metrics))
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
                       x_train=None, y_train=None,
                       # existing jitter baseline params
                       jitter_n=None, jitter_scale: float = 0.20,
                       # NEW alpha toggles/params
                       run_alpha_raw_original: bool = True,
                       run_alpha_plus_real: bool = True,
                       alpha_plus_real_n: int = 2000,
                       # NEW CTGAN baseline toggles/params
                       run_alpha_plus_ctgan: bool = True,
                       alpha_plus_ctgan_n: int = 2000,
                       ctgan_epochs: int = 300,
                       cap_ctgan_train: int | None = None,
                       # NEW CTABGAN baseline toggles/params
                       run_ctabgan: bool = True,
                       alpha_plus_ctabgan_n: int = 2000,
                       # CTABGAN subprocess wiring
                       ctab_python: str = "/home/epigou/envs/ctabgan/bin/python",
                       ctab_repo: str = "/home/epigou/CTAB-GAN-Plus-DP",
                       ctab_runner: str | None = None,
                       # dataset settings used to rebuild original pool
                       data_path: str = "census+income/adult.data",
                       bias_pct: float = 0.20,
                       val_frac: float = 0.20,
                       test_frac: float = 0.20,
                       train_size: int | None = None,
                       # additional CTABGAN batch/seed/pca-related params
                       batch_size: int = 64,
                       pca_components: int = None,
                       seed: int | None = None):

        tests = TestSuite(
            seed_dir=self.seed_dir,
            experiment_dir=self.experiment_dir,
            seed=self.seed,
            run_id=self.run_id,
            beta_factory=self.beta_factory,
            best_beta_path=self.best_beta_path,
            alpha_factory=self.alpha_factory if hasattr(self, "alpha_factory") else self.beta_factory,
            dataset=self.dataset
        )

        return tests.log_final_test(
            alpha_model=alpha_model,
            x_test=x_test,
            y_test=y_test,
            f1_thresh=f1_thresh,
            prefer_best_beta=prefer_best_beta,
            beta_model=beta_model,
            x_train=x_train,
            y_train=y_train,
            # existing jitter params
            jitter_n=jitter_n,
            jitter_scale=jitter_scale,
            # NEW alpha toggles/params
            run_alpha_raw_original=run_alpha_raw_original,
            run_alpha_plus_real=run_alpha_plus_real,
            alpha_plus_real_n=alpha_plus_real_n,
            # NEW CTGAN baseline toggles/params
            run_alpha_plus_ctgan=run_alpha_plus_ctgan,
            alpha_plus_ctgan_n=alpha_plus_ctgan_n,
            ctgan_epochs=ctgan_epochs,
            cap_ctgan_train=cap_ctgan_train,
            # NEW CTABGAN baseline toggles/params
            run_ctabgan=run_ctabgan,
            alpha_plus_ctabgan_n=alpha_plus_ctabgan_n,
            # CTABGAN subprocess wiring
            ctab_python=ctab_python,
            ctab_repo=ctab_repo,
            ctab_runner=ctab_runner,
            # dataset settings for original pool rebuild
            data_path=data_path,
            bias_pct=bias_pct,
            val_frac=val_frac,
            test_frac=test_frac,
            train_size=train_size,
            # additional CTABGAN batch/seed/pca params
            batch_size=batch_size,
            pca_components=pca_components,
            seed=self.seed if seed is None else seed  # allow override
        )




    def save_alpha_state_dict(self, alpha_model, config=None, n_pca_components=None):
        """
        Save alpha model weights to <seed_dir>/alpha_state_dict.pt and, if provided,
        a JSON-safe config to <seed_dir>/ffnn_meta.json (including n_pca_components).
        """
        # --- save weights ---
        try:
            model_obj = getattr(alpha_model, "model", alpha_model)  # support FFNNAgent or raw nn.Module
            state_dict = model_obj.state_dict()
        except Exception as e:
            raise AttributeError(
                "alpha_model must be an FFNNAgent or a torch.nn.Module exposing .state_dict()."
            ) from e

        torch.save(state_dict, self.alpha_state_dict_path)
        print(f"[Tracker] Alpha model state_dict saved to {self.alpha_state_dict_path}")

        # --- optionally save JSON-safe config ---
        if (config is not None) and (n_pca_components is not None):
            # Make a shallow copy so we don't mutate the original
            meta = dict(config)

            # Add required fields
            meta["n_pca_components"] = int(n_pca_components)
            meta["saved_at"] = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Sanitize to JSON-safe types
            def _to_jsonable(obj):
                import numpy as _np
                from pathlib import Path as _Path

                # Primitives
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj

                # torch types
                if isinstance(obj, torch.device):
                    return str(obj)
                if isinstance(obj, torch.dtype):
                    return str(obj)
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()

                # numpy types
                if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()

                # containers
                if isinstance(obj, dict):
                    return {str(k): _to_jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [_to_jsonable(v) for v in obj]

                # paths
                if isinstance(obj, _Path):
                    return str(obj)

                # fallback: string representation
                return str(obj)

            safe_meta = _to_jsonable(meta)

            ffnn_meta_path = self.seed_dir / "ffnn_meta.json"
            with open(ffnn_meta_path, "w", encoding="utf-8") as f:
                json.dump(safe_meta, f, indent=2)
            print(f"[Tracker] Alpha model config saved to {ffnn_meta_path}")