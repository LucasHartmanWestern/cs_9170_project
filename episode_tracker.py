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


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")

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

    # NEW: include spec name early so it appears in the parent folder
    add_key("SPEC", "SPEC_NAME", lambda s: _slug(str(s)))

    add_key("EP",    "EPISODES",       int)
    add_key("PCA",   "pca_components", int)
    add_key("REW",   "reward_mode",    lambda s: str(s).lower())
    add_key("minID", "minority_id",    int)
    add_key("majID", "majority_id",    int)
    add_key("thirdID","third_id",      int)
    add_key("TRJ",   "TRAJ_LENGTH",    int)
    add_key("REAL",  "REAL_DATA_SIZE", int)
    add_key("BIAS",  "BIAS_PCT",       lambda x: str(x).rstrip("0").rstrip(".") if isinstance(x, float) else x)

    # OPTIONAL: keep a short group stamp, but don't truncate away useful info
    # If EXP_GROUP already includes G2026..., keep just the time marker:
    add_key("G", "EXP_GROUP", lambda s: str(s)[-13:])  # e.g., "__G202601061430"

    slug = "_".join(parts) if parts else "exp"

    j = json.dumps(rs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(j).hexdigest()[:8]
    experiment_id = f"{slug}_{h}"
    return experiment_id, rs


def _flatten_with_prefix(prefix: str, d: dict):
    # Return dict with "prefix.key" -> value. Safe on None.
    d = d or {}
    return {f"{prefix}.{k}": v for k, v in d.items()}

def _flatten_nested(d: dict, *, sep: str = ".", prefix: str = "") -> dict:
    """
    Recursively flatten nested dicts:
      {"global":{"a":1}, "x":2} -> {"global.a":1, "x":2}
    Lists/tuples are kept as-is (stringified later by pandas if needed).
    """
    out = {}
    d = d or {}
    for k, v in d.items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_nested(v, sep=sep, prefix=kk))
        else:
            out[kk] = v
    return out

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
        self._phase_best = {}  # per-phase best metric tracking
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

        keys = list(flat_row.keys())

        def sort_key(k):
            # new canonical groups from diagnostics
            if   k.startswith("global."):   g = 0
            elif k.startswith("utility."):  g = 1
            elif k.startswith("fairness."): g = 2
            elif k.startswith("local."):    g = 3
            elif k.startswith("extra."):    g = 4
            elif k.startswith("align."):    g = 5
            elif k.startswith("meta."):     g = 6
            else:                           g = 9
            return (g, k)

        keys_sorted = sorted(keys, key=sort_key)

        self._csv_columns = ["episode", "wall_seconds"] + keys_sorted

        self.seed_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(self._csv_columns) + "\n")

        self._header_written = True

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

    # --- inside EpisodeTracker class, replace log_episode with this version ---
    def log_episode(self, episode_num: int, diagnostics: dict, alignment_metrics: dict | None = None, extra_metrics: dict | None = None):
        """
        Accepts the diagnostics dict returned by compute_reward() directly.

        - diagnostics: expected nested dict with groups like global/utility/fairness/local/extra
        - alignment_metrics: optional dict (will be stored under align.*)
        - extra_metrics: optional dict for anything else you want (stored under meta.*)
        """
        alignment_metrics = alignment_metrics or {}
        extra_metrics = extra_metrics or {}

        flat = {}
        flat.update(_flatten_nested(diagnostics))                       # global.*, utility.*, fairness.*, local.*, extra.*
        flat.update(_flatten_with_prefix("align", alignment_metrics))   # align.*
        flat.update(_flatten_with_prefix("meta", extra_metrics))        # meta.* (optional)

        # first-use header
        self._ensure_header(flat)

        # write row
        csv_row = self._row_for_csv(episode_num, flat)

        # cache avg reward if present
        # (your compute_reward diagnostics doesn't include avg_reward; caller can pass via extra_metrics if desired)
        # so we store the compare metric value later from flat row anyway.
        self.episode_rewards.append(float(csv_row.get("meta.avg_reward", np.nan)))

        # console one-liner
        ep_ret  = csv_row.get("meta.episode_return", np.nan)
        g_obj   = csv_row.get("global.global_obj", np.nan)
        l_mean  = csv_row.get("global.local_reward", np.nan)
        f1m     = csv_row.get("utility.f1_macro_beta", np.nan)
        worst   = csv_row.get("fairness.worst_loss_beta", np.nan)

        if np.isnan(ep_ret):
            print(f"[Tracker] Ep {episode_num:4d} | Global {g_obj:.4f} | Local {l_mean:.4f} | F1_macro {f1m:.4f} | WorstLoss {worst:.4f}")
        else:
            print(f"[Tracker] Ep {episode_num:4d} | Return {ep_ret:.2f} | Global {g_obj:.4f} | Local {l_mean:.4f} | F1_macro {f1m:.4f} | WorstLoss {worst:.4f}")

        pd.DataFrame([csv_row]).to_csv(self.csv_path, mode="a", header=False, index=False)

        self._since_last_flush = 0
        self._last_flat_row = flat


    # --- inside EpisodeTracker class, update legacy metric mapping to new keys (optional but recommended) ---
    def _metric_from_flat_row(self, flat_row: dict):
        """
        Compare by any column name. Back-compat mapping updated for new diagnostics keys.
        """
        key = self.compare_metric
        if key in flat_row:
            v = flat_row[key]
        else:
            legacy = {
                # old
                "average_reward": "meta.avg_reward",
                "reward.avg_reward": "meta.avg_reward",

                # new preferred
                "global_obj": "global.global_obj",
                "local_reward": "global.local_reward",
                "worst_loss_beta": "fairness.worst_loss_beta",
                "macro_f1_beta": "utility.f1_macro_beta",
                "f1_minority_beta": "utility.f1_minority_beta",

                # some old names you used previously
                "obj1": "global.global_obj",
                "obj2_mean": "global.local_reward",
                "global_f1": "utility.f1_macro_beta",
            }
            mapped = legacy.get(key, "meta.avg_reward")
            v = flat_row.get(mapped, np.nan)

        try:
            return float(v)
        except Exception:
            return -float("inf")

    # --- inside EpisodeTracker class, update maybe_save_synthetic to not require obj1/obj2/etc ---
    def maybe_save_synthetic(
        self,
        episode_num,
        x_syn,
        y_syn,
        *,
        feature_names=None,
        beta_model=None,
        metrics_flat_override: dict | None = None,
        phase_label: str | None = None,
    ):
        """
        Save synthetic data checkpoints.

        When phase_label is set (e.g. "phase1_class1", "phase2_class0"):
        - Uses phase-specific paths: best_synthetic_{phase_label}.npz, etc.
        - Tracks per-phase best metric independently via self._phase_best[phase_label]
        When phase_label is None: existing behavior (backward compatible).
        """
        save_snap = (episode_num % self.ckpt_every == 0)

        if metrics_flat_override is not None:
            flat_row = dict(metrics_flat_override)
        else:
            flat_row = getattr(self, "_last_flat_row", {})

        metric_val = self._metric_from_flat_row(flat_row)

        # Determine best-metric tracking and file paths based on phase_label
        if phase_label is not None:
            phase_best = self._phase_best.get(phase_label, -float("inf"))
            is_best = metric_val > phase_best
            npz_path = self.seed_dir / f"best_synthetic_{phase_label}.npz"
            csv_path = self.seed_dir / f"best_synthetic_{phase_label}.csv"
            meta_path = self.seed_dir / f"best_synthetic_{phase_label}.json"
            beta_path = self.seed_dir / f"best_beta_state_dict_{phase_label}.pt"
            beta_meta = self.seed_dir / f"best_beta_meta_{phase_label}.json"
        else:
            is_best = metric_val > self.best_metric
            npz_path = self.best_npz_path
            csv_path = self.best_csv_path
            meta_path = self.best_meta_path
            beta_path = self.best_beta_path
            beta_meta = self.best_beta_meta

        if not (save_snap or is_best):
            return

        # Convert to numpy once
        if hasattr(x_syn, "detach"): x_syn = x_syn.detach().cpu().numpy()
        if hasattr(y_syn, "detach"): y_syn = y_syn.detach().cpu().numpy()
        x_syn = np.asarray(x_syn)
        y_syn = np.asarray(y_syn).reshape(-1)

        # Periodic snapshot
        if save_snap:
            suffix = f"_{phase_label}" if phase_label else ""
            snap_npz = self.snap_dir / f"synthetic_ep{episode_num:04d}{suffix}.npz"
            np.savez_compressed(snap_npz, x=x_syn, y=y_syn)
            if self.snapshot_csv:
                if feature_names is None:
                    feature_names = [f"pca_{i}" for i in range(x_syn.shape[1])]
                df = pd.DataFrame(x_syn, columns=feature_names)
                df["target"] = y_syn
                snap_csv = self.snap_dir / f"synthetic_ep{episode_num:04d}{suffix}.csv"
                df.to_csv(snap_csv, index=False)
            print(f"[Tracker] Saved snapshot: ep{episode_num:04d}{suffix}")

        # Best-so-far
        if is_best:

            if phase_label is not None:
                self._phase_best[phase_label] = metric_val
            else:
                self.best_metric = metric_val

            np.savez_compressed(npz_path, x=x_syn, y=y_syn)
            if feature_names is None:
                feature_names = [f"pca_{i}" for i in range(x_syn.shape[1])]
            df_best = pd.DataFrame(x_syn, columns=feature_names)
            df_best["target"] = y_syn
            df_best.to_csv(csv_path, index=False)
            with open(meta_path, "w") as f:
                json.dump({
                    "episode": episode_num,
                    "metric": self.compare_metric,
                    "metric_value": metric_val,
                    "phase_label": phase_label,
                    "updated_at": time.strftime("%Y-%m-%d_%H-%M-%S")
                }, f, indent=2)
            label_str = f" [{phase_label}]" if phase_label else ""
            print(f"[Tracker] New BEST synthetic{label_str} (by {self.compare_metric}: {metric_val:.6f}) saved.")

            if beta_model is not None:
                torch.save(beta_model.model.state_dict(), beta_path)
                with open(beta_meta, "w") as f:
                    json.dump({
                        "episode": episode_num,
                        "metric": self.compare_metric,
                        "metric_value": metric_val,
                        "phase_label": phase_label,
                        "updated_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
                        "checkpoint": str(beta_path.name)
                    }, f, indent=2)
                print(f"[Tracker] BEST β weights{label_str} saved -> {beta_path}")

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

    def log_final_test(
        self, alpha_model, x_test, y_test, f1_thresh: float = 0.5,
        prefer_best_beta: bool = True, beta_model=None,
        x_train=None, y_train=None,
        # existing jitter baseline params
        jitter_n=None, jitter_scale: float = 0.20,
        # alpha toggles/params
        run_alpha_raw_original: bool = True,
        run_alpha_plus_real: bool = True,
        alpha_plus_real_n: int = 2000,
        # CTGAN baseline toggles/params
        run_alpha_plus_ctgan: bool = False,
        alpha_plus_ctgan_n: int = 2000,
        ctgan_epochs: int = 300,
        cap_ctgan_train: int | None = None,
        # CTABGAN baseline toggles/params
        run_ctabgan: bool = False,
        alpha_plus_ctabgan_n: int = 2000,
        # CTABGAN subprocess wiring (IMPORTANT: default None)
        ctab_python: str | None = None,
        ctab_repo: str | None = None,
        ctab_runner: str | None = None,
        # dataset settings used to rebuild original pool
        data_path: str = "census+income/adult.data",
        bias_pct = None,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        # additional CTABGAN batch/seed/pca-related params
        batch_size: int = 64,
        pca_components: int | None = None,
        seed: int | None = None,
        a_test: torch.Tensor | None = None,   # <--- NEW
    ):
        # Prefer phase-labelled best checkpoint (phase1_class1) if it exists,
        # since maybe_save_synthetic always writes the phase-labelled path.
        _phase1_path = self.seed_dir / "best_beta_state_dict_phase1_class1.pt"
        _best_beta_path = _phase1_path if _phase1_path.exists() else self.best_beta_path

        tests = TestSuite(
            seed_dir=self.seed_dir,
            experiment_dir=self.experiment_dir,
            seed=self.seed,
            run_id=self.run_id,
            beta_factory=self.beta_factory,
            best_beta_path=_best_beta_path,
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
            # jitter params
            jitter_n=jitter_n,
            jitter_scale=jitter_scale,
            # alpha toggles/params
            run_alpha_raw_original=run_alpha_raw_original,
            run_alpha_plus_real=run_alpha_plus_real,
            alpha_plus_real_n=alpha_plus_real_n,
            # CTGAN
            run_alpha_plus_ctgan=run_alpha_plus_ctgan,
            alpha_plus_ctgan_n=alpha_plus_ctgan_n,
            ctgan_epochs=ctgan_epochs,
            cap_ctgan_train=cap_ctgan_train,
            # CTABGAN
            run_ctabgan=run_ctabgan,
            alpha_plus_ctabgan_n=alpha_plus_ctabgan_n,
            # CTABGAN wiring
            ctab_python=ctab_python,
            ctab_repo=ctab_repo,
            ctab_runner=ctab_runner,
            # dataset rebuild settings
            data_path=data_path,
            bias_pct=bias_pct,
            val_frac=val_frac,
            test_frac=test_frac,
            train_size=train_size,
            # pass through batch/pca/seed
            batch_size=batch_size,
            pca_components=pca_components,
            seed=self.seed if seed is None else seed,
            a_test=a_test
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