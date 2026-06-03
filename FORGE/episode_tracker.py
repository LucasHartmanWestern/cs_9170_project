import os
import re
import json
import time
import sys
import uuid
import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from test_suite import TestSuite


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

    def isatty(self):
        return False


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")


def _read_json(path: Path) -> dict:
    try:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _fingerprint_run_stats(
    run_stats: dict,
    exclude_keys=("seed", "device", "notes", "started_at", "ts", "timestamp", "time", "date"),
) -> tuple[str, dict]:
    rs = deepcopy(run_stats) if run_stats else {}
    for k in exclude_keys:
        rs.pop(k, None)

    parts = []

    def _add(tag, key, fmt=None):
        if key in run_stats and run_stats[key] is not None:
            v = run_stats[key]
            parts.append(f"{tag}{fmt(v) if fmt else v}")

    _add("SPEC",  "SPEC_NAME",      lambda s: _slug(str(s)))
    _add("EP",    "episodes",       int)
    _add("PCA",   "pca_components", int)
    _add("K",     "sigmoid_k",      lambda x: str(x).rstrip("0").rstrip(".") if isinstance(x, float) else x)
    _add("minID", "minority_id",    int)
    _add("majID", "majority_id",    int)
    _add("TRJ",   "traj_length",    int)
    _add("REAL",  "real_data_size", int)
    _add("G",     "EXP_GROUP",      lambda s: str(s)[-13:])

    slug = "_".join(parts) if parts else "exp"
    h    = hashlib.sha1(json.dumps(rs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    return f"{slug}_{h}", rs


def _flatten_nested(d: dict, *, sep: str = ".", prefix: str = "") -> dict:
    """Recursively flatten nested dicts: {"a": {"b": 1}} → {"a.b": 1}."""
    out = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_nested(v, sep=sep, prefix=kk))
        else:
            out[kk] = v
    return out


class EpisodeTracker:
    def __init__(
        self,
        run_stats: dict,
        dataset,
        save_dir: str = "runs",
        capture_console: bool = True,
        ckpt_every: int = 5,
        beta_ckpt_every: int = 20,
        compare_metric: str = "reward.wgl_reward",
        beta_factory=None,
        process_label: str = "Process",
        seed: int = 42,
    ):
        self.dataset       = dataset
        self.process_label = process_label
        self.capture_console = capture_console
        self.compare_metric  = compare_metric
        self.ckpt_every      = max(1, int(ckpt_every))
        self.beta_ckpt_every = max(1, int(beta_ckpt_every))
        self.beta_factory    = beta_factory
        self.t0              = time.time()

        self.seed = run_stats.get("seed", seed)

        self.start_ts     = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_id, filtered_stats = _fingerprint_run_stats(dict(run_stats))
        self.uid          = f"{os.getpid()}_{str(uuid.uuid4())[:8]}"

        self.experiment_dir = Path(save_dir) / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir = self.experiment_dir / f"seed_{self.seed}"
        self.seed_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = f"{self.start_ts}_seed{self.seed}_{self.uid}"

        # Experiment-level metadata (merge across seeds)
        meta_path = self.experiment_dir / "meta.json"
        merged = dict(_read_json(meta_path))
        merged.update({"experiment_id": self.experiment_id,
                        "last_started_at": self.start_ts,
                        "config": filtered_stats})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

        seeds_info = _read_json(self.experiment_dir / "seeds.json")
        seeds_set  = set(seeds_info.get("seeds", []))
        seeds_set.add(str(self.seed))
        with open(self.experiment_dir / "seeds.json", "w", encoding="utf-8") as f:
            json.dump({"seeds": sorted(seeds_set)}, f, indent=2)

        # Per-seed metadata
        with open(self.seed_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "seed": self.seed,
                       "started_at": self.start_ts, **run_stats}, f, indent=2)

        # CSV state
        self.csv_path       = self.seed_dir / "metrics.csv"
        self._csv_columns   = None
        self._header_written = False
        self._last_flat_row  = {}

        # Console mirroring
        self._orig_stdout = self._orig_stderr = self._log_file = None
        if self.capture_console:
            self._log_file = open(self.seed_dir / "console.log", "a",
                                  buffering=1, encoding="utf-8", errors="replace")
            self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _TeeLogger(sys.__stdout__, self._log_file)
            sys.stderr = _TeeLogger(sys.__stderr__, self._log_file)
            print(f"[Tracker {self.process_label}] {self.experiment_id}")
            print(f"[Tracker {self.process_label}] seed dir: {self.seed_dir}")
            print(f"[Tracker {self.process_label}] metrics:  {self.csv_path}")

        # Checkpoint state
        self._phase_best   = {}
        self.best_metric   = -float("inf")
        self.snap_dir      = self.seed_dir / "synthetic_snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.best_beta_path      = self.seed_dir / "best_beta_state_dict.pt"
        self.alpha_state_dict_path = self.seed_dir / "alpha_state_dict.pt"

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    # ------------------------------------------------------------------ CSV

    def _ensure_header(self, flat_row: dict):
        if self._header_written:
            return

        def _sort_key(k):
            for i, prefix in enumerate(("reward.", "performance.", "fairness.",
                                         "generation.", "align.", "meta.")):
                if k.startswith(prefix):
                    return (i, k)
            return (9, k)

        self._csv_columns = ["episode", "wall_seconds"] + sorted(flat_row.keys(), key=_sort_key)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(self._csv_columns) + "\n")
        self._header_written = True
        with open(self.seed_dir / "metrics_header.json", "w", encoding="utf-8") as f:
            json.dump({"columns": self._csv_columns}, f, indent=2)
        print(f"[Tracker {self.process_label}] CSV header: {len(self._csv_columns)} columns")

    def _row_for_csv(self, episode_num: int, flat_row: dict) -> dict:
        row = {"episode": episode_num, "wall_seconds": round(time.time() - self.t0, 2)}
        for k in self._csv_columns[2:]:
            v = flat_row.get(k, np.nan)
            if hasattr(v, "item"):
                try: v = v.item()
                except Exception: pass
            row[k] = v
        return row

    # ---------------------------------------------------------------- logging

    def log_episode(
        self,
        episode_num: int,
        diagnostics: dict,
        alignment_metrics: dict | None = None,
        extra_metrics: dict | None = None,
    ):
        flat = _flatten_nested(diagnostics)
        flat.update(_flatten_nested({"align": alignment_metrics or {}}))
        flat.update(_flatten_nested({"meta":  extra_metrics  or {}}))

        self._ensure_header(flat)
        csv_row = self._row_for_csv(episode_num, flat)
        self._last_flat_row = flat

        wgl_r = csv_row.get("reward.wgl_reward",   np.nan)
        f1w   = csv_row.get("performance.f1_macro", np.nan)
        wgl_b = csv_row.get("fairness.wgl_beta",    np.nan)
        eo    = csv_row.get("fairness.eo_gap",       np.nan)
        print(f"[Tracker {self.process_label}] "
              f"Ep {episode_num:4d} | WGL_reward={wgl_r:.4f} | F1w={f1w:.4f} | "
              f"WGL_beta={wgl_b:.4f} | EO={eo:.4f}")

        pd.DataFrame([csv_row]).to_csv(self.csv_path, mode="a", header=False, index=False)

    def _metric_from_flat_row(self, flat_row: dict) -> float:
        key = self.compare_metric
        v = flat_row.get(key)
        if v is None:
            legacy = {
                "wgl_reward":       "reward.wgl_reward",
                "average_reward":   "meta.avg_reward",
                "global_obj":       "reward.wgl_reward",
                "worst_loss_beta":  "fairness.wgl_beta",
                "macro_f1_beta":    "performance.f1_macro",
                "f1_minority_beta": "performance.f1_minority",
            }
            v = flat_row.get(legacy.get(key, "meta.avg_reward"), np.nan)
        try:
            return float(v)
        except Exception:
            return -float("inf")

    # ---------------------------------------------------------- checkpointing

    def maybe_save_synthetic(
        self,
        episode_num: int,
        x_syn,
        y_syn,
        *,
        feature_names=None,
        beta_model=None,
        phase_label: str | None = None,
    ):
        save_snap      = (episode_num % self.ckpt_every == 0)
        save_beta_snap = (episode_num % self.beta_ckpt_every == 0)
        metric_val     = self._metric_from_flat_row(self._last_flat_row)

        if phase_label is not None:
            is_best   = metric_val > self._phase_best.get(phase_label, -float("inf"))
            suffix    = f"_{phase_label}"
            npz_path  = self.seed_dir / f"best_synthetic{suffix}.npz"
            beta_path = self.seed_dir / f"best_beta_state_dict{suffix}.pt"
            meta_path = self.seed_dir / f"best_beta_meta{suffix}.json"
        else:
            is_best   = metric_val > self.best_metric
            suffix    = ""
            npz_path  = self.seed_dir / "best_synthetic.npz"
            beta_path = self.best_beta_path
            meta_path = self.seed_dir / "best_beta_meta.json"

        if not (save_snap or save_beta_snap or is_best):
            return

        # Convert once
        if hasattr(x_syn, "detach"): x_syn = x_syn.detach().cpu().numpy()
        if hasattr(y_syn, "detach"): y_syn = y_syn.detach().cpu().numpy()
        x_syn = np.asarray(x_syn)
        y_syn = np.asarray(y_syn).reshape(-1)
        if feature_names is None:
            feature_names = [f"pca_{i}" for i in range(x_syn.shape[1])]

        if save_snap:
            np.savez_compressed(self.snap_dir / f"synthetic_ep{episode_num:04d}{suffix}.npz",
                                x=x_syn, y=y_syn)
            print(f"[Tracker {self.process_label}] snapshot ep{episode_num:04d}{suffix}")

        if save_beta_snap and beta_model is not None:
            torch.save(beta_model.model.state_dict(),
                       self.snap_dir / f"beta_ep{episode_num:04d}{suffix}.pt")

        if is_best:
            if phase_label is not None:
                self._phase_best[phase_label] = metric_val
            else:
                self.best_metric = metric_val

            np.savez_compressed(npz_path, x=x_syn, y=y_syn)
            with open(meta_path, "w") as f:
                json.dump({"episode": episode_num, "metric": self.compare_metric,
                           "metric_value": metric_val, "phase_label": phase_label,
                           "updated_at": time.strftime("%Y-%m-%d_%H-%M-%S")}, f, indent=2)
            print(f"[Tracker {self.process_label}] "
                  f"new best{suffix} ({self.compare_metric}={metric_val:.6f})")

            if beta_model is not None:
                torch.save(beta_model.model.state_dict(), beta_path)
                print(f"[Tracker {self.process_label}] best β{suffix} → {beta_path}")

    # ----------------------------------------------------------------- final

    def log_final_test(
        self,
        alpha_model,
        x_test,
        y_test,
        f1_thresh: float = 0.5,
        prefer_best_beta: bool = True,
        beta_model=None,
        a_test: torch.Tensor | None = None,
        **_ignored,
    ):
        tests = TestSuite(
            seed_dir=self.seed_dir,
            beta_factory=self.beta_factory,
            best_beta_path=self.best_beta_path,
            dataset=self.dataset,
        )
        return tests.log_final_test(
            alpha_model=alpha_model,
            x_test=x_test,
            y_test=y_test,
            f1_thresh=f1_thresh,
            prefer_best_beta=prefer_best_beta,
            beta_model=beta_model,
            a_test=a_test,
        )

    def save_alpha_state_dict(self, alpha_model, config=None, n_pca_components=None):
        model_obj  = getattr(alpha_model, "model", alpha_model)
        torch.save(model_obj.state_dict(), self.alpha_state_dict_path)
        print(f"[Tracker {self.process_label}] alpha weights → {self.alpha_state_dict_path}")

        if config is not None and n_pca_components is not None:
            meta = {**config, "n_pca_components": int(n_pca_components),
                    "saved_at": time.strftime("%Y-%m-%d_%H-%M-%S")}

            def _jsonable(obj):
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj
                if isinstance(obj, (torch.device, torch.dtype)):
                    return str(obj)
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {str(k): _jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [_jsonable(v) for v in obj]
                if isinstance(obj, Path):
                    return str(obj)
                return str(obj)

            out = self.seed_dir / "ffnn_meta.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(_jsonable(meta), f, indent=2)
            print(f"[Tracker {self.process_label}] alpha config  → {out}")

    def summary_path(self) -> str:
        return str(self.seed_dir)

    def close(self):
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
