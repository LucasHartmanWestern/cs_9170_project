import os, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

class _TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data: str):
        # Write to all streams; only flush on newline to reduce syscalls.
        for s in self.streams:
            s.write(data)
            if data.endswith("\n"):
                s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

class EpisodeTracker:
    """
    Per-run folder with:
      meta.json, metrics.csv, console.log, and synthetic-data checkpoints.
    """
    def __init__(self, run_stats: dict, save_dir: str = "runs",
                 capture_console: bool = True,
                 ckpt_every: int = 5,
                 compare_metric: str = "average_reward",
                 flush_every_episodes: int = 10,
                 snapshot_csv: bool = False):
        # --- run id & dirs ---
        self.start_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = (
            f"{self.start_ts}"
            f"_EP{run_stats.get('EPISODES')}"
            f"_TRJ{run_stats.get('TRAJ_LENGTH')}"
            f"_REAL{run_stats.get('REAL_DATA_SIZE')}"
            f"_BIAS{run_stats.get('BIAS_PCT')}"
            f"_L{run_stats.get('lambda_')}"
        )
        self.run_dir = Path(save_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # --- meta & metrics ---
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump({"run_id": self.run_id, "started_at": self.start_ts, **run_stats}, f, indent=2)

        self.csv_path = self.run_dir / "metrics.csv"
        # open once; line-buffered for reasonably live updates
        self._csv_fh = open(self.csv_path, "w", buffering=1, encoding="utf-8", newline="")
        self._csv_fh.write("episode,average_reward,obj1_minority_f1,obj2_mean,global_f1,wall_seconds\n")

        self.flush_every_episodes = max(1, int(flush_every_episodes))
        self._since_last_flush = 0

        self.episode_rewards = []
        self.t0 = time.time()

        # --- console mirroring ---
        self.capture_console = capture_console
        self._orig_stdout = self._orig_stderr = self._log_file = None
        if self.capture_console:
            # line-buffered file; we flush on newline in _TeeLogger
            self._log_file = open(self.run_dir / "console.log", "a", buffering=1, encoding="utf-8", errors="replace")
            self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _TeeLogger(sys.__stdout__, self._log_file)
            sys.stderr = _TeeLogger(sys.__stderr__, self._log_file)
            print(f"[Tracker] Run: {self.run_id}")
            print(f"[Tracker] Logging metrics to: {self.csv_path}")
            print(f"[Tracker] Mirroring console to: {self.run_dir / 'console.log'}")

        # --- synthetic checkpoint config ---
        self.ckpt_every = max(1, int(ckpt_every))
        self.compare_metric = compare_metric  # column key we’ll compare on
        self.best_metric = -float("inf")
        self.best_meta_path = self.run_dir / "best_synthetic.json"
        self.best_csv_path  = self.run_dir / "best_synthetic.csv"
        self.best_npz_path  = self.run_dir / "best_synthetic.npz"
        self.snap_dir = self.run_dir / "synthetic_snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.snapshot_csv = bool(snapshot_csv)  # optional CSV snapshots

    # Context manager support
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def log_episode(self, episode_num, avg_reward, obj1, obj2_mean, global_f1):
        self.episode_rewards.append(float(avg_reward))
        elapsed = time.time() - self.t0

        # write a single CSV line; keep file open
        self._csv_fh.write(
            f"{episode_num},{float(avg_reward):.6f},"
            f"{float(obj1):.6f},{float(obj2_mean):.6f},"
            f"{float(global_f1):.6f},"
            f"{elapsed:.2f}\n"
        )
        self._since_last_flush += 1
        if self._since_last_flush >= self.flush_every_episodes:
            self._csv_fh.flush()
            self._since_last_flush = 0

        # one-line console summary
        print(
            f"[Tracker] Ep {episode_num:4d} | "
            f"AvgR {float(avg_reward):.4f} | "
            f"F1_minority {float(obj1):.4f} | "
            f"Mean(1-Brier)_synth {float(obj2_mean):.4f} | "
            f"F1_macro {float(global_f1):.4f}"
        )

    def _metric_from_args(self, avg_reward, obj1, obj2_mean, global_f1):
        return {
            "average_reward": float(avg_reward),
            "obj1": float(obj1),
            "obj2_mean": float(obj2_mean),
            "global_f1": float(global_f1)
        }[self.compare_metric]

    def maybe_save_synthetic(self, episode_num, x_syn, y_syn,
                             avg_reward, obj1, obj2_mean, global_f1,
                             feature_names=None):
        """
        Save snapshot every ckpt_every episodes and update 'best' if metric improves.
        x_syn: np.ndarray or torch.Tensor [T, D]
        y_syn: np.ndarray or torch.Tensor [T]
        """
        # Decide if we need to save anything first (avoid conversions when not saving)
        save_snap = (episode_num % self.ckpt_every == 0)
        metric_val = self._metric_from_args(avg_reward, obj1, obj2_mean, global_f1)
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

        # Best-so-far checkpoint
        if is_best:
            self.best_metric = metric_val
            np.savez_compressed(self.best_npz_path, x=x_syn, y=y_syn)
            # CSV for best (nice to inspect)
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

    def summary_path(self): return str(self.run_dir)

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
