import os, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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
      meta.json, metrics.csv, console.log, synthetic snapshots, best synthetic, and best β weights.
    """
    def __init__(self, run_stats: dict, save_dir: str = "runs",
                 capture_console: bool = True,
                 ckpt_every: int = 5,
                 compare_metric: str = "average_reward",
                 flush_every_episodes: int = 10,
                 snapshot_csv: bool = False,
                 beta_factory=None):
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
        self.compare_metric = compare_metric  # which column we compare for "best"
        self.best_metric = -float("inf")
        self.best_meta_path = self.run_dir / "best_synthetic.json"
        self.best_csv_path  = self.run_dir / "best_synthetic.csv"
        self.best_npz_path  = self.run_dir / "best_synthetic.npz"
        self.snap_dir = self.run_dir / "synthetic_snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.snapshot_csv = bool(snapshot_csv)  # optional CSV snapshots

        # --- β factory & best-β checkpoint paths ---
        self.beta_factory = beta_factory
        self.best_beta_path = self.run_dir / "best_beta_state_dict.pt"
        self.best_beta_meta = self.run_dir / "best_beta_meta.json"

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
                             feature_names=None, beta_model=None):
        """
        Save snapshot every ckpt_every episodes and update 'best' if metric improves.
        x_syn: np.ndarray or torch.Tensor [T, D]
        y_syn: np.ndarray or torch.Tensor [T]
        beta_model: optional agent (e.g., FFNNAgent); if provided and this is BEST,
                    also save its .model state_dict as best_beta_state_dict.pt
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

    def log_final_test(self, alpha_model, x_test, y_test, f1_thresh: float = 0.5,
                       prefer_best_beta: bool = True, beta_model=None):
        """
        End-of-run θ_test evaluation.
        If prefer_best_beta and a checkpoint exists and a beta_factory was provided,
        instantiate a fresh β, load best weights, and evaluate.
        Otherwise, if beta_model is given, evaluate that. Always evaluates α.

        Computes and logs:
          F1(minority=1), F1(majority=0), F1(weighted), F1(macro), Brier
        """
        # --- pick which beta to evaluate ---
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
                logits = agent.model(x)             # [N,2]
                probs  = torch.softmax(logits, -1)  # [N,2]
                return probs[..., 1]                # [N]

        def _all_f1_from_probs(y_true, p1, threshold=0.5):
            """
            Returns: f1_minority (class 1), f1_majority (class 0),
                     f1_weighted, f1_macro
            """
            y_true = y_true.to(p1.device).long()
            y_pred_pos = (p1 >= threshold).long()  # predicted class-1

            # Confusion for class 1 (minority)
            tp1 = ((y_pred_pos == 1) & (y_true == 1)).sum().float()
            fp1 = ((y_pred_pos == 1) & (y_true == 0)).sum().float()
            fn1 = ((y_pred_pos == 0) & (y_true == 1)).sum().float()
            prec1 = tp1 / (tp1 + fp1 + eps)
            rec1  = tp1 / (tp1 + fn1 + eps)
            f1_1  = (2 * prec1 * rec1) / (prec1 + rec1 + eps)

            # Confusion for class 0 (majority): y_pred_neg = 1 - y_pred_pos
            tp0 = ((y_pred_pos == 0) & (y_true == 0)).sum().float()
            fp0 = ((y_pred_pos == 0) & (y_true == 1)).sum().float()
            fn0 = ((y_pred_pos == 1) & (y_true == 0)).sum().float()
            prec0 = tp0 / (tp0 + fn0 + eps)
            rec0  = tp0 / (tp0 + fp0 + eps)
            f1_0  = (2 * prec0 * rec0) / (prec0 + rec0 + eps)

            # Supports
            n1 = (y_true == 1).sum().float()
            n0 = (y_true == 0).sum().float()
            n  = n0 + n1 + eps

            f1_macro    = 0.5 * (f1_1 + f1_0)
            f1_weighted = (f1_1 * (n1 / n)) + (f1_0 * (n0 / n))

            # Convert to plain floats
            return float(f1_1), float(f1_0), float(f1_weighted), float(f1_macro)

        def _brier_mean(y_true, p1):
            y_true = y_true.to(p1.device).float()
            return float(((p1 - y_true) ** 2).mean())

        with torch.no_grad():
            # --- Alpha on TEST ---
            p1_alpha = _p1_from_agent(alpha_model, x_test)
            a_f1_min, a_f1_maj, a_f1_w, a_f1_macro = _all_f1_from_probs(y_test, p1_alpha, f1_thresh)
            a_brier = _brier_mean(y_test, p1_alpha)

            # --- Beta on TEST ---
            if beta_for_eval is not None:
                p1_beta = _p1_from_agent(beta_for_eval, x_test)
                b_f1_min, b_f1_maj, b_f1_w, b_f1_macro = _all_f1_from_probs(y_test, p1_beta, f1_thresh)
                b_brier = _brier_mean(y_test, p1_beta)
            else:
                b_f1_min = b_f1_maj = b_f1_w = b_f1_macro = b_brier = float('nan')

        # Console summary
        print("\n[TEST] Alpha -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
              % (a_f1_min, a_f1_maj, a_f1_w, a_f1_macro, a_brier))
        if beta_for_eval is not None:
            print("[TEST] Beta  -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
                  % (b_f1_min, b_f1_maj, b_f1_w, b_f1_macro, b_brier))
            winner = "beta" if b_f1_min > a_f1_min else "alpha"
            print(f"[TEST] Winner (by F1 minority): {winner}")
        else:
            print("[TEST] Beta  -> (no checkpoint/factory)")

        # Append to dedicated CSV (keeps per-episode metrics.csv unchanged)
        final_csv = self.run_dir / "final_test_metrics.csv"
        row = {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
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

            "winner_by_f1_minority": ("beta" if (not np.isnan(b_f1_min) and b_f1_min > a_f1_min) else "alpha")
        }
        pd.DataFrame([row]).to_csv(final_csv, mode="a", header=not final_csv.exists(), index=False)
        print(f"[Tracker] Final test metrics appended to: {final_csv}")
        return row
