# benchmarks/smote_baseline.py
"""
SMOTE baseline — oversamples minority-group positive examples in PCA space,
then trains a standard FFNN (ERM).  Designed to directly test whether simple
near-distribution augmentation is sufficient, as an ablation against the RL
two-phase framework.

Two variants controlled by smote config:
  - phase1_only  (default False): only oversample disadvantaged-group y=1
  - two_phase    (default True):  also oversample majority-group y=0 to match
                                  the RL gen_both_classes=True setup

n_synthetic controls total synthetic samples per phase (default 2000 to match
RL traj_length).  Feature space is PCA-10 throughout, identical to RL.
"""

import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.ffnn_agent2 import FFNNAgent
from dataset import Dataset
from test_suite import TestSuite


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def _smote_augment(x_np: np.ndarray, y_np: np.ndarray, a_np: np.ndarray,
                   minority_group: int, target_class: int,
                   n_synthetic: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SMOTE on (minority_group, target_class) examples and return
    (x_syn, y_syn) as numpy arrays of shape (n_synthetic, D) and (n_synthetic,).

    Uses SMOTE's interpolation in the provided feature space (PCA-10).
    Falls back to random oversampling if fewer than k_neighbors+1 samples exist.
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    mask = (a_np == minority_group) & (y_np == target_class)
    n_real = mask.sum()

    if n_real < 2:
        # Cannot synthesise — return empty
        return np.empty((0, x_np.shape[1])), np.empty((0,))

    # We want to oversample just the target group/class combination.
    # Treat everything else as a single "other" class so SMOTE only
    # interpolates within the target group.
    binary_y = mask.astype(int)          # 1 = target, 0 = everything else
    n_other  = (~mask).sum()

    # SMOTE requires at least k_neighbors+1 samples in the minority class
    k = min(5, n_real - 1)
    n_target_after = n_other + n_real + n_synthetic   # desired total

    try:
        sm = SMOTE(
            sampling_strategy={1: n_real + n_synthetic},
            k_neighbors=k,
            random_state=seed,
        )
        x_res, binary_y_res = sm.fit_resample(x_np, binary_y)
    except Exception:
        ros = RandomOverSampler(
            sampling_strategy={1: n_real + n_synthetic},
            random_state=seed,
        )
        x_res, binary_y_res = ros.fit_resample(x_np, binary_y)

    # Extract only the newly generated rows (SMOTE appends them)
    x_syn = x_res[len(x_np):][:n_synthetic]
    y_syn = np.full(len(x_syn), target_class, dtype=np.int64)
    return x_syn, y_syn


class SMOTEBaselineTrainer:
    def __init__(
        self,
        exp_group: str,
        spec_name: str,
        dataset_name: str,
        seed: int,
        device: str,
        minority_id=None,
        majority_id=None,
        third_id=None,
        bias_pct=None,
        da_pct=None,
        real_data_size: int = 3000,
        ffnn: dict = None,
        smote: dict = None,
        multiclass: bool = False,
        use_pca: bool = True,
        pca_components: int = 10,
        win_seconds: float = 5.0,
        step_seconds: float = 2.5,
        dp_protected_col: str | None = None,
    ):
        self.exp_group    = exp_group
        self.spec_name    = spec_name
        self.dataset_name = dataset_name
        self.seed         = seed
        self.device       = torch.device(device)
        self.minority_id  = minority_id
        self.majority_id  = majority_id
        self.third_id     = third_id
        self.bias_pct     = bias_pct
        self.da_pct       = da_pct
        self.real_data_size = real_data_size
        self.multiclass   = multiclass
        self.win_seconds  = win_seconds
        self.step_seconds = step_seconds
        self.dp_protected_col = dp_protected_col
        self.project_root = _PROJECT_ROOT

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True

        DEFAULT_SMOTE = {
            "n_synthetic":  2000,   # synthetic samples per phase — matches RL traj_length
            "two_phase":    True,   # also oversample majority class (mirrors gen_both_classes)
            "epochs":       100,
            "lr":           0.001,
            "batch_size":   64,
        }
        self.smote_config = {**DEFAULT_SMOTE, **(smote or {})}
        self.ffnn_overrides = ffnn or {}
        self.pca_components = pca_components

        self.dataset = Dataset(
            dataset_name,
            multiclass=multiclass,
            minority_id=minority_id,
            majority_id=majority_id,
            third_id=third_id,
            pca_components=pca_components,
            seed=seed,
            device=self.device,
            use_pca=True,      # always PCA — same space as RL agent
        )

    # ------------------------------------------------------------------ #

    def __call__(self):
        start_time = time.time()

        x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
            train_size=self.real_data_size,
            bias_pct=self.bias_pct,
            da_pct=self.da_pct,
            pca_components=self.pca_components,
            drop_protected=False,
            protected_cols=self.dataset.protected_attributes,
            bias_val=True,
            win_seconds=self.win_seconds,
            step_seconds=self.step_seconds,
            **({"dp_protected_col": self.dp_protected_col} if self.dp_protected_col is not None else {}),
        )
        feature_dim = x_train.shape[1]

        a_train = self.dataset.a_train
        a_val   = self.dataset.a_val
        a_test  = getattr(self.dataset, "a_test", None)

        # minority_id=0 means disadvantaged group is a=0
        minority_group = int(self.minority_id) if self.minority_id is not None else 0
        majority_group = int(self.majority_id) if self.majority_id is not None else 1

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": float(self.smote_config["lr"]),
            "batch_size":    int(self.smote_config["batch_size"]),
            "epochs":        int(self.smote_config["epochs"]),
            "type":          "classification",
            "classes":       [0, 1, 2] if self.multiclass else [0, 1],
            "device":        self.device,
            "seed":          self.seed,
        }
        self.ffnn_config = ffnn_config

        seed_dir, experiment_dir = self._setup_output_dirs()

        log_path = seed_dir / "console.log"
        log_fh   = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        orig_stdout, orig_stderr = sys.stdout, sys.stderr

        class _Tee:
            def __init__(self, *s): self.s = s
            def write(self, d):
                for s in self.s: s.write(d)
                if d.endswith("\n"):
                    for s in self.s: s.flush()
            def flush(self):
                for s in self.s: s.flush()

        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

        try:
            n_synthetic = int(self.smote_config["n_synthetic"])
            two_phase   = bool(self.smote_config["two_phase"])

            print(f"[SMOTE] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[SMOTE] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[SMOTE] smote_config={self.smote_config}")
            print(f"[SMOTE] seed_dir={seed_dir}")

            # Convert to numpy for SMOTE
            x_np = x_train.cpu().numpy()
            y_np = y_train.cpu().numpy().astype(np.int64)
            a_np = a_train.cpu().numpy().astype(np.int64)

            # ---- Phase 1: oversample minority-group positives (y=1) ----
            x_syn1, y_syn1 = _smote_augment(
                x_np, y_np, a_np,
                minority_group=minority_group,
                target_class=1,
                n_synthetic=n_synthetic,
                seed=self.seed,
            )
            print(f"[SMOTE] Phase-1 synthetic (minority y=1): {len(x_syn1)} samples")

            # ---- Phase 2: oversample majority-group negatives (y=0) ----
            if two_phase and len(x_syn1) > 0:
                x_syn0, y_syn0 = _smote_augment(
                    x_np, y_np, a_np,
                    minority_group=majority_group,
                    target_class=0,
                    n_synthetic=n_synthetic,
                    seed=self.seed + 1,
                )
                print(f"[SMOTE] Phase-2 synthetic (majority y=0): {len(x_syn0)} samples")
            else:
                x_syn0, y_syn0 = np.empty((0, feature_dim)), np.empty((0,))
                print("[SMOTE] Phase-2 skipped (two_phase=False or phase-1 empty)")

            # ---- Combine real + synthetic ----
            parts_x = [x_np]
            parts_y = [y_np]
            if len(x_syn1) > 0:
                parts_x.append(x_syn1)
                parts_y.append(y_syn1)
            if len(x_syn0) > 0:
                parts_x.append(x_syn0)
                parts_y.append(y_syn0)

            x_aug = torch.tensor(np.concatenate(parts_x), dtype=torch.float32).to(self.device)
            y_aug = torch.tensor(np.concatenate(parts_y), dtype=torch.long).to(self.device)
            print(f"[SMOTE] Combined training set: {len(x_aug)} samples "
                  f"(real={len(x_np)}, syn1={len(x_syn1)}, syn0={len(x_syn0)})")

            # ---- Train FFNN on augmented data ----
            beta = FFNNAgent(**ffnn_config)
            gen  = torch.Generator(device="cpu").manual_seed(self.seed)
            loader = DataLoader(
                TensorDataset(x_aug, y_aug),
                batch_size=ffnn_config["batch_size"],
                shuffle=True,
                generator=gen,
            )
            beta.train(loader)

            # Save model
            model_path = seed_dir / "smote_model.pt"
            torch.save(beta.model.state_dict(), model_path)
            print(f"[SMOTE] Model saved -> {model_path}")

            # ---- Alpha (ERM on real data only) ----
            erm_config = dict(ffnn_config)
            erm_config["epochs"] = int(self.ffnn_overrides.get("epochs", 100))
            alpha = FFNNAgent(**erm_config)
            gen_a = torch.Generator(device="cpu").manual_seed(self.seed)
            alpha_loader = DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=ffnn_config["batch_size"],
                shuffle=True,
                generator=gen_a,
            )
            alpha.train(alpha_loader)

            # ---- TestSuite ----
            _ffnn_cfg = dict(ffnn_config)

            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                return ag

            tests = TestSuite(
                seed_dir=seed_dir,
                experiment_dir=experiment_dir,
                seed=self.seed,
                run_id=f"smote_seed{self.seed}",
                beta_factory=_beta_factory,
                best_beta_path=model_path,
                alpha_factory=lambda: FFNNAgent(**_ffnn_cfg),
                dataset=self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model=alpha,
                x_test=x_test,
                y_test=y_test,
                f1_thresh=0.5,
                prefer_best_beta=True,
                beta_model=beta,
                x_train=x_train,
                y_train=y_train,
                bias_pct=self.bias_pct,
                train_size=self.real_data_size,
                seed=self.seed,
                a_test=a_test,
            )

            summary = {
                "baseline":     "smote",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "n_synthetic":  n_synthetic,
                "two_phase":    two_phase,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items() if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[SMOTE] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[SMOTE] Results in {seed_dir}")

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_fh.close()

    # ------------------------------------------------------------------ #

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "smote",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "smote_config": self.smote_config,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        with open(seed_dir / "meta.json", "w") as f:
            json.dump({**meta, "seed": self.seed}, f, indent=2)

        return seed_dir, experiment_dir
