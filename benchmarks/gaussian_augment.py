# benchmarks/gaussian_augment.py
"""
Gaussian augmentation baseline — the 'no-search' ablation against CMA-ES.

Fits a diagonal Gaussian N(mu, diag(sigma^2)) to the real disadvantaged
minority training samples in PCA space, then draws n_synthetic i.i.d. samples
from it. Trains the same FFNN as CMA-ES beta (20 epochs, same architecture).

Phase 1: augment disadvantaged positive class  (mirrors CMA-ES phase 1)
Phase 2: augment advantaged negative class     (mirrors gen_both_classes=True)

This isolates the contribution of CMA-ES optimization vs. simply augmenting
with a Gaussian fitted to the real minority distribution (CMA-ES episode 0).
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


def _gaussian_sample(
    x_np: np.ndarray,
    y_np: np.ndarray,
    a_np: np.ndarray,
    group: int,
    target_class: int,
    n_synthetic: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a diagonal Gaussian to (group, target_class) samples and draw
    n_synthetic i.i.d. samples. Returns (x_syn, y_syn) as numpy arrays.
    """
    mask = (a_np == group) & (y_np == target_class)
    x_group = x_np[mask]

    if x_group.shape[0] < 2:
        return np.empty((0, x_np.shape[1])), np.empty((0,), dtype=np.int64)

    mu    = x_group.mean(axis=0)
    sigma = x_group.std(axis=0).clip(min=1e-6)

    rng     = np.random.default_rng(seed)
    noise   = rng.standard_normal((n_synthetic, x_np.shape[1]))
    x_syn   = mu + sigma * noise
    y_syn   = np.full(n_synthetic, target_class, dtype=np.int64)
    return x_syn, y_syn


class GaussianAugmentTrainer:
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
        n_synthetic: int = 2000,
        multiclass: bool = False,
        use_pca: bool = True,
        pca_components: int = 10,
        win_seconds: float = 5.0,
        step_seconds: float = 2.5,
        dp_protected_col: str | None = None,
    ):
        self.exp_group      = exp_group
        self.spec_name      = spec_name
        self.dataset_name   = dataset_name
        self.seed           = seed
        self.device         = torch.device(device)
        self.minority_id    = minority_id
        self.majority_id    = majority_id
        self.third_id       = third_id
        self.bias_pct       = bias_pct
        self.da_pct         = da_pct
        self.real_data_size = real_data_size
        self.n_synthetic    = int(n_synthetic)
        self.multiclass     = multiclass
        self.win_seconds    = win_seconds
        self.step_seconds   = step_seconds
        self.dp_protected_col = dp_protected_col
        self.project_root   = _PROJECT_ROOT
        self.pca_components = pca_components
        self.ffnn_overrides = ffnn or {}

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark     = False
            torch.backends.cudnn.deterministic = True

        self.dataset = Dataset(
            dataset_name,
            multiclass=multiclass,
            minority_id=minority_id,
            majority_id=majority_id,
            third_id=third_id,
            pca_components=pca_components,
            seed=seed,
            device=self.device,
            use_pca=True,
        )

    def __call__(self):
        start_time = time.time()

        x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
            train_size=self.real_data_size,
            bias_pct=self.bias_pct,
            da_pct=self.da_pct,
            pca_components=self.pca_components,
            drop_protected=False,
            protected_cols=self.dataset.protected_attributes,
            bias_val=False,   # match CMA-ES default
            win_seconds=self.win_seconds,
            step_seconds=self.step_seconds,
            **({"dp_protected_col": self.dp_protected_col} if self.dp_protected_col is not None else {}),
        )
        feature_dim = x_train.shape[1]

        a_train = self.dataset.a_train
        a_test  = getattr(self.dataset, "a_test", None)

        minority_group = int(self.minority_id) if self.minority_id is not None else 0
        majority_group = int(self.majority_id) if self.majority_id is not None else 1

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": float(self.ffnn_overrides.get("learning_rate", 0.001)),
            "batch_size":    int(self.ffnn_overrides.get("batch_size", 64)),
            "epochs":        int(self.ffnn_overrides.get("epochs", 20)),
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
            print(f"[GaussAugment] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[GaussAugment] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[GaussAugment] n_synthetic={self.n_synthetic}  pca_components={self.pca_components}")

            x_np = x_train.cpu().numpy()
            y_np = y_train.cpu().numpy().astype(np.int64)
            a_np = a_train.cpu().numpy().astype(np.int64)

            # ---- Phase 1: sample from Gaussian fitted to minority-group positives ----
            x_syn1, y_syn1 = _gaussian_sample(
                x_np, y_np, a_np,
                group=minority_group,
                target_class=1,
                n_synthetic=self.n_synthetic,
                seed=self.seed,
            )
            n_real_min = int(((a_np == minority_group) & (y_np == 1)).sum())
            print(f"[GaussAugment] Phase-1: {n_real_min} real minority positives "
                  f"-> {len(x_syn1)} synthetic samples")

            # ---- Phase 2: sample from Gaussian fitted to majority-group negatives ----
            x_syn0, y_syn0 = _gaussian_sample(
                x_np, y_np, a_np,
                group=majority_group,
                target_class=0,
                n_synthetic=self.n_synthetic,
                seed=self.seed + 1,
            )
            print(f"[GaussAugment] Phase-2: {len(x_syn0)} synthetic majority negatives")

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
            print(f"[GaussAugment] Combined: {len(x_aug)} total "
                  f"(real={len(x_np)}, syn1={len(x_syn1)}, syn0={len(x_syn0)})")

            # ---- Train beta on augmented data ----
            beta = FFNNAgent(**ffnn_config)
            gen  = torch.Generator(device="cpu").manual_seed(self.seed)
            loader = DataLoader(
                TensorDataset(x_aug, y_aug),
                batch_size=ffnn_config["batch_size"],
                shuffle=True,
                generator=gen,
            )
            beta.train(loader)

            model_path = seed_dir / "beta_model.pt"
            torch.save(beta.model.state_dict(), model_path)

            # ---- Alpha (ERM on real data only, same epochs as beta) ----
            alpha = FFNNAgent(**ffnn_config)
            gen_a = torch.Generator(device="cpu").manual_seed(self.seed)
            alpha.train(DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=ffnn_config["batch_size"],
                shuffle=True,
                generator=gen_a,
            ))

            # ---- Evaluate ----
            _ffnn_cfg = dict(ffnn_config)

            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(torch.load(model_path, map_location=self.device))
                return ag

            tests = TestSuite(
                seed_dir=seed_dir,
                experiment_dir=experiment_dir,
                seed=self.seed,
                run_id=f"gauss_augment_seed{self.seed}",
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
                "baseline":     "gaussian_augment",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "n_synthetic":  self.n_synthetic,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items() if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"[GaussAugment] Done. seed={self.seed} | "
                  f"time={time.time()-start_time:.1f}s")

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_fh.close()

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "paper_results_v4" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "gaussian_augment",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "n_synthetic":  self.n_synthetic,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        return seed_dir, experiment_dir
