# benchmarks/gaussian_ot_repair.py
"""
Gaussian OT Repair baseline (Gordaliza et al. 2019,
"Obtaining Fairness using Optimal Transport Theory").

Algorithm (Gaussian approximation):
  1. Fit N(μ_g, Σ_g) to each group's training features.
  2. Compute the Wasserstein barycenter N(μ*, Σ*) via fixed-point iteration:
         Σ*  = Σ_g w_g (Σ*^{1/2} Σ_g Σ*^{1/2})^{1/2}
         μ*  = Σ_g w_g μ_g
  3. Compute the affine OT map for each group:
         T_g(x) = μ* + A_g (x - μ_g)
         A_g = Σ_g^{-1/2} (Σ_g^{1/2} Σ* Σ_g^{1/2})^{1/2} Σ_g^{-1/2}
  4. Partially repair features:
         x_rep = (1 - λ) x + λ T_g(x)
  5. Train a standard ERM classifier on (X_repaired, y).
  6. Repair test features with the same maps, evaluate.
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.ffnn_agent2 import FFNNAgent
from dataset import Dataset
from test_suite import TestSuite


# ------------------------------------------------------------------ #
#  Matrix utilities (all operate on 2-D torch tensors)               #
# ------------------------------------------------------------------ #

def _sym_mat_sqrt(A: torch.Tensor, reg: float = 1e-7) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    A = 0.5 * (A + A.T)                               # enforce symmetry
    A = A + reg * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp(min=0.0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _sym_mat_sqrt_inv(A: torch.Tensor, reg: float = 1e-7) -> torch.Tensor:
    """Symmetric PSD matrix square-root inverse."""
    A = 0.5 * (A + A.T)
    A = A + reg * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp(min=reg)
    return eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T


def _wasserstein_barycenter_cov(
    covs: list,
    weights: torch.Tensor,
    n_iter: int = 100,
    reg: float = 1e-7,
    tol: float = 1e-9,
) -> torch.Tensor:
    """
    Fixed-point iteration for the Wasserstein barycenter covariance:
        S_{t+1} = Σ_g w_g (S_t^{1/2} Σ_g S_t^{1/2})^{1/2}

    Converges to the unique PD barycenter covariance.
    """
    # Initialise with weighted mean of covariances
    S = sum(float(w) * C for w, C in zip(weights, covs))

    for _ in range(n_iter):
        S_sqrt = _sym_mat_sqrt(S, reg=reg)
        S_new  = torch.zeros_like(S)
        for w, C in zip(weights, covs):
            M      = S_sqrt @ C @ S_sqrt          # S^{1/2} Σ_g S^{1/2}
            S_new += float(w) * _sym_mat_sqrt(M, reg=reg)

        diff = torch.norm(S_new - S)
        S    = S_new
        if diff < tol * (torch.norm(S) + 1e-12):
            break

    return S


def _ot_map_matrix(
    cov_g: torch.Tensor,
    cov_star: torch.Tensor,
    reg: float = 1e-7,
) -> torch.Tensor:
    """
    Affine OT map matrix A_g:
        T_g(x) = μ* + A_g (x - μ_g)
        A_g = Σ_g^{-1/2} (Σ_g^{1/2} Σ* Σ_g^{1/2})^{1/2} Σ_g^{-1/2}
    """
    Sg_sqrt     = _sym_mat_sqrt(cov_g,     reg=reg)
    Sg_sqrt_inv = _sym_mat_sqrt_inv(cov_g, reg=reg)
    M           = Sg_sqrt @ cov_star @ Sg_sqrt
    return Sg_sqrt_inv @ _sym_mat_sqrt(M, reg=reg) @ Sg_sqrt_inv


def _repair_features(
    X:       torch.Tensor,           # [N, D]  float32
    a:       torch.Tensor,           # [N]     long, {0, …, G-1}
    mu_list: list,                   # G × [D] float64 tensors
    A_list:  list,                   # G × [D, D] float64 tensors
    mu_star: torch.Tensor,           # [D]     float64
    lam:     float,
) -> torch.Tensor:
    """Apply partial repair: x_rep = (1-λ)x + λ T_g(x).
    OT math is done in float64; result is cast back to X's original dtype."""
    orig_dtype = X.dtype
    X64   = X.double()
    X_rep = X64.clone()
    for g, (mu_g, A_g) in enumerate(zip(mu_list, A_list)):
        mask = (a == g)
        if mask.sum() == 0:
            continue
        X_g         = X64[mask]                            # [n_g, D]
        T_g         = mu_star + (X_g - mu_g) @ A_g.T      # [n_g, D]
        X_rep[mask] = (1.0 - lam) * X_g + lam * T_g
    return X_rep.to(orig_dtype)


# ------------------------------------------------------------------ #
#  Main trainer                                                       #
# ------------------------------------------------------------------ #

class GaussianOTRepairTrainer:
    def __init__(
        self,
        exp_group:     str,
        spec_name:     str,
        dataset_name:  str,
        seed:          int,
        device:        str,
        # data
        minority_id  = None,
        majority_id  = None,
        third_id     = None,
        bias_pct     = None,
        da_pct       = None,
        real_data_size: int = 3000,
        # model
        ffnn:         dict = None,
        # OT-Repair hyperparams
        ot_repair:    dict = None,
        # misc
        multiclass:   bool = False,
        use_pca:      bool = False,
        pca_components: int = 10,
        win_seconds:  float = 5.0,
        step_seconds: float = 2.5,
        dp_protected_col: str | None = None,
        acs_states: list = None,
        fold_idx: int = None,
        n_folds: int = 5,
    ):
        self.exp_group     = exp_group
        self.spec_name     = spec_name
        self.dataset_name  = dataset_name
        self.seed          = seed
        self.device        = torch.device(device)
        self.minority_id   = minority_id
        self.majority_id   = majority_id
        self.third_id      = third_id
        self.bias_pct      = bias_pct
        self.da_pct        = da_pct
        self.real_data_size = real_data_size
        self.multiclass    = multiclass
        self.win_seconds   = win_seconds
        self.step_seconds  = step_seconds
        self.dp_protected_col = dp_protected_col
        self.acs_states    = acs_states
        self.fold_idx      = fold_idx
        self.n_folds       = n_folds
        self.project_root  = _PROJECT_ROOT

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark   = False
            torch.backends.cudnn.deterministic = True

        DEFAULT_OT = {
            "lam":        1.0,    # repair strength ∈ [0, 1]
            "n_groups":   2,
            "cov_reg":    1e-5,   # regularisation added to each group covariance
            "barycenter_iters": 100,
            "repair_test": True,  # also repair test features before evaluation
        }
        self.ot_config  = {**DEFAULT_OT, **(ot_repair or {})}
        self.ffnn_overrides = ffnn or {}
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.dataset = Dataset(
            dataset_name,
            multiclass   = multiclass,
            minority_id  = minority_id,
            majority_id  = majority_id,
            third_id     = third_id,
            pca_components = pca_components,
            seed         = seed,
            device       = self.device,
            use_pca      = use_pca,
        )

    # ---------------------------------------------------------------- #

    def __call__(self):
        start_time = time.time()

        # ---- data ----
        x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
            train_size     = self.real_data_size,
            bias_pct       = self.bias_pct,
            da_pct         = self.da_pct,
            pca_components = self.pca_components,
            drop_protected = False,
            protected_cols = self.dataset.protected_attributes,
            bias_val       = True,
            win_seconds    = self.win_seconds,
            step_seconds   = self.step_seconds,
            **({"dp_protected_col": self.dp_protected_col} if self.dp_protected_col is not None else {}),
            **({"acs_states": self.acs_states} if self.acs_states is not None else {}),
            **({"fold_idx": self.fold_idx, "n_folds": self.n_folds} if self.fold_idx is not None else {}),
        )
        feature_dim = x_train.shape[1]

        a_train = self.dataset.a_train
        a_val   = self.dataset.a_val
        a_test  = getattr(self.dataset, "a_test", None)

        # ---- FFNN config ----
        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": self.ffnn_overrides.get("lr", 1e-3),
            "batch_size":    self.ffnn_overrides.get("batch_size", 64),
            "epochs":        self.ffnn_overrides.get("epochs", 100),
            "type":          "classification",
            "classes":       [0, 1, 2] if self.multiclass else [0, 1],
            "device":        self.device,
            "seed":          self.seed,
        }
        self.ffnn_config = ffnn_config

        # ---- output dirs ----
        seed_dir, experiment_dir = self._setup_output_dirs()

        # Console mirror
        log_fh = open(seed_dir / "console.log", "a", buffering=1,
                      encoding="utf-8", errors="replace")
        orig_out, orig_err = sys.stdout, sys.stderr

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
            print(f"[OTRepair] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[OTRepair] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[OTRepair] ot_config={self.ot_config}")
            print(f"[OTRepair] seed_dir={seed_dir}")

            lam        = float(self.ot_config["lam"])
            n_groups   = int(self.ot_config["n_groups"])
            cov_reg    = float(self.ot_config["cov_reg"])
            bc_iters   = int(self.ot_config["barycenter_iters"])
            repair_test = bool(self.ot_config["repair_test"])

            # ---- fit Gaussian per group (on training data only) ----
            mu_list, cov_list = self._fit_group_gaussians(
                x_train, a_train, n_groups, cov_reg
            )

            # ---- Wasserstein barycenter ----
            weights = torch.ones(n_groups, dtype=torch.float64) / n_groups
            mu_star  = sum(float(w) * mu for w, mu in zip(weights, mu_list))
            cov_star = _wasserstein_barycenter_cov(
                cov_list, weights, n_iter=bc_iters, reg=cov_reg
            )
            print(f"[OTRepair] Barycenter computed. μ* norm={mu_star.norm():.4f}")

            # ---- OT map matrices per group ----
            A_list = []
            for g, cov_g in enumerate(cov_list):
                A_g = _ot_map_matrix(cov_g, cov_star, reg=cov_reg)
                A_list.append(A_g)
                print(f"[OTRepair] Group {g}: A_g computed, "
                      f"||A_g - I||={torch.norm(A_g - torch.eye(A_g.shape[0], dtype=A_g.dtype, device=A_g.device)):.4f}")

            # Save the repair summary
            repair_meta = {
                "lam":       lam,
                "n_groups":  n_groups,
                "cov_reg":   cov_reg,
                "mu_star":   mu_star.cpu().tolist(),
                "mu_groups": [m.cpu().tolist() for m in mu_list],
            }
            with open(seed_dir / "ot_repair_meta.json", "w") as f:
                json.dump(repair_meta, f, indent=2)

            # ---- repair features ----
            x_train_rep = _repair_features(
                x_train, a_train, mu_list, A_list, mu_star, lam
            )
            x_val_rep   = _repair_features(
                x_val, a_val, mu_list, A_list, mu_star, lam
            )

            if repair_test and a_test is not None:
                x_test_eval = _repair_features(
                    x_test, a_test, mu_list, A_list, mu_star, lam
                )
                print("[OTRepair] Test features repaired.")
            else:
                x_test_eval = x_test
                print("[OTRepair] Test features NOT repaired (repair_test=False or no a_test).")

            # ---- train ERM on repaired train features ----
            model_agent = self._train_erm(ffnn_config, x_train_rep, y_train)
            print("[OTRepair] ERM trained on repaired features.")

            # ---- alpha: ERM on original (unrepaired) features for comparison ----
            alpha_agent = self._train_erm(ffnn_config, x_train, y_train)
            print("[OTRepair] Alpha (ERM on original features) trained.")

            # ---- save model ----
            model_path = seed_dir / "otrep_model.pt"
            torch.save(model_agent.model.state_dict(), model_path)

            _ffnn_cfg = dict(ffnn_config)
            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                return ag

            # ---- final evaluation ----
            tests = TestSuite(
                seed_dir       = seed_dir,
                experiment_dir = experiment_dir,
                seed           = self.seed,
                run_id         = f"otrep_seed{self.seed}",
                beta_factory   = _beta_factory,
                best_beta_path = model_path,
                alpha_factory  = lambda: FFNNAgent(**_ffnn_cfg),
                dataset        = self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model       = alpha_agent,
                x_test            = x_test_eval,   # beta evaluated on repaired test features
                x_test_alpha      = x_test,         # alpha evaluated on original test features
                y_test            = y_test,
                f1_thresh         = 0.5,
                prefer_best_beta  = True,
                beta_model        = model_agent,
                x_train           = x_train_rep,
                y_train           = y_train,
                bias_pct          = self.bias_pct,
                train_size        = self.real_data_size,
                seed              = self.seed,
                a_test            = a_test,
            )

            summary = {
                "baseline":     "gaussian_ot_repair",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "lam":          lam,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items()
                   if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[OTRepair] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[OTRepair] Results in {seed_dir}")

        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
            log_fh.close()

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _fit_group_gaussians(
        self,
        X:        torch.Tensor,
        a:        torch.Tensor,
        n_groups: int,
        cov_reg:  float,
    ):
        """
        Fit (μ_g, Σ_g) for each group on the CPU using float64 for
        numerical stability, then move to self.device.
        """
        mu_list  = []
        cov_list = []
        X64 = X.double().cpu()
        a_cpu = a.cpu()

        for g in range(n_groups):
            mask = (a_cpu == g)
            n_g  = int(mask.sum().item())
            if n_g == 0:
                raise RuntimeError(
                    f"[OTRepair] Group {g} has no samples in training data. "
                    "Check minority_id/majority_id and bias settings."
                )
            X_g = X64[mask]      # [n_g, D]
            mu  = X_g.mean(dim=0)

            if n_g == 1:
                cov = torch.zeros(X_g.shape[1], X_g.shape[1], dtype=torch.float64)
            else:
                X_c  = X_g - mu
                cov  = (X_c.T @ X_c) / (n_g - 1)

            # Regularise
            cov = cov + cov_reg * torch.eye(cov.shape[0], dtype=torch.float64)

            mu_list.append(mu.to(self.device))
            cov_list.append(cov.to(self.device))

            print(f"[OTRepair] Group {g}: n={n_g}, "
                  f"||μ||={mu.norm():.4f}, "
                  f"cov_diag=[{cov.diag().min():.4f}, {cov.diag().max():.4f}]")

        return mu_list, cov_list

    def _train_erm(self, ffnn_config: dict, x_train: torch.Tensor, y_train: torch.Tensor) -> FFNNAgent:
        agent = FFNNAgent(**ffnn_config)
        gen   = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size = int(ffnn_config["batch_size"]),
            shuffle    = True,
            generator  = gen,
        )
        agent.train(loader)
        return agent

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "gaussian_ot_repair",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "ot_config":    self.ot_config,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        with open(seed_dir / "meta.json", "w") as f:
            json.dump({**meta, "seed": self.seed}, f, indent=2)

        return seed_dir, experiment_dir
