# benchmarks/ctgan_baseline.py
"""
CTGAN baseline for fairness evaluation.

Algorithm:
  1. Train CTGAN on biased training data (same data all other methods see).
  2. Conditionally sample synthetic disadvantaged-group positive (y=1, a=minority_id) examples.
  3. Concatenate synthetic samples with real biased training data.
  4. Train a standard ERM classifier on the augmented dataset.
  5. Evaluate fairness / utility with TestSuite.

Requires: sdv >= 1.0  (tested with 1.27)
"""

import csv
import json
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

from agents.ffnn_agent import FFNNAgent
from dataset import Dataset
from test_suite import TestSuite


# ------------------------------------------------------------------ #
#  CTGAN helpers                                                      #
# ------------------------------------------------------------------ #

def _fit_ctgan(X_np: np.ndarray, y_np: np.ndarray, a_np: np.ndarray,
               label_col: str, group_col: str,
               ctgan_epochs: int, batch_size: int, seed: int):
    """
    Fit a CTGAN on the joint (X, y, a) distribution.
    Label and group columns are stored as strings so SDV treats them as categorical.
    Returns the fitted synthesizer and the list of feature column names.
    """
    import pandas as pd
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    n_feat = X_np.shape[1]
    feat_cols = [f"f{i}" for i in range(n_feat)]

    df = pd.DataFrame(X_np, columns=feat_cols)
    df[label_col] = y_np.astype(str)   # categorical label
    df[group_col] = a_np.astype(str)   # categorical group

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(column_name=label_col, sdtype="categorical")
    metadata.update_column(column_name=group_col, sdtype="categorical")

    synth = CTGANSynthesizer(
        metadata,
        epochs=ctgan_epochs,
        batch_size=batch_size,
        verbose=False,
    )
    synth.fit(df)
    return synth, feat_cols


def _sample_conditional(synth, label_col: str, group_col: str,
                        label_value: int, group_value: int,
                        n_samples: int, feat_cols: list) -> np.ndarray:
    """
    Sample `n_samples` rows conditioned on (y=label_value, a=group_value).
    Returns a float32 numpy array of shape [n_samples, n_features].
    """
    from sdv.sampling import Condition

    cond = Condition(
        column_values={label_col: str(label_value), group_col: str(group_value)},
        num_rows=int(n_samples),
    )
    out_df = synth.sample_from_conditions([cond])
    out_df = out_df.replace([np.inf, -np.inf], np.nan).dropna()
    return out_df[feat_cols].values.astype(np.float32)


# ------------------------------------------------------------------ #
#  Main trainer                                                       #
# ------------------------------------------------------------------ #

class CTGANBaselineTrainer:
    def __init__(
        self,
        exp_group:      str,
        spec_name:      str,
        dataset_name:   str,
        seed:           int,
        device:         str,
        # data
        minority_id  = None,
        majority_id  = None,
        da_pct       = None,
        real_data_size: int = 3000,
        # model
        ffnn:           dict = None,
        # CTGAN hyperparams
        ctgan:          dict = None,
        # misc
        use_pca:        bool = False,
        pca_components: int  = 10,
        dp_protected_col: str | None = None,
        fold_idx: int = None,
        n_folds: int = 5,
        fold_rng_seed: int = None,
    ):
        self.exp_group      = exp_group
        self.spec_name      = spec_name
        self.dataset_name   = dataset_name
        self.seed           = seed
        self.device         = torch.device(device)
        self.minority_id    = minority_id
        self.majority_id    = majority_id
        self.da_pct         = da_pct
        self.real_data_size = real_data_size
        self.dp_protected_col = dp_protected_col
        self.fold_idx       = fold_idx
        self.n_folds        = n_folds
        self.fold_rng_seed  = fold_rng_seed
        self.project_root   = _PROJECT_ROOT

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

        DEFAULT_CTGAN = {
            "epochs":               300,
            "batch_size":           500,
            "n_synthetic":          2000,  # total synthetic disadvantaged-positive samples to add
            "label_col":            "__y__",
            "group_col":            "__a__",
        }
        self.ctgan_config   = {**DEFAULT_CTGAN, **(ctgan or {})}
        self.ffnn_overrides = ffnn or {}
        self.use_pca        = use_pca
        self.pca_components = pca_components

        self.dataset = Dataset(
            dataset_name,
            minority_id    = minority_id,
            majority_id    = majority_id,
            pca_components = pca_components,
            seed           = seed,
            device         = self.device,
            use_pca        = use_pca,
        )

    # ---------------------------------------------------------------- #

    def __call__(self):
        start_time = time.time()

        # ---- data ----
        x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
            train_size     = self.real_data_size,
            da_pct         = self.da_pct,
            pca_components = self.pca_components,
            drop_protected = False,
            protected_cols = self.dataset.protected_attributes,
            **({"dp_protected_col": self.dp_protected_col} if self.dp_protected_col is not None else {}),
            **({"fold_idx": self.fold_idx, "n_folds": self.n_folds,
                "fold_rng_seed": self.fold_rng_seed} if self.fold_idx is not None else {}),
        )
        feature_dim = x_train.shape[1]

        a_test = getattr(self.dataset, "a_test", None)

        # ---- FFNN config ----
        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size": 2,
            "learning_rate": self.ffnn_overrides.get("learning_rate", 1e-3),
            "batch_size":    self.ffnn_overrides.get("batch_size", 64),
            "epochs":        self.ffnn_overrides.get("epochs", 100),
            "type":          "classification",
            "classes": [0, 1],
            "device":        self.device,
            "seed":          self.seed,
        }
        self.ffnn_config = ffnn_config

        # ---- output dirs ----
        seed_dir, experiment_dir = self._setup_output_dirs()

        # Console mirror
        log_fh   = open(seed_dir / "console.log", "a", buffering=1,
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
            print(f"[CTGAN] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[CTGAN] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[CTGAN] ctgan_config={self.ctgan_config}")
            print(f"[CTGAN] seed_dir={seed_dir}")

            X_train_np = x_train.cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            a_train_np = self.dataset.a_train.cpu().numpy()

            n_pos  = int((y_train_np == 1).sum())
            n_neg  = int((y_train_np == 0).sum())
            n_da_pos = int(((y_train_np == 1) & (a_train_np == self.minority_id)).sum())
            print(f"[CTGAN] Training set: n_pos={n_pos}, n_neg={n_neg}, "
                  f"n_disadvantaged_pos={n_da_pos}, total={len(y_train_np)}")

            label_col    = self.ctgan_config["label_col"]
            group_col    = self.ctgan_config["group_col"]
            ctgan_epochs = int(self.ctgan_config["epochs"])
            ctgan_batch  = int(self.ctgan_config["batch_size"])
            n_synthetic  = int(self.ctgan_config["n_synthetic"])

            # ---- fit CTGAN on biased training data (with group column) ----
            print(f"[CTGAN] Fitting CTGAN for {ctgan_epochs} epochs …")
            t0 = time.time()
            synth, feat_cols = _fit_ctgan(
                X_train_np, y_train_np, a_train_np, label_col, group_col,
                ctgan_epochs, ctgan_batch, self.seed
            )
            print(f"[CTGAN] CTGAN fitted in {time.time() - t0:.1f}s")

            # ---- sample synthetic disadvantaged-group positive (y=1, a=minority_id) ----
            print(f"[CTGAN] Sampling {n_synthetic} synthetic (y=1, a={self.minority_id}) samples …")
            X_syn_np = _sample_conditional(
                synth, label_col, group_col,
                label_value=1, group_value=self.minority_id,
                n_samples=n_synthetic, feat_cols=feat_cols
            )
            y_syn_np = np.ones(len(X_syn_np), dtype=np.int64)
            print(f"[CTGAN] Got {len(X_syn_np)} synthetic disadvantaged-positive samples.")

            # ---- augment training data ----
            X_aug_np = np.concatenate([X_train_np, X_syn_np], axis=0)
            y_aug_np = np.concatenate([y_train_np, y_syn_np], axis=0)

            x_train_aug = torch.tensor(X_aug_np, dtype=torch.float32,
                                       device=self.device)
            y_train_aug = torch.tensor(y_aug_np, dtype=torch.long,
                                       device=self.device)

            n_pos_aug = int((y_aug_np == 1).sum())
            print(f"[CTGAN] Augmented set: n_pos={n_pos_aug}, "
                  f"total={len(y_aug_np)}")

            # ---- train ERM on augmented data ----
            model_agent = self._train_erm(ffnn_config, x_train_aug, y_train_aug)
            print("[CTGAN] ERM trained on augmented features.")

            # ---- alpha: ERM on original biased data (for comparison) ----
            alpha_agent = self._train_erm(ffnn_config, x_train, y_train)
            print("[CTGAN] Alpha (ERM on original biased data) trained.")

            # ---- save model ----
            model_path = seed_dir / "ctgan_model.pt"
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
                run_id         = f"ctgan_seed{self.seed}",
                beta_factory   = _beta_factory,
                best_beta_path = model_path,
                alpha_factory  = lambda: FFNNAgent(**_ffnn_cfg),
                dataset        = self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model      = alpha_agent,
                x_test           = x_test,
                y_test           = y_test,
                f1_thresh        = 0.5,
                prefer_best_beta = True,
                beta_model       = model_agent,
                x_train          = x_train_aug,
                y_train          = y_train_aug,
                train_size       = self.real_data_size,
                seed             = self.seed,
                a_test           = a_test,
            )

            summary = {
                "baseline":       "ctgan",
                "dataset":        self.dataset_name,
                "seed":           self.seed,
                "n_synthetic":    n_synthetic,
                "ctgan_epochs":   ctgan_epochs,
                "total_time_s":   round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items()
                   if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[CTGAN] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[CTGAN] Results in {seed_dir}")

        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
            log_fh.close()

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _train_erm(self, ffnn_config: dict,
                   x_train: torch.Tensor, y_train: torch.Tensor) -> FFNNAgent:
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
            "baseline":     "ctgan",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "ctgan_config": self.ctgan_config,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        with open(seed_dir / "meta.json", "w") as f:
            json.dump({**meta, "seed": self.seed}, f, indent=2)

        return seed_dir, experiment_dir
