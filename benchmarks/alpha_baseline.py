# benchmarks/alpha_baseline.py
"""
Alpha baseline — ERM classifier trained on the biased training set with no
fairness intervention. Serves as the no-intervention reference point in the
radar figure and comparison tables.

Uses identical data splits, FFNN architecture, and epoch counts as all other
benchmarks so that differences reflect the intervention, not the setup.
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.ffnn_agent2 import FFNNAgent
from dataset import Dataset
from test_suite import TestSuite


class AlphaBaselineTrainer:
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
        multiclass: bool = False,
        use_pca: bool = True,
        pca_components: int = 10,
        win_seconds: float = 5.0,
        step_seconds: float = 2.5,
        dp_protected_col: str | None = None,
        acs_states: list = None,
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
        self.third_id       = third_id
        self.bias_pct       = bias_pct
        self.da_pct         = da_pct
        self.real_data_size = real_data_size
        self.multiclass     = multiclass
        self.win_seconds    = win_seconds
        self.step_seconds   = step_seconds
        self.dp_protected_col = dp_protected_col
        self.acs_states     = acs_states
        self.fold_idx       = fold_idx
        self.n_folds        = n_folds
        self.fold_rng_seed  = fold_rng_seed
        self.pca_components = pca_components
        self.ffnn_overrides = ffnn or {}
        self.project_root   = _PROJECT_ROOT

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
            **({"acs_states": self.acs_states} if self.acs_states is not None else {}),
            **({"fold_idx": self.fold_idx, "n_folds": self.n_folds,
                "fold_rng_seed": self.fold_rng_seed} if self.fold_idx is not None else {}),
        )
        feature_dim = x_train.shape[1]
        a_test = getattr(self.dataset, "a_test", None)

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": float(self.ffnn_overrides.get("learning_rate", 1e-3)),
            "batch_size":    int(self.ffnn_overrides.get("batch_size", 64)),
            "epochs":        int(self.ffnn_overrides.get("epochs", 20)),
            "type":          "classification",
            "classes":       [0, 1, 2] if self.multiclass else [0, 1],
            "device":        self.device,
            "seed":          self.seed,
        }

        seed_dir, experiment_dir = self._setup_output_dirs()
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
            print(f"[Alpha] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[Alpha] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[Alpha] ffnn_config epochs={ffnn_config['epochs']} lr={ffnn_config['learning_rate']}")

            # Train ERM on biased training data
            model = FFNNAgent(**ffnn_config)
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            loader = DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=ffnn_config["batch_size"],
                shuffle=True,
                generator=gen,
            )
            model.train(loader)

            # Save model weights
            model_path = seed_dir / "alpha_model.pt"
            torch.save(model.model.state_dict(), model_path)

            # TestSuite — alpha and beta are the same ERM model
            def _model_factory():
                ag = FFNNAgent(**ffnn_config)
                ag.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                return ag

            tests = TestSuite(
                seed_dir=seed_dir,
                experiment_dir=experiment_dir,
                seed=self.seed,
                run_id=f"alpha_seed{self.seed}",
                beta_factory=_model_factory,
                best_beta_path=model_path,
                alpha_factory=_model_factory,
                dataset=self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model=model,
                x_test=x_test,
                y_test=y_test,
                f1_thresh=0.5,
                prefer_best_beta=True,
                beta_model=model,
                x_train=x_train,
                y_train=y_train,
                bias_pct=self.bias_pct,
                train_size=self.real_data_size,
                seed=self.seed,
                a_test=a_test,
            )

            summary = {
                "baseline":     "alpha",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items() if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[Alpha] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[Alpha] Results in {seed_dir}")

        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
            log_fh.close()

    # ------------------------------------------------------------------ #

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "alpha",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "ffnn_config":  self.ffnn_overrides,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        with open(seed_dir / "meta.json", "w") as f:
            json.dump({**meta, "seed": self.seed}, f, indent=2)

        return seed_dir, experiment_dir
