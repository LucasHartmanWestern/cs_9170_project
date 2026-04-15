# benchmarks/fairness_loss_balancing.py
"""
Fairness-Aware Loss Balancing via Batch Sampling baseline.

Kim et al. (2023) — "Fair Classification by Loss Balancing via Fairness-Aware
Batch Sampling", Neurocomputing, Vol. 518, pp. 231-241.
DOI: 10.1016/j.neucom.2022.08.040

Algorithm per epoch:
  1. Divide training data into (a, y) subgroups.
  2. Compute per-group mean BCE loss on the current batch.
  3. Update per-group sampling probabilities proportional to group losses:
       p_g  ←  p_g * exp(eta * L_g)  then renormalise.
  4. Sample the next batch by drawing uniformly within each group according
     to the updated probabilities (proportional group sizes in each batch).
  5. Backprop on standard (unweighted) BCE over the resampled batch.

Key difference from GroupDRO: GroupDRO reweights the *loss*; this method
reweights the *sampling* so that high-loss groups contribute more samples to
each batch. The loss itself remains unweighted.
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

from agents.ffnn_agent2 import FFNNAgent, FFNNModel
from dataset import Dataset
import reward_helpers as rh
from test_suite import TestSuite


class FairnessLossBalancingTrainer:
    def __init__(
        self,
        exp_group: str,
        spec_name: str,
        dataset_name: str,
        seed: int,
        device: str,
        # data
        minority_id=None,
        majority_id=None,
        third_id=None,
        bias_pct=None,
        da_pct=None,
        real_data_size: int = 3000,
        # model
        ffnn: dict = None,
        # FLB hyperparams
        flb: dict = None,
        # misc
        multiclass: bool = False,
        use_pca: bool = False,
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
        self.multiclass     = multiclass
        self.win_seconds    = win_seconds
        self.step_seconds   = step_seconds
        self.dp_protected_col = dp_protected_col
        self.project_root   = _PROJECT_ROOT

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True

        DEFAULT_FLB = {
            "lr":         1e-3,
            "epochs":     200,
            "batch_size": 64,
            "eta":        0.01,   # sampling-weight update rate (same scale as GroupDRO eta)
            "n_groups":   4,      # (a, y) subgroups
        }
        self.flb_config     = {**DEFAULT_FLB, **(flb or {})}
        self.ffnn_overrides = ffnn or {}
        self.use_pca        = use_pca
        self.pca_components = pca_components

        self.dataset = Dataset(
            dataset_name,
            multiclass     = multiclass,
            minority_id    = minority_id,
            majority_id    = majority_id,
            third_id       = third_id,
            pca_components = pca_components,
            seed           = seed,
            device         = self.device,
            use_pca        = use_pca,
        )

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self):
        start_time = time.time()

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
        )
        feature_dim = x_train.shape[1]

        a_train = self.dataset.a_train
        a_val   = self.dataset.a_val
        a_test  = getattr(self.dataset, "a_test", None)

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": self.flb_config["lr"],
            "batch_size":    self.flb_config["batch_size"],
            "epochs":        self.flb_config["epochs"],
            "type":          "classification",
            "classes":       [0, 1, 2] if self.multiclass else [0, 1],
            "device":        self.device,
            "seed":          self.seed,
        }
        self.ffnn_config = ffnn_config

        seed_dir, experiment_dir = self._setup_output_dirs()

        log_fh = open(seed_dir / "console.log", "a", buffering=1,
                      encoding="utf-8", errors="replace")
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
            print(f"[FLB] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[FLB] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[FLB] flb_config={self.flb_config}")
            print(f"[FLB] seed_dir={seed_dir}")

            model = FFNNModel(
                ffnn_config["input_size"],
                ffnn_config["hidden_sizes"],
                ffnn_config["output_size"],
            ).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.flb_config["lr"])

            n_groups = int(self.flb_config["n_groups"])
            eta      = float(self.flb_config["eta"])

            # Per-group sampling probabilities (initialised uniform)
            p = torch.ones(n_groups, device="cpu") / n_groups

            metrics_rows, best_state = self._train_flb(
                model, optimizer, p, eta, n_groups,
                x_train, y_train, a_train,
                x_val,   y_val,   a_val,
            )

            self._save_metrics_csv(seed_dir / "metrics.csv", metrics_rows)

            best_model_path = seed_dir / "flb_model.pt"
            torch.save(best_state, best_model_path)
            print(f"[FLB] Best model saved -> {best_model_path}")

            # Alpha: ERM on original biased data
            erm_config = dict(ffnn_config)
            erm_config["epochs"]        = int(self.ffnn_overrides.get("epochs", 10))
            erm_config["learning_rate"] = float(self.ffnn_overrides.get("learning_rate", 1e-3))
            alpha_agent = self._train_erm(erm_config, x_train, y_train)

            beta_agent = FFNNAgent(**ffnn_config)
            beta_agent.model.load_state_dict(best_state)
            beta_agent.model.to(self.device)

            _ffnn_cfg = dict(ffnn_config)

            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(
                    torch.load(best_model_path, map_location=self.device)
                )
                return ag

            tests = TestSuite(
                seed_dir       = seed_dir,
                experiment_dir = experiment_dir,
                seed           = self.seed,
                run_id         = f"flb_seed{self.seed}",
                beta_factory   = _beta_factory,
                best_beta_path = best_model_path,
                alpha_factory  = lambda: FFNNAgent(**_ffnn_cfg),
                dataset        = self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model      = alpha_agent,
                x_test           = x_test,
                y_test           = y_test,
                f1_thresh        = 0.5,
                prefer_best_beta = True,
                beta_model       = beta_agent,
                x_train          = x_train,
                y_train          = y_train,
                bias_pct         = self.bias_pct,
                train_size       = self.real_data_size,
                seed             = self.seed,
                a_test           = a_test,
            )

            summary = {
                "baseline":     "fairness_loss_balancing",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items()
                   if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[FLB] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[FLB] Results in {seed_dir}")

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_fh.close()

    # ------------------------------------------------------------------ #
    #  FLB training loop                                                   #
    # ------------------------------------------------------------------ #

    def _train_flb(
        self,
        model, optimizer, p, eta, n_groups,
        x_train, y_train, a_train,
        x_val,   y_val,   a_val,
    ):
        epochs     = int(self.flb_config["epochs"])
        batch_size = int(self.flb_config["batch_size"])

        # Pre-compute group membership indices on CPU for efficient sampling
        # Group index: g = a * 2 + y  →  0=(a=0,y=0), 1=(a=0,y=1),
        #                                2=(a=1,y=0), 3=(a=1,y=1)
        y_cpu = y_train.cpu().numpy().astype(int)
        a_cpu = a_train.cpu().numpy().astype(int)
        g_ids_np = a_cpu * 2 + y_cpu   # [N]

        group_indices = [
            np.where(g_ids_np == g)[0] for g in range(n_groups)
        ]
        group_sizes = [len(idx) for idx in group_indices]
        print(f"[FLB] Group sizes (a=0/y=0, a=0/y=1, a=1/y=0, a=1/y=1): "
              f"{group_sizes}")

        # Move full train tensors to device once
        x_tr = x_train.to(self.device)
        y_tr = y_train.to(self.device).long()

        rng = np.random.default_rng(self.seed)

        best_val_worst = float("inf")
        best_state     = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rows = []

        for epoch in range(1, epochs + 1):
            model.train()

            # ---- Build resampled batch according to p ----
            # Allocate batch_size slots proportionally to p, then sample
            # uniformly within each group to fill those slots.
            p_np   = p.numpy()
            alloc  = np.maximum((p_np * batch_size).astype(int), 0)
            # Top up to exactly batch_size by adding to largest group
            deficit = batch_size - alloc.sum()
            if deficit > 0:
                alloc[np.argmax(alloc)] += deficit

            sampled_indices = []
            for g in range(n_groups):
                n_g = int(alloc[g])
                if n_g == 0 or group_sizes[g] == 0:
                    continue
                chosen = rng.choice(group_indices[g], size=n_g, replace=True)
                sampled_indices.append(chosen)

            if not sampled_indices:
                continue

            idx = np.concatenate(sampled_indices)
            rng.shuffle(idx)
            idx_t = torch.from_numpy(idx).long()

            x_b = x_tr[idx_t]
            y_b = y_tr[idx_t]
            a_b = a_train.to(self.device)[idx_t].long()
            g_ids_b = a_b * 2 + y_b

            # ---- Forward pass ----
            logits = model(x_b)
            probs  = torch.softmax(logits, dim=-1)
            p1     = probs[:, 1]

            per_sample_loss = rh.bce_per_sample_from_probs(y_b, p1)

            # ---- Standard (unweighted) BCE on resampled batch ----
            loss = per_sample_loss.mean()

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Update sampling probabilities using per-group losses ----
            with torch.no_grad():
                for g in range(n_groups):
                    mask = (g_ids_b == g)
                    if mask.sum() > 0:
                        lg = per_sample_loss[mask].mean().item()
                        if np.isfinite(lg):
                            p[g] *= np.exp(eta * lg)
                p_sum = p.sum().item()
                if p_sum > 0:
                    p /= p_sum
                else:
                    p = torch.ones(n_groups) / n_groups

            # ---- Validation ----
            if epoch % 5 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    val_p1 = torch.softmax(
                        model(x_val.to(self.device)), dim=-1
                    )[:, 1]
                    val_losses = rh.bce_per_sample_from_probs(
                        y_val.to(self.device).long(), val_p1
                    )
                    g_ids_val = (a_val.to(self.device) * 2
                                 + y_val.to(self.device).long())
                    val_worst_t, val_per_g = rh.worst_group_loss(
                        val_losses, g_ids_val,
                        group_values=tuple(range(n_groups)),
                    )

                val_worst = (float(val_worst_t.item())
                             if torch.isfinite(val_worst_t) else float("nan"))

                if val_worst == val_worst and val_worst < best_val_worst:
                    best_val_worst = val_worst
                    best_state = {
                        k: v.detach().clone()
                        for k, v in model.state_dict().items()
                    }

                p_str = ",".join(f"{p[g]:.3f}" for g in range(n_groups))
                if epoch % 20 == 0 or epoch == 1:
                    print(
                        f"[FLB] epoch={epoch:4d} | "
                        f"loss={loss.item():.4f} | "
                        f"val_worst={val_worst:.4f} | "
                        f"p=[{p_str}]"
                    )

                row = {
                    "epoch":                epoch,
                    "train_loss":           float(loss.item()),
                    "val_worst_group_loss": val_worst,
                    **{f"p{g}": float(p[g].item()) for g in range(n_groups)},
                    **{f"val_group{g}_loss": val_per_g.get(g, float("nan"))
                       for g in range(n_groups)},
                }
                rows.append(row)

        print(f"[FLB] Training done. Best val_worst={best_val_worst:.4f}")
        return rows, best_state

    # ------------------------------------------------------------------ #
    #  ERM baseline (alpha)                                                #
    # ------------------------------------------------------------------ #

    def _train_erm(self, ffnn_config: dict, x_train, y_train) -> FFNNAgent:
        alpha  = FFNNAgent(**ffnn_config)
        gen    = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size = int(ffnn_config["batch_size"]),
            shuffle    = True,
            generator  = gen,
        )
        alpha.train(loader)
        return alpha

    # ------------------------------------------------------------------ #
    #  I/O helpers                                                         #
    # ------------------------------------------------------------------ #

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "fairness_loss_balancing",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "flb_config":   self.flb_config,
        }
        meta_path = experiment_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        with open(seed_dir / "meta.json", "w") as f:
            json.dump({**meta, "seed": self.seed}, f, indent=2)

        return seed_dir, experiment_dir

    def _save_metrics_csv(self, path: Path, rows: list):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[FLB] Training metrics -> {path}")
