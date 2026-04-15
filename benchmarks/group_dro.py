# benchmarks/group_dro.py
"""
Group DRO baseline (Sagawa et al. 2020).

Algorithm per minibatch:
  1. Compute per-sample BCE losses.
  2. Compute per-group mean losses L_g.
  3. Update group weights: q_g *= exp(eta * L_g); normalise.
  4. Backprop on sum_g q_g * L_g.
"""

import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.ffnn_agent2 import FFNNAgent, FFNNModel
from dataset import Dataset
import reward_helpers as rh
from test_suite import TestSuite


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


class GroupDROTrainer:
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
        # model architecture overrides
        ffnn: dict = None,
        # Group DRO hyperparams
        group_dro: dict = None,
        # misc
        multiclass: bool = False,
        use_pca: bool = False,
        pca_components: int = 10,
        win_seconds: float = 5.0,
        step_seconds: float = 2.5,
        dp_protected_col: str | None = None,
    ):
        self.exp_group = exp_group
        self.spec_name = spec_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.device = torch.device(device)
        self.minority_id = minority_id
        self.majority_id = majority_id
        self.third_id = third_id
        self.bias_pct = bias_pct
        self.da_pct = da_pct
        self.real_data_size = real_data_size
        self.multiclass = multiclass
        self.win_seconds = win_seconds
        self.step_seconds = step_seconds
        self.dp_protected_col = dp_protected_col
        self.project_root = _PROJECT_ROOT

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Group DRO hyperparams
        DEFAULT_GDRO = {
            "lr": 1e-3,
            "epochs": 200,
            "batch_size": 64,
            "eta": 0.01,
            "n_groups": 4,   # (a, y) subgroups: a∈{0,1} × y∈{0,1}
        }
        self.gdro_config = {**DEFAULT_GDRO, **(group_dro or {})}

        # FFNN architecture overrides (applied later once feature_dim is known)
        self.ffnn_overrides = ffnn or {}

        self.use_pca = use_pca
        self.pca_components = pca_components

        # Dataset (feature_dim resolved after get_data_splits)
        self.dataset = Dataset(
            dataset_name,
            multiclass=multiclass,
            minority_id=minority_id,
            majority_id=majority_id,
            third_id=third_id,
            pca_components=pca_components,
            seed=seed,
            device=self.device,
            use_pca=use_pca,
        )

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self):
        start_time = time.time()

        # ---- data ----
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

        a_train = self.dataset.a_train   # [N] long, {0, 1}
        a_val   = self.dataset.a_val
        a_test  = getattr(self.dataset, "a_test", None)

        # ---- FFNN config ----
        ffnn_config = {
            "input_size":   feature_dim,
            "hidden_sizes": self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":  3 if self.multiclass else 2,
            "learning_rate": self.gdro_config["lr"],
            "batch_size":   self.gdro_config["batch_size"],
            "epochs":       self.gdro_config["epochs"],
            "type":         "classification",
            "classes":      [0, 1, 2] if self.multiclass else [0, 1],
            "device":       self.device,
            "seed":         self.seed,
        }
        self.ffnn_config = ffnn_config

        # ---- output dirs ----
        seed_dir, experiment_dir = self._setup_output_dirs()

        # Redirect stdout/stderr to console.log (mirrors EpisodeTracker behaviour)
        log_path = seed_dir / "console.log"
        log_fh = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
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
            print(f"[GroupDRO] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[GroupDRO] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[GroupDRO] gdro_config={self.gdro_config}")
            print(f"[GroupDRO] seed_dir={seed_dir}")

            # ---- build model & optimiser ----
            model = FFNNModel(
                ffnn_config["input_size"],
                ffnn_config["hidden_sizes"],
                ffnn_config["output_size"],
            ).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.gdro_config["lr"])

            # ---- group weights (initialised uniform) ----
            n_groups = int(self.gdro_config["n_groups"])
            eta      = float(self.gdro_config["eta"])
            q = torch.ones(n_groups, device=self.device) / n_groups

            # ---- train Group DRO ----
            metrics_rows, best_state = self._train_group_dro(
                model, optimizer, q, eta, n_groups,
                x_train, y_train, a_train,
                x_val,   y_val,   a_val,
            )

            # ---- save training metrics CSV ----
            self._save_metrics_csv(seed_dir / "metrics.csv", metrics_rows)

            # ---- save best model ----
            best_model_path = seed_dir / "gdro_model.pt"
            torch.save(best_state, best_model_path)
            print(f"[GroupDRO] Best model saved -> {best_model_path}")

            # ---- alpha (ERM) for comparison in TestSuite ----
            # Use ffnn_overrides for epochs/lr so alpha matches training.py's ERM config,
            # not the DRO training config (which uses 200 epochs by default).
            erm_config = dict(ffnn_config)
            erm_config["epochs"] = int(self.ffnn_overrides.get("epochs", 10))
            erm_config["learning_rate"] = float(self.ffnn_overrides.get("learning_rate", 1e-3))
            alpha_agent = self._train_erm(erm_config, x_train, y_train)

            # ---- wrap trained model as FFNNAgent for TestSuite ----
            beta_agent = FFNNAgent(**ffnn_config)
            beta_agent.model.load_state_dict(best_state)
            beta_agent.model.to(self.device)

            # ---- beta_factory: load best weights from disk ----
            _ffnn_cfg = dict(ffnn_config)
            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                return ag

            tests = TestSuite(
                seed_dir=seed_dir,
                experiment_dir=experiment_dir,
                seed=self.seed,
                run_id=f"gdro_seed{self.seed}",
                beta_factory=_beta_factory,
                best_beta_path=best_model_path,
                alpha_factory=lambda: FFNNAgent(**_ffnn_cfg),
                dataset=self.dataset,
            )

            test_row = tests.log_final_test(
                alpha_model=alpha_agent,
                x_test=x_test,
                y_test=y_test,
                f1_thresh=0.5,
                prefer_best_beta=True,
                beta_model=beta_agent,
                x_train=x_train,
                y_train=y_train,
                bias_pct=self.bias_pct,
                train_size=self.real_data_size,
                seed=self.seed,
                a_test=a_test,
            )

            # ---- summary JSON ----
            summary = {
                "baseline": "group_dro",
                "dataset":  self.dataset_name,
                "seed":     self.seed,
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items() if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            elapsed = time.time() - start_time
            print(f"[GroupDRO] Done. seed={self.seed} | time={elapsed:.1f}s")
            print(f"[GroupDRO] Results in {seed_dir}")

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_fh.close()

    # ------------------------------------------------------------------ #
    #  Group DRO training loop                                            #
    # ------------------------------------------------------------------ #

    def _train_group_dro(
        self,
        model, optimizer, q, eta, n_groups,
        x_train, y_train, a_train,
        x_val,   y_val,   a_val,
    ):
        epochs     = int(self.gdro_config["epochs"])
        batch_size = int(self.gdro_config["batch_size"])

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train, a_train),
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
        )

        best_val_worst = float("inf")
        best_state     = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rows = []

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_dro_loss   = 0.0
            epoch_worst_loss = 0.0
            batch_count      = 0

            for x_b, y_b, a_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device).long()
                a_b = a_b.to(self.device).long()

                # Composite group index: g = a*2 + y  →  0=(a=0,y=0), 1=(a=0,y=1),
                #                                        2=(a=1,y=0), 3=(a=1,y=1)
                g_ids_b = a_b * 2 + y_b

                # Forward → P(y=1|x)
                logits = model(x_b)
                probs  = torch.softmax(logits, dim=-1)
                p1     = probs[:, 1]

                # Per-sample BCE
                losses = rh.bce_per_sample_from_probs(y_b, p1)

                # Per-group mean losses; track which groups are present in this batch
                group_losses = []
                present      = []
                for g in range(n_groups):
                    lg = rh.group_mean_loss(losses, g_ids_b, g)
                    if not torch.isfinite(lg):
                        group_losses.append(torch.tensor(0.0, device=self.device))
                        present.append(False)
                    else:
                        group_losses.append(lg)
                        present.append(True)
                group_losses_t = torch.stack(group_losses)   # [G]

                # Update q only for groups present in this batch; skip absent ones
                # so their weight is not silently treated as zero-loss.
                with torch.no_grad():
                    for g in range(n_groups):
                        if present[g]:
                            q[g] *= torch.exp(eta * group_losses_t[g].detach())
                    q.div_(q.sum())

                # DRO loss
                dro_loss = (q.detach() * group_losses_t).sum()

                if not torch.isfinite(dro_loss):
                    continue

                optimizer.zero_grad()
                dro_loss.backward()
                optimizer.step()

                epoch_dro_loss   += float(dro_loss.item())
                epoch_worst_loss += float(group_losses_t.max().item())
                batch_count      += 1

            # ---- validation ----
            model.eval()
            with torch.no_grad():
                val_p1 = torch.softmax(model(x_val.to(self.device)), dim=-1)[:, 1]
                val_losses = rh.bce_per_sample_from_probs(
                    y_val.to(self.device).long(), val_p1
                )
                # Composite group index for validation set
                g_ids_val = a_val.to(self.device) * 2 + y_val.to(self.device).long()
                val_worst_t, val_per_g = rh.worst_group_loss(
                    val_losses,
                    g_ids_val,
                    group_values=tuple(range(n_groups)),
                )

            val_worst = float(val_worst_t.item()) if torch.isfinite(val_worst_t) else float("nan")

            # ---- checkpoint best by val worst-group loss ----
            if val_worst == val_worst and val_worst < best_val_worst:
                best_val_worst = val_worst
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            q_vals = {f"q{g}": float(q[g].item()) for g in range(n_groups)}
            val_g_vals = {f"val_group{g}_loss": val_per_g.get(g, float("nan")) for g in range(n_groups)}
            row = {
                "epoch":                  epoch,
                "train_dro_loss":         epoch_dro_loss   / max(batch_count, 1),
                "train_worst_group_loss": epoch_worst_loss / max(batch_count, 1),
                **q_vals,
                "val_worst_group_loss":   val_worst,
                **val_g_vals,
            }
            rows.append(row)

            if epoch % 20 == 0 or epoch == 1:
                q_str = ",".join(f"{q[g]:.3f}" for g in range(n_groups))
                print(
                    f"[GroupDRO] epoch={epoch:4d} | "
                    f"dro={row['train_dro_loss']:.4f} | "
                    f"val_worst={val_worst:.4f} | "
                    f"q=[{q_str}]"
                )

        print(f"[GroupDRO] Training done. Best val_worst={best_val_worst:.4f}")
        return rows, best_state

    # ------------------------------------------------------------------ #
    #  ERM baseline (alpha)                                               #
    # ------------------------------------------------------------------ #

    def _train_erm(self, ffnn_config: dict, x_train, y_train) -> FFNNAgent:
        alpha = FFNNAgent(**ffnn_config)
        gen   = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=int(ffnn_config["batch_size"]),
            shuffle=True,
            generator=gen,
        )
        alpha.train(loader)
        return alpha

    # ------------------------------------------------------------------ #
    #  I/O helpers                                                        #
    # ------------------------------------------------------------------ #

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "group_dro",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "gdro_config":  self.gdro_config,
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
        print(f"[GroupDRO] Training metrics -> {path}")
