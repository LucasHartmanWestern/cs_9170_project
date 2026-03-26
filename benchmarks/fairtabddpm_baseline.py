# benchmarks/fairtabddpm_baseline.py
"""
FairTabDDPM baseline — conditional denoising diffusion for fair tabular generation.

Adapted from:
  Yang et al. (2025) "Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models"
  Transactions on Machine Learning Research (TMLR), February 2025.
  https://github.com/comp-well-org/fair-tab-diffusion

The original library targets mixed-type (numerical + categorical) tabular data with a
complex file-system-centric pipeline. Since our features are already all-numerical
(PCA-transformed), we implement the core algorithm — Gaussian DDPM with joint
(class label, sensitive attribute) conditioning — directly in PyTorch without the
mixed-type encoding machinery.

Algorithm:
  1. Train a conditional denoising network f_θ(x_t, t, y, a) on (X_train, y_train, a_train).
  2. Standard DDPM with cosine noise schedule; loss = MSE(noise, predicted noise).
  3. To generate fair synthetic minority samples: sample conditioned on (y=1, a=minority).
  4. Add synthetic samples to the biased training set; train ERM classifier; evaluate.

The is_fair conditioning (simultaneous class + sensitive-attr guidance) is the key
contribution of Yang et al. vs. standard class-conditional CTGAN/TabDDPM.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.ffnn_agent2 import FFNNAgent
from dataset import Dataset
from test_suite import TestSuite


# ------------------------------------------------------------------ #
#  Cosine noise schedule (Nichol & Dhariwal 2021)                     #
# ------------------------------------------------------------------ #

def _cosine_betas(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


class DDPMSchedule:
    def __init__(self, timesteps: int = 100, device: torch.device = torch.device("cpu")):
        self.T = timesteps
        betas = _cosine_betas(timesteps).to(device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)          # [T]
        self.sqrt_ab    = self.alphas_bar.sqrt()
        self.sqrt_1mab  = (1.0 - self.alphas_bar).sqrt()
        self.betas      = betas
        self.device     = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ā_t)*x0 + sqrt(1-ā_t)*ε"""
        ab  = self.sqrt_ab[t].view(-1, 1)
        mab = self.sqrt_1mab[t].view(-1, 1)
        return ab * x0 + mab * noise


# ------------------------------------------------------------------ #
#  Conditional denoising network                                       #
# ------------------------------------------------------------------ #

class CondDenoiseNet(nn.Module):
    """
    MLP that predicts the noise ε given (x_t, t, y, a).

    Conditioning:
      - t   : sinusoidal time embedding
      - y   : class label (0/1) — one-hot → linear projection
      - a   : sensitive attribute (0/1) — one-hot → linear projection

    Architecture follows Yang et al.: embed each input separately, sum
    embeddings, pass through residual MLP.
    """

    def __init__(self, d_x: int, d_emb: int = 128, n_layers: int = 4):
        super().__init__()
        self.d_x = d_x

        # Sinusoidal time embedding → linear projection
        self.t_proj = nn.Sequential(
            nn.Linear(d_emb, d_emb), nn.SiLU(), nn.Linear(d_emb, d_emb)
        )
        # Class and group embeddings
        self.y_emb = nn.Embedding(2, d_emb)
        self.a_emb = nn.Embedding(2, d_emb)

        # Input projection
        self.x_proj = nn.Linear(d_x, d_emb)

        # Residual MLP
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_emb, d_emb), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)

        # Output
        self.out = nn.Linear(d_emb, d_x)

    def _sinusoidal_embedding(self, t: torch.Tensor, d: int) -> torch.Tensor:
        half = d // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        t_emb = self._sinusoidal_embedding(t, self.t_proj[0].in_features)
        h = self.x_proj(x_t) + self.t_proj(t_emb) + self.y_emb(y) + self.a_emb(a)
        h = self.mlp(h)
        return self.out(h)


# ------------------------------------------------------------------ #
#  Main trainer                                                        #
# ------------------------------------------------------------------ #

class FairTabDDPMTrainer:
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
        real_data_size: int = 3000,
        # model
        ffnn: dict = None,
        # FairTabDDPM hyperparams
        fairtabddpm: dict = None,
        # misc
        multiclass: bool = False,
        use_pca: bool = False,
        pca_components: int = 10,
        win_seconds: float = 5.0,
        step_seconds: float = 2.5,
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
        self.real_data_size = real_data_size
        self.multiclass     = multiclass
        self.win_seconds    = win_seconds
        self.step_seconds   = step_seconds
        self.project_root   = _PROJECT_ROOT

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True

        DEFAULT_FTDDPM = {
            "timesteps":      100,     # diffusion steps (Yang et al. use 100)
            "d_emb":          128,     # embedding dimension
            "n_layers":       4,       # residual MLP depth
            "lr":             3.9e-4,  # Adam lr (Yang et al. default)
            "weight_decay":   0.0,
            "diff_epochs":    1000,    # diffusion model training epochs
            "batch_size":     256,
            "n_synthetic":    2000,    # synthetic minority samples to add
        }
        self.cfg        = {**DEFAULT_FTDDPM, **(fairtabddpm or {})}
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

    # ---------------------------------------------------------------- #

    def __call__(self):
        start_time = time.time()

        x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
            train_size     = self.real_data_size,
            bias_pct       = self.bias_pct,
            pca_components = self.pca_components,
            drop_protected = False,
            protected_cols = self.dataset.protected_attributes,
            bias_val       = True,
            win_seconds    = self.win_seconds,
            step_seconds   = self.step_seconds,
        )
        feature_dim = x_train.shape[1]
        a_train = self.dataset.a_train
        a_test  = getattr(self.dataset, "a_test", None)

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size":   3 if self.multiclass else 2,
            "learning_rate": self.ffnn_overrides.get("learning_rate", 1e-3),
            "batch_size":    self.ffnn_overrides.get("batch_size", 64),
            "epochs":        self.ffnn_overrides.get("epochs", 20),
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
            print(f"[FairTabDDPM] exp_group={self.exp_group}  seed={self.seed}")
            print(f"[FairTabDDPM] dataset={self.dataset_name}  feature_dim={feature_dim}")
            print(f"[FairTabDDPM] cfg={self.cfg}")

            # ---- Build and train diffusion model ----
            schedule = DDPMSchedule(
                timesteps=int(self.cfg["timesteps"]), device=self.device
            )
            net = CondDenoiseNet(
                d_x      = feature_dim,
                d_emb    = int(self.cfg["d_emb"]),
                n_layers = int(self.cfg["n_layers"]),
            ).to(self.device)

            print(f"[FairTabDDPM] Training diffusion model for "
                  f"{self.cfg['diff_epochs']} epochs ...")
            t0 = time.time()
            diff_metrics = self._train_diffusion(net, schedule, x_train, y_train, a_train)
            print(f"[FairTabDDPM] Diffusion training done in {time.time()-t0:.1f}s")

            # ---- Sample synthetic minority + sensitive-group samples ----
            n_syn = int(self.cfg["n_synthetic"])
            # Auto-detect disadvantaged group: whichever a-group has fewer positive
            # training examples. This is robust to minority_id encoding conventions.
            _a_np = a_train.cpu().numpy()
            _y_np = y_train.cpu().numpy()
            _n_pos_g0 = int(((_a_np == 0) & (_y_np == 1)).sum())
            _n_pos_g1 = int(((_a_np == 1) & (_y_np == 1)).sum())
            minority_a = 0 if _n_pos_g0 <= _n_pos_g1 else 1
            print(f"[FairTabDDPM] Disadvantaged group: a={minority_a} "
                  f"(pos_g0={_n_pos_g0}, pos_g1={_n_pos_g1})")
            print(f"[FairTabDDPM] Sampling {n_syn} fair synthetic samples "
                  f"(y=1, a={minority_a}) ...")
            X_syn = self._sample_conditional(
                net, schedule, feature_dim, n_syn,
                y_val=1, a_val=minority_a,
            )
            y_syn = torch.ones(len(X_syn), dtype=torch.long, device=self.device)
            print(f"[FairTabDDPM] Got {len(X_syn)} synthetic samples.")

            # ---- Augment ----
            x_train_aug = torch.cat([x_train, X_syn], dim=0)
            y_train_aug = torch.cat([y_train, y_syn], dim=0)
            print(f"[FairTabDDPM] Augmented: {len(x_train_aug)} total "
                  f"({int((y_train_aug==1).sum())} pos)")

            # ---- Train ERM on augmented data ----
            beta_agent = self._train_erm(ffnn_config, x_train_aug, y_train_aug)
            alpha_agent = self._train_erm(ffnn_config, x_train, y_train)

            # ---- Save model ----
            model_path = seed_dir / "fairtabddpm_model.pt"
            torch.save(beta_agent.model.state_dict(), model_path)

            _ffnn_cfg = dict(ffnn_config)
            def _beta_factory():
                ag = FFNNAgent(**_ffnn_cfg)
                ag.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                return ag

            # ---- Evaluate ----
            tests = TestSuite(
                seed_dir       = seed_dir,
                experiment_dir = experiment_dir,
                seed           = self.seed,
                run_id         = f"fairtabddpm_seed{self.seed}",
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
                beta_model       = beta_agent,
                x_train          = x_train_aug,
                y_train          = y_train_aug,
                bias_pct         = self.bias_pct,
                train_size       = self.real_data_size,
                seed             = self.seed,
                a_test           = a_test,
            )

            self._save_metrics_csv(seed_dir / "diff_metrics.csv", diff_metrics)

            summary = {
                "baseline":     "fairtabddpm",
                "dataset":      self.dataset_name,
                "seed":         self.seed,
                "n_synthetic":  n_syn,
                "diff_epochs":  self.cfg["diff_epochs"],
                "total_time_s": round(time.time() - start_time, 2),
                **{k: v for k, v in test_row.items()
                   if isinstance(v, (int, float, str, bool))},
            }
            with open(seed_dir / "test_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"[FairTabDDPM] Done. seed={self.seed} | "
                  f"time={time.time()-start_time:.1f}s")

        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
            log_fh.close()

    # ---------------------------------------------------------------- #
    #  Diffusion training                                               #
    # ---------------------------------------------------------------- #

    def _train_diffusion(
        self,
        net: CondDenoiseNet,
        schedule: DDPMSchedule,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        a_train: torch.Tensor,
    ) -> list:
        epochs     = int(self.cfg["diff_epochs"])
        batch_size = int(self.cfg["batch_size"])
        T          = schedule.T

        opt = optim.Adam(
            net.parameters(),
            lr           = float(self.cfg["lr"]),
            weight_decay = float(self.cfg["weight_decay"]),
        )

        # Normalise features to [-1, 1] for diffusion stability
        x_min = x_train.min(0).values
        x_max = x_train.max(0).values
        x_range = (x_max - x_min).clamp(min=1e-8)
        self._x_min   = x_min
        self._x_range = x_range
        x0 = 2.0 * (x_train - x_min) / x_range - 1.0

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x0, y_train.long(), a_train.long()),
            batch_size = batch_size,
            shuffle    = True,
            generator  = gen,
        )

        rows = []
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        for epoch in range(1, epochs + 1):
            net.train()
            total_loss = 0.0
            n_batches  = 0

            for x_b, y_b, a_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                a_b = a_b.to(self.device)

                # Sample random timesteps
                t = torch.randint(0, T, (len(x_b),), device=self.device)
                noise = torch.randn_like(x_b)
                x_t   = schedule.q_sample(x_b, t, noise)

                # Predict noise
                noise_pred = net(x_t, t, y_b, a_b)
                loss = F.mse_loss(noise_pred, noise)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

                total_loss += float(loss.item())
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

            if epoch % 100 == 0 or epoch == 1:
                print(f"[FairTabDDPM] epoch={epoch:5d} | loss={avg_loss:.5f}")

            rows.append({"epoch": epoch, "diff_loss": avg_loss})

        net.load_state_dict(best_state)
        print(f"[FairTabDDPM] Diffusion done. Best loss={best_loss:.5f}")
        return rows

    # ---------------------------------------------------------------- #
    #  Conditional sampling (DDPM reverse process)                     #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def _sample_conditional(
        self,
        net: CondDenoiseNet,
        schedule: DDPMSchedule,
        d_x: int,
        n_samples: int,
        y_val: int,
        a_val: int,
        batch_size: int = 256,
    ) -> torch.Tensor:
        net.eval()
        T    = schedule.T
        all_x = []

        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = torch.randn(n, d_x, device=self.device)
            y_cond = torch.full((n,), y_val, dtype=torch.long, device=self.device)
            a_cond = torch.full((n,), a_val, dtype=torch.long, device=self.device)

            for t_idx in reversed(range(T)):
                t_tensor = torch.full((n,), t_idx, dtype=torch.long,
                                      device=self.device)
                noise_pred = net(x, t_tensor, y_cond, a_cond)

                beta_t  = schedule.betas[t_idx]
                ab_t    = schedule.alphas_bar[t_idx]
                ab_prev = schedule.alphas_bar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0)

                # DDPM reverse step
                x0_pred  = (x - (1.0 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
                x0_pred  = x0_pred.clamp(-1.5, 1.5)
                dir_xt   = (1.0 - ab_prev - beta_t).clamp(min=0.0).sqrt() * noise_pred
                x = ab_prev.sqrt() * x0_pred + dir_xt

                if t_idx > 0:
                    x = x + beta_t.sqrt() * torch.randn_like(x)

            all_x.append(x)

        x_norm = torch.cat(all_x, dim=0)[:n_samples]

        # Inverse-normalise from [-1, 1] back to original PCA scale
        x_out = (x_norm + 1.0) / 2.0 * self._x_range.to(self.device) \
                + self._x_min.to(self.device)
        return x_out.float()

    # ---------------------------------------------------------------- #
    #  ERM helper                                                       #
    # ---------------------------------------------------------------- #

    def _train_erm(self, ffnn_config: dict, x_train, y_train) -> FFNNAgent:
        agent  = FFNNAgent(**ffnn_config)
        gen    = torch.Generator(device="cpu").manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size = int(ffnn_config["batch_size"]),
            shuffle    = True,
            generator  = gen,
        )
        agent.train(loader)
        return agent

    # ---------------------------------------------------------------- #
    #  I/O helpers                                                      #
    # ---------------------------------------------------------------- #

    def _setup_output_dirs(self):
        experiment_dir = self.project_root / "training_runs" / self.exp_group
        seed_dir       = experiment_dir / f"seed_{self.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "exp_group":    self.exp_group,
            "spec_name":    self.spec_name,
            "baseline":     "fairtabddpm",
            "dataset_name": self.dataset_name,
            "seed":         self.seed,
            "cfg":          self.cfg,
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
        print(f"[FairTabDDPM] Diffusion metrics -> {path}")
