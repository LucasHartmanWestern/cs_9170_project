# benchmarks/fairtabddpm_baseline.py
"""
FairTabDDPM baseline — conditional denoising diffusion for fair tabular generation.

Adapted from:
  Yang et al. (2025) "Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models"
  Transactions on Machine Learning Research (TMLR), February 2025.
  https://github.com/comp-well-org/fair-tab-diffusion

Since our features are already all-numerical (PCA-transformed) we replace the original
mixed-type encoder/U-Net/decoder stack with a residual MLP, but preserve the core
fairness mechanism exactly as implemented in estimator.py:

  3-path forward (unconditional | label-conditioned | sensitive-conditioned):
    z_uncond = f(x_t, t, ∅)
    z_label  = f(x_t, t, y)
    z_cond   = f(x_t, t, a)

  Label guidance:
    label_guid = z_label − z_uncond                              (Eq. 5)

  Sensitive guidance with security gate μ (Eq. 6 / §4.1):
    scale      = clamp(|z_label − z_cond| * w_s,  max=1)
    gate       = [(z_label − z_cond) ≤ λ]          (element-wise)
    cond_guid  = (z_cond − z_uncond) * scale * gate
    cond_guid  = 0  if  t < warmup_steps            (warmup)

  Final guided estimate:
    ε̃ = z_uncond + w_g * (label_guid + cond_guid)

  The same forward pass is used during both training and sampling.

Algorithm:
  1. Train f_θ(x_t, t, y, a) on (X_train, y_train, a_train).
     Loss = MSE(ε̃_θ(x_t, t, y, a), ε_true).
  2. DDPM cosine noise schedule.
  3. Sample conditioned on (y=1, a=minority) to generate fair synthetic data.
  4. Augment biased training set; train ERM classifier; evaluate.
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

from agents.ffnn_agent import FFNNAgent
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
#  Residual block and CFG denoising network                           #
# ------------------------------------------------------------------ #

class ResidualBlock(nn.Module):
    """Residual MLP block with timestep injection."""

    def __init__(self, d_model: int, d_t_emb: int):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.fc1    = nn.Linear(d_model, d_model)
        self.t_proj = nn.Linear(d_t_emb, d_model)
        self.act    = nn.SiLU()
        self.norm2  = nn.LayerNorm(d_model)
        self.fc2    = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x) + self.t_proj(t_emb)))
        return x + self.norm2(self.fc2(h))


class CondDenoiseNet(nn.Module):
    """
    Residual MLP denoising network with CFG-based multivariate guidance.

    Faithfully implements the 3-path guidance of Yang et al. (2025) §4.1
    (estimator.py in the reference repo), adapted for 1-D numerical features.

    Three paths share the same MLP weights but receive different condition
    embeddings (unconditional / label / sensitive), mirroring the original
    3× feature-concatenation trick used with the U-Net.
    """

    def __init__(
        self,
        d_x:                  int,
        d_emb:                int   = 128,
        n_layers:             int   = 4,
        overall_guid_weight:  float = 1.0,
        cond_guid_weight:     float = 1.0,   # w_s  — security-gate scale
        cond_guid_threshold:  float = 0.1,   # λ    — security-gate threshold
        warmup_steps:         int   = 10,    # δ    — warmup timestep
    ):
        super().__init__()
        self.d_x                 = d_x
        self.overall_guid_weight = overall_guid_weight
        self.cond_guid_weight    = cond_guid_weight
        self.cond_guid_threshold = cond_guid_threshold
        self.warmup_steps        = warmup_steps

        # Sinusoidal time embedding → projected
        self.t_proj = nn.Sequential(
            nn.Linear(d_emb, d_emb), nn.SiLU(), nn.Linear(d_emb, d_emb)
        )

        # Condition embeddings: unconditional token, class label y, sensitive attr a
        self.uncond_emb = nn.Embedding(1, d_emb)
        self.y_emb      = nn.Embedding(2, d_emb)
        self.a_emb      = nn.Embedding(2, d_emb)

        # Input projection (feature space → embedding space)
        self.x_proj = nn.Linear(d_x, d_emb)

        # Residual MLP backbone — shared across all three paths
        self.blocks = nn.ModuleList([
            ResidualBlock(d_emb, d_emb) for _ in range(n_layers)
        ])

        # Output projection (embedding space → feature space)
        self.out = nn.Linear(d_emb, d_x)

    # ---------------------------------------------------------------- #

    def _sinusoidal_embedding(self, t: torch.Tensor, d: int) -> torch.Tensor:
        half = d // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _run_path(
        self,
        x_t:      torch.Tensor,   # [B, d_x]
        t_emb:    torch.Tensor,   # [B, d_emb]
        cond_emb: torch.Tensor,   # [B, d_emb]
    ) -> torch.Tensor:            # [B, d_x]
        """Single conditioning path through the shared backbone."""
        h = self.x_proj(x_t) + t_emb + cond_emb
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)

    def forward(
        self,
        x_t: torch.Tensor,   # [B, d_x]  noisy features
        t:   torch.Tensor,   # [B]        integer diffusion timesteps
        y:   torch.Tensor,   # [B]        class labels (long)
        a:   torch.Tensor,   # [B]        sensitive attribute (long)
    ) -> torch.Tensor:       # [B, d_x]  guided noise estimate
        """
        3-path CFG forward (Yang et al. 2025, Eq. 5-6 / estimator.py).

        Returns the guided noise estimate to be used as the target-noise
        predictor in both training (loss = MSE vs true noise) and sampling.
        """
        t_emb = self.t_proj(self._sinusoidal_embedding(t, self.t_proj[0].in_features))

        zeros      = torch.zeros(x_t.shape[0], dtype=torch.long, device=x_t.device)
        uncond_emb = self.uncond_emb(zeros)   # ε̂(x_t)       — no condition
        label_emb  = self.y_emb(y)            # ε̂(x_t, c)    — label guided
        sens_emb   = self.a_emb(a)            # ε̂(x_t, s)    — sensitive guided

        z_uncond = self._run_path(x_t, t_emb, uncond_emb)
        z_label  = self._run_path(x_t, t_emb, label_emb)
        z_cond   = self._run_path(x_t, t_emb, sens_emb)

        # ---- label guidance (Eq. 5) ----
        label_guid = z_label - z_uncond

        # ---- sensitive guidance with security gate (Eq. 6 / §4.1) ----
        # scale ∝ |ε̂(x_t,c) − ε̂(x_t,s)|, clamped to [0, 1]
        scale = torch.clamp(
            torch.abs(z_label - z_cond) * self.cond_guid_weight,
            max=1.0,
        )
        # gate: zero out element-wise wherever (label − sensitive) > λ
        gate      = ((z_label - z_cond) <= self.cond_guid_threshold).float()
        cond_guid = (z_cond - z_uncond) * scale * gate

        # warmup: suppress sensitive guidance for t < δ (early reverse steps)
        warm_mask = (t >= self.warmup_steps).float().unsqueeze(-1)   # [B, 1]
        cond_guid = cond_guid * warm_mask

        # ---- combine (Eq. 5) ----
        total_guid = label_guid + cond_guid
        return z_uncond + total_guid * self.overall_guid_weight


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
        da_pct=None,
        real_data_size: int = 3000,
        # model
        ffnn: dict = None,
        # FairTabDDPM hyperparams
        fairtabddpm: dict = None,
        # misc
        use_pca: bool = False,
        pca_components: int = 10,
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
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True

        DEFAULT_FTDDPM = {
            "timesteps":            100,     # T — diffusion steps
            "d_emb":                128,     # embedding dimension
            "n_layers":             4,       # residual MLP depth
            "lr":                   3.9e-4,  # Adam lr (Yang et al. default)
            "weight_decay":         0.0,
            "diff_epochs":          1000,    # diffusion model training epochs
            "batch_size":           256,
            "n_synthetic":          2000,    # synthetic minority samples to add
            # CFG guidance parameters (estimator.py / §4.1)
            "overall_guid_weight":  1.0,     # w_g — overall guidance weight
            "cond_guid_weight":     1.0,     # w_s — security-gate scale weight
            "cond_guid_threshold":  0.1,     # λ   — security-gate threshold
            "warmup_steps":         10,      # δ   — warmup timestep
        }
        self.cfg        = {**DEFAULT_FTDDPM, **(fairtabddpm or {})}
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
        a_train = self.dataset.a_train
        a_test  = getattr(self.dataset, "a_test", None)

        ffnn_config = {
            "input_size":    feature_dim,
            "hidden_sizes":  self.ffnn_overrides.get("hidden_sizes", [32, 16]),
            "output_size": 2,
            "learning_rate": self.ffnn_overrides.get("learning_rate", 1e-3),
            "batch_size":    self.ffnn_overrides.get("batch_size", 64),
            "epochs":        self.ffnn_overrides.get("epochs", 20),
            "type":          "classification",
            "classes": [0, 1],
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

            # ---- Build diffusion schedule and denoising network ----
            schedule = DDPMSchedule(
                timesteps=int(self.cfg["timesteps"]), device=self.device
            )
            net = CondDenoiseNet(
                d_x                 = feature_dim,
                d_emb               = int(self.cfg["d_emb"]),
                n_layers            = int(self.cfg["n_layers"]),
                overall_guid_weight = float(self.cfg["overall_guid_weight"]),
                cond_guid_weight    = float(self.cfg["cond_guid_weight"]),
                cond_guid_threshold = float(self.cfg["cond_guid_threshold"]),
                warmup_steps        = int(self.cfg["warmup_steps"]),
            ).to(self.device)

            print(f"[FairTabDDPM] Training diffusion model for "
                  f"{self.cfg['diff_epochs']} epochs ...")
            t0 = time.time()
            diff_metrics = self._train_diffusion(net, schedule, x_train, y_train, a_train)
            print(f"[FairTabDDPM] Diffusion training done in {time.time()-t0:.1f}s")

            # ---- Sample synthetic minority + sensitive-group samples ----
            n_syn = int(self.cfg["n_synthetic"])
            # Auto-detect disadvantaged group: whichever a-group has fewer positives.
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
            beta_agent  = self._train_erm(ffnn_config, x_train_aug, y_train_aug)
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
        net:     CondDenoiseNet,
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

        # Normalise features to [-1, 1] per dimension for diffusion stability.
        x_min   = x_train.min(0).values
        x_max   = x_train.max(0).values
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
        best_loss  = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        for epoch in range(1, epochs + 1):
            net.train()
            total_loss = 0.0
            n_batches  = 0

            for x_b, y_b, a_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                a_b = a_b.to(self.device)

                t     = torch.randint(0, T, (len(x_b),), device=self.device)
                noise = torch.randn_like(x_b)
                x_t   = schedule.q_sample(x_b, t, noise)

                # Guided noise prediction (all 3 paths)
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
        net:       CondDenoiseNet,
        schedule:  DDPMSchedule,
        d_x:       int,
        n_samples: int,
        y_val:     int,
        a_val:     int,
        batch_size: int = 256,
    ) -> torch.Tensor:
        net.eval()
        T     = schedule.T
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
                ab_prev = schedule.alphas_bar[t_idx - 1] if t_idx > 0 \
                          else torch.tensor(1.0, device=self.device)

                # DDPM reverse step (x_0 prediction form)
                x0_pred = (x - (1.0 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
                x0_pred = x0_pred.clamp(-1.5, 1.5)
                dir_xt  = (1.0 - ab_prev - beta_t).clamp(min=0.0).sqrt() * noise_pred
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
