# --- Imports ---

# Standard library
import os
import sys
import time
import uuid
import subprocess
import json
import multiprocessing as mp
from datetime import datetime

# Third-party: numpy, pandas, sklearn, torch
import numpy as np
import pandas as pd
import torch
import copy
from copy import deepcopy

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from math import exp
from collections import deque
import itertools

# Local modules
from env import Environment
from dataset import Dataset
from agents.reinforce_agent import ReinforceAgent
# from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker

# ---------------- Config ----------------

class Training:
    def __init__(
        self,
        exp_group=None,
        spec_name=None,

        # ---- core experiment flags ----
        curriculum_learning=True,
        multiclass=False,
        dataset_name="census_income",

        # ---- class / bias ----
        minority_id=None,
        majority_id=None,
        third_id=None,
        bias_pct=0.18,

        # ---- PCA / trajectory ----
        pca_components=2,
        traj_length=500,
        real_data_size=1500,
        total_episodes=1,

        # ---- reward ----
        reward_mode="gauss_penalty",
        lambda_schedule=(0.8, 0.8),

        # ---- ENV hyperparams (NEW) ----
        use_delta_actions=True,
        delta_scale=0.10,
        delta_clip=0.20,
        pca_clip=None,
        radius_clip=None,

        # ---- misc ----
        seed=42,
        device='cpu',
    ):
        self.exp_group = exp_group
        self.spec_name = spec_name
        self.seed = seed
        self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.bias_pct = bias_pct
        self.pca_components = pca_components
        self.reward_mode = reward_mode
        self.lambda_schedule = lambda_schedule
        self.curriculum_learning = curriculum_learning
        self.multiclass = multiclass
        self.dataset_name = dataset_name

        self.minority_id = minority_id
        self.majority_id = majority_id
        self.third_id = third_id
        self.bias_pct = bias_pct

        self.pca_components = pca_components
        self.traj_length = traj_length
        self.real_data_size = real_data_size
        self.episodes = total_episodes
        if self.curriculum_learning:
            # frac_done + pca + editable_mask
            self.state_dim = 1 + 2 * self.pca_components
        else:
            self.state_dim = 2
        # ---- ENV hyperparams (NEW) ----
        self.use_delta_actions = use_delta_actions
        self.delta_scale = delta_scale
        self.delta_clip = delta_clip
        self.pca_clip = pca_clip
        self.radius_clip = radius_clip
        from pathlib import Path
        self.project_root = Path(__file__).resolve().parent


        # -----------------------------
        # dataset
        # -----------------------------
        self.dataset = Dataset(
            dataset_name,
            multiclass=self.multiclass,
            minority_id=self.minority_id,
            majority_id=self.majority_id,
            third_id=self.third_id,
            pca_components=self.pca_components,
            seed=self.seed,
            device=self.device,
        )

        # -----------------------------
        # FFNN (alpha / beta) config
        # -----------------------------
        self.ffnn_config = {
            "input_size": self.pca_components,
            "hidden_sizes": [32, 16],
            "output_size": 3 if self.multiclass else 2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 10,
            "type": "classification",
            "classes": [0, 1, 2] if self.multiclass else [0, 1],
            "device": self.device,
            "seed": self.seed,
        }

        # -----------------------------
        # REINFORCE config
        # -----------------------------
        reinforce_config = {
            "state_size": self.state_dim,
            "action_size": self.pca_components,
            "hidden_sizes": [64, 64],
            "total_episodes": self.episodes,
            "lr": 3e-4,
            "gamma": 0.99,
            "entropy_start": 1e-2,
            "entropy_end": 0.0,
            "seed": self.seed,
            "device": self.device,
        }

        self.dl_generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # -----------------------------
        # agents
        # -----------------------------
        self.agent = ReinforceAgent(**reinforce_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)

        # -----------------------------
        # buffers
        # -----------------------------
        self._corr_window = 40
        self._local_buf = deque(maxlen=self._corr_window)
        self._delta_buf = deque(maxlen=self._corr_window)

    def _corr_local_delta(self):
        # need a few points for a stable estimate
        if len(self._local_buf) < 8:
            return float('nan')
        x = torch.tensor(list(self._local_buf), dtype=torch.float32)
        y = torch.tensor(list(self._delta_buf), dtype=torch.float32)
        # z-score both; guard against zero variance
        x = x - x.mean()
        y = y - y.mean()
        xs = x.std(unbiased=False)
        ys = y.std(unbiased=False)
        if xs.item() < 1e-8 or ys.item() < 1e-8:
            return float('nan')
        x = x / xs
        y = y / ys
        return float((x * y).mean())  # Pearson corr over the rolling window

    def train_predictor_model(self, model, x_train, y_train):
        train_dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            generator=self.dl_generator,
        )
        model.train(loader)
        return model

    # ---------------- Reward helpers ----------------
    # Generic "get P(y=1|x)" from an FFNNAgent (alpha or beta)
    def _p1_from_agent(self, agent, x):
        agent.model.eval()
        with torch.no_grad():
            logits = agent.model(x)             # [N, 2]
            probs  = torch.softmax(logits, -1)  # [N, 2]
            return probs[..., 1]                # [N]

    # Binary F1 for the positive class, from probabilities and a threshold
    def f1_from_probs(self, y_true, p1, threshold=0.5):
        y_true = y_true.to(p1.device).long()
        y_pred = (p1 >= threshold).long()
        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        fn = ((y_pred == 0) & (y_true == 1)).sum().float()
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1

    # Per-sample Brier (MSE on probabilities)
    def brier_per_sample(self, y_true, p1):
        y_true = y_true.to(p1.device).float()
        return (p1 - y_true) ** 2  # in [0,1]

    # Beta model F1 minority class score + local term (no EMA)
    def compute_reward(
        self,
        alpha_model, beta_model, stale_beta_model,   # stale_beta_model unused (kept for signature compat)
        x_theta_val, y_theta_val,                    # θ-space VAL set (labels are 0/1 or 0/1/2)
        x_phi, y_phi,                                # synthetic batch (y_phi should be 1 for rope)
        progress: float,                             # (episode+1)/EPISODES in [0,1]
        f1_thresh: float = 0.5,

        # schedules / gates
        lambda_schedule=(0.30, 0.95),                # (start, end) across the run

        # hinge-penalty for majority/weighted scores (used only in *_penalty mode)
        epsilon_majority: float = 0.01,   # allow up to -1.0 pp on majority F1
        epsilon_weighted: float = 0.005,  # allow up to -0.5 pp on weighted F1
        c_majority: float = 0.30,         # penalty weight for majority violation
        c_weighted: float = 0.30,         # penalty weight for weighted violation
        class_mode: str = "binary",   # "binary" or "multiclass"
    ):
        """
        Reward combines:
        - Global term: ΔF1(minority, rope) on θ_val between β and α.
        - Local term: Gaussian usefulness around 0.5 on Pα(y=1|x_phi), with optional slice gate.

        class_mode:
        - "binary": y_theta_val is 0/1; we evaluate directly.
        - "multiclass": y_theta_val is 0/1/2; we evaluate as (rope vs non-rope):
                y_val_bin = 1 if y==1 else 0    # {0,2} -> 0, 1->1
        """
        # ---------- Mode selection ----------
        alias = {
            "gauss_nopen": "local_gauss",
            "gauss_penalty": "local_gauss_penalty",
            "judge_conf": "local_gauss",
        }
        mode = alias.get(self.reward_mode, self.reward_mode)
        valid = {"local_gauss", "local_gauss_penalty", "local_slices"}
        if mode not in valid:
            raise ValueError(f"reward_mode must be one of {valid}, got {self.reward_mode!r}")

        # --- λ schedule ---
        lambda_start, lambda_end = lambda_schedule
        lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

        # --- Build rope vs non-rope labels for evaluation ---
        if class_mode not in ("binary", "multiclass"):
            raise ValueError("class_mode must be 'binary' or 'multiclass'")

        if class_mode == "binary":
            # y_theta_val already 0/1
            y_val_bin = y_theta_val.long()
        else:
            # collapse 3 classes into rope vs non-rope: {0,2}→0, 1→1
            y_val_bin = (y_theta_val == 1).long()

        # shared Gaussian width (also used in diagnostics)
        tau = 0.10

        with torch.no_grad():
            # ---- Global term on θ_val ----

            # Always use Pα(y=1|x) and Pβ(y=1|x). For 3-class, this is softmax column 1 (rope).
            p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val  = self._p1_from_agent(beta_model,  x_theta_val)

            # Minority / majority / macro (α vs β) on rope-vs-rest binarization
            f1_minority_alpha = self.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
            f1_minority_beta  = self.f1_from_probs(y_val_bin, p1_beta_val,  f1_thresh)
            f1_majority_alpha = self.f1_from_probs(1 - y_val_bin, 1 - p1_alpha_val, 1 - f1_thresh)
            f1_majority_beta  = self.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val,  1 - f1_thresh)

            f1_macro_beta = 0.5 * (f1_minority_beta + f1_majority_beta)

            # Weighted F1 (support-weighted) using rope fraction on θ_val
            pos_frac = float(y_val_bin.float().mean().item())
            neg_frac = 1.0 - pos_frac
            f1_weighted_alpha = pos_frac * float(f1_minority_alpha) + neg_frac * float(f1_majority_alpha)
            f1_weighted_beta  = pos_frac * float(f1_minority_beta)  + neg_frac * float(f1_majority_beta)

            # Deltas (β − α)
            delta_f1_minority = float(f1_minority_beta - f1_minority_alpha)
            delta_f1_majority = float(f1_majority_beta - f1_majority_alpha)
            delta_f1_weighted = float(f1_weighted_beta - f1_weighted_alpha)

            # ---- Local term on synthetic Φ (judge = α) ----
            # y_phi should be 1 for rope samples; we score usefulness by α's uncertainty.
            p = self._p1_from_agent(alpha_model, x_phi)  # P_alpha(y=1|x_phi) in [0,1]

            # Symmetric Gaussian around 0.5
            m = torch.abs(p - 0.5)
            score_gauss = torch.exp(-0.5 * (m / tau) ** 2)
            score_local_raw = score_gauss.clone()

            # Cap local score
            LOCAL_CAP = 1.0
            cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
            over_mask = (score_local_raw > cap_t).float()
            local_cap_frac = float(over_mask.mean().item())
            score_local = torch.minimum(score_local_raw, cap_t)

            # ---- Combine global + local ----
            base_reward = lambda_t * delta_f1_minority + (1.0 - lambda_t) * score_local  # [T]

            # Penalty (optional mode)
            maj_violation = 0.0
            wtd_violation = 0.0
            penalty = 0.0
            if mode == "local_gauss_penalty":
                maj_violation = max(0.0, -(delta_f1_majority + epsilon_majority))
                wtd_violation = max(0.0, -(delta_f1_weighted + epsilon_weighted))
                penalty = c_majority * maj_violation + c_weighted * wtd_violation

            reward = base_reward - penalty
            mean_local = float(score_local.mean().item())

            # keep these around for diagnostics
            alpha_wrong_rate = float(
                ((p >= 0.5).float() != (y_phi.to(p.device).float())).float().mean().item()
            )
            judge_conf_mean = float(score_gauss.mean().item())
            uncert_alpha_mean = float((1.0 - (2.0 * m).clamp(0, 1)).mean().item())

        # -------------------------------------------------------
        # Extra diagnostics: confidence, radius, and grad norm
        # (computed outside no_grad, using detached copy of x_phi)
        # -------------------------------------------------------
        diag_mean_conf_all  = float("nan")
        diag_frac_mid_conf  = float("nan")
        diag_gen_radius_mean = float("nan")
        diag_grad_norm      = float("nan")

        try:
            x_phi_det = x_phi.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                p_diag = self._p1_from_agent(alpha_model, x_phi_det)  # [T]

                # 1) confidence over generator samples: max(p, 1-p)
                conf = torch.maximum(p_diag, 1.0 - p_diag)
                diag_mean_conf_all = float(conf.mean().item())

                # fraction near decision boundary (0.4–0.6)
                mid_band = ((p_diag >= 0.4) & (p_diag <= 0.6)).float()
                diag_frac_mid_conf = float(mid_band.mean().item())

                # 2) radial distance in PCA space for generator samples
                radius = torch.linalg.norm(x_phi_det, dim=1)  # [T]
                diag_gen_radius_mean = float(radius.mean().item())

                # 3) gradient norm of local Gaussian reward w.r.t x
                m_diag = torch.abs(p_diag - 0.5)
                score_gauss_diag = torch.exp(-0.5 * (m_diag / tau) ** 2)
                local_mean_diag = score_gauss_diag.mean()  # scalar
                grad_x, = torch.autograd.grad(
                    local_mean_diag,
                    x_phi_det,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                diag_grad_norm = float(torch.linalg.norm(grad_x, dim=1).mean().item())
        except Exception as e:
            # If anything goes wrong, keep NaNs but don't break training
            # print(f"[warn] diag metrics failed: {e}")
            pass

        diagnostics = {
            "reward_mode": mode,
            "global_reward": float(f1_minority_beta),     # F1 rope (β) on θ_val
            "local_reward": mean_local,                   # mean local score
            "f1_macro_beta": float(f1_macro_beta),

            # Details (existing)
            "judge_conf_mean": judge_conf_mean,
            "uncert_alpha_mean": uncert_alpha_mean,
            "alpha_wrong_rate": alpha_wrong_rate,
            "delta_f1_val": delta_f1_minority,
            "delta_f1_majority": delta_f1_majority,
            "delta_f1_weighted": delta_f1_weighted,
            "f1_minority_alpha": float(f1_minority_alpha),
            "f1_minority_beta_stale": 0,
            "local_cap_frac": local_cap_frac,

            # Penalty diagnostics
            "epsilon_majority": epsilon_majority if mode == "local_gauss_penalty" else None,
            "epsilon_weighted": epsilon_weighted if mode == "local_gauss_penalty" else None,
            "penalty": penalty if mode == "local_gauss_penalty" else 0.0,
            "maj_violation": maj_violation if mode == "local_gauss_penalty" else 0.0,
            "wtd_violation": wtd_violation if mode == "local_gauss_penalty" else 0.0,
            "corr_factor_mean": None,

            # NEW diagnostic metrics
            "diag_mean_conf_all": diag_mean_conf_all,          # avg max(p, 1-p) over generator samples
            "diag_frac_mid_conf": diag_frac_mid_conf,          # fraction with p in [0.4, 0.6]
            "diag_gen_radius_mean": diag_gen_radius_mean,      # mean PCA radius of generated samples
            "diag_grad_norm": diag_grad_norm,                  # mean ||∂ local reward / ∂x||
        }
        return reward, diagnostics

    # ---------------- Training loop ----------------
    def __call__(self):
        start_time = time.time()

        run_stats = {
            # ---- experiment identity ----
            "EXP_GROUP": self.exp_group,
            "SPEC_NAME": self.spec_name,
            "dataset_name": self.dataset_name,
            "multiclass": self.multiclass,

            "CURRICULUM_LEARNING": self.curriculum_learning,
            "EPISODES": self.episodes,

            # ---- data / trajectory ----
            "TRAJ_LENGTH": self.traj_length,
            "REAL_DATA_SIZE": self.real_data_size,
            "BIAS_PCT": self.bias_pct,

            "reward_mode": self.reward_mode,
            "lambda_schedule": self.lambda_schedule,

            "pca_components": self.pca_components,
            "minority_id": self.minority_id,
            "majority_id": self.majority_id,
            "third_id": self.third_id,

            "use_delta_actions": self.use_delta_actions,
            "delta_scale": self.delta_scale,
            "delta_clip": self.delta_clip,
            "pca_clip": self.pca_clip,
            "radius_clip": self.radius_clip,

            "seed": self.seed,
        }


        # create beta factory once (so tracker can rehydrate best-beta for final test)
        beta_factory = lambda: FFNNAgent(**self.ffnn_config)

        with EpisodeTracker(
            run_stats,
            dataset=self.dataset,
            save_dir="training_runs",
            compare_metric="average_reward",
            beta_factory=beta_factory,
            seed=self.seed,
        ) as tracker:
            self.tracker = tracker

            # ---------------- Data splits ----------------
            x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = (
                self.dataset.get_data_splits(
                    train_size=self.real_data_size,
                    bias_pct=self.bias_pct,
                    pca_components=self.pca_components,
                )
            )
            minority_mask = (y_theta_train == 1)
            real_minority_samples = x_theta_train[minority_mask]

            total_data = len(x_theta_train) + self.traj_length
            real_percentage = (len(x_theta_train) / total_data) * 100
            synthetic_percentage = (self.traj_length / total_data) * 100
            print(
                f"Beta will train with: {real_percentage:.2f}% real data, "
                f"{synthetic_percentage:.2f}% synthetic data: "
            )

            # Train α once on real-only train set
            self.alpha_model = self.train_predictor_model(
                self.alpha_model, x_theta_train, y_theta_train
            )
            self.tracker.save_alpha_state_dict(
                self.alpha_model, self.ffnn_config, self.pca_components
            )

            try:
                pca_means = getattr(self.dataset, "pca_means_tensor", None)
            except Exception:
                pca_means = None

            # Build a multi-stage curriculum: dims 2..A (or up to A if A < 10)
            if self.curriculum_learning:
                A = self.pca_components
                start_dim = min(2, A)
                max_dim = min(20, A)   # if A < 10, stop at A

                ks = list(range(start_dim, max_dim + 1))  # [2,3,...,10]
                num_stages = len(ks)

                curriculum_stages = []
                for i, k in enumerate(ks):
                    # thresholds spaced across [0, 1); last stage covers the tail
                    th = i / float(num_stages)
                    curriculum_stages.append((th, k))
            else:
                curriculum_stages = None

            env = Environment(
                curriculum=self.curriculum_learning,
                target=1,
                max_actions=self.traj_length,
                total_episodes=self.episodes,
                device=self.device,
                seed=self.seed,

                # PCA / curriculum
                pca_components=self.pca_components,
                pca_means=pca_means,
                curriculum_stages=curriculum_stages,
                real_minority_samples=real_minority_samples,

                # ---- NEW: delta-action + clipping controls ----
                use_delta_actions=self.use_delta_actions,     # bool
                delta_scale=self.delta_scale,                 # e.g. 0.10
                delta_clip=self.delta_clip,                   # e.g. 0.20 or None
                pca_clip=self.pca_clip,                       # e.g. None, 5.0, 8.0
                use_radius_clip=(self.radius_clip is not None),
                radius_clip=self.radius_clip,                 # e.g. None, 8.0, 10.0
            )

            # ---------------- Episodes ----------------
            for episode in range(self.episodes):
                A = self.pca_components
                if self.curriculum_learning:                 # <<< CHANGED
                    D = 1 + A + A                            # <<< CHANGED
                else:
                    D = 2                                    # <<< CHANGED (legacy)

                # Pre-allocate (GPU tensors)
                states = torch.zeros(
                    (self.traj_length, D),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.zeros(
                    (self.traj_length, A),
                    dtype=torch.float32,
                    device=self.device,
                )
                next_states = torch.zeros(
                    (self.traj_length, D),
                    dtype=torch.float32,
                    device=self.device,
                )
                dones = torch.zeros(
                    self.traj_length, dtype=torch.bool, device=self.device
                )

                x_syn_tensor = torch.zeros(
                    (self.traj_length, A),
                    dtype=torch.float32,
                    device=self.device,
                )
                y_syn_tensor = torch.zeros(
                    self.traj_length, dtype=torch.long, device=self.device
                )

                # Reset env — pass episode index so curriculum can schedule stages
                if self.curriculum_learning:                 # <<< CHANGED
                    state = env.reset(episode_idx=episode)   # <<< CHANGED
                else:
                    state = env.reset()                      # <<< CHANGED (still works; env has default)

                # Reset beta to its initial snapshot each episode for a stable baseline
                stale_beta_model = copy.deepcopy(self.beta_model)
                self.beta_model.reset()

                for t in range(self.traj_length):
                    action = self.agent.predict(state)
                    next_state, done, info = env.step(action, (t + 1))

                    states[t] = state
                    actions[t] = action
                    next_states[t] = next_state
                    dones[t] = done

                    # In curriculum mode, use env's current_pca as the *actual* synthetic PCA point.
                    # In non-curriculum mode, we keep the legacy behaviour: action == PCA point.
                    if self.curriculum_learning:             # <<< CHANGED
                        pca_sample = info.get("current_pca", action)
                    else:
                        pca_sample = action

                    x_syn_tensor[t] = pca_sample             # <<< CHANGED (was: action)
                    y_syn_tensor[t] = info["sampled_target"]  # 1 for minority

                    state = next_state
                    if done:
                        break

                T = t + 1 if done else self.traj_length
                x_phi_t = x_syn_tensor[:T]
                y_phi_t = y_syn_tensor[:T]

                # Train beta on hybrid (real + synthetic for this episode)
                x_hybrid = torch.cat([x_theta_train, x_phi_t])
                y_hybrid = torch.cat([y_theta_train, y_phi_t])
                self.beta_model = self.train_predictor_model(
                    self.beta_model, x_hybrid, y_hybrid
                )

                progress = (episode + 1) / self.episodes

                # Rewards (alpha baseline global + per-step local)
                rewards, diagnostics = self.compute_reward(
                    self.alpha_model,
                    self.beta_model,
                    stale_beta_model,
                    x_theta_val,
                    y_theta_val,
                    x_phi_t,
                    y_phi_t,
                    progress=progress,
                    f1_thresh=0.5,
                    lambda_schedule=self.lambda_schedule,
                    class_mode=("multiclass" if self.multiclass else "binary"),
                )

                # Truncate episode tensors and learn
                states = states[:T]
                actions = actions[:T]
                next_states = next_states[:T]
                dones = dones[:T]
                rewards = rewards[:T]

                self.agent.learn_trajectory(
                    states, actions, rewards, next_states, dones, episode
                )

                reward_metrics = {
                    "avg_reward": float(torch.mean(rewards)),
                    "obj1_f1_minority_beta": float(diagnostics["global_reward"]),
                    "obj2_local_useful_mean": float(diagnostics["local_reward"]),
                    "macro_f1_beta": float(diagnostics["f1_macro_beta"]),
                }

                new_reward_metrics = {
                    "judge_conf_mean": float(
                        diagnostics.get("judge_conf_mean", float("nan"))
                    ),
                    "uncert_alpha_mean": float(
                        diagnostics.get("uncert_alpha_mean", float("nan"))
                    ),
                    "alpha_wrong_rate": float(
                        diagnostics.get("alpha_wrong_rate", float("nan"))
                    ),
                    "corr_factor_mean": float(
                        0.5 + 0.5 * diagnostics.get("alpha_wrong_rate", 0.0)
                    ),
                    "f1_minority_alpha": float(diagnostics["f1_minority_alpha"]),
                    "f1_minority_beta_stale": float(
                        diagnostics["f1_minority_beta_stale"]
                    ),
                    "local_cap_frac": float(diagnostics["local_cap_frac"]),

                    # 🔹 NEW diagnostics actually written to metrics.csv (prefixed as newr.*)
                    "diag_mean_conf_all": float(
                        diagnostics.get("diag_mean_conf_all", float("nan"))
                    ),
                    "diag_frac_mid_conf": float(
                        diagnostics.get("diag_frac_mid_conf", float("nan"))
                    ),
                    "diag_gen_radius_mean": float(
                        diagnostics.get("diag_gen_radius_mean", float("nan"))
                    ),
                    "diag_grad_norm": float(
                        diagnostics.get("diag_grad_norm", float("nan"))
                    ),
                }

                # Pull episode-level values from diagnostics
                local_mean = float(diagnostics["local_reward"])
                delta_val = float(diagnostics["delta_f1_val"])

                # Update rolling buffers and compute correlation
                self._local_buf.append(local_mean)
                self._delta_buf.append(delta_val)
                corr_local_delta = self._corr_local_delta()

                alignment_metrics = {
                    "delta_f1_val": delta_val,
                    "corr_local_delta": float(corr_local_delta),
                    "curriculum_stage": int(env.current_stage),
                }

                # Log
                avg_reward = torch.mean(rewards)
                self.tracker.log_episode(
                    episode + 1,
                    reward_metrics,      # Avg reward across trajectory
                    new_reward_metrics,  # local/details
                    alignment_metrics,   # global ΔF1, corr
                )
                # Save periodic snapshot and best-so-far synthetic
                self.tracker.maybe_save_synthetic(
                    episode_num=episode + 1,
                    x_syn=x_phi_t,
                    y_syn=y_phi_t,
                    avg_reward=float(avg_reward),
                    obj1=float(diagnostics["global_reward"]),
                    obj2_mean=float(diagnostics["local_reward"]),
                    global_f1=float(diagnostics["f1_macro_beta"]),
                    feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                    beta_model=self.beta_model,
                )

            # ---------------- Final test ----------------
            self.tracker.log_final_test(
                alpha_model=self.alpha_model,
                x_test=x_theta_test,
                y_test=y_theta_test,
                f1_thresh=0.5,

                prefer_best_beta=True,
                beta_model=self.beta_model,      # optional; best checkpoint will override if present

                x_train=x_theta_train,
                y_train=y_theta_train,

                # existing jitter baseline params (use defaults or override as needed)
                jitter_n=None,
                jitter_scale=0.20,

                # NEW alpha toggles/params
                run_alpha_raw_original=False,
                run_alpha_plus_real=False,
                alpha_plus_real_n=2000,

                # NEW CTGAN baseline toggles/params
                run_alpha_plus_ctgan=False,
                alpha_plus_ctgan_n=self.traj_length,   # or your chosen synth budget
                ctgan_epochs=self.episodes,
                cap_ctgan_train=None,

                # NEW CTABGAN baseline toggles/params
                run_ctabgan=False,
                alpha_plus_ctabgan_n=self.traj_length,  # same budget as CTGAN for fairness

                # CTABGAN subprocess wiring (optional if you set defaults in signature)
                ctab_python="/home/epigou/envs/ctabgan/bin/python",
                ctab_repo="/home/epigou/CTAB-GAN-Plus-DP",
                ctab_runner=str(self.project_root / "benchmarks" / "ctabgan" / "run_ctabgan.py"),

                # Dataset rebuild params (must match CTGAN baseline rebuild)
                data_path=None,  # set if needed, or rely on default
                bias_pct=self.bias_pct,
                val_frac=0.20,
                test_frac=0.20,
                train_size=self.real_data_size,

                # additional CTABGAN batch/seed/pca-related params
                batch_size=64,
                pca_components=None,
                seed=self.seed,
            )


        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")