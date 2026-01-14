import os
import sys
import time
import uuid
import subprocess
import json
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import copy
from copy import deepcopy
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from math import exp
from collections import deque
import itertools

from env import Environment
from dataset import Dataset
from agents.reinforce_agent import ReinforceAgent
# from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker

class Training:
    def __init__(
        self,
        exp_group=None,
        spec_name=None,

        #flags
        curriculum_learning=True,
        multiclass=False,
        dataset_name="census_income",

        #class / bias
        minority_id=None,
        majority_id=None,
        third_id=None,
        bias_pct=0.18,

        #PCA / trajectory
        pca_components=2,
        traj_length=500,
        real_data_size=1500,
        total_episodes=1,

        #reward
        reward_mode="gauss_penalty",
        lambda_schedule=(0.8, 0.8),

        #ENV hyperparams
        use_delta_actions=True,
        delta_scale=0.10,
        delta_clip=0.20,
        pca_clip=None,
        radius_clip=None,

        #Config dictionaries
        ffnn=None,         
        reinforce=None,    
        curriculum=None,   
        benchmarks=None,   

        #misc
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

        # ENV hyperparams
        self.use_delta_actions = use_delta_actions
        self.delta_scale = delta_scale
        self.delta_clip = delta_clip
        self.pca_clip = pca_clip
        self.radius_clip = radius_clip


        self.project_root = Path(__file__).resolve().parent

        self.ffnn_overrides = ffnn or {}
        self.reinforce_overrides = reinforce or {}
        self.curriculum_overrides = curriculum or {}
        self.benchmarks_overrides = benchmarks or {}

        # dataset
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

        # Curriculum schedule config
        DEFAULT_CURRICULUM = {
            "start_dim": 2,
            "max_dim_cap": 14,
            "stage_count": 5,
            "schedule": "linear",
        }
        self.curriculum_config = {**DEFAULT_CURRICULUM, **self.curriculum_overrides}

        # Benchmarks config
        DEFAULT_BENCHMARKS = {
            "run_ctgan": False,
            "alpha_plus_ctgan_n": 2000,
            "ctgan_epochs": 300,
            "cap_ctgan_train": None,

            "run_ctabgan": False,
            "alpha_plus_ctabgan_n": 2000,
            "ctab_python": None,
            "ctab_repo": None,
        }
        self.benchmarks_config = {**DEFAULT_BENCHMARKS, **self.benchmarks_overrides}

        # FFNN (alpha / beta) config
        DEFAULT_FFNN = {
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
        self.ffnn_config = {**DEFAULT_FFNN, **self.ffnn_overrides}

        # keep these aligned
        self.ffnn_config["input_size"] = self.pca_components
        self.ffnn_config["output_size"] = 3 if self.multiclass else 2
        self.ffnn_config["classes"] = [0, 1, 2] if self.multiclass else [0, 1]
        self.ffnn_config["device"] = self.device
        self.ffnn_config["seed"] = self.seed

        # REINFORCE config
        DEFAULT_REINFORCE = {
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
        reinforce_config = {**DEFAULT_REINFORCE, **self.reinforce_overrides}

        # keep these aligned
        reinforce_config["state_size"] = self.state_dim
        reinforce_config["action_size"] = self.pca_components
        reinforce_config["total_episodes"] = self.episodes
        reinforce_config["seed"] = self.seed
        reinforce_config["device"] = self.device

        self.dl_generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # agents
        self.agent = ReinforceAgent(**reinforce_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)

        # buffers
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
            batch_size=int(self.ffnn_config["batch_size"]),
            shuffle=True,
            generator=self.dl_generator,
        )
        model.train(loader)
        return model

    # ---------------- Reward helpers ----------------
    # Generic "get P(y=1|x)" from an FFNNAgent (alpha or beta)
    def _p1_from_agent(self, agent, x, *, no_grad=True):
        agent.model.eval()
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            logits = agent.model(x)
            probs  = torch.softmax(logits, -1)
            return probs[..., 1]


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
        
    # ---------------- helpers (external----------------
    def _dp_diff_from_probs(self, a01: torch.Tensor, p1: torch.Tensor, thresh: float = 0.5) -> float:
        a01 = a01.to(p1.device).long().view(-1)
        p1 = p1.view(-1)
        g0 = (a01 == 0)
        g1 = (a01 == 1)
        if g0.sum()==0 or g1.sum()==0:
            return float("nan")
        r0 = p1[g0].mean()
        r1 = p1[g1].mean()
        return float(torch.abs(r0 - r1).item())

    # Beta model F1 minority class score + local term (no EMA)
    def compute_reward(
        self,
        alpha_model, beta_model,
        x_theta_val, y_theta_val,                    
        x_phi, y_phi,                                
        progress: float,                             
        f1_thresh: float = 0.5,            # (start, end) across the run
        # hinge-penalty for majority/weighted scores (used only in *_penalty mode)
        epsilon_majority: float = 0.01,   # allow up to -1.0 pp on majority F1
        epsilon_weighted: float = 0.005,  # allow up to -0.5 pp on weighted F1
        c_majority: float = 0.30,         # penalty weight for majority violation
        c_weighted: float = 0.30,         # penalty weight for weighted violation
        class_mode: str = "binary",   # "binary" or "multiclass"
        dp_scale: float = 50.0,
        fairness_style: str = "improvement",
        prev_dp_beta: float | None = None
    ):
        """
        Reward combines:
        - Global term: ΔF1minority on θ_val between beta and alpha.
        - Local term: Gaussian usefulness around 0.5 on Pα(y=1|x_phi).
        """
        mode = self.reward_mode
        valid = {"local_gauss", "local_gauss_penalty", "fairness"}
        if mode not in valid:
            raise ValueError(f"reward_mode must be one of {valid}, got {self.reward_mode!r}")

        # lambda schedule
        lambda_start, lambda_end = self.lambda_schedule
        lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

        if class_mode not in ("binary", "multiclass"):
            raise ValueError("class_mode must be 'binary' or 'multiclass'")

        if class_mode == "binary":
            # y_theta_val already 0/1
            y_val_bin = y_theta_val.long()
        else:
            y_val_bin = (y_theta_val == 1).long()

        # shared Gaussian width (also used in diagnostics)
        tau = 0.10

        with torch.no_grad():
            #Global term on θ_val 
            p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val  = self._p1_from_agent(beta_model,  x_theta_val)

            # Minority / majority / macro (alpha vs beta)
            f1_minority_alpha = self.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
            f1_minority_beta  = self.f1_from_probs(y_val_bin, p1_beta_val,  f1_thresh)
            f1_majority_alpha = self.f1_from_probs(1 - y_val_bin, 1 - p1_alpha_val, 1 - f1_thresh)
            f1_majority_beta  = self.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val,  1 - f1_thresh)

            f1_macro_beta = 0.5 * (f1_minority_beta + f1_majority_beta)

            # Weighted F1
            pos_frac = float(y_val_bin.float().mean().item())
            neg_frac = 1.0 - pos_frac
            f1_weighted_alpha = pos_frac * float(f1_minority_alpha) + neg_frac * float(f1_majority_alpha)
            f1_weighted_beta  = pos_frac * float(f1_minority_beta)  + neg_frac * float(f1_majority_beta)

            # Deltas 
            delta_f1_minority = float(f1_minority_beta - f1_minority_alpha)
            delta_f1_majority = float(f1_majority_beta - f1_majority_alpha)
            delta_f1_weighted = float(f1_weighted_beta - f1_weighted_alpha)

        # -------- fairness global objective (DP) --------
        dp_alpha = float("nan")
        dp_beta  = float("nan")

        delta_dp = 0.0                 # dp_alpha - dp_beta
        dp_improve = 0.0               # abs

        if mode == "fairness":
            a_theta_val = self.dataset.a_val
            assert len(a_theta_val) == x_theta_val.shape[0], "a_val misaligned with x_theta_val"

            dp_alpha = self._dp_diff_from_probs(a_theta_val, p1_alpha_val, thresh=f1_thresh)
            dp_beta  = self._dp_diff_from_probs(a_theta_val, p1_beta_val,  thresh=f1_thresh)

            # NaN guard (missing group etc.)
            if (dp_alpha != dp_alpha) or (dp_beta != dp_beta):
                delta_dp = 0.0
                dp_improve = 0.0
            else:
                dp_alpha = float(dp_alpha)
                dp_beta  = float(dp_beta)

                # model comparison: positive if beta reduced DP gap vs alpha
                delta_dp = dp_alpha - dp_beta

                # (2) time-improvement shaping: positive if DP gap shrinks this step
                # Prefer using the caller-provided previous dp_beta; otherwise use a stored value.
                if prev_dp_beta is None:
                    prev = getattr(self, "_prev_dp_beta", None)
                else:
                    prev = prev_dp_beta

                if prev is None or (prev != prev):   # None or NaN
                    dp_improve = 0.0
                else:
                    prev = float(prev)
                    dp_improve = abs(prev) - abs(dp_beta)

                # store for next call (so you don’t have to thread prev_dp_beta everywhere)
                self._prev_dp_beta = dp_beta



        # ---- Local term on synthetic Φ----
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
        if mode == "fairness":
            if fairness_style == "improvement":
                global_term = dp_improve
            elif fairness_style == "delta":
                global_term = delta_dp
            else:
                raise ValueError("fairness_style must be 'improvement' or 'delta'")
        else:
            global_term = delta_f1_minority


        if mode == "fairness":
            global_term_scaled = dp_scale * float(global_term)
        else:
            global_term_scaled = float(global_term)

        base_reward = lambda_t * global_term_scaled + (1.0 - lambda_t) * score_local


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

        # Extra diagnostics: confidence, radius, and grad norm
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
        except Exception:
            # keep NaNs but don't break training
            pass

        global_obj = float(global_term_scaled)



        diagnostics = {
            "global_obj": global_obj,
            "f1_minority_beta": float(f1_minority_beta),
            "f1_minority_alpha": float(f1_minority_alpha),

            "local_reward": mean_local,
            "f1_macro_beta": float(f1_macro_beta),

            # Details (existing)
            "judge_conf_mean": judge_conf_mean,
            "uncert_alpha_mean": uncert_alpha_mean,
            "alpha_wrong_rate": alpha_wrong_rate,
            "delta_f1_val": delta_f1_minority,
            "delta_f1_majority": delta_f1_majority,
            "delta_f1_weighted": delta_f1_weighted,
            "f1_minority_beta_stale": 0,
            "local_cap_frac": local_cap_frac,
            "dp_alpha": dp_alpha,
            "dp_beta": dp_beta,
            "delta_dp": delta_dp,
            "diag_mean_abs_margin": float(torch.abs(p_diag - 0.5).mean().item()),
            "dp_improve": dp_improve,
            "dp_scale": dp_scale if mode == "fairness" else None,
            "fairness_style": fairness_style if mode == "fairness" else None,

            # Penalty diagnostics
            "epsilon_majority": epsilon_majority if mode == "local_gauss_penalty" else None,
            "epsilon_weighted": epsilon_weighted if mode == "local_gauss_penalty" else None,
            "penalty": penalty if mode == "local_gauss_penalty" else 0.0,
            "maj_violation": maj_violation if mode == "local_gauss_penalty" else 0.0,
            "wtd_violation": wtd_violation if mode == "local_gauss_penalty" else 0.0,
            "corr_factor_mean": None,

            # extra diagnostic metrics
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

            #Data splits 
            x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = (
                self.dataset.get_data_splits(
                    train_size=self.real_data_size,
                    bias_pct=self.bias_pct,
                    pca_components=self.pca_components,
                    drop_protected=False,
                    protected_cols=self.dataset.protected_attributes
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

            # Train alpha once on real-only train set
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

            # ---------------- Curriculum stages ----------------
            if self.curriculum_learning:
                A = self.pca_components

                start_dim = int(self.curriculum_config.get("start_dim", 2))
                max_dim_cap = int(self.curriculum_config.get("max_dim_cap", 20))
                stage_count = int(self.curriculum_config.get("stage_count", 5))
                schedule = str(self.curriculum_config.get("schedule", "linear")).lower()

                kmax = min(max_dim_cap, A)
                start_dim = max(1, min(start_dim, kmax))  # guard

                if stage_count <= 1:
                    ks = [kmax]
                else:
                    if schedule == "linear":
                        ks = np.linspace(start_dim, kmax, stage_count)
                    else:
                        # fallback to linear for now
                        ks = np.linspace(start_dim, kmax, stage_count)

                    ks = [int(round(k)) for k in ks]
                    ks = sorted(set(max(start_dim, min(kmax, k)) for k in ks))
                    if ks[-1] != kmax:
                        ks.append(kmax)

                num_stages = len(ks)
                curriculum_stages = []
                for i, k in enumerate(ks):
                    th = i / float(num_stages)  # thresholds spaced across [0, 1)
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

                #delta-action + clipping controls 
                use_delta_actions=self.use_delta_actions,
                delta_scale=self.delta_scale,
                delta_clip=self.delta_clip,
                pca_clip=self.pca_clip,
                use_radius_clip=(self.radius_clip is not None),
                radius_clip=self.radius_clip,
            )

            # ---------------- Episodes Loop ----------------
            for episode in range(self.episodes):
                A = self.pca_components
                if self.curriculum_learning:
                    D = 1 + A + A
                else:
                    D = 2

                # Pre-allocate GPU tensors
                states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
                actions = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
                next_states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
                dones = torch.zeros(self.traj_length, dtype=torch.bool, device=self.device)

                x_syn_tensor = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
                y_syn_tensor = torch.zeros(self.traj_length, dtype=torch.long, device=self.device)

                # Reset env — pass episode index so curriculum can schedule stages
                if self.curriculum_learning:
                    state = env.reset(episode_idx=episode)
                else:
                    state = env.reset()

                self.beta_model.reset()

                for t in range(self.traj_length):
                    action = self.agent.predict(state)
                    next_state, done, info = env.step(action, (t + 1))

                    states[t] = state
                    actions[t] = action
                    next_states[t] = next_state
                    dones[t] = done

                    if self.curriculum_learning:
                        pca_sample = info.get("current_pca", action)
                    else:
                        pca_sample = action

                    x_syn_tensor[t] = pca_sample
                    y_syn_tensor[t] = info["sampled_target"]

                    state = next_state
                    if done:
                        break

                T = t + 1 if done else self.traj_length
                x_phi_t = x_syn_tensor[:T]
                y_phi_t = y_syn_tensor[:T]

                # Train beta on hybrid (real + synthetic for this episode)
                x_hybrid = torch.cat([x_theta_train, x_phi_t])
                y_hybrid = torch.cat([y_theta_train, y_phi_t])
                self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

                progress = (episode + 1) / self.episodes

                # Rewards
                rewards, diagnostics = self.compute_reward(
                    self.alpha_model,
                    self.beta_model,
                    x_theta_val,
                    y_theta_val,
                    x_phi_t,
                    y_phi_t,
                    progress=progress,
                    f1_thresh=0.5,
                    class_mode=("multiclass" if self.multiclass else "binary"),
                )

                # Truncate episode tensors and learn
                states = states[:T]
                actions = actions[:T]
                next_states = next_states[:T]
                dones = dones[:T]
                rewards = rewards[:T]

                self.agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

                reward_metrics = {
                    "avg_reward": float(torch.mean(rewards)),
                    "obj1_global": float(diagnostics["global_obj"]),
                    "f1_minority_beta": float(diagnostics["f1_minority_beta"]),
                    "obj2_local_useful_mean": float(diagnostics["local_reward"]),
                    "macro_f1_beta": float(diagnostics["f1_macro_beta"]),

                    # NEW: always present for header stability
                    "dp_alpha": float(diagnostics.get("dp_alpha", float("nan"))),
                    "dp_beta": float(diagnostics.get("dp_beta", float("nan"))),
                    "delta_dp": float(diagnostics.get("delta_dp", float("nan"))),
                }


                new_reward_metrics = {
                    "judge_conf_mean": float(diagnostics.get("judge_conf_mean", float("nan"))),
                    "uncert_alpha_mean": float(diagnostics.get("uncert_alpha_mean", float("nan"))),
                    "alpha_wrong_rate": float(diagnostics.get("alpha_wrong_rate", float("nan"))),
                    "alpha_wrong_rate_scaled": float(0.5 + 0.5 * diagnostics.get("alpha_wrong_rate", 0.0)),
                    "f1_minority_alpha": float(diagnostics["f1_minority_alpha"]),
                    "f1_minority_beta_stale": float(diagnostics["f1_minority_beta_stale"]),
                    "local_cap_frac": float(diagnostics["local_cap_frac"]),
                    "diag_mean_conf_all": float(diagnostics.get("diag_mean_conf_all", float("nan"))),
                    "diag_frac_mid_conf": float(diagnostics.get("diag_frac_mid_conf", float("nan"))),
                    "diag_gen_radius_mean": float(diagnostics.get("diag_gen_radius_mean", float("nan"))),
                    "diag_grad_norm": float(diagnostics.get("diag_grad_norm", float("nan"))),
                    "diag_mean_abs_margin":float(diagnostics.get("diag_mean_abs_margin", float("nan"))),
                    "dp_alpha": float(diagnostics.get("dp_alpha", float("nan"))),
                    "dp_beta": float(diagnostics.get("dp_beta", float("nan"))),
                    "delta_dp": float(diagnostics.get("delta_dp", 0.0)),
                }

                local_mean = float(diagnostics["local_reward"])
                if self.reward_mode == "fairness":
                    delta_val = float(diagnostics.get("dp_improve", 0.0))  # not delta_dp
                else:
                    delta_val = float(diagnostics["delta_f1_val"])


                self._local_buf.append(local_mean)
                self._delta_buf.append(delta_val)

                corr_local_delta = self._corr_local_delta()

                alignment_metrics = {
                    "delta_global": float(delta_val),   # delta_f1_minority OR delta_dp depending on mode
                    "corr_local_delta": float(corr_local_delta),
                    "curriculum_stage": int(env.current_stage),
                }
                lambda_start, lambda_end = self.lambda_schedule
                lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

                alignment_metrics.update({
                    "reward_mode": self.reward_mode,
                    "lambda_t": lambda_t,
                    "global_term_mag": abs(float(diagnostics["global_obj"])),
                    "local_term_mag": float(diagnostics["local_reward"]),
                    "global_contrib_est": lambda_t * abs(float(diagnostics["global_obj"])),
                    "local_contrib_est": (1.0 - lambda_t) * float(diagnostics["local_reward"]),
                })

                avg_reward = torch.mean(rewards)
                self.tracker.log_episode(
                    episode + 1,
                    reward_metrics,
                    new_reward_metrics,
                    alignment_metrics,
                )
                self.tracker.maybe_save_synthetic(
                    episode_num=episode + 1,
                    x_syn=x_phi_t,
                    y_syn=y_phi_t,
                    avg_reward=float(avg_reward),
                    obj1=float(diagnostics["global_obj"]),
                    obj2_mean=float(diagnostics["local_reward"]),
                    global_f1=float(diagnostics["f1_macro_beta"]),
                    feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                    beta_model=self.beta_model,
                )

            # ---------------- Final test (UPDATED to use self.benchmarks_config) ----------------
            b = self.benchmarks_config

            self.tracker.log_final_test(
                alpha_model=self.alpha_model,
                x_test=x_theta_test,
                y_test=y_theta_test,
                f1_thresh=0.5,

                prefer_best_beta=True,
                beta_model=self.beta_model,

                x_train=x_theta_train,
                y_train=y_theta_train,

                # existing jitter baseline params
                jitter_n=None,
                jitter_scale=0.20,

                # alpha toggles/params 
                run_alpha_raw_original=False,
                run_alpha_plus_real=False,
                alpha_plus_real_n=2000,

                # CTGAN baseline toggles/params 
                run_alpha_plus_ctgan=bool(b.get("run_ctgan", False)),
                alpha_plus_ctgan_n=int(b.get("alpha_plus_ctgan_n", self.traj_length)),
                ctgan_epochs=int(b.get("ctgan_epochs", 300)),
                cap_ctgan_train=b.get("cap_ctgan_train", None),

                # CTABGAN baseline toggles/params 
                run_ctabgan=bool(b.get("run_ctabgan", False)),
                alpha_plus_ctabgan_n=int(b.get("alpha_plus_ctabgan_n", self.traj_length)),

                # CTABGAN subprocess wiring 
                ctab_python=b.get("ctab_python", None),
                ctab_repo=b.get("ctab_repo", None),
                ctab_runner=str(self.project_root / "benchmarks" / "ctabgan" / "run_ctabgan.py"),

                # Dataset rebuild params 
                data_path=None,
                bias_pct=self.bias_pct,
                val_frac=0.20,
                test_frac=0.20,
                train_size=self.real_data_size,

                # additional params 
                batch_size=64,
                pca_components=None,
                seed=self.seed,
            )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")
