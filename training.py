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
        reward_mode="fairness",
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
        
    # ---------------- Fairness (Equal Opportunity) helpers ----------------

    def _tpr_from_probs(self, a, y_true, p1, *, thresh=0.5, group_value=0):
        """
        True Positive Rate for a specific group:
        TPR(a=g) = P(ŷ=1 | y=1, A=g)
        Returns NaN if denominator is 0 (no positives for that group).
        """
        device = p1.device
        a = torch.as_tensor(a, device=device).long()
        y_true = y_true.to(device).long()
        y_pred = (p1 >= thresh).long()

        group_mask = (a == int(group_value))
        pos_mask   = (y_true == 1)
        denom_mask = group_mask & pos_mask

        denom = denom_mask.sum().float()
        if denom.item() == 0:
            return torch.tensor(float("nan"), device=device)

        tp = ((y_pred == 1) & denom_mask).sum().float()
        return tp / (denom + 1e-8)

    #Uses _tpr_from_probs
    def _eo_gap_from_probs(self, a, y_true, p1, *, thresh=0.5, group0=0, group1=1):
        """
        Equal Opportunity gap:
        ΔEO = |TPR(A=group0) - TPR(A=group1)|
        Returns NaN if one of the groups has no positives (can't compute TPR).
        """
        tpr0 = self._tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group0)
        tpr1 = self._tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group1)

        # if either is NaN, return NaN
        if (tpr0 != tpr0) or (tpr1 != tpr1):
            return torch.tensor(float("nan"), device=p1.device)

        return torch.abs(tpr0 - tpr1)

    #Uses _tpr_from_probs
    def _eo_signed_diff_from_probs(self, a, y_true, p1, *, thresh=0.5, group0=0, group1=1):
        """
        Signed difference (useful for debugging directionality):
        TPR(group0) - TPR(group1)
        """
        tpr0 = self._tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group0)
        tpr1 = self._tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group1)
        if (tpr0 != tpr0) or (tpr1 != tpr1):
            return torch.tensor(float("nan"), device=p1.device)
        return (tpr0 - tpr1)

    def init_disadvantaged_group(self, alpha_model, x_theta_val, y_theta_val, thresh=0.5):
        with torch.no_grad():
            p1 = self._p1_from_agent(alpha_model, x_theta_val)
            y = y_theta_val.long()
            a = self.dataset.a_val  # aligned with x_theta_val

            tpr_g0 = self._tpr_from_probs(a, y, p1, thresh=thresh, group_value=0)
            tpr_g1 = self._tpr_from_probs(a, y, p1, thresh=thresh, group_value=1)

        # pick disadvantaged = lower TPR (handle NaNs defensively)
        if (tpr_g0 != tpr_g0):  # NaN
            disadv = 1
        elif (tpr_g1 != tpr_g1):
            disadv = 0
        else:
            disadv = 0 if float(tpr_g0) < float(tpr_g1) else 1

        self.disadv_group_value = disadv
        self.adv_group_value = 1 - disadv
        self.disadv_tpr_alpha_g0 = float(tpr_g0) if (tpr_g0 == tpr_g0) else float("nan")
        self.disadv_tpr_alpha_g1 = float(tpr_g1) if (tpr_g1 == tpr_g1) else float("nan")

    def _nearest_anchor_dist(self, x: torch.Tensor, anchors: torch.Tensor, *, chunk=512) -> torch.Tensor:
        """
        Returns min Euclidean distance from each x[i] to any anchor.
        x:       [T, A]
        anchors: [N, A]
        output:  [T]
        VRAM-safe via chunking.
        """
        device = x.device
        anchors = anchors.to(device)

        T = x.shape[0]
        out = torch.empty((T,), device=device, dtype=x.dtype)

        # Precompute anchor norms for fast distance: ||x-a||^2 = ||x||^2 + ||a||^2 - 2 x·a
        a2 = (anchors * anchors).sum(dim=1)  # [N]

        for s in range(0, T, chunk):
            xb = x[s:s+chunk]                # [b, A]
            x2 = (xb * xb).sum(dim=1, keepdim=True)  # [b,1]
            # squared distances: [b,N]
            d2 = x2 + a2.unsqueeze(0) - 2.0 * (xb @ anchors.t())
            d2 = torch.clamp(d2, min=0.0)
            out[s:s+chunk] = torch.sqrt(d2.min(dim=1).values + 1e-12)

        return out


    def _diversity_penalty(self, x: torch.Tensor, *, max_pts=128, rho=0.5) -> torch.Tensor:
        """
        Scalar penalty high when samples are too similar.
        Uses a subsample + gaussian similarity exp(-||xi-xj||^2/(2*rho^2)).
        """
        T = x.shape[0]
        if T <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        if T > max_pts:
            idx = torch.linspace(0, T - 1, steps=max_pts, device=x.device).long()
            Xd = x[idx]
        else:
            Xd = x

        # Pairwise distances (M x M)
        D = torch.cdist(Xd, Xd)
        M = D.shape[0]
        D = D + torch.eye(M, device=D.device, dtype=D.dtype) * 1e6  # ignore diagonal

        rho_t = torch.tensor(float(rho), device=D.device, dtype=D.dtype)
        sim = torch.exp(-0.5 * (D / (rho_t + 1e-8)) ** 2)
        return sim.mean()

    # Beta model F1 minority class score + local term (no EMA)
    def compute_reward(
        self,
        alpha_model, beta_model,
        x_theta_val, y_theta_val,
        x_phi, y_phi,
        progress: float,
        f1_thresh: float = 0.5,
        # hinge-penalty for majority/weighted scores (used only in *_penalty mode)
        epsilon_majority: float = 0.01,
        epsilon_weighted: float = 0.005,
        c_majority: float = 0.30,
        c_weighted: float = 0.30,
        class_mode: str = "binary",
        eo_scale: float = 50.0,

        # --- local fairness objective weights / scales ---
        w_anchor: float = 0.60,         # encourage closeness to disadv-group positives
        w_hard: float = 0.30,           # encourage "hard positives" for alpha
        w_div: float = 0.05,            # discourage mode collapse (too-similar samples)
        sigma_anchor: float = 0.85,     # PCA-distance scale for anchor proximity
        rho_div: float = 0.60,          # PCA-distance scale for diversity penalty
        hard_margin: float = 0.65,      # p1 <= hard_margin is considered "hard" positive for alpha
    ):
        """
        Reward combines:
        - Global term
        - Local term
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

        # shared Gaussian width
        tau = 0.10

    # ---------------- Global term on θ_val ----------------
        with torch.no_grad():
            p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val  = self._p1_from_agent(beta_model,  x_theta_val)

            # Minority / majority / macro
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

        # -------- fairness global objective (EO) --------
        eo_alpha = float("nan")
        eo_beta  = float("nan")
        delta_eo = 0.0                 # eo_alpha - eo_beta
        tpr_alpha_g0 = float("nan")
        tpr_alpha_g1 = float("nan")
        tpr_beta_g0  = float("nan")
        tpr_beta_g1  = float("nan")
        eo_signed_beta = float("nan")

        if mode == "fairness":
            a_theta_val = self.dataset.a_val
            assert len(a_theta_val) == x_theta_val.shape[0], "a_val misaligned with x_theta_val"

            eo_alpha_t = self._eo_gap_from_probs(
                a_theta_val, y_val_bin, p1_alpha_val, thresh=f1_thresh, group0=0, group1=1
            )
            eo_beta_t  = self._eo_gap_from_probs(
                a_theta_val, y_val_bin, p1_beta_val,  thresh=f1_thresh, group0=0, group1=1
            )

            # per-group TPRs (diagnostics)
            tpr_a0 = self._tpr_from_probs(a_theta_val, y_val_bin, p1_alpha_val, thresh=f1_thresh, group_value=0)
            tpr_a1 = self._tpr_from_probs(a_theta_val, y_val_bin, p1_alpha_val, thresh=f1_thresh, group_value=1)
            tpr_b0 = self._tpr_from_probs(a_theta_val, y_val_bin, p1_beta_val,  thresh=f1_thresh, group_value=0)
            tpr_b1 = self._tpr_from_probs(a_theta_val, y_val_bin, p1_beta_val,  thresh=f1_thresh, group_value=1)
            eo_signed_b = self._eo_signed_diff_from_probs(a_theta_val, y_val_bin, p1_beta_val, thresh=f1_thresh, group0=0, group1=1)

            if (eo_alpha_t != eo_alpha_t) or (eo_beta_t != eo_beta_t):
                delta_eo = 0.0
            else:
                eo_alpha = float(eo_alpha_t)
                eo_beta  = float(eo_beta_t)
                delta_eo = eo_alpha - eo_beta

            if (tpr_a0 == tpr_a0): tpr_alpha_g0 = float(tpr_a0)
            if (tpr_a1 == tpr_a1): tpr_alpha_g1 = float(tpr_a1)
            if (tpr_b0 == tpr_b0): tpr_beta_g0  = float(tpr_b0)
            if (tpr_b1 == tpr_b1): tpr_beta_g1  = float(tpr_b1)
            if (eo_signed_b == eo_signed_b): eo_signed_beta = float(eo_signed_b)

        # ---------------- Local term on Φ ----------------
        with torch.no_grad():
            p = self._p1_from_agent(alpha_model, x_phi)

        # Default placeholders for diagnostics
        local_cap_frac = float("nan")
        anchor_reward_mean = float("nan")
        hard_reward_mean = float("nan")
        div_pen_mean = float("nan")
        min_anchor_dist_mean = float("nan")
        min_anchor_dist_p50 = float("nan")
        min_anchor_dist_p90 = float("nan")
        anchors_used = 0

        if mode in ("local_gauss", "local_gauss_penalty"):
            #symmetric Gaussian around 0.5
            m = torch.abs(p - 0.5)
            score_gauss = torch.exp(-0.5 * (m / tau) ** 2)
            score_local_raw = score_gauss.clone()

            # Cap local score
            LOCAL_CAP = 1.0
            cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
            over_mask = (score_local_raw > cap_t).float()
            local_cap_frac = float(over_mask.mean().item())
            score_local = torch.minimum(score_local_raw, cap_t)

            judge_conf_mean = float(score_gauss.mean().item())
            uncert_alpha_mean = float((1.0 - (2.0 * m).clamp(0, 1)).mean().item())

        else:
            # Fairness mode local objective (NO protected-attr forcing):
            # local = w_anchor * anchor_prox + w_hard * hard_pos - w_div * diversity_pen

            if not hasattr(self, "disadv_group_value"):
                self.init_disadvantaged_group(alpha_model, x_theta_val, y_val_bin, thresh=f1_thresh)

            anchors = getattr(self, "disadv_pos_anchors", None)

            # --- anchor proximity ---
            if anchors is None or anchors.numel() == 0:
                anchors_used = 0
                min_d = torch.full((x_phi.shape[0],), float("nan"), device=x_phi.device, dtype=x_phi.dtype)
                anchor_reward = torch.zeros((x_phi.shape[0],), device=x_phi.device, dtype=x_phi.dtype)
            else:
                anchors_used = int(anchors.shape[0])
                min_d = self._nearest_anchor_dist(x_phi, anchors, chunk=512)  # [T]
                sig = torch.tensor(float(sigma_anchor), device=min_d.device, dtype=min_d.dtype)
                anchor_reward = torch.exp(-0.5 * (min_d / (sig + 1e-8)) ** 2)

            # --- hard-positive reward (alpha finds these "hard") ---
            hm = float(hard_margin)
            hard_reward = ((hm - p) / max(hm, 1e-8)).clamp(0.0, 1.0)  # [T]

            # --- diversity penalty (mode collapse) ---
            div_pen = self._diversity_penalty(x_phi, max_pts=128, rho=rho_div)  # scalar

            # combine
            score_local = (
                w_anchor * anchor_reward +
                w_hard   * hard_reward -
                w_div    * div_pen
            ).clamp(0.0, 1.0)

            # diagnostics
            m = torch.abs(p - 0.5)
            judge_conf_mean = float(torch.exp(-0.5 * (m / tau) ** 2).mean().item())
            uncert_alpha_mean = float((1.0 - (2.0 * m).clamp(0, 1)).mean().item())
            local_cap_frac = 0.0

            anchor_reward_mean = float(anchor_reward.mean().item())
            hard_reward_mean = float(hard_reward.mean().item())
            div_pen_mean = float(div_pen.item()) if torch.is_tensor(div_pen) else float(div_pen)

            if torch.isfinite(min_d).any():
                md = min_d[torch.isfinite(min_d)]
                min_anchor_dist_mean = float(md.mean().item())
                try:
                    min_anchor_dist_p50 = float(torch.quantile(md, 0.50).item())
                    min_anchor_dist_p90 = float(torch.quantile(md, 0.90).item())
                except Exception:
                    pass


        # ---------------- Combine global + local ----------------
        if mode == "fairness":
            global_term = eo_scale * float(delta_eo)
        else:
            global_term = float(delta_f1_minority)

        # make local term per-step (score_local is [T])
        base_reward = lambda_t * global_term + (1.0 - lambda_t) * score_local


        # ---------------- Optional performance penalty ----------------
        maj_violation = 0.0
        wtd_violation = 0.0
        penalty = 0.0
        if mode == "local_gauss_penalty":
            maj_violation = max(0.0, -(delta_f1_majority + epsilon_majority))
            wtd_violation = max(0.0, -(delta_f1_weighted + epsilon_weighted))
            penalty = c_majority * maj_violation + c_weighted * wtd_violation

        reward = base_reward - penalty
        mean_local = float(score_local.mean().item())

        # ---------------- Diagnostics ----------------
        # alpha_wrong_rate is meaningful only if y_phi is aligned to the classifier target;
        # you force y_phi=1, so this is essentially FN rate on generated positives.
        alpha_wrong_rate = float(
            ((p >= 0.5).float() != (y_phi.to(p.device).float())).float().mean().item()
        )

        diag_mean_conf_all  = float("nan")
        diag_frac_mid_conf  = float("nan")
        diag_gen_radius_mean = float("nan")
        diag_grad_norm      = float("nan")

        try:
            x_phi_det = x_phi.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                p_diag = self._p1_from_agent(alpha_model, x_phi_det, no_grad=False)  # [T]

                # confidence over generator samples: max(p, 1-p)
                conf = torch.maximum(p_diag, 1.0 - p_diag)
                diag_mean_conf_all = float(conf.mean().item())

                # fraction near decision boundary (0.4–0.6)
                mid_band = ((p_diag >= 0.4) & (p_diag <= 0.6)).float()
                diag_frac_mid_conf = float(mid_band.mean().item())

                # radial distance in PCA space for generator samples
                radius = torch.linalg.norm(x_phi_det, dim=1)  # [T]
                diag_gen_radius_mean = float(radius.mean().item())

                # gradient norm of local reward w.r.t x (use the mean local score)
                if mode in ("local_gauss", "local_gauss_penalty"):
                    m_diag = torch.abs(p_diag - 0.5)
                    score_gauss_diag = torch.exp(-0.5 * (m_diag / tau) ** 2)
                    local_mean_diag = score_gauss_diag.mean()
                else:
                    # fairness-local: rebuild differentiable local mean (approx)
                    # NOTE: anchor distances via cdist are differentiable, but we computed them in no_grad above.
                    # For diagnostics, use a simple differentiable proxy: hardness only.
                    hm = float(hard_margin)
                    hard_reward_diag = ((hm - p_diag) / max(hm, 1e-8)).clamp(0.0, 1.0)
                    local_mean_diag = hard_reward_diag.mean()

                grad_x, = torch.autograd.grad(
                    local_mean_diag,
                    x_phi_det,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                diag_grad_norm = float(torch.linalg.norm(grad_x, dim=1).mean().item())
        except Exception:
            p_diag = p  # keep something defined
            pass

        diagnostics = {
            "global_obj": float(global_term),
            "f1_minority_beta": float(f1_minority_beta),
            "f1_minority_alpha": float(f1_minority_alpha),
            "local_reward": float(mean_local),
            "f1_macro_beta": float(f1_macro_beta),

            # legacy/local diagnostics
            "judge_conf_mean": float(judge_conf_mean),
            "uncert_alpha_mean": float(uncert_alpha_mean),
            "alpha_wrong_rate": float(alpha_wrong_rate),

            "delta_f1_val": float(delta_f1_minority),
            "delta_f1_majority": float(delta_f1_majority),
            "delta_f1_weighted": float(delta_f1_weighted),
            "f1_minority_beta_stale": 0.0,
            "local_cap_frac": float(local_cap_frac) if local_cap_frac == local_cap_frac else float("nan"),

            # fairness diagnostics
            "eo_alpha": float(eo_alpha),
            "eo_beta": float(eo_beta),
            "delta_eo": float(delta_eo),
            "eo_scale": float(eo_scale) if mode == "fairness" else None,
            "tpr_alpha_g0": tpr_alpha_g0,
            "tpr_alpha_g1": tpr_alpha_g1,
            "tpr_beta_g0":  tpr_beta_g0,
            "tpr_beta_g1":  tpr_beta_g1,
            "eo_signed_beta": eo_signed_beta,

            # fairness-local diagnostics
            "anchors_used": int(anchors_used),
            "anchor_reward_mean": float(anchor_reward_mean),
            "hard_reward_mean": float(hard_reward_mean),
            "div_pen_mean": float(div_pen_mean),
            "min_anchor_dist_mean": float(min_anchor_dist_mean),
            "min_anchor_dist_p50": float(min_anchor_dist_p50),
            "min_anchor_dist_p90": float(min_anchor_dist_p90),

            # penalty diagnostics
            "epsilon_majority": epsilon_majority if mode == "local_gauss_penalty" else None,
            "epsilon_weighted": epsilon_weighted if mode == "local_gauss_penalty" else None,
            "penalty": float(penalty) if mode == "local_gauss_penalty" else 0.0,
            "maj_violation": float(maj_violation) if mode == "local_gauss_penalty" else 0.0,
            "wtd_violation": float(wtd_violation) if mode == "local_gauss_penalty" else 0.0,
            "corr_factor_mean": None,

            # extra diagnostic metrics
            "diag_mean_conf_all": float(diag_mean_conf_all),
            "diag_frac_mid_conf": float(diag_frac_mid_conf),
            "diag_gen_radius_mean": float(diag_gen_radius_mean),
            "diag_grad_norm": float(diag_grad_norm),
            "diag_mean_abs_margin": float(torch.abs(p_diag - 0.5).mean().item()) if torch.is_tensor(p_diag) else float("nan"),
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
            # lock disadvantaged group based on alpha on validation set
            self.init_disadvantaged_group(self.alpha_model, x_theta_val, y_theta_val, thresh=0.5)

            # Build anchor set: real TRAIN points that are (y=1) AND (a=disadvantaged group)
            a_train = self.dataset.a_train  # torch tensor aligned with x_theta_train
            disadv = int(self.disadv_group_value)

            anchor_mask = (y_theta_train == 1) & (a_train == disadv)

            anchors = x_theta_train[anchor_mask]

            # Optional: cap anchors to keep distance calcs fast
            MAX_ANCHORS = 2000
            if anchors.shape[0] > MAX_ANCHORS:
                g = torch.Generator(device="cpu").manual_seed(self.seed)
                idx = torch.randperm(anchors.shape[0], generator=g)[:MAX_ANCHORS]
                anchors = anchors[idx.to(anchors.device)]

            self.disadv_pos_anchors = anchors.detach()  # [N, A]
            print(f"Anchors: {self.disadv_pos_anchors.shape[0]} (disadv={disadv})")

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
                    "avg_reward": float(torch.mean(rewards).item()),
                    "obj1_global": float(diagnostics["global_obj"]),
                    "f1_minority_beta": float(diagnostics["f1_minority_beta"]),
                    "obj2_local_useful_mean": float(diagnostics["local_reward"]),
                    "macro_f1_beta": float(diagnostics["f1_macro_beta"]),

                    # NEW: always present for header stability
                    "eo_alpha": float(diagnostics.get("eo_alpha", float("nan"))),
                    "eo_beta": float(diagnostics.get("eo_beta", float("nan"))),
                    "delta_eo": float(diagnostics.get("delta_eo", float("nan"))),
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
                    "eo_alpha": float(diagnostics.get("eo_alpha", float("nan"))),
                    "eo_beta": float(diagnostics.get("eo_beta", float("nan"))),
                    "delta_eo": float(diagnostics.get("delta_eo", 0.0)),
                }

                local_mean = float(diagnostics["local_reward"])
                if self.reward_mode == "fairness":
                    delta_val = float(diagnostics.get("delta_eo", 0.0))  # not delta_dp
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
