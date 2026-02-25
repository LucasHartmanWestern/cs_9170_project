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
import reward_helpers as rh

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
        bias_pct=None,

        #PCA / trajectory
        pca_components=2,
        traj_length=500,
        real_data_size=1500,
        total_episodes=1,

        #reward
        reward_mode="fairness",
        lambda_schedule=(0.2, 0.8),
        global_reward_mode="delta",   # "exp_neg" | "neg" | "delta"
        terminal_global=True,         # assign global only at terminal step
        local_squash_k=4.0,          # sigmoid sharpness; 0 = clamp (old behavior)
        local_squash_center=0.5,     # centering point for sigmoid squashing
        hard_from_beta=False,         # use beta for hardness; False = alpha (stable)

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

        #feature space
        use_pca=True,
        bias_val=True,

        #two-phase generation
        gen_both_classes=False,

        #local reward weights
        w_anchor: float = 0.60,
        w_hard: float = 0.30,
        w_div: float = 0.05,
        sigma_anchor: float = 0.85,
        rho_div: float = 0.60,
        hard_margin: float = 0.65,

        #misc
        seed=42,
        device='cpu',
    ):
        self.exp_group = exp_group
        self.spec_name = spec_name
        self.seed = seed
        self.device = torch.device(device)

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision("highest")

        self.gen_both_classes = gen_both_classes
        self.bias_pct = bias_pct
        self.pca_components = pca_components
        self.reward_mode = reward_mode
        self.lambda_schedule = lambda_schedule
        self.global_reward_mode = global_reward_mode
        self.terminal_global = terminal_global
        self.local_squash_k = local_squash_k
        self.local_squash_center = local_squash_center
        self.hard_from_beta = hard_from_beta
        self.w_anchor = w_anchor
        self.w_hard = w_hard
        self.w_div = w_div
        self.sigma_anchor = sigma_anchor
        self.rho_div = rho_div
        self.hard_margin = hard_margin
        self.curriculum_learning = curriculum_learning
        self.multiclass = multiclass
        self.dataset_name = dataset_name

        self.use_pca = use_pca
        self.bias_val = bias_val

        self.minority_id = minority_id
        self.majority_id = majority_id
        self.third_id = third_id

        self.traj_length = traj_length
        self.real_data_size = real_data_size
        self.episodes = total_episodes

        # state_dim and agent configs are finalized in __call__ after data loading
        # (feature_dim may differ from pca_components when use_pca=False)
        if self.curriculum_learning:
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
            use_pca=self.use_pca,
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
        self.reinforce_config = reinforce_config

        self.dl_generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # agents (will be rebuilt in __call__ after feature_dim is known)
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



    # Beta model F1 minority class score + local term (no EMA)
    def compute_reward(
        self,
        alpha_model, beta_model,
        x_theta_val, y_theta_val,
        x_phi, y_phi,
        progress: float,
        f1_thresh: float = 0.5,
        class_mode: str = "binary",
        dro_scale: float = 1.0,

        # --- local fairness objective weights / scales ---
        w_anchor: float = 0.60,
        w_hard: float = 0.30,
        w_div: float = 0.05,
        sigma_anchor: float = 0.85,
        rho_div: float = 0.60,
        hard_margin: float = 0.65,

        # --- reward shaping ---
        global_reward_mode: str = "delta",
        terminal_global: bool = True,
        local_squash_k: float = 4.0,
        local_squash_center: float = 0.5,
        hard_from_beta: bool = False,
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

        # y in {0,1}
        if class_mode == "binary":
            y_val_bin = y_theta_val.long()
        else:
            y_val_bin = (y_theta_val == 1).long()

        # shared Gaussian width (for local_gauss variants + some diagnostics)
        tau = 0.10

        # ---------------- Global metrics on θ_val ----------------
        with torch.no_grad():
            p1_alpha_val = rh.p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val  = rh.p1_from_agent(beta_model,  x_theta_val)

            # Utility sanity checks (keep regardless of mode)
            f1_minority_beta  = rh.f1_from_probs(y_val_bin, p1_beta_val,  f1_thresh)
            f1_majority_beta  = rh.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val, 1 - f1_thresh)
            f1_macro_beta     = 0.5 * (f1_minority_beta + f1_majority_beta)

        # ---------------- Fairness (Group DRO) globals ----------------
        worst_loss_beta = float("nan")
        group_loss_beta_g0 = float("nan")
        group_loss_beta_g1 = float("nan")
        worst_group_beta = None
        group_loss_gap_beta = float("nan")
        bce_mean_beta = float("nan")
        n_g0 = 0
        n_g1 = 0

        if mode == "fairness":
            a_theta_val = self.dataset.a_val
            assert len(a_theta_val) == x_theta_val.shape[0], "a_val misaligned with x_theta_val"

            with torch.no_grad():
                loss_beta_vec = rh.bce_per_sample_from_probs(y_val_bin, p1_beta_val)

                worst_b_t, per_b = rh.worst_group_loss(loss_beta_vec, a_theta_val, group_values=(0, 1))

                # per-group mean losses (floats/nan)
                group_loss_beta_g0 = per_b.get(0, float("nan"))
                group_loss_beta_g1 = per_b.get(1, float("nan"))

                # overall mean BCE
                bce_mean_beta = float(loss_beta_vec.mean().item())

                # group counts
                a_t = torch.as_tensor(a_theta_val, device=loss_beta_vec.device).long()
                n_g0 = int((a_t == 0).sum().item())
                n_g1 = int((a_t == 1).sum().item())

                # worst-group + gap (only if both groups present)
                if (group_loss_beta_g0 == group_loss_beta_g0) and (group_loss_beta_g1 == group_loss_beta_g1):
                    worst_group_beta = 1 if group_loss_beta_g1 >= group_loss_beta_g0 else 0
                    group_loss_gap_beta = float(abs(group_loss_beta_g1 - group_loss_beta_g0))
                else:
                    worst_group_beta = None
                    group_loss_gap_beta = float("nan")

                if worst_b_t == worst_b_t:
                    worst_loss_beta = float(worst_b_t.item())
                else:
                    worst_loss_beta = float("nan")

        # ---------------- Local term on Φ ----------------
        with torch.no_grad():
            if hard_from_beta:
                p = rh.p1_from_agent(beta_model, x_phi)
            else:
                p = rh.p1_from_agent(alpha_model, x_phi)

        # defaults for diagnostics
        anchors_used = 0
        anchor_reward_mean = float("nan")
        hard_reward_mean = float("nan")
        div_pen_mean = float("nan")
        min_anchor_dist_mean = float("nan")

        if mode in ("local_gauss", "local_gauss_penalty"):
            m = torch.abs(p - 0.5)
            raw_local_gauss = torch.exp(-0.5 * (m / tau) ** 2)
            if local_squash_k > 0:
                score_local = torch.sigmoid(local_squash_k * (raw_local_gauss - local_squash_center))
            else:
                score_local = raw_local_gauss.clamp(0.0, 1.0)

            # judge/uncertainty (kept only if you still want them; not logged in lean set)
            # judge_conf_mean = float(score_local.mean().item())
        else:
            # fairness-local: anchor + hard - diversity
            if not hasattr(self, "disadv_group_value"):
                disadv, adv, per_g, worst = rh.disadvantaged_group_from_alpha(
                    alpha_model, x_theta_val, y_val_bin, self.dataset.a_val, group_values=(0, 1)
                )
                self.disadv_group_value = disadv
                self.adv_group_value = adv
                self.disadv_loss_alpha_g0 = per_g.get(0, float("nan"))
                self.disadv_loss_alpha_g1 = per_g.get(1, float("nan"))
                self.disadv_worst_loss_alpha = worst

            anchors = getattr(self, "disadv_pos_anchors", None)

            # anchor proximity
            if anchors is None or anchors.numel() == 0:
                anchors_used = 0
                min_d = torch.full((x_phi.shape[0],), float("nan"), device=x_phi.device, dtype=x_phi.dtype)
                anchor_reward = torch.zeros((x_phi.shape[0],), device=x_phi.device, dtype=x_phi.dtype)
            else:
                anchors_used = int(anchors.shape[0])
                min_d = rh.nearest_anchor_dist(x_phi, anchors, chunk=512)  # [T]
                sig = torch.tensor(float(sigma_anchor), device=min_d.device, dtype=min_d.dtype)
                anchor_reward = torch.exp(-0.5 * (min_d / (sig + 1e-8)) ** 2)

            # hard-positive reward
            hm = float(hard_margin)
            hard_reward = ((hm - p) / max(hm, 1e-8)).clamp(0.0, 1.0)  # [T]

            # diversity penalty (scalar)
            div_pen = rh.diversity_penalty(x_phi, max_pts=128, rho=rho_div)  # scalar

            # diversity-only isolation: invert penalty to positive reward
            if self.w_anchor == 0.0 and self.w_hard == 0.0 and self.w_div > 0.0:
                raw_local = (1.0 - w_div * div_pen).expand_as(hard_reward)
            else:
                raw_local = (
                    w_anchor * anchor_reward +
                    w_hard   * hard_reward -
                    w_div    * div_pen
                )

            if local_squash_k > 0:
                score_local = torch.sigmoid(local_squash_k * (raw_local - local_squash_center))
            else:
                score_local = raw_local.clamp(0.0, 1.0)

            # local diagnostics
            anchor_reward_mean = float(anchor_reward.mean().item())
            hard_reward_mean = float(hard_reward.mean().item())
            div_pen_mean = float(div_pen.item()) if torch.is_tensor(div_pen) else float(div_pen)

            if torch.isfinite(min_d).any():
                md = min_d[torch.isfinite(min_d)]
                min_anchor_dist_mean = float(md.mean().item())

        mean_local = float(score_local.mean().item())

        # local clipping rates (helpful to see if local is saturating)
        local_clip_frac_0 = float((score_local <= 0.0 + 1e-12).float().mean().item())
        local_clip_frac_1 = float((score_local >= 1.0 - 1e-12).float().mean().item())

        # ---------------- Combine global + local ----------------
        if mode == "fairness":
            if global_reward_mode == "exp_neg":
                global_term = float(torch.exp(-torch.tensor(worst_loss_beta)).item())
            elif global_reward_mode == "neg":
                global_term = -worst_loss_beta
            else:  # "delta" (default)
                alpha_baseline = getattr(self, "disadv_worst_loss_alpha", float("nan"))
                global_term = alpha_baseline - worst_loss_beta
        else:
            with torch.no_grad():
                f1_minority_alpha = rh.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
            global_term = float(f1_minority_beta - f1_minority_alpha)

        if terminal_global:
            reward = (1.0 - lambda_t) * score_local
            reward = reward.clone()
            reward[-1] = reward[-1] + lambda_t * float(global_term)
        else:
            reward = lambda_t * float(global_term) + (1.0 - lambda_t) * score_local

        # ---------------- Extra diagnostics (lean set) ----------------
        diag_frac_mid_conf = float("nan")
        diag_mean_abs_margin = float("nan")
        diag_gen_radius_mean = float("nan")

        try:
            x_phi_det = x_phi.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                p_diag = rh.p1_from_agent(alpha_model, x_phi_det, no_grad=False)  # [T]

                mid_band = ((p_diag >= 0.4) & (p_diag <= 0.6)).float()
                diag_frac_mid_conf = float(mid_band.mean().item())

                diag_mean_abs_margin = float(torch.abs(p_diag - 0.5).mean().item())

                radius = torch.linalg.norm(x_phi_det, dim=1)
                diag_gen_radius_mean = float(radius.mean().item())
        except Exception:
            pass

        # ---------------- Lean diagnostics dict ----------------
        diagnostics = {
            "global": {
                "global_obj": float(global_term),          # global objective (delta, -loss, or exp(-loss))
                "local_reward": float(mean_local),         # mean local reward per step
                "global_reward_mode": global_reward_mode,  # which formulation was used
                "terminal_global": terminal_global,        # whether global was terminal-only
            },
            "utility": {
                "f1_macro_beta": float(f1_macro_beta),         # macro F1 for beta model
                "f1_minority_beta": float(f1_minority_beta),   # F1 score on minority class for beta model
            },
            "fairness": {
                "worst_loss_beta": float(worst_loss_beta),     # highest groupwise loss for beta model (DRO objective)
                "group_loss_beta_g0": float(group_loss_beta_g0),   # group 0 loss for beta model
                "group_loss_beta_g1": float(group_loss_beta_g1),   # group 1 loss for beta model
                "worst_group_beta": worst_group_beta,           # group id with worst loss for beta
                "group_loss_gap_beta": float(group_loss_gap_beta),   # difference between group loss (gap)
                "bce_mean_beta": float(bce_mean_beta),          # average binary cross-entropy loss beta
                "n_g0": int(n_g0),                              # count of samples from group 0
                "n_g1": int(n_g1),                              # count of samples from group 1
                "dro_scale": float(dro_scale) if mode == "fairness" else None,   # DRO scaling factor
                "worst_loss_alpha_baseline": float(getattr(self, "disadv_worst_loss_alpha", float("nan"))),
            },
            "local": {
                "anchors_used": int(anchors_used),                  # number of anchor points used
                "anchor_reward_mean": float(anchor_reward_mean),     # mean anchor reward
                "hard_reward_mean": float(hard_reward_mean),         # mean hard reward (challenging samples)
                "div_pen_mean": float(div_pen_mean),                 # mean diversity penalty
                "min_anchor_dist_mean": float(min_anchor_dist_mean), # mean min distance to an anchor
                "local_clip_frac_0": float(local_clip_frac_0),       # fraction of local rewards clipped at zero
                "local_clip_frac_1": float(local_clip_frac_1),       # fraction of local rewards clipped at one
            },
            "extra": {
                "diag_frac_mid_conf": float(diag_frac_mid_conf),         # fraction of samples with confidence in mid (0.4, 0.6)
                "diag_mean_abs_margin": float(diag_mean_abs_margin),     # mean abs(p-0.5) margin
                "diag_gen_radius_mean": float(diag_gen_radius_mean),     # mean l2 norm of generated samples
            },
        }

        return reward, diagnostics


    # ---------------- Single-phase episode loop ----------------
    def _run_phase(
        self,
        *,
        target_class: int,
        env,
        agent,
        x_theta_train,
        y_theta_train,
        x_theta_val,
        y_theta_val,
        prior_synthetic: tuple | None,   # (x_syn, y_syn) from earlier phase
        phase_label: str,                 # "phase1_class1" or "phase2_class0"
    ) -> tuple:
        """Run a full episode loop for one phase. Returns (best_x_syn, best_y_syn) tensors."""
        # Clear correlation buffers at start of each phase
        self._local_buf.clear()
        self._delta_buf.clear()

        best_phase_reward = -float("inf")
        best_x_syn = None
        best_y_syn = None

        print(f"\n{'='*60}")
        print(f"[Phase] Starting {phase_label} | target_class={target_class} | episodes={self.episodes}")
        print(f"{'='*60}")

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
                action = agent.predict(state)
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

            # Train beta on hybrid (real + prior_synthetic + current_synthetic)
            parts_x = [x_theta_train]
            parts_y = [y_theta_train]
            if prior_synthetic is not None:
                parts_x.append(prior_synthetic[0])
                parts_y.append(prior_synthetic[1])
            parts_x.append(x_phi_t)
            parts_y.append(y_phi_t)
            x_hybrid = torch.cat(parts_x)
            y_hybrid = torch.cat(parts_y)
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
                w_anchor=self.w_anchor,
                w_hard=self.w_hard,
                w_div=self.w_div,
                sigma_anchor=self.sigma_anchor,
                rho_div=self.rho_div,
                hard_margin=self.hard_margin,
                global_reward_mode=self.global_reward_mode,
                terminal_global=self.terminal_global,
                local_squash_k=self.local_squash_k,
                local_squash_center=self.local_squash_center,
                hard_from_beta=self.hard_from_beta,
            )

            # Truncate episode tensors and learn
            states = states[:T]
            actions = actions[:T]
            next_states = next_states[:T]
            dones = dones[:T]
            rewards = rewards[:T]

            agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

            # alignment metrics
            g = diagnostics.get("global", {})
            f = diagnostics.get("fairness", {})

            local_mean = float(g.get("local_reward", float("nan")))
            if self.reward_mode == "fairness":
                delta_val = float(-f.get("worst_loss_beta", float("nan")))
            else:
                delta_val = float("nan")

            self._local_buf.append(local_mean)
            self._delta_buf.append(delta_val)

            corr_local_delta = self._corr_local_delta()

            lambda_start, lambda_end = self.lambda_schedule
            lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

            alignment_metrics = {
                "delta_global": float(delta_val),
                "corr_local_delta": float(corr_local_delta),
                "curriculum_stage": int(env.current_stage),
                "reward_mode": self.reward_mode,
                "lambda_t": float(lambda_t),
                "global_term_mag": abs(float(g.get("global_obj", 0.0))),
                "local_term_mag": float(g.get("local_reward", 0.0)),
                "global_contrib_est": float(lambda_t) * abs(float(g.get("global_obj", 0.0))),
                "local_contrib_est": (1.0 - float(lambda_t)) * float(g.get("local_reward", 0.0)),
            }
            avg_reward = float(torch.mean(rewards).item())
            meta_metrics = {"avg_reward": avg_reward, "phase": phase_label}

            self.tracker.log_episode(
                episode + 1,
                diagnostics=diagnostics,
                alignment_metrics=alignment_metrics,
                extra_metrics=meta_metrics,
            )

            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_phi_t,
                y_syn=y_phi_t,
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                beta_model=self.beta_model,
                phase_label=phase_label,
            )

            # Track best in-memory for return
            if avg_reward > best_phase_reward:
                best_phase_reward = avg_reward
                best_x_syn = x_phi_t.detach().clone()
                best_y_syn = y_phi_t.detach().clone()

        print(f"[Phase] Finished {phase_label} | best avg_reward={best_phase_reward:.4f}")
        return (best_x_syn, best_y_syn)

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
            "gen_both_classes": self.gen_both_classes,
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
                    protected_cols=self.dataset.protected_attributes,
                    bias_val=self.bias_val,
                )
            )

            # Resolve actual feature dimension (may differ from pca_components when use_pca=False)
            feature_dim = x_theta_train.shape[1]
            self.pca_components = feature_dim

            if self.curriculum_learning:
                self.state_dim = 1 + 2 * feature_dim
            else:
                self.state_dim = 2

            # Update configs and rebuild agents with correct dimensions
            self.ffnn_config["input_size"] = feature_dim
            self.reinforce_config["state_size"] = self.state_dim
            self.reinforce_config["action_size"] = feature_dim

            self.agent = ReinforceAgent(**self.reinforce_config)
            self.alpha_model = FFNNAgent(**self.ffnn_config)
            self.beta_model = FFNNAgent(**self.ffnn_config)

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
            disadv, adv, per_g, worst = rh.disadvantaged_group_from_alpha(
                self.alpha_model,
                x_theta_val,
                y_theta_val,
                self.dataset.a_val,     
                group_values=(0, 1),
            )

            self.disadv_group_value = disadv
            self.adv_group_value = adv
            self.disadv_loss_alpha_g0 = per_g.get(0, float("nan"))
            self.disadv_loss_alpha_g1 = per_g.get(1, float("nan"))
            self.disadv_worst_loss_alpha = worst

            print(f"[disadv] group={disadv} per_g={per_g} worst={worst:.4f}")


            # Build anchor set: real TRAIN points that are (y=1) AND (a=disadvantaged group)
            a_train = self.dataset.a_train  
            disadv = int(self.disadv_group_value)

            anchor_mask = (y_theta_train == 1) & (a_train == disadv)

            anchors = x_theta_train[anchor_mask]

            #cap anchors to keep distance calcs fast
            MAX_ANCHORS = 2000
            if anchors.shape[0] > MAX_ANCHORS:
                g = torch.Generator(device="cpu").manual_seed(self.seed)
                idx = torch.randperm(anchors.shape[0], generator=g)[:MAX_ANCHORS]
                anchors = anchors[idx.to(anchors.device)]

            self.disadv_pos_anchors = anchors.detach()

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

            # Helper to build an Environment with given target class and seed samples
            def _make_env(target, seed_samples):
                return Environment(
                    curriculum=self.curriculum_learning,
                    target=target,
                    max_actions=self.traj_length,
                    total_episodes=self.episodes,
                    device=self.device,
                    seed=self.seed,
                    pca_components=self.pca_components,
                    pca_means=pca_means,
                    curriculum_stages=curriculum_stages,
                    real_minority_samples=seed_samples,
                    use_delta_actions=self.use_delta_actions,
                    delta_scale=self.delta_scale,
                    delta_clip=self.delta_clip,
                    pca_clip=self.pca_clip,
                    use_radius_clip=(self.radius_clip is not None),
                    radius_clip=self.radius_clip,
                )

            # === Phase 1: target=1 (minority) ===
            phase1_env = _make_env(target=1, seed_samples=real_minority_samples)

            best_syn_1 = self._run_phase(
                target_class=1,
                env=phase1_env,
                agent=self.agent,
                x_theta_train=x_theta_train,
                y_theta_train=y_theta_train,
                x_theta_val=x_theta_val,
                y_theta_val=y_theta_val,
                prior_synthetic=None,
                phase_label="phase1_class1",
            )

            best_syn_0 = None
            if self.gen_both_classes:
                # === Phase 2: target=0 (majority) ===
                # Build anchors: y=0 & a=disadvantaged
                anchor_mask_0 = (y_theta_train == 0) & (a_train == disadv)
                anchors_0 = x_theta_train[anchor_mask_0]
                MAX_ANCHORS = 2000
                if anchors_0.shape[0] > MAX_ANCHORS:
                    g = torch.Generator(device="cpu").manual_seed(self.seed)
                    idx = torch.randperm(anchors_0.shape[0], generator=g)[:MAX_ANCHORS]
                    anchors_0 = anchors_0[idx.to(anchors_0.device)]
                self.disadv_pos_anchors = anchors_0.detach()
                print(f"Phase 2 anchors: {self.disadv_pos_anchors.shape[0]} (y=0 & disadv={disadv})")

                # Seed samples for env.reset() — majority class
                real_majority_samples = x_theta_train[y_theta_train == 0]

                # Fresh RL agent for phase 2
                phase2_agent = ReinforceAgent(**self.reinforce_config)

                # Fresh beta model for phase 2
                self.beta_model = FFNNAgent(**self.ffnn_config)

                phase2_env = _make_env(target=0, seed_samples=real_majority_samples)

                best_syn_0 = self._run_phase(
                    target_class=0,
                    env=phase2_env,
                    agent=phase2_agent,
                    x_theta_train=x_theta_train,
                    y_theta_train=y_theta_train,
                    x_theta_val=x_theta_val,
                    y_theta_val=y_theta_val,
                    prior_synthetic=best_syn_1,
                    phase_label="phase2_class0",
                )

            # === Final test ===
            if self.gen_both_classes and best_syn_0 is not None and best_syn_1 is not None:
                # Train a fresh beta on real + best_syn_1 + best_syn_0
                combined_beta = FFNNAgent(**self.ffnn_config)
                parts_x = [x_theta_train, best_syn_1[0], best_syn_0[0]]
                parts_y = [y_theta_train, best_syn_1[1], best_syn_0[1]]
                x_combined = torch.cat(parts_x)
                y_combined = torch.cat(parts_y)
                combined_beta = self.train_predictor_model(combined_beta, x_combined, y_combined)

                # Overwrite best_beta_state_dict.pt with the combined-trained beta
                torch.save(
                    combined_beta.model.state_dict(),
                    self.tracker.best_beta_path,
                )
                print(f"[Combined] Saved combined beta -> {self.tracker.best_beta_path}")

                # Save combined synthetic
                combined_npz_path = self.tracker.seed_dir / "best_synthetic_combined.npz"
                x_all_syn = torch.cat([best_syn_1[0], best_syn_0[0]])
                y_all_syn = torch.cat([best_syn_1[1], best_syn_0[1]])
                np.savez_compressed(
                    combined_npz_path,
                    x=x_all_syn.detach().cpu().numpy(),
                    y=y_all_syn.detach().cpu().numpy(),
                )
                print(f"[Combined] Saved combined synthetic -> {combined_npz_path}")

                # Use combined beta for final test
                final_beta = combined_beta
            else:
                final_beta = self.beta_model

            # ---------------- Final test ----------------
            b = self.benchmarks_config

            self.tracker.log_final_test(
                alpha_model=self.alpha_model,
                x_test=x_theta_test,
                y_test=y_theta_test,
                f1_thresh=0.5,

                prefer_best_beta=True,
                beta_model=final_beta,

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
                a_test=getattr(self.dataset, "a_test", None)
            )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")
