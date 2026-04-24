# Training v2 enhancing code for main v2
import gc
import os
import sys
import time
import uuid
import subprocess
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
from agents.cmaes_agent import CMAESAgent
from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker
import reward_helpers as rh

class Training:
    def __init__(
        self,
        exp_group,
        spec_name,
        spec,
        output_dir,
        seed=42,    
        process_label="Training process -0",
        device='cpu'
        ):
        self.exp_group = exp_group
        self.spec_name = spec_name
        #misc
        self.seed = seed
        self.process_label = process_label
        self.device = torch.device(device)
        self.save_dir = output_dir

        self._get_specs(spec)

        #Seeding procces
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision("highest")
        print(f"[{self.process_label}] ---- running seed={self.seed} ----")
            

        # state_dim and agent configs are finalized in __call__ after data loading
        # (feature_dim may differ from pca_components when use_pca=False)
        self.state_dim = 1 + 2 * self.pca_components


        self.project_root = Path(__file__).resolve().parent

        self.ffnn_overrides = self.ffnn or {}
        self.reinforce_overrides = self.reinforce or {}
        self.curriculum_overrides = self.curriculum or {}
        self.benchmarks_overrides = self.benchmarks or {}

        # dataset
        self.dataset = Dataset(
            self.dataset_name,
            multiclass=self.multiclass,
            minority_id=self.minority_id,
            majority_id=self.majority_id,
            third_id=self.third_id,
            pca_components=self.pca_components,
            seed=self.seed,
            device=self.device,
            use_pca=self.use_pca,
            whiten_pca=self.whiten_pca,
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
            "optimizer": "adam",
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
            "optimizer": "adam",
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

        DEFAULT_CMAES = {"sigma0": 1.0, "popsize": None, "cmaes_opts": {}}
        self.cmaes_config = {**DEFAULT_CMAES, **(self.cmaes or {})}

        DEFAULT_PPO = {
            "state_size": self.state_dim,
            "action_size": self.pca_components,
            "hidden_size": 64,
            "lr": 3e-4,
            "gamma": 1.0,
            "clip_epsilon": 0.2,
            "update_epochs": 4,
            "batch_size": 64,
            "c1": 0.5,
            "c2": 0.01,
            "device": self.device,
            "seed": self.seed,
        }
        ppo_config = {**DEFAULT_PPO, **(self.ppo or {})}
        ppo_config["state_size"] = self.state_dim
        ppo_config["action_size"] = self.pca_components
        ppo_config["device"] = self.device
        ppo_config["seed"] = self.seed
        self.ppo_config = ppo_config

        # agents (will be rebuilt in __call__ after feature_dim is known)
        if self.use_ppo:
            self.agent = PPOAgent(**self.ppo_config)
        else:
            self.agent = ReinforceAgent(**reinforce_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)

        # buffers
        self._corr_window = 40
        self._local_buf = deque(maxlen=self._corr_window)
        self._delta_buf = deque(maxlen=self._corr_window)

    def _get_specs(self, spec):
        """
        Get the specs from the spec dictionary
        """
        lw = spec.get("local_weights", {})
        rs = spec.get("reward_shaping", {})
            
        #flags
        self.curriculum_learning=spec.get("curriculum_learning", True)
        self.multiclass=spec.get("multiclass", False)

        #dataset
        self.dataset_name=spec["dataset_name"]
            
        #class / bias
        self.minority_id=spec.get("minority_id")
        self.majority_id=spec.get("majority_id")
        self.third_id=spec.get("third_id")
        self.bias_pct=spec.get("bias_pct")
        self.da_pct=spec.get("da_pct")

        #PCA / trajectory
        self.pca_components=spec["pca_components"]
        self.traj_length=spec["traj_length"]
        self.real_data_size=spec["real_data_size"]
        self.episodes=spec["total_episodes"]

        #reward
        self.reward_mode=spec["reward_mode"]
        self.lambda_schedule=tuple(spec["lambda_schedule"])
        self.local_squash_k=float(rs.get("local_squash_k", 4.0))
        self.local_squash_center=float(rs.get("local_squash_center", 0.5))
        self.hard_from_beta=bool(rs.get("hard_from_beta", False))
        
        #ENV hyperparams
        self.use_delta_actions=spec.get("use_delta_actions", True)
        self.delta_scale=spec.get("delta_scale", 0.10)
        self.delta_clip=spec.get("delta_clip", 0.20)
        self.pca_clip=spec.get("pca_clip", None)
        self.radius_clip=spec.get("radius_clip", None)

        #feature space
        self.use_pca=spec.get("use_pca", True)
        self.whiten_pca=spec.get("whiten_pca", False)
        self.bias_val=spec.get("bias_val", True)

        #PAMAP2 windowing
        self.win_seconds=spec.get("win_seconds", 5.0)
        self.step_seconds=spec.get("step_seconds", 2.5)
        
        #Config dictionaries
        self.ffnn=spec["ffnn"]         
        self.reinforce=spec["reinforce"]    
        self.curriculum=spec["curriculum"]   
        self.benchmarks=spec["benchmarks"]   

        #dataset protected attribute column (passed to get_data_splits)
        self.dp_protected_col=spec.get("dp_protected_col", None)

        #FairJob: target positive fraction for neg-undersampling of train+val
        self.pool_pos_fraction=spec.get("pool_pos_fraction", None)

        #two-phase generation
        self.gen_both_classes=spec.get("gen_both_classes", False)

        #local reward weights
        self.w_anchor=float(lw.get("w_anchor", 0.60))
        self.w_hard=float(lw.get("w_hard", 0.30))
        self.w_div=float(lw.get("w_div", 0.05))
        self.sigma_anchor=float(lw.get("sigma_anchor", 0.85))
        self.rho_div=float(lw.get("rho_div", 0.60))
        self.hard_margin=float(lw.get("hard_margin", 0.65))
        self.use_uncertainty_anchors=bool(lw.get("use_uncertainty_anchors", False))
        self.uncertainty_warmup_episodes=int(lw.get("uncertainty_warmup_episodes", 0))
        self.sigma_calibration_factor=float(lw["sigma_calibration_factor"]) if lw.get("sigma_calibration_factor") is not None else None
        self.anchor_refresh_interval=int(lw.get("anchor_refresh_interval", 0))
        self.anchor_refresh_top_k=int(lw.get("anchor_refresh_top_k", 500))
        self.anchor_selection_mode=str(lw.get("anchor_selection_mode", "all"))
        self.anchor_selection_top_k=int(lw.get("anchor_selection_top_k", 200))
        
        #DVRL-inspired local reward (v10)
        self.use_dvrl_local=bool(lw.get("use_dvrl_local", False))
        self.dvrl_max_bce=float(lw.get("dvrl_max_bce", 0.693))
        self.dvrl_scale=float(lw.get("dvrl_scale", 1.0))

        #asymmetric phase episodes (gen_both_classes only)
        self.phase2_episodes=spec.get("phase2_episodes", None)  # None = use total_episodes for both phases

        #reward shaping
        self.global_sigmoid_k=float(lw.get("global_sigmoid_k", 10.0))
        self.utility_guard_min_factor=float(lw.get("utility_guard_min_factor", 1.0))
        self.roc_eo_lambda=float(lw.get("roc_eo_lambda", 0.5))

        #CMA-ES optimizer (replaces REINFORCE when use_cmaes=True)
        self.use_cmaes=bool(spec.get("use_cmaes", False))
        self.cmaes=spec.get("cmaes", None)

        #PPO optimizer (replaces REINFORCE when use_ppo=True)
        self.use_ppo=bool(spec.get("use_ppo", False))
        self.ppo=spec.get("ppo", None)

        #beta warm-start   
        self.beta_reset_interval=int(spec.get("beta_reset_interval", 1))
        self.beta_warmstart_from_alpha=bool(spec.get("beta_warmstart_from_alpha", False))

        #OT-inspired local reward
        self.w_ot=float(lw.get("w_ot", 0.0))

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
        local_squash_k: float = 4.0,
        local_squash_center: float = 0.5,
        hard_from_beta: bool = False,
        use_uncertainty_anchors: bool = False,
        global_sigmoid_k: float = 10.0,
        use_dvrl_local: bool = False,
        dvrl_max_bce: float = 0.693,
        dvrl_scale: float = 1.0,
        utility_guard_min_factor: float = 1.0,
        roc_eo_lambda: float = 0.5,
    ):
        """
        Reward combines:
        - Global term
        - Local term
        """

        mode = self.reward_mode
        valid = {"local_gauss", "local_gauss_penalty", "wgl", "fairness", "roc_eo"}
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
            acc_beta          = rh.acc_from_probs(y_val_bin, p1_beta_val, f1_thresh)
            auc_beta          = rh.roc_auc_from_probs(y_val_bin, p1_beta_val)

        # ---------------- Fairness (Group DRO) globals ----------------
        worst_loss_beta = float("nan")
        group_loss_beta_g0 = float("nan")
        group_loss_beta_g1 = float("nan")
        worst_group_beta = None
        group_loss_gap_beta = float("nan")
        bce_mean_beta = float("nan")
        n_g0 = 0
        n_g1 = 0
        ep_dp_diff = float("nan")
        ep_eo_tpr_diff = float("nan")
        ep_eod_max_diff = float("nan")
        ep_eod_avg_diff = float("nan")
        ep_soft_eo_beta = float("nan")

        if mode in ("wgl", "fairness", "roc_eo"):
            a_theta_val = self.dataset.a_val
            assert len(a_theta_val) == x_theta_val.shape[0], "a_val misaligned with x_theta_val"

            with torch.no_grad():
                loss_beta_vec = rh.bce_per_sample_from_probs(y_val_bin, p1_beta_val)

                g_ids_val = torch.as_tensor(a_theta_val, device=loss_beta_vec.device).long() * 2 + y_val_bin
                worst_b_t, per_b = rh.worst_group_loss(loss_beta_vec, g_ids_val, group_values=(0, 1, 2, 3))

                # per-group mean losses (floats/nan)
                group_loss_beta_g0 = per_b.get(0, float("nan"))
                group_loss_beta_g1 = per_b.get(1, float("nan"))

                # overall mean BCE
                bce_mean_beta = float(loss_beta_vec.mean().item())

                # group counts (static — val split never changes; cache after first compute)
                if not hasattr(self, "_cached_n_g0"):
                    a_t = torch.as_tensor(a_theta_val, device=loss_beta_vec.device).long()
                    self._cached_n_g0 = int((a_t == 0).sum().item())
                    self._cached_n_g1 = int((a_t == 1).sum().item())
                n_g0 = self._cached_n_g0
                n_g1 = self._cached_n_g1

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

                # Classification-based fairness metrics on val (hard threshold, for logging)
                _fcls = rh.fairness_classification_metrics(
                    a_theta_val, y_val_bin, p1_beta_val, threshold=f1_thresh
                )
                ep_dp_diff      = _fcls["dp_diff"]
                ep_eo_tpr_diff  = _fcls["eo_tpr_diff"]
                ep_eod_max_diff = _fcls["eod_max_diff"]
                ep_eod_avg_diff = _fcls["eod_avg_diff"]

                # Soft EO for beta (threshold-free, used as reward signal)
                _soft_eo_b = rh.soft_eo_gap(a_theta_val, y_val_bin, p1_beta_val)
                ep_soft_eo_beta = float(_soft_eo_b.item()) if (_soft_eo_b == _soft_eo_b) else float("nan")

        # ---------------- Local term on Φ ----------------
        with torch.no_grad():
            if hard_from_beta:
                p = rh.p1_from_agent(beta_model, x_phi)
            else:
                p = rh.p1_from_agent(alpha_model, x_phi)

        # defaults for diagnostics
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
                # 4-group alpha baseline to match 4-group beta worst-loss
                with torch.no_grad():
                    _loss_a = rh.bce_per_sample_from_probs(y_val_bin, rh.p1_from_agent(alpha_model, x_theta_val))
                    _g_ids_a = torch.as_tensor(self.dataset.a_val, device=_loss_a.device).long() * 2 + y_val_bin
                    _worst_4g_a, _ = rh.worst_group_loss(_loss_a, _g_ids_a, group_values=(0, 1, 2, 3))
                self.disadv_worst_loss_alpha = float(_worst_4g_a.item()) if _worst_4g_a == _worst_4g_a else float("nan")
                self.bce_mean_alpha_overall = float(_loss_a.mean().item())
                self.auc_alpha_overall = rh.roc_auc_from_probs(y_val_bin, p1_alpha_val)
                # EO alpha baseline (computed once, used as reference for EO-direct global reward)
                with torch.no_grad():
                    _p1_a = rh.p1_from_agent(alpha_model, x_theta_val)
                    _eo_fcls = rh.fairness_classification_metrics(
                        self.dataset.a_val, y_val_bin, _p1_a, threshold=f1_thresh
                    )
                self.eo_alpha_baseline = float(_eo_fcls["eo_tpr_diff"])

            if use_dvrl_local:
                # DVRL-inspired local reward: beta's BCE loss on each generated sample.
                # High loss = beta currently fails here = high retraining value.
                # Seeds are pre-filtered to disadvantaged minority (Option B in training
                # setup), so all generated samples are implicitly group-targeted.
                score_local = rh.group_conditional_beta_loss(
                    x_phi, y_phi, beta_model, max_bce=dvrl_max_bce
                ) * dvrl_scale
                anchor_reward_mean = float("nan")
                hard_reward_mean = float("nan")
                div_pen_mean = float("nan")
                min_anchor_dist_mean = float("nan")
            else:
                # anchor/hard/diversity local reward (legacy)
                anchors = getattr(self, "disadv_pos_anchors", None)

                # anchor proximity
                if anchors is None or anchors.numel() == 0:
                    min_d = torch.full((x_phi.shape[0],), float("nan"), device=x_phi.device, dtype=x_phi.dtype)
                    anchor_reward = torch.zeros((x_phi.shape[0],), device=x_phi.device, dtype=x_phi.dtype)
                elif use_uncertainty_anchors:
                    with torch.no_grad():
                        p_beta_anchors = rh.p1_from_agent(beta_model, anchors)  # [N_anchors]
                    uncertainty_weights = (1.0 - p_beta_anchors).clamp(0.0, 1.0)
                    min_d, nearest_idx = rh.nearest_anchor_dist_and_idx(x_phi, anchors, chunk=512)
                    sig = torch.tensor(float(sigma_anchor), device=min_d.device, dtype=min_d.dtype)
                    anchor_reward = torch.exp(-0.5 * (min_d / (sig + 1e-8)) ** 2) * uncertainty_weights[nearest_idx]
                else:
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

        # OT-inspired local reward: replaces score_local entirely when enabled
        if self.w_ot > 0.0 and self._ot_mean is not None:
            score_local = rh.ot_local_reward(
                x_phi, self._ot_mean, self._ot_log_var, self._ot_ref_log_prob
            )

        mean_local = float(score_local.mean().item())

        # local clipping rates (helpful to see if local is saturating)
        local_clip_frac_0 = float((score_local <= 0.0 + 1e-12).float().mean().item())
        local_clip_frac_1 = float((score_local >= 1.0 - 1e-12).float().mean().item())

        # ---------------- Combine global + local ----------------
        # Global: sigmoid(k * (worst_loss_alpha - worst_loss_beta)) ∈ (0, 1)
        # 0.5 = no change vs alpha, >0.5 = beta has lower worst-group loss (better), <0.5 = worse
        # Uses 4-group (group × class) worst-case loss: directly aligned with DRO objective.
        if mode in ("wgl", "fairness"):
            wgl_alpha = getattr(self, "disadv_worst_loss_alpha", float("nan"))
            wgl_beta  = worst_loss_beta
            if wgl_alpha == wgl_alpha and wgl_beta == wgl_beta:
                if global_sigmoid_k == 0:
                    # Normalized reward: relative WGL improvement, continuous and scale-invariant.
                    # Positive = beta improved over alpha; negative = beta worse than alpha.
                    global_term = float((wgl_alpha - wgl_beta) / (wgl_alpha + 1e-8))
                else:
                    global_term = float(torch.sigmoid(torch.tensor(global_sigmoid_k * (wgl_alpha - wgl_beta))).item())
            else:
                global_term = 0.0  # neutral fallback when worst-group loss unavailable
        elif mode == "roc_eo":
            # G(θ) = λ·AUC − (1−λ)·EO  (supervisor proposal)
            # Directly optimizes the fairness-utility tradeoff without a reference baseline.
            # Uses hard EO (threshold-based TPR diff) — consistent with what we report.
            eo = ep_eo_tpr_diff if ep_eo_tpr_diff == ep_eo_tpr_diff else float("nan")
            if auc_beta == auc_beta and eo == eo:
                global_term = float(roc_eo_lambda * auc_beta - (1.0 - roc_eo_lambda) * eo)
            else:
                global_term = 0.0
        else:
            with torch.no_grad():
                f1_minority_alpha = rh.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
            delta_f1 = float(f1_minority_beta - f1_minority_alpha)
            global_term = float(torch.sigmoid(torch.tensor(10.0 * delta_f1)).item())

        # Utility guard: scale global_term down when beta's AUC regresses vs alpha.
        # utility_factor = clamp(auc_beta / auc_alpha, min_factor, 1.0)
        # = 1.0 when beta ≥ alpha AUC (no penalty); <1.0 when beta is worse.
        # min_factor prevents reward from collapsing to zero (keeps gradient signal).
        # Disabled when utility_guard_min_factor == 1.0 (default).
        auc_alpha_ref = getattr(self, "auc_alpha_overall", float("nan"))
        utility_factor = 1.0
        if (utility_guard_min_factor < 1.0
                and auc_alpha_ref == auc_alpha_ref and auc_alpha_ref > 0
                and auc_beta == auc_beta):
            raw_factor = auc_beta / auc_alpha_ref
            utility_factor = float(max(utility_guard_min_factor, min(1.0, raw_factor)))
        global_term = global_term * utility_factor

        # Spread both signals across all steps so cumulative sums stay comparable:
        # G_0 ≈ (1 - λ) * mean(local) + λ * global
        T_traj = float(score_local.shape[0])
        reward = (1.0 - lambda_t) / T_traj * score_local + lambda_t / T_traj * float(global_term)

        # ---------------- Extra diagnostics (lean set) ----------------
        # Reuse `p` (alpha/beta on x_phi) already computed for local reward — no extra forward pass needed.
        try:
            diag_frac_mid_conf    = float(((p >= 0.4) & (p <= 0.6)).float().mean().item())
            diag_mean_abs_margin  = float(torch.abs(p - 0.5).mean().item())
            diag_gen_radius_mean  = float(torch.linalg.norm(x_phi, dim=1).mean().item())
        except Exception:
            diag_frac_mid_conf = diag_mean_abs_margin = diag_gen_radius_mean = float("nan")

        # ---------------- Lean diagnostics dict ----------------
        diagnostics = {
            "global": {
                "global_obj": float(global_term),        # fairness signal × utility_factor
                "local_reward": float(mean_local),       # mean local reward per step
                "utility_factor": float(utility_factor), # multiplicative utility guard (1.0 = no penalty)
            },
            "utility": {
                "f1_macro_beta": float(f1_macro_beta),         # macro F1 for beta model
                "f1_minority_beta": float(f1_minority_beta),   # F1 score on minority class for beta model
                "acc_beta": float(acc_beta),                   # accuracy for beta model
                "auc_beta": float(auc_beta),                   # ROC-AUC for beta model
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
                "worst_loss_alpha_baseline": float(getattr(self, "disadv_worst_loss_alpha", float("nan"))),
                "eo_alpha_baseline": float(getattr(self, "eo_alpha_baseline", float("nan"))),
                "dp_diff": float(ep_dp_diff),                   # |P(ŷ=1|a=0) - P(ŷ=1|a=1)|
                "eo_tpr_diff": float(ep_eo_tpr_diff),           # |TPR(a=0) - TPR(a=1)|
                "soft_eo_beta": float(ep_soft_eo_beta),         # |E[p1|y=1,a=0] - E[p1|y=1,a=1]| (no threshold)
                "eod_max_diff": float(ep_eod_max_diff),         # max(|TPR diff|, |FPR diff|)
                "eod_avg_diff": float(ep_eod_avg_diff),         # avg(|TPR diff|, |FPR diff|)
            },
            "local": {
                "anchors_used": int(getattr(self, "_cached_anchors_used", 0)),  # number of anchor points (static after setup)
                "anchor_reward_mean": float(anchor_reward_mean),     # mean anchor reward
                "hard_reward_mean": float(hard_reward_mean),         # mean hard reward (challenging samples)
                "div_pen_mean": float(div_pen_mean),                 # mean diversity penalty
                "min_anchor_dist_mean": float(min_anchor_dist_mean), # mean min distance to an anchor
                "local_clip_frac_0": float(local_clip_frac_0),       # fraction of local rewards clipped at zero
                "local_clip_frac_1": float(local_clip_frac_1),       # fraction of local rewards clipped at one
                "sigma_anchor_used": float(sigma_anchor),            # effective sigma this episode
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
        episodes: int | None = None,      # override self.episodes for this phase
    ) -> tuple:
        """Run a full episode loop for one phase. Returns (best_x_syn, best_y_syn) tensors."""
        n_episodes = episodes if episodes is not None else self.episodes

        # Clear correlation buffers at start of each phase
        self._local_buf.clear()
        self._delta_buf.clear()

        best_phase_reward = -float("inf")
        best_x_syn = None
        best_y_syn = None

        print(f"\n{'='*60}")
        print(f"[Phase] Starting {phase_label} | target_class={target_class} | episodes={n_episodes}")
        print(f"{'='*60}")

        for episode in range(n_episodes):
            A = self.pca_components
            D = 1 + 2 * A

            # Pre-allocate GPU tensors
            states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            actions = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            next_states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            dones = torch.zeros(self.traj_length, dtype=torch.bool, device=self.device)

            x_syn_tensor = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            y_syn_tensor = torch.zeros(self.traj_length, dtype=torch.long, device=self.device)
            log_probs = torch.zeros(self.traj_length, dtype=torch.float32, device=self.device)

            # Reset env — pass episode index so curriculum can schedule stages
            if self.curriculum_learning:
                state = env.reset(episode_idx=episode)
            else:
                state = env.reset()

            if episode % self.beta_reset_interval == 0:
                if self.beta_warmstart_from_alpha and hasattr(self, "alpha_model"):
                    # Warm-start beta from alpha weights — prevents the random-init deadzone
                    # where beta is worse than alpha for hundreds of episodes.
                    self.beta_model.model.load_state_dict(
                        self.alpha_model.model.state_dict()
                    )
                    self.beta_model.optimizer = type(self.beta_model.optimizer)(
                        self.beta_model.model.parameters(),
                        **self.beta_model._optim_cfg,
                    )
                else:
                    self.beta_model.reset()

            for t in range(self.traj_length):
                if self.use_ppo:
                    action, lp = agent.predict(state)
                    log_probs[t] = lp.to(self.device)
                else:
                    action = agent.predict(state)
                next_state, done, info = env.step(action, (t + 1))

                states[t] = state
                actions[t] = action
                next_states[t] = next_state
                dones[t] = done

                pca_sample = info["current_pca"]

                x_syn_tensor[t] = pca_sample
                y_syn_tensor[t] = info["sampled_target"]

                state = next_state
                if done:
                    break

            T = self.traj_length
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
            del x_hybrid, y_hybrid

            progress = (episode + 1) / n_episodes

            # Dynamic anchor refresh: re-select most uncertain anchors from candidate pool
            if (self.anchor_refresh_interval > 0
                    and episode > 0
                    and episode % self.anchor_refresh_interval == 0
                    and getattr(self, "_anchor_candidate_pool", None) is not None
                    and self._anchor_candidate_pool.shape[0] > 0):
                with torch.no_grad():
                    p_pool = rh.p1_from_agent(self.beta_model, self._anchor_candidate_pool)
                uncertainty = torch.abs(p_pool - 0.5)  # low = uncertain (near decision boundary)
                top_k = min(self.anchor_refresh_top_k, self._anchor_candidate_pool.shape[0])
                _, uncertain_idx = uncertainty.topk(top_k, largest=False)
                self.disadv_pos_anchors = self._anchor_candidate_pool[uncertain_idx].detach()
                self._anchor_refresh_count = getattr(self, "_anchor_refresh_count", 0) + 1
                # Recalibrate sigma for the new anchor set if auto-calibration is enabled
                if self.sigma_calibration_factor is not None and self.disadv_pos_anchors.shape[0] >= 2:
                    n_s = min(300, self.disadv_pos_anchors.shape[0])
                    sub = self.disadv_pos_anchors[:n_s].to(self.device)
                    D_r = torch.cdist(sub, sub)
                    D_r.fill_diagonal_(float("inf"))
                    median_d_r = float(D_r.min(dim=1).values.median().item())
                    self.sigma_anchor = median_d_r * self.sigma_calibration_factor
                print(f"[anchor_refresh] ep={episode} refresh #{self._anchor_refresh_count}: "
                      f"{self.disadv_pos_anchors.shape[0]} most-uncertain anchors "
                      f"sigma_anchor={self.sigma_anchor:.4f}")

            # UA warm-up: disable uncertainty weighting for first N episodes
            use_ua_this_episode = (
                self.use_uncertainty_anchors
                and (episode >= self.uncertainty_warmup_episodes)
            )

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
                local_squash_k=self.local_squash_k,
                local_squash_center=self.local_squash_center,
                hard_from_beta=self.hard_from_beta,
                use_uncertainty_anchors=use_ua_this_episode,
                global_sigmoid_k=self.global_sigmoid_k,
                use_dvrl_local=self.use_dvrl_local,
                dvrl_max_bce=self.dvrl_max_bce,
                dvrl_scale=self.dvrl_scale,
                utility_guard_min_factor=self.utility_guard_min_factor,
                roc_eo_lambda=self.roc_eo_lambda,
            )

            # Truncate episode tensors and learn
            states = states[:T]
            actions = actions[:T]
            next_states = next_states[:T]
            dones = dones[:T]
            rewards = rewards[:T]

            if self.use_ppo:
                agent.learn_trajectory(states, actions, log_probs[:T], rewards, next_states, dones)
            else:
                agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

            # alignment metrics
            g = diagnostics.get("global", {})
            f = diagnostics.get("fairness", {})

            local_mean = float(g.get("local_reward", float("nan")))
            if self.reward_mode in ("wgl", "fairness"):
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
                "lambda_t": float(lambda_t),
                "anchor_refresh_count": int(getattr(self, "_anchor_refresh_count", 0)),
                "ua_warmup_active": int(self.use_uncertainty_anchors and episode < self.uncertainty_warmup_episodes),
            }
            avg_reward = float(torch.mean(rewards).item())
            episode_return = float(rewards.sum().item())
            meta_metrics = {"avg_reward": avg_reward, "episode_return": episode_return, "phase": phase_label}

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

            # Track best in-memory for return — use global_obj (fairness metric) not episode_return,
            # so the synthetic data returned to the final combined test is from the best fairness episode.
            best_select_val = diagnostics.get("global", {}).get("global_obj", episode_return)
            if best_select_val > best_phase_reward:
                best_phase_reward = best_select_val
                best_x_syn = x_phi_t.detach().clone()
                best_y_syn = y_phi_t.detach().clone()

            del states, actions, next_states, dones, rewards, x_syn_tensor, y_syn_tensor, log_probs, x_phi_t, y_phi_t
            if episode % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"[Phase] Finished {phase_label} | best global_obj={best_phase_reward:.4f}")
        return (best_x_syn, best_y_syn)

    # ---------------- CMA-ES episode loop ----------------
    def _run_phase_cmaes(
        self,
        *,
        target_class: int,
        agent,                         # CMAESAgent
        x_theta_train,
        y_theta_train,
        x_theta_val,
        y_theta_val,
        prior_synthetic: tuple | None,
        phase_label: str,
        episodes: int | None = None,
    ) -> tuple:
        """Run a full episode loop using CMA-ES. Returns (best_x_syn, best_y_syn)."""
        n_episodes = episodes if episodes is not None else self.episodes

        self._local_buf.clear()
        self._delta_buf.clear()

        best_phase_reward = -float("inf")
        best_x_syn = None
        best_y_syn = None

        print(f"\n{'='*60}")
        print(f"[CMA-ES Phase] {phase_label} | target_class={target_class} | "
              f"episodes={n_episodes} | popsize={agent.popsize}")
        print(f"{'='*60}")

        converged = False

        for episode in range(n_episodes):
            if converged:
                break

            # Vary DataLoader shuffle seed per episode so candidates in the same
            # generation don't receive identical beta training (real data dominates)
            self.dl_generator = torch.Generator(device="cpu").manual_seed(
                self.seed + episode
            )

            # Get current candidate and sample synthetic data
            candidate_params = agent.ask()
            x_phi_t, y_phi_t = agent.sample_synthetic(candidate_params, target_class)

            # Reset beta each episode (same policy as REINFORCE path)
            if episode % self.beta_reset_interval == 0:
                if self.beta_warmstart_from_alpha and hasattr(self, "alpha_model"):
                    self.beta_model.model.load_state_dict(
                        self.alpha_model.model.state_dict()
                    )
                    self.beta_model.optimizer = type(self.beta_model.optimizer)(
                        self.beta_model.model.parameters(),
                        **self.beta_model._optim_cfg,
                    )
                else:
                    self.beta_model.reset()

            # Train beta on real + prior_synthetic + current synthetic
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
            del x_hybrid, y_hybrid

            progress = (episode + 1) / n_episodes

            # Compute reward — identical to REINFORCE path
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
                local_squash_k=self.local_squash_k,
                local_squash_center=self.local_squash_center,
                hard_from_beta=self.hard_from_beta,
                use_uncertainty_anchors=False,
                global_sigmoid_k=self.global_sigmoid_k,
                use_dvrl_local=self.use_dvrl_local,
                dvrl_max_bce=self.dvrl_max_bce,
                dvrl_scale=self.dvrl_scale,
                utility_guard_min_factor=self.utility_guard_min_factor,
                roc_eo_lambda=self.roc_eo_lambda,
            )

            global_obj = diagnostics.get("global", {}).get("global_obj", 0.0)
            candidate_idx = episode % agent.popsize
            # Use the full episode return (global + local) so that OT / other local
            # rewards actually influence CMA-ES search. When lambda=1 or w_ot=0 this
            # reduces to global_obj (no behaviour change for pure-global runs).
            cmaes_obj = float(rewards.sum().item())
            agent.tell(candidate_idx, cmaes_obj)
            agent.advance()

            # End of generation — update CMA-ES only when full generation is complete
            is_full_gen_done = ((episode + 1) % agent.popsize == 0)
            if is_full_gen_done:
                stop = agent.step_generation()
                if stop:
                    print(f"[CMA-ES] Converged at episode {episode+1}: {stop}")
                    converged = True

            # Alignment metrics
            f = diagnostics.get("fairness", {})
            g = diagnostics.get("global", {})
            local_mean = float(g.get("local_reward", float("nan")))
            delta_val = float(-f.get("worst_loss_beta", float("nan")))
            self._local_buf.append(local_mean)
            self._delta_buf.append(delta_val)

            lambda_start, lambda_end = self.lambda_schedule
            lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

            alignment_metrics = {
                "delta_global": float(delta_val),
                "corr_local_delta": self._corr_local_delta(),
                "curriculum_stage": 0,
                "lambda_t": float(lambda_t),
                "anchor_refresh_count": 0,
                "ua_warmup_active": 0,
                "cmaes_generation": episode // agent.popsize,
                "cmaes_candidate_idx": candidate_idx,
                "cmaes_sigma": agent.sigma,
            }

            avg_reward = float(torch.mean(rewards).item())
            episode_return = float(rewards.sum().item())
            meta_metrics = {
                "avg_reward": avg_reward,
                "episode_return": episode_return,
                "phase": phase_label,
            }

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

            best_select_val = global_obj
            if best_select_val > best_phase_reward:
                best_phase_reward = best_select_val
                best_x_syn = x_phi_t.detach().clone()
                best_y_syn = y_phi_t.detach().clone()

            del x_phi_t, y_phi_t
            if episode % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"[CMA-ES Phase] Finished {phase_label} | best global_obj={best_phase_reward:.4f}")
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
            "DA_PCT": self.da_pct,

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

            # reward shaping
            "global_sigmoid_k": self.global_sigmoid_k,

            # dataset-specific windowing (capture24); needed to reconstruct PCA post-hoc
            "win_seconds": self.win_seconds,
            "step_seconds": self.step_seconds,
        }

        # create beta factory once (so tracker can rehydrate best-beta for final test)
        beta_factory = lambda: FFNNAgent(**self.ffnn_config)

        # Snapshot every N episodes — auto-scaled so long runs don't create thousands of files.
        # Target ~100 snapshots per run regardless of episode count.
        # Gen-curve (check_run.py) picks the nearest available snapshot, so coarser is fine.
        _ckpt_every = 150

        with EpisodeTracker(
            run_stats,
            dataset=self.dataset,
            save_dir=getattr(self, "save_dir", "training_runs"),
            compare_metric="global.global_obj",
            beta_factory=beta_factory,
            seed=self.seed,
            ckpt_every=_ckpt_every,
        ) as tracker:
            self.tracker = tracker

            #Data splits
            x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = (
                self.dataset.get_data_splits(
                    train_size=self.real_data_size,
                    bias_pct=self.bias_pct,
                    da_pct=self.da_pct,
                    pca_components=self.pca_components,
                    drop_protected=False,
                    protected_cols=self.dataset.protected_attributes,
                    bias_val=self.bias_val,
                    win_seconds=self.win_seconds,
                    step_seconds=self.step_seconds,
                    **({"dp_protected_col": self.dp_protected_col} if self.dp_protected_col is not None else {}),
                    **({"pool_pos_fraction": self.pool_pos_fraction} if self.pool_pos_fraction is not None else {}),
                )
            )

            # Free the GAN view cache when no GAN-based baselines will run.
            # For Capture-24, X_train_unbiased_df holds ~595K windows (~157MB) for the
            # entire training lifetime; releasing it here prevents CPU RAM accumulation
            # across multi-seed runs on SLURM nodes with limited RAM.
            b = self.benchmarks_config
            if not b.get("run_ctgan", False) and not b.get("run_ctabgan", False):
                gv = getattr(self.dataset, "_gan_view_cache", None)
                if gv and gv.get("supported"):
                    gv["X_train_unbiased_df"] = None
                    gv["y_train_unbiased"] = None

            # Resolve actual feature dimension (may differ from pca_components when use_pca=False)
            feature_dim = x_theta_train.shape[1]
            self.pca_components = feature_dim

            self.state_dim = 1 + 2 * feature_dim

            # Update configs and rebuild agents with correct dimensions
            self.ffnn_config["input_size"] = feature_dim
            self.reinforce_config["state_size"] = self.state_dim
            self.reinforce_config["action_size"] = feature_dim
            self.ppo_config["state_size"] = self.state_dim
            self.ppo_config["action_size"] = feature_dim

            if self.use_ppo:
                self.agent = PPOAgent(**self.ppo_config)
            else:
                self.agent = ReinforceAgent(**self.reinforce_config)
            self.alpha_model = FFNNAgent(**self.ffnn_config)
            self.beta_model = FFNNAgent(**self.ffnn_config)

            # real_minority_samples is populated after disadv_group_value is known (see below)
            real_minority_samples = None  # placeholder; set after alpha/disadv setup

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
            # 4-group alpha baseline to match 4-group beta worst-loss
            with torch.no_grad():
                _y_long = y_theta_val.long()
                _p1_a = rh.p1_from_agent(self.alpha_model, x_theta_val)
                _loss_a = rh.bce_per_sample_from_probs(_y_long, _p1_a)
                _g_ids_a = torch.as_tensor(self.dataset.a_val, device=_loss_a.device).long() * 2 + _y_long
                _worst_4g_a, _ = rh.worst_group_loss(_loss_a, _g_ids_a, group_values=(0, 1, 2, 3))
            self.disadv_worst_loss_alpha = float(_worst_4g_a.item()) if _worst_4g_a == _worst_4g_a else float("nan")
            self.auc_alpha_overall = rh.roc_auc_from_probs(_y_long, _p1_a)

            # Soft EO alpha baseline (threshold-free, differentiable proxy for EO gap)
            with torch.no_grad():
                _p1_a_eo = rh.p1_from_agent(self.alpha_model, x_theta_val)
                _soft_eo_a = rh.soft_eo_gap(self.dataset.a_val, y_theta_val, _p1_a_eo)
            self.eo_alpha_baseline = float(_soft_eo_a.item()) if (_soft_eo_a == _soft_eo_a) else float("nan")

            print(f"[disadv] group={disadv} per_g={per_g} worst_4g={self.disadv_worst_loss_alpha:.4f} soft_eo_alpha={self.eo_alpha_baseline:.4f}")


            # Build anchor set: real TRAIN points that are (y=1) AND (a=disadvantaged group)
            a_train = self.dataset.a_train
            disadv = int(self.disadv_group_value)

            # Option B: seed env trajectories only from disadvantaged minority samples so
            # all generated samples are implicitly targeted at the right group (no group
            # labels needed at step time for DVRL local reward).
            disadv_minority_mask = (y_theta_train == 1) & (a_train == disadv)
            real_minority_samples = x_theta_train[disadv_minority_mask]

            anchor_mask = (y_theta_train == 1) & (a_train == disadv)
            anchor_pool = x_theta_train[anchor_mask]

            # Store full candidate pool (pre-cap) for dynamic anchor refresh
            self._anchor_candidate_pool = anchor_pool.detach()
            self._anchor_refresh_count = 0

            MAX_ANCHORS = 2000
            if self.anchor_selection_mode == "hard_positive":
                # Select anchors where alpha is most uncertain/wrong:
                # sort by p_alpha ascending (lowest = alpha gets it most wrong),
                # so anchor proximity and hard_reward align toward the decision boundary.
                with torch.no_grad():
                    p_alpha_pool = rh.p1_from_agent(self.alpha_model, anchor_pool)
                n_sel = min(self.anchor_selection_top_k, anchor_pool.shape[0])
                _, hard_idx = p_alpha_pool.topk(n_sel, largest=False)
                anchors = anchor_pool[hard_idx]
                print(f"[anchor_selection] hard_positive: {n_sel}/{anchor_pool.shape[0]} anchors selected "
                      f"(p_alpha range [{float(p_alpha_pool[hard_idx].min()):.3f}, "
                      f"{float(p_alpha_pool[hard_idx].max()):.3f}])")
            else:
                # "all": random cap (original behaviour)
                anchors = anchor_pool
                if anchors.shape[0] > MAX_ANCHORS:
                    g = torch.Generator(device="cpu").manual_seed(self.seed)
                    idx = torch.randperm(anchors.shape[0], generator=g)[:MAX_ANCHORS]
                    anchors = anchors[idx.to(anchors.device)]

            self.disadv_pos_anchors = anchors.detach()
            self._cached_anchors_used = int(self.disadv_pos_anchors.shape[0])

            # OT local reward: precompute target Gaussian from advantaged minority samples
            if self.w_ot > 0.0:
                adv_minority_mask = (y_theta_train == 1) & (a_train == int(self.adv_group_value))
                adv_minority = x_theta_train[adv_minority_mask]
                if adv_minority.shape[0] >= 2 and real_minority_samples.shape[0] >= 2:
                    self._ot_mean, self._ot_log_var, self._ot_ref_log_prob = rh.compute_ot_target(
                        real_minority_samples, adv_minority
                    )
                    print(f"[OT] target fitted: adv_pos={adv_minority.shape[0]} "
                          f"disadv_pos={real_minority_samples.shape[0]} "
                          f"ref_log_prob={self._ot_ref_log_prob:.3f}")
                else:
                    print(f"[OT] insufficient samples (adv={adv_minority.shape[0]}, "
                          f"disadv={real_minority_samples.shape[0]}) — OT local reward disabled")
                    self.w_ot = 0.0

            # Sigma auto-calibration: set sigma to median nearest-neighbour distance * factor
            if self.sigma_calibration_factor is not None and anchors.shape[0] >= 2:
                n_sample = min(300, anchors.shape[0])
                g_cal = torch.Generator(device="cpu").manual_seed(self.seed)
                cal_idx = torch.randperm(anchors.shape[0], generator=g_cal)[:n_sample]
                sub = anchors[cal_idx].to(self.device)
                D_cal = torch.cdist(sub, sub)
                D_cal.fill_diagonal_(float("inf"))
                nn_dists = D_cal.min(dim=1).values
                median_d = float(nn_dists.median().item())
                self.sigma_anchor = median_d * self.sigma_calibration_factor
                print(f"[sigma_calib] n_anchors={anchors.shape[0]} median_nn_dist={median_d:.4f} "
                      f"sigma_anchor={self.sigma_anchor:.4f} (factor={self.sigma_calibration_factor})")

            print(f"Anchors: {self.disadv_pos_anchors.shape[0]} (disadv={disadv}) sigma_anchor={self.sigma_anchor:.4f}")

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

            # === Build CMA-ES agents (if use_cmaes=True) ===
            if self.use_cmaes:
                # Compute init_mean from real_minority_samples mean in PCA space
                if real_minority_samples is not None and real_minority_samples.shape[0] > 0:
                    init_mean_1 = real_minority_samples.mean(dim=0).cpu().numpy().astype(np.float64)
                    init_log_std_1 = np.zeros(self.pca_components, dtype=np.float64)
                else:
                    init_mean_1 = np.zeros(self.pca_components, dtype=np.float64)
                    init_log_std_1 = np.zeros(self.pca_components, dtype=np.float64)

                cmaes_cfg = self.cmaes_config or {}
                phase1_cmaes_agent = CMAESAgent(
                    pca_components=self.pca_components,
                    n_samples=self.traj_length,
                    init_mean=init_mean_1,
                    init_log_std=init_log_std_1,
                    sigma0=float(cmaes_cfg.get("sigma0", 1.0)),
                    popsize=cmaes_cfg.get("popsize", None),
                    seed=self.seed,
                    device=self.device,
                )
                phase1_cmaes_agent.reset(phase_label="phase1_class1")

            # === Phase 1: target=1 (minority) ===
            if self.use_cmaes:
                best_syn_1 = self._run_phase_cmaes(
                    target_class=1,
                    agent=phase1_cmaes_agent,
                    x_theta_train=x_theta_train,
                    y_theta_train=y_theta_train,
                    x_theta_val=x_theta_val,
                    y_theta_val=y_theta_val,
                    prior_synthetic=None,
                    phase_label="phase1_class1",
                )
            else:
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
                self._cached_anchors_used = int(self.disadv_pos_anchors.shape[0])
                print(f"Phase 2 anchors: {self.disadv_pos_anchors.shape[0]} (y=0 & disadv={disadv})")

                # Seed samples for env.reset() — disadvantaged group y=0 only (mirrors Option B)
                real_majority_samples = x_theta_train[(y_theta_train == 0) & (a_train == disadv)]
                if real_majority_samples.shape[0] == 0:
                    real_majority_samples = x_theta_train[y_theta_train == 0]

                if self.use_cmaes:
                    if real_majority_samples.shape[0] > 0:
                        init_mean_0 = real_majority_samples.mean(dim=0).cpu().numpy().astype(np.float64)
                        init_log_std_0 = np.zeros(self.pca_components, dtype=np.float64)
                    else:
                        init_mean_0 = np.zeros(self.pca_components, dtype=np.float64)
                        init_log_std_0 = np.zeros(self.pca_components, dtype=np.float64)

                    phase2_cmaes_agent = CMAESAgent(
                        pca_components=self.pca_components,
                        n_samples=self.traj_length,
                        init_mean=init_mean_0,
                        init_log_std=init_log_std_0,
                        sigma0=float(cmaes_cfg.get("sigma0", 1.0)),
                        popsize=cmaes_cfg.get("popsize", None),
                        seed=self.seed + 1,
                        device=self.device,
                    )
                    phase2_cmaes_agent.reset(phase_label="phase2_class0")

                    # Fresh beta model for phase 2
                    self.beta_model = FFNNAgent(**self.ffnn_config)

                    best_syn_0 = self._run_phase_cmaes(
                        target_class=0,
                        agent=phase2_cmaes_agent,
                        x_theta_train=x_theta_train,
                        y_theta_train=y_theta_train,
                        x_theta_val=x_theta_val,
                        y_theta_val=y_theta_val,
                        prior_synthetic=best_syn_1,
                        phase_label="phase2_class0",
                        episodes=self.phase2_episodes,
                    )
                else:
                    # Fresh RL agent for phase 2
                    if self.use_ppo:
                        phase2_agent = PPOAgent(**self.ppo_config)
                    else:
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
                        episodes=self.phase2_episodes,
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
        return True