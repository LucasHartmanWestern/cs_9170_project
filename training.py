# --- Imports ---

# Standard library
from audioop import bias
import os
import time

# Third-party: numpy, pandas, sklearn, torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import copy

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
import multiprocessing as mp
from datetime import datetime
import uuid

# ---------------- Config ----------------
class Training:
    def __init__(
        self,
        exp_group=None,
        multiclass=False,
        dataset_name="census_income",
        minority_id=None,
        majority_id=None,
        third_id=None,
        bias_pct=0.18,
        total_pca_components=2,
        episodes=2000,
        traj_length=500,
        real_data_size=1500,
        reward_mode="gauss_penalty",
        seed=42,
        device='cpu'
    ):
        self.exp_group=exp_group
        self.seed = seed
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.bias_pct=bias_pct
        self.total_pca_components = total_pca_components
        self.episodes = episodes
        self.reward_mode = reward_mode
        self.lambda_schedule = (0.8, 0.8)
        self.multiclass = multiclass
        self.minority_id = minority_id
        self.majority_id = majority_id
        self.third_id = third_id
        self.traj_length = traj_length
        self.real_data_size = real_data_size

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.dataset = Dataset(dataset_name, multiclass=self.multiclass, minority_id=minority_id, majority_id=majority_id, third_id=third_id, pca_components=self.total_pca_components, seed=self.seed, device=self.device)

        # FFNN (alpha/beta) config
        self.ffnn_config = {
            'input_size': self.total_pca_components,            
            'hidden_sizes': [32, 16],
            'output_size': 3 if self.multiclass else 2,  # 3 for multiclass, 2 for binary
            'learning_rate': 1e-3,
            'batch_size': 64,           
            'epochs': 10,                
            'type': 'classification',
            'classes': [0, 1, 2] if self.multiclass else [0, 1],
            'device': self.device,
            'seed': self.seed,
        }

        # REINFORCE config
        reinforce_config = {
            'state_size': 2,
            'action_size': self.total_pca_components,
            'hidden_sizes': [64, 64],
            'total_episodes': episodes,
            'lr': 3e-4,
            'gamma': 0.99,
            'entropy_start': 1e-2,
            'entropy_end': 0.0,
            'seed': self.seed,
            'device': self.device
        }

        self.dl_generator = torch.Generator(device='cpu').manual_seed(self.seed)

        # Agents
        self.agent = ReinforceAgent(**reinforce_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model  = FFNNAgent(**self.ffnn_config)

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

    # # ---------------- Reward helpers ----------------
    # # Generic "get P(y=1|x)" from an FFNNAgent (alpha or beta)
    # def _p1_from_agent(self, agent, x):
    #     agent.model.eval()
    #     with torch.no_grad():
    #         logits = agent.model(x)             # [N, 2]
    #         probs  = torch.softmax(logits, -1)  # [N, 2]
    #         return probs[..., 1]                # [N]

    # # Binary F1 for the positive class, from probabilities and a threshold
    # def f1_from_probs(self, y_true, p1, threshold=0.5):
    #     y_true = y_true.to(p1.device).long()
    #     y_pred = (p1 >= threshold).long()
    #     tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    #     fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    #     fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    #     eps = 1e-8
    #     precision = tp / (tp + fp + eps)
    #     recall    = tp / (tp + fn + eps)
    #     f1 = 2 * precision * recall / (precision + recall + eps)
    #     return f1

    # # Per-sample Brier (MSE on probabilities)
    # def brier_per_sample(self, y_true, p1):
    #     y_true = y_true.to(p1.device).float()
    #     return (p1 - y_true) ** 2  # in [0,1]

    # # Beta model F1 minority class score + local term (no EMA)
    # def compute_reward(
    #     self,
    #     alpha_model, beta_model, stale_beta_model,   # stale_beta_model unused (kept for signature compat)
    #     x_theta_val, y_theta_val,                    # θ-space VAL set (labels are 0/1 or 0/1/2)
    #     x_phi, y_phi,                                # synthetic batch (y_phi should be 1 for rope)
    #     progress: float,                             # (episode+1)/EPISODES in [0,1]
    #     f1_thresh: float = 0.5,

    #     # schedules / gates
    #     lambda_schedule=(0.30, 0.95),                # (start, end) across the run

    #     # hinge-penalty for majority/weighted scores (used only in *_penalty mode)
    #     epsilon_majority: float = 0.01,   # allow up to -1.0 pp on majority F1
    #     epsilon_weighted: float = 0.005,  # allow up to -0.5 pp on weighted F1
    #     c_majority: float = 0.30,         # penalty weight for majority violation
    #     c_weighted: float = 0.30,         # penalty weight for weighted violation
    #     class_mode: str = "binary",   # "binary" or "multiclass"
    # ):
    #     """
    #     Reward combines:
    #     - Global term: ΔF1(minority, rope) on θ_val between β and α.
    #     - Local term: Gaussian usefulness around 0.5 on Pα(y=1|x_phi), with optional slice gate.

    #     class_mode:
    #     - "binary": y_theta_val is 0/1; we evaluate directly.
    #     - "multiclass": y_theta_val is 0/1/2; we evaluate as (rope vs non-rope):
    #             y_val_bin = 1 if y==1 else 0    # {0,2} -> 0, 1->1

    #     Notes:
    #     * We always treat class 1 = rope as the minority of interest.
    #     * stale_beta_model/EMA params are accepted for compatibility but not used.
    #     """
    #     # ---------- Mode selection ----------
    #     alias = {
    #         "gauss_nopen": "local_gauss",
    #         "gauss_penalty": "local_gauss_penalty",
    #         "judge_conf": "local_gauss",
    #     }
    #     mode = alias.get(self.reward_mode, self.reward_mode)
    #     valid = {"local_gauss", "local_gauss_penalty", "local_slices"}
    #     if mode not in valid:
    #         raise ValueError(f"reward_mode must be one of {valid}, got {self.reward_mode!r}")

    #     # --- λ schedule ---
    #     lambda_start, lambda_end = lambda_schedule
    #     lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

    #     # --- Build rope vs non-rope labels for evaluation ---
    #     if class_mode not in ("binary", "multiclass"):
    #         raise ValueError("class_mode must be 'binary' or 'multiclass'")

    #     if class_mode == "binary":
    #         # y_theta_val already 0/1
    #         y_val_bin = y_theta_val.long()
    #     else:
    #         # collapse 3 classes into rope vs non-rope: {0,2}→0, 1→1
    #         y_val_bin = (y_theta_val == 1).long()

    #     with torch.no_grad():
    #         # ---- Global term on θ_val ----

    #         # Always use Pα(y=1|x) and Pβ(y=1|x). For 3-class, this is softmax column 1 (rope).
    #         p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
    #         p1_beta_val  = self._p1_from_agent(beta_model,  x_theta_val)

    #         # Minority / majority / macro (α vs β) on rope-vs-rest binarization
    #         f1_minority_alpha = self.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
    #         f1_minority_beta  = self.f1_from_probs(y_val_bin, p1_beta_val,  f1_thresh)
    #         f1_majority_alpha = self.f1_from_probs(1 - y_val_bin, 1 - p1_alpha_val, 1 - f1_thresh)
    #         f1_majority_beta  = self.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val,  1 - f1_thresh)

    #         f1_macro_beta = 0.5 * (f1_minority_beta + f1_majority_beta)

    #         # Weighted F1 (support-weighted) using rope fraction on θ_val
    #         pos_frac = float(y_val_bin.float().mean().item())
    #         neg_frac = 1.0 - pos_frac
    #         f1_weighted_alpha = pos_frac * float(f1_minority_alpha) + neg_frac * float(f1_majority_alpha)
    #         f1_weighted_beta  = pos_frac * float(f1_minority_beta)  + neg_frac * float(f1_majority_beta)

    #         # Deltas (β − α)
    #         delta_f1_minority = float(f1_minority_beta - f1_minority_alpha)
    #         delta_f1_majority = float(f1_majority_beta - f1_majority_alpha)
    #         delta_f1_weighted = float(f1_weighted_beta - f1_weighted_alpha)

    #         # ---- Local term on synthetic Φ (judge = α) ----
    #         # y_phi should be 1 for rope samples; we score usefulness by α's uncertainty.
    #         p = self._p1_from_agent(alpha_model, x_phi)  # P_alpha(y=1|x_phi) in [0,1]

    #         # Symmetric Gaussian around 0.5
    #         tau = 0.10
    #         m = torch.abs(p - 0.5)
    #         score_gauss = torch.exp(-0.5 * (m / tau) ** 2)
    #         score_local_raw = score_gauss.clone()

    #         # Optional slice-based radial gate
    #         w_slice = torch.ones_like(score_local_raw)
    #         if mode == "local_slices":
    #             if hasattr(self, "slice_pairs") and len(self.slice_pairs) > 0:
    #                 gates = []
    #                 for idx, (i, j) in enumerate(self.slice_pairs):
    #                     r_ij = torch.linalg.norm(x_phi[:, [i, j]], dim=1)  # [T]
    #                     denom = (self.slice_gain * self.slice_r_iqr[idx]).clamp_min(1e-6)
    #                     g_ij = torch.sigmoid(((r_ij - self.slice_r_med[idx]) / denom).clamp(-10, 10))
    #                     gates.append(g_ij)
    #                 w_slice = torch.stack(gates, dim=1).amax(dim=1)      # union of “outside cores”
    #                 score_local_raw = score_local_raw * w_slice
    #             # else keep w_slice = 1

    #         # Cap local score
    #         LOCAL_CAP = 1.0
    #         cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
    #         over_mask = (score_local_raw > cap_t).float()
    #         local_cap_frac = float(over_mask.mean().item())
    #         score_local = torch.minimum(score_local_raw, cap_t)

    #     # ---- Combine global + local ----
    #     base_reward = lambda_t * delta_f1_minority + (1.0 - lambda_t) * score_local  # [T]

    #     # Penalty (optional mode)
    #     maj_violation = 0.0
    #     wtd_violation = 0.0
    #     penalty = 0.0
    #     if mode == "local_gauss_penalty":
    #         maj_violation = max(0.0, -(delta_f1_majority + epsilon_majority))
    #         wtd_violation = max(0.0, -(delta_f1_weighted + epsilon_weighted))
    #         penalty = c_majority * maj_violation + c_weighted * wtd_violation

    #     reward = base_reward - penalty
    #     mean_local = float(score_local.mean().item())

    #     diagnostics = {
    #         "reward_mode": mode,
    #         "global_reward": float(f1_minority_beta),     # F1 rope (β) on θ_val
    #         "local_reward": mean_local,                   # mean local score
    #         "f1_macro_beta": float(f1_macro_beta),

    #         # Details
    #         "judge_conf_mean": float(score_gauss.mean().item()),
    #         "uncert_alpha_mean": float((1.0 - (2.0 * m).clamp(0, 1)).mean().item()),
    #         "alpha_wrong_rate": float(((p >= 0.5).float() != (y_phi.to(p.device).float())).float().mean().item()),
    #         "delta_f1_val": delta_f1_minority,
    #         "delta_f1_majority": delta_f1_majority,
    #         "delta_f1_weighted": delta_f1_weighted,
    #         "f1_minority_alpha": float(f1_minority_alpha),
    #         "f1_minority_beta_stale": 0,
    #         "local_cap_frac": local_cap_frac,

    #         # Penalty diagnostics
    #         "epsilon_majority": epsilon_majority if mode == "local_gauss_penalty" else None,
    #         "epsilon_weighted": epsilon_weighted if mode == "local_gauss_penalty" else None,
    #         "penalty": penalty if mode == "local_gauss_penalty" else 0.0,
    #         "maj_violation": maj_violation if mode == "local_gauss_penalty" else 0.0,
    #         "wtd_violation": wtd_violation if mode == "local_gauss_penalty" else 0.0,

    #         # Slice gate signal
    #         "w_slice_mean": float(w_slice.mean().item()) if mode == "local_slices" else 1.0,

    #         "corr_factor_mean": None,
    #     }
    #     return reward, diagnostics


    # ---------------- Training loop ----------------
    def __call__(self):
        start_time = time.time()

        run_stats = {
            "EXP_GROUP": self.exp_group,
            "EPISODES": self.episodes,
            "TRAJ_LENGTH": self.traj_length,
            "REAL_DATA_SIZE": self.real_data_size,
            "BIAS_PCT": self.bias_pct,
            "lambda_schedule": self.lambda_schedule,
            "seed": self.seed,
            "pca_components": self.total_pca_components,
            "reward_mode": self.reward_mode,     
            "minority_id": self.minority_id,
            "majority_id":self.majority_id,
            "third_id":self.third_id
        }
        num_stages = self.total_pca_components - 1
        stage_episodes = self.episodes // num_stages
        
        # create beta factory once (so tracker can rehydrate best-beta for final test)
        beta_factory = lambda: FFNNAgent(**self.ffnn_config)

        with EpisodeTracker(
            run_stats,
            dataset=self.dataset,
            save_dir="training_runs",
            compare_metric="average_reward",
            beta_factory=beta_factory,
            seed=self.seed
        ) as tracker:
            self.tracker = tracker  


        x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = self.dataset.get_data_splits(
            train_size=self.real_data_size, bias_pct=self.bias_pct, pca_components=self.total_pca_components, 
        )

        total_data = len(x_theta_train) + self.traj_length
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (self.traj_length / total_data) * 100
        print(f"Beta will train with: {real_percentage:.2f}% real data, {synthetic_percentage:.2f}% synthetic data: ")

        # Environment (state is 2D PCA coords; target is 1 class for synthetic generation)
        env = Environment(
            target=1,
            max_actions=self.traj_length,
            device=self.device,
            seed=self.seed
        )

        for stage in range(num_stages):
            if stage == 0:
                feature_length = 2
            else:
                feature_length = feature_length + 1
            # Train α baseline on masked data for each stage on real-only train set
            self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)
            self.tracker.save_alpha_state_dict(self.alpha_model, self.ffnn_config, self.total_pca_components)

            for episode in range(stage_episodes):
                # Pre-allocate (GPU tensors)
                D = 2
                A = self.total_pca_components
                states      = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
                actions     = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
                next_states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
                dones       = torch.zeros(self.traj_length, dtype=torch.bool, device=self.device)

                x_syn_tensor = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
                y_syn_tensor = torch.zeros(self.traj_length, dtype=torch.long, device=self.device)

                state = env.reset()

                # Reset beta to its initial snapshot each episode for a stable baseline
                stale_beta_model = copy.deepcopy(self.beta_model)
                self.beta_model.reset()

                for t in range(self.traj_length):
                    action = self.agent.predict(state)                      
                    next_state, done, info = env.step(action, (t + 1))

                    states[t]      = state
                    actions[t]     = action
                    next_states[t] = next_state
                    dones[t]       = done

                    x_syn_tensor[t] = action
                    y_syn_tensor[t] = info['sampled_target']  # 1 for minority

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

                # Rewards (alpha baseline global + per-step local)
                rewards, diagnostics = self.compute_reward(
                    self.alpha_model, self.beta_model, stale_beta_model,
                    x_theta_val, y_theta_val,
                    x_phi_t, y_phi_t,
                    progress=progress,
                    f1_thresh=0.5,
                    lambda_schedule=self.lambda_schedule,
                    class_mode=("multiclass" if self.multiclass else "binary")
                )

                # Truncate episode tensors and learn
                states      = states[:T]
                actions     = actions[:T]
                next_states = next_states[:T]
                dones       = dones[:T]
                rewards     = rewards[:T]

                self.agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

                reward_metrics = {
                    "avg_reward": float(torch.mean(rewards)),
                    "obj1_f1_minority_beta": float(diagnostics["global_reward"]),
                    "obj2_local_useful_mean": float(diagnostics["local_reward"]),
                    "macro_f1_beta": float(diagnostics["f1_macro_beta"]),
                }

                new_reward_metrics = {
                    "judge_conf_mean": float(diagnostics.get("judge_conf_mean", float('nan'))),
                    "uncert_alpha_mean": float(diagnostics.get("uncert_alpha_mean", float('nan'))),
                    "alpha_wrong_rate": float(diagnostics.get("alpha_wrong_rate", float('nan'))),
                    "corr_factor_mean": float(0.5 + 0.5 * diagnostics.get("alpha_wrong_rate", 0.0)),
                    "f1_minority_alpha": float(diagnostics["f1_minority_alpha"]),
                    "f1_minority_beta_stale": float(diagnostics["f1_minority_beta_stale"]),
                    "local_cap_frac": float(diagnostics["local_cap_frac"]), 
                }

                # Pull episode-level values from diagnostics
                local_mean = float(diagnostics["local_reward"])     
                delta_val  = float(diagnostics["delta_f1_val"])     

                # Update rolling buffers and compute correlation
                self._local_buf.append(local_mean)
                self._delta_buf.append(delta_val)
                corr_local_delta = self._corr_local_delta()

                alignment_metrics = {
                    "delta_f1_val": delta_val,
                    "corr_local_delta": float(corr_local_delta),
                }
                
                # Log
                avg_reward = torch.mean(rewards)
                self.tracker.log_episode(
                    episode + 1,
                    reward_metrics,                     # Avg reward across trajectory
                    new_reward_metrics,         # Mean (1 - Brier) on synthetic Φ (β)
                    alignment_metrics           # macro F1 on θ_val (β), for context
                )
                # Save periodic snapshot and best-so-far synthetic
                self.tracker.maybe_save_synthetic(
                    episode_num=episode + 1,
                    x_syn=x_phi_t,
                    y_syn=y_phi_t,
                    avg_reward=float(avg_reward),
                    obj1=float(diagnostics['global_reward']),
                    obj2_mean=float(diagnostics['local_reward']),
                    global_f1=float(diagnostics['f1_macro_beta']),
                    feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                    beta_model=self.beta_model               
                )

        self.tracker.log_final_test(
            alpha_model=self.alpha_model,
            x_test=x_theta_test,
            y_test=y_theta_test,
            f1_thresh=0.5,                       # or a θ_val-fixed threshold
            x_train=x_theta_train, y_train=y_theta_train,

            # Existing jitter baseline (β)
            jitter_n=self.traj_length,                # match per-episode synthetic count
            jitter_scale=0.20,                   # 0.10–0.30 typical

            # (A) α_raw_original on ORIGINAL TRAIN (unbiased) mapped to θ
            run_alpha_raw_original=True,

            # (B) α_same+REAL: add real minority from ORIGINAL TRAIN pool
            run_alpha_plus_real=True,
            alpha_plus_real_n=self.traj_length,       # keep parity with jitter/ctgan counts

            # (C) Alpha+CTGAN: add conditional CTGAN minority (trained in raw space)
            run_alpha_plus_ctgan=False,
            alpha_plus_ctgan_n=self.traj_length,      # same augmentation count for fairness
            ctgan_epochs=300,                    # bump if you want more stable synth
            cap_ctgan_train=None,                # or an int to cap CTGAN training rows

            # Dataset settings for rebuilding the original pool
            data_path=self.dataset.data_path,
            bias_pct=self.bias_pct,
            val_frac=0.20,
            test_frac=0.20,
            train_size=self.real_data_size                    
        )


        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")


import re

def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', s).strip('-')

def run_one(dataset_name: str, device_str: str, seed: int, reward_mode: str,
            multiclass: bool, majority_id: int, minority_id: int,
            third_id: int | None, bias_pct: float, exp_group: str,
            pca_components: int, traj_length: int, real_data_size: int):
    print(f"\n=== RUN start | group={exp_group} | dev={device_str} | seed={seed} | mode={reward_mode} "
          f"| multi={multiclass} | maj={majority_id} | min={minority_id} | third={third_id} | "
          f"bias={bias_pct} | PCA={pca_components} | T={traj_length} | R={real_data_size} | ds={dataset_name} ===")

    trainer = Training(
        exp_group=exp_group,
        dataset_name=dataset_name,
        seed=seed,
        device=device_str,
        reward_mode=reward_mode,
        multiclass=multiclass,
        majority_id=majority_id,
        minority_id=minority_id,
        third_id=third_id,
        bias_pct=bias_pct,
        pca_components=pca_components,
        traj_length=traj_length,
        real_data_size=real_data_size,
    )
    trainer()

    if torch.cuda.is_available() and 'cuda' in device_str:
        torch.cuda.synchronize(torch.device(device_str))
        torch.cuda.empty_cache()
    time.sleep(2)


if __name__ == "__main__":
    base_group = datetime.now().strftime("%Y%m%d%H%M%S")
    devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    # Requested updates
    seeds = [42, 123, 999, 2020, 5555]
    reward_mode = "local_gauss"

    pca_list = [2, 4, 6, 8]
    bias_list = [0.25]  # keep as before unless you want a different bias grid
    TRAJ_LENGTH = 2000
    REAL_DATA_SIZE = 3000

    # Sensor dataset only (activity IDs apply here)
    sensor_pairs = [(4, 13), (12, 13)]

    exps = []
    for pca in pca_list:
        for bias in bias_list:
            for (min_id, maj_id) in sensor_pairs:
                name = f"pamap2_PCA{pca}_Bias{bias}_Min{min_id}_Maj{maj_id}_T{TRAJ_LENGTH}_R{REAL_DATA_SIZE}"
                exps.append(dict(
                    name=name,
                    dataset_name="pamap2",       # <-- adjust to your Dataset class key if different
                    multiclass=False,
                    minority_id=min_id,
                    majority_id=maj_id,
                    third_id=None,
                    bias_pct=bias,
                    pca_components=pca,
                    traj_length=TRAJ_LENGTH,
                    real_data_size=REAL_DATA_SIZE,
                ))

    # Dispatcher
    for i, spec in enumerate(exps):
        for seed in seeds:
            dev = devices[i % len(devices)]
            exp_group = f"{base_group}__{_slug(spec['name'])}"
            print(f"\n[dispatcher] launching {spec['name']} (seed={seed}) | group={exp_group} on {dev}")

            run_one(
                dataset_name=spec["dataset_name"],
                device_str=dev,
                seed=seed,
                reward_mode=reward_mode,
                multiclass=spec["multiclass"],
                majority_id=spec["majority_id"],
                minority_id=spec["minority_id"],
                third_id=spec["third_id"],
                bias_pct=spec["bias_pct"],
                exp_group=exp_group,
                pca_components=spec["pca_components"],
                traj_length=spec["traj_length"],
                real_data_size=spec["real_data_size"],
            )

    print("\n[dispatcher] all sequential runs complete.")



