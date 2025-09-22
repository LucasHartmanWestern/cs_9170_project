# --- Imports ---

# Standard library
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
EPISODES        = 1
TRAJ_LENGTH     = 2000
REAL_DATA_SIZE  = 3000
BIAS_PCT        = 0.35 #Final percentage of minority data in the dataset

class EMATracker:
    def __init__(self, tau=0.07, init=0.0):
        self.tau = float(tau)
        self.value = float(init)
        self.initialized = False

    def update(self, x: float):
        if not self.initialized:
            self.value = float(x)
            self.initialized = True
        else:
            self.value = (1.0 - self.tau) * self.value + self.tau * float(x)
        return self.value

    def get(self) -> float:
        return float(self.value)


class Training:
    def __init__(self, exp_group=None, dataset_name="census_income", lambda_schedule=(0.8, 0.8), tau=0.03, init=0.0, w=0.0, w_max_step=0.05, w_margin=0.01, w_k=6.0, w_burnin=0.15, reward_mode="gauss_penalty", seed=42, device='cpu'):
        self.exp_group=exp_group
        self.seed = seed
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.pca_components = 6
        self.reward_mode = reward_mode
        self.lambda_schedule = (0.8, 0.8)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # inside Training.__init__
        self.ema = EMATracker(tau=tau, init=init)  # tau=update rate(High trusts newest episodes more, Low is more cautious)
        self.w = w                                # current judge weight (Starts at 0, 0=all alpha, 1=all staleβ)
        self.w_max_step = w_max_step              # clamp |Δw| per episode (Max % weight change per episode)
        self.w_margin = w_margin                  # minimum EMA margin required before trusting stale-beta. (Deadzone near zero)
        self.w_k = w_k                            # steepness of sigmoid that maps EMA to current weight (Higher more abrupt, lower more gradual)
        self.w_burnin = w_burnin                  # % of episodes to force alpha only judge (Prevents an early noisy beta model being used)

        self.dataset = Dataset(dataset_name, self.seed, self.device)

        # FFNN (alpha/beta) config
        self.ffnn_config = {
            'input_size': self.pca_components,
            'hidden_sizes': [32, 16],
            'output_size': 2,               # logits for 2 classes
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'type': 'classification',
            'classes': [0, 1],
            'device': self.device,
            'seed': self.seed
        }

        # REINFORCE config
        reinforce_config = {
            'state_size': 2,
            'action_size': self.pca_components,
            'hidden_sizes': [64, 64],
            'total_episodes': EPISODES,
            'lr': 3e-4,
            'gamma': 0.99,
            'entropy_start': 1e-2,
            'entropy_end': 0.0,
            'seed': self.seed,
            'device': self.device
        }

        self.dl_generator = torch.Generator(device='cpu').manual_seed(self.seed)

        # Agents
        # self.agent = PPOAgent(**ppo_config)
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

    #Beta model F1 minority class score + brier score (MSE) of judge on synthetic data. Judge determined by higher brier score on real test set
    def compute_reward(
        self,
        alpha_model, beta_model, stale_beta_model,
        x_theta_val, y_theta_val,
        x_phi, y_phi,
        ema,                 # EMATracker instance
        w_current: float,    # current judge weight in [0,1]
        progress: float,     # (episode+1)/EPISODES in [0,1]
        f1_thresh: float = 0.5,

        # schedules / gates
        lambda_schedule=(0.30, 0.95),   # (start, end) across the run
        burnin_frac: float = 0.15,      # fraction of run to force alpha-only judge
        w_k: float = 10.0,              # sigmoid steepness
        w_margin: float = 0.01,         # margin needed before trusting stale-β
        w_max_step: float = 0.05,       # max |Δw| per episode

        # hinge-penalty for majority/weighted scores (used only in gauss_penalty)
        epsilon_majority: float = 0.01,   # allow up to -1.0 pp on majority F1
        epsilon_weighted: float = 0.005,  # allow up to -0.5 pp on weighted F1
        c_majority: float = 0.30,         # penalty weight for majority violation
        c_weighted: float = 0.30,         # penalty weight for weighted violation
    ):
        """
        reward_mode:
        - "gauss_penalty": Gaussian shaping (alpha-centric) + hinge penalties
        - "gauss_nopen":   Gaussian shaping (alpha-centric), no penalties
        - "judge_conf":    judge confidence = 1 - brier (higher is better), no penalties
        """
        # ---------------- Mode selection (back-compat) ----------------
        if self.reward_mode is None:
            self.reward_mode = "gauss_penalty" if getattr(self, "use_new_reward", False) else "judge_conf"
        valid_modes = {"gauss_penalty", "gauss_nopen", "judge_conf"}
        if self.reward_mode not in valid_modes:
            raise ValueError(f"reward_mode must be one of {valid_modes}, got {self.reward_mode!r}")

        # --- λ schedule ---
        lambda_start, lambda_end = lambda_schedule
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress

        with torch.no_grad():
            # ---- Global term on θ_val ----
            p1_alpha_val      = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val       = self._p1_from_agent(beta_model,  x_theta_val)
            p1_stale_beta_val = self._p1_from_agent(stale_beta_model, x_theta_val)

            # Minority / majority / macro
            f1_minority_alpha      = self.f1_from_probs(y_theta_val, p1_alpha_val, f1_thresh)
            f1_minority_beta       = self.f1_from_probs(y_theta_val, p1_beta_val,  f1_thresh)
            f1_minority_beta_stale = self.f1_from_probs(y_theta_val, p1_stale_beta_val, f1_thresh)

            f1_majority_alpha = self.f1_from_probs(1 - y_theta_val, 1 - p1_alpha_val, 1 - f1_thresh)
            f1_majority_beta  = self.f1_from_probs(1 - y_theta_val, 1 - p1_beta_val,  1 - f1_thresh)

            f1_macro_beta = 0.5 * (f1_minority_beta + f1_majority_beta)

            # Weighted F1 (support-weighted)
            pos_frac = float(y_theta_val.float().mean().item())
            neg_frac = 1.0 - pos_frac
            f1_weighted_alpha = pos_frac * float(f1_minority_alpha) + neg_frac * float(f1_majority_alpha)
            f1_weighted_beta  = pos_frac * float(f1_minority_beta)  + neg_frac * float(f1_majority_beta)

            # Deltas
            delta_f1_minority = float(f1_minority_beta - f1_minority_alpha)
            delta_f1_majority = float(f1_majority_beta - f1_majority_alpha)
            delta_f1_weighted = float(f1_weighted_beta - f1_weighted_alpha)

            # ---- Judge gating (EMA) ----
            margin_sa  = float(f1_minority_beta_stale - f1_minority_alpha)
            ema_val    = ema.update(margin_sa)
            m_gate     = ema_val - w_margin
            w_target   = 1.0 / (1.0 + exp(-w_k * m_gate))
            if progress < burnin_frac:
                w_target = 0.0
            delta      = max(-w_max_step, min(w_max_step, w_target - w_current))
            new_w      = float(min(1.0, max(0.0, w_current + delta)))

            # ---- Local term on synthetic Φ ----
            p1_phi_alpha = self._p1_from_agent(alpha_model,      x_phi)   # [T]
            p1_phi_stale = self._p1_from_agent(stale_beta_model, x_phi)   # [T]
            p1_phi_blend = new_w * p1_phi_stale + (1.0 - new_w) * p1_phi_alpha

            # Base confidence: 1 - Brier (higher is better)
            brier_t    = self.brier_per_sample(y_phi, p1_phi_blend)       # [T]
            judge_conf = 1.0 - brier_t

            # Defaults for diagnostics (filled per-mode)
            uncert_alpha   = torch.zeros_like(judge_conf)
            alpha_wrong    = torch.zeros_like(judge_conf)
            local_cap_frac = 0.0

            if self.reward_mode.startswith("gauss"):
                # Alpha-centric Gaussian shaping around the 0.5 decision boundary
                p = p1_phi_alpha
                m = torch.abs(p - 0.5)
                d = 0.5 - p
                alpha_pred  = (p >= 0.5).float()
                alpha_wrong = (alpha_pred != y_phi.to(p.device).float()).float()

                # Gaussian parameters
                mu_wrong, mu_right = 0.03, 0.00
                tau_wrong, tau_right = 0.10, 0.06
                w_wrong,  w_right  = 1.0, 0.4

                g_wrong = torch.exp(-0.5 * ((d - mu_wrong)/tau_wrong)**2)
                g_right = torch.exp(-0.5 * ((d - mu_right)/tau_right)**2)
                score_local_raw = alpha_wrong * (w_wrong * g_wrong) + (1.0 - alpha_wrong) * (w_right * g_right)

                # Cap
                LOCAL_CAP = 1.00
                cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
                over_mask = (score_local_raw > cap_t).float()
                local_cap_frac = float(over_mask.mean().item())
                score_local = torch.minimum(score_local_raw, cap_t)

                # Diagnostics
                judge_conf   = score_local_raw
                uncert_alpha = 1.0 - (2.0 * m).clamp(0, 1)
            else:
                # "judge_conf" mode: just use confidence (no shaping, no cap)
                score_local = judge_conf

        # ---- Combine global + local ----
        base_reward = lambda_t * delta_f1_minority + (1.0 - lambda_t) * score_local  # [T]

        # Penalty only in "gauss_penalty"
        maj_violation = 0.0
        wtd_violation = 0.0
        penalty = 0.0
        if self.reward_mode == "gauss_penalty":
            maj_violation = max(0.0, -(delta_f1_majority + epsilon_majority))
            wtd_violation = max(0.0, -(delta_f1_weighted + epsilon_weighted))
            penalty = c_majority * maj_violation + c_weighted * wtd_violation

        reward = base_reward - penalty
        mean_local = float(score_local.mean().item())

        diagnostics = {
            # Reward metrics
            "reward_mode": self.reward_mode,
            "global_reward": float(f1_minority_beta),
            "local_reward": mean_local,
            "f1_macro_beta": float(f1_macro_beta),

            # EMA / gating
            "margin_stale_minus_alpha": margin_sa,
            "ema_margin": ema_val,
            "w_target": w_target,
            "w_used": new_w,
            "lambda_t": lambda_t,

            # Details for analysis
            "judge_conf_mean": float(judge_conf.mean()),
            "uncert_alpha_mean": float(uncert_alpha.mean()),
            "alpha_wrong_rate": float(alpha_wrong.mean()),
            "delta_f1_val": delta_f1_minority,
            "delta_f1_majority": delta_f1_majority,
            "delta_f1_weighted": delta_f1_weighted,
            "f1_minority_alpha": float(f1_minority_alpha),
            "f1_minority_beta_stale": float(f1_minority_beta_stale),
            "local_cap_frac": local_cap_frac,

            # Penalty diagnostics (zero unless gauss_penalty)
            "epsilon_majority": epsilon_majority,
            "epsilon_weighted": epsilon_weighted,
            "penalty": penalty,
            "maj_violation": maj_violation,
            "wtd_violation": wtd_violation,

            # legacy placeholder (not used here; keep key if your logging expects it)
            "corr_factor_mean": None,
        }
        return reward, diagnostics

    # ---------------- Training loop ----------------
    def __call__(self):
        start_time = time.time()

        run_stats = {
            "EXP_GROUP": self.exp_group,
            "EPISODES": EPISODES,
            "TRAJ_LENGTH": TRAJ_LENGTH,
            "REAL_DATA_SIZE": REAL_DATA_SIZE,
            "BIAS_PCT": BIAS_PCT,
            "lambda_schedule": self.lambda_schedule,
            "seed": self.seed,
            "pca_components": self.pca_components,
            "reward_mode": self.reward_mode,     
            "ema_tau": self.ema.tau,
            "w_max_step": self.w_max_step,
            "w_margin": self.w_margin,
            "w_k": self.w_k,
            "w_burnin": self.w_burnin,
        }


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
            train_size=REAL_DATA_SIZE, bias_pct=BIAS_PCT, pca_components=self.pca_components
        )

        total_data = len(x_theta_train) + TRAJ_LENGTH
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (TRAJ_LENGTH / total_data) * 100
        print(f"Beta will train with: {real_percentage:.2f}% real data, {synthetic_percentage:.2f}% synthetic data: ")

        # Train α once on real-only train set
        self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)
        self.tracker.save_alpha_state_dict(self.alpha_model, self.ffnn_config, self.pca_components)

        # Environment (state is 2D PCA coords; target is 1 class for synthetic generation)
        env = Environment(
            target=1,
            max_actions=TRAJ_LENGTH,
            device=self.device,
            seed=self.seed
        )

        for episode in range(EPISODES):
            # Pre-allocate (GPU tensors)
            D = 2
            A = self.pca_components
            states      = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            actions     = torch.zeros((TRAJ_LENGTH, A), dtype=torch.float32, device=self.device)
            next_states = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            dones       = torch.zeros(TRAJ_LENGTH, dtype=torch.bool, device=self.device)

            x_syn_tensor = torch.zeros((TRAJ_LENGTH, A), dtype=torch.float32, device=self.device)
            y_syn_tensor = torch.zeros(TRAJ_LENGTH, dtype=torch.long, device=self.device)

            state = env.reset()

            # Reset beta to its initial snapshot each episode for a stable baseline
            stale_beta_model = copy.deepcopy(self.beta_model)
            self.beta_model.reset()

            for t in range(TRAJ_LENGTH):
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

            T = t + 1 if done else TRAJ_LENGTH
            x_phi_t = x_syn_tensor[:T]
            y_phi_t = y_syn_tensor[:T]

            # Train beta on hybrid (real + synthetic for this episode)
            x_hybrid = torch.cat([x_theta_train, x_phi_t])
            y_hybrid = torch.cat([y_theta_train, y_phi_t])
            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

            progress = (episode + 1) / EPISODES

            # Rewards (alpha baseline global + per-step local)
            rewards, diagnostics = self.compute_reward(
                self.alpha_model, self.beta_model, stale_beta_model,
                x_theta_val, y_theta_val,
                x_phi_t, y_phi_t,
                ema=self.ema,
                w_current=self.w,
                progress=progress,
                f1_thresh=0.5,
                lambda_schedule=self.lambda_schedule,
                burnin_frac=self.w_burnin,
                w_k=self.w_k,
                w_margin=self.w_margin,
                w_max_step=self.w_max_step,
            )
            self.w = float(diagnostics["w_used"])

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

            ema_metrics = {
                "margin_stale_minus_alpha": diagnostics.get("margin_stale_minus_alpha", float('nan')),
                "ema_margin": float(diagnostics.get("ema_margin", float('nan'))),
                "w_used": float(diagnostics.get("w_used", float('nan'))),
                "w_target": float(diagnostics.get("w_target", float('nan'))),
                "lambda_t": float(diagnostics.get("lambda_t", float('nan'))),
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
            local_mean = float(diagnostics["local_reward"])     # score_local.mean()
            delta_val  = float(diagnostics["delta_f1_val"])     # F1_β − F1_α on θ_val

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
                ema_metrics,                    # F1 for positive class on θ_val (β)
                new_reward_metrics,         # Mean (1 - Brier) on synthetic Φ (β)
                alignment_metrics           # macro F1 on θ_val (β), for context
            )
            # Save periodic snapshot and best-so-far synthetic
            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_phi_t,
                y_syn=y_phi_t,
                avg_reward=float(avg_reward),
                obj1=float(diagnostics['global_reward']),                 # use compare_metric="obj1" if you want best-on-val F1
                obj2_mean=float(diagnostics['local_reward']),
                global_f1=float(diagnostics['f1_macro_beta']),
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                beta_model=self.beta_model               # <-- NEW: saves best β when is_best=True
            )


        self.tracker.log_final_test(
            alpha_model=self.alpha_model,
            x_test=x_theta_test,
            y_test=y_theta_test,
            f1_thresh=0.5,                       # or a θ_val-fixed threshold
            x_train=x_theta_train, y_train=y_theta_train,

            # Existing jitter baseline (β)
            jitter_n=TRAJ_LENGTH,                # match per-episode synthetic count
            jitter_scale=0.20,                   # 0.10–0.30 typical

            # (A) α_raw_original on ORIGINAL TRAIN (unbiased) mapped to θ
            run_alpha_raw_original=True,

            # (B) α_same+REAL: add real minority from ORIGINAL TRAIN pool
            run_alpha_plus_real=True,
            alpha_plus_real_n=TRAJ_LENGTH,       # keep parity with jitter/ctgan counts

            # (C) Alpha+CTGAN: add conditional CTGAN minority (trained in raw space)
            run_alpha_plus_ctgan=False,
            alpha_plus_ctgan_n=TRAJ_LENGTH,      # same augmentation count for fairness
            ctgan_epochs=300,                    # bump if you want more stable synth
            cap_ctgan_train=None,                # or an int to cap CTGAN training rows

            # Dataset settings for rebuilding the original pool
            data_path=self.dataset.data_path,
            bias_pct=BIAS_PCT,
            val_frac=0.20,
            test_frac=0.20,
            train_size=REAL_DATA_SIZE                    
        )


        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")


# ---------- SEQUENTIAL LAUNCHER ----------
def run_one(datase_name: str,
            device_str: str,
            seed: int,
            reward_mode: str,
            w_burnin: float,
            w_start: float,
            w_max_step: float,
            tau: float = 0.1):
    print(f"\n=== RUN start | dev={device_str} | seed={seed} | mode={reward_mode} | "
          f"w_burnin={w_burnin} | w_start={w_start} | w_max_step={w_max_step} ===")

    trainer = Training(
        exp_group=exp_group,
        dataset_name=dataset_name,
        seed=seed,
        device=device_str,
        reward_mode=reward_mode,
        tau=tau,
        w_burnin=w_burnin,   # >1 => alpha-only (early)
        w=w_start,           # 1.0 => beta-only if w_max_step=0
        w_max_step=w_max_step
    )
    trainer()

    if torch.cuda.is_available() and 'cuda' in device_str:
        torch.cuda.synchronize(torch.device(device_str))
        torch.cuda.empty_cache()
    time.sleep(2)  # small breather for IO/logs


if __name__ == "__main__":
    # device pool (rotate, but only one run at a time)
    exp_group = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_name = "pamap2"
    num_cuda = torch.cuda.device_count()
    if num_cuda >= 2:
        devices = ["cuda:0", "cuda:1"]
    elif num_cuda == 1:
        devices = ["cuda:0"]
    else:
        devices = ["cpu"]

    seeds = [42]
    tau   = 0.1

    # Three reward modes:
    # 1) Gaussian shaping + penalties
    # cfg_gauss_pen = dict(reward_mode="gauss_penalty", w_burnin=3.0, w_start=0.0, w_max_step=0.05)
    # 2) Gaussian shaping, no penalties
    cfg_gauss_np  = dict(reward_mode="gauss_nopen",   w_burnin=3.0, w_start=0.0, w_max_step=0.05)
    # 3) Judge confidence (1 - Brier), no penalties
    # cfg_jconf     = dict(reward_mode="judge_conf",    w_burnin=3.0, w_start=0.0, w_max_step=0.05)

    job_specs = []
    for s in seeds:
        # job_specs.append(("gauss_penalty", s, cfg_gauss_pen))
        job_specs.append(("gauss_nopen",   s, cfg_gauss_np))
        # job_specs.append(("judge_conf",    s, cfg_jconf))

    # Run iteratively, rotating devices
    for idx, (tag, seed, cfg) in enumerate(job_specs):
        dev = devices[idx % len(devices)]
        print(f"[dispatcher] launching {tag} | seed={seed} on {dev}")
        run_one(dataset_name, dev, seed, cfg["reward_mode"], cfg["w_burnin"], cfg["w_start"], cfg["w_max_step"], tau=tau)

    print("[dispatcher] all sequential runs complete.")
