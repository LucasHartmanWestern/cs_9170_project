# --- Imports ---

# Standard library
import os
import time

# Third-party: numpy, pandas, sklearn, torch
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
from agents.reinforce_agent import ReinforceAgent
# from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker
import multiprocessing as mp


# ---------------- Config ----------------
EPISODES        = 1000
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
    def __init__(self, lambda_schedule=(0.8, 0.8), tau=0.03, init=0.0, w=0.0, w_max_step=0.05, w_margin=0.01, w_k=6.0, w_burnin=0.15, use_new_reward: bool = True, seed=42, device='cpu'):
        self.seed = seed
        self.device = torch.device(device)  # honor argument
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.pca_components = 2
        self.use_new_reward = use_new_reward
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

        self._corr_window = 40   # ~13% of 300 eps; tweak if you like
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

    # ---------------- Data prep ----------------
    def split_dataset(self, train_size=None, bias_pct=0.2,
                    val_frac=0.20, test_frac=0.20):
        """
        Loads and processes the Adult Census dataset, applies bias to the minority class,
        and returns PCA-transformed train/val/test splits as torch tensors.

        PIPELINE:
        1) Load raw data.
        2) Split raw data into θ_train / θ_val / θ_test (stratified by the original labels).
        3) Within each split, apply the *same bias* by downsampling the minority class:
            - Keep all majority samples.
            - Keep only (1 - bias_pct) of minority samples (e.g., bias_pct=0.75 => keep 25% of minority).
            (We avoid any pre-split upsampling to prevent duplicates across splits.)
        4) Fit OneHotEncoder + StandardScaler on θ_train *only*; transform θ_val and θ_test with those fitted transformers (no leakage).
        5) Fit PCA on θ_train *only*; transform θ_val and θ_test with that PCA (no leakage).
        6) Optionally subsample θ_train to a fixed size *after* biasing and before fitting PCA (still no leakage).
        7) Return torch tensors for train/val/test on self.device.

        Returns:
        X_train_theta, X_val_theta, X_test_theta,
        y_train_theta, y_val_theta, y_test_theta
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."

        # 1) Load raw data
        data_path = "census+income/adult.data"
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        X_df_raw = pd.read_csv(data_path, header=None, names=column_names, na_values="?", skipinitialspace=True)
        y_raw = np.where(X_df_raw["income"].isin(['>50K', '>50K.']), 1, 0).astype(int)
        X_df_raw = X_df_raw.drop(columns=["income"])

        # Identify column types once (consistent across splits)
        cat_cols = [c for c in X_df_raw.columns if X_df_raw[c].dtype.name in ['category', 'object', 'bool']]
        num_cols = [c for c in X_df_raw.columns if np.issubdtype(X_df_raw[c].dtype, np.number)]

        # 2) Split raw into θ_train / θ_temp, then θ_val / θ_test (stratified)
        X_train_df, X_temp_df, y_train, y_temp = train_test_split(
            X_df_raw, y_raw, test_size=(val_frac + test_frac),
            random_state=self.seed, stratify=y_raw
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test = train_test_split(
            X_temp_df, y_temp, test_size=rel_test,
            random_state=self.seed, stratify=y_temp
        )

        # Helper: apply the SAME bias inside a split (keep all majority; keep fraction of minority)
        def apply_bias(df_split, y_split, target_minority_pct):
            df = df_split.copy()
            df["__y__"] = y_split
            df_major = df[df["__y__"] == 0]
            df_minor = df[df["__y__"] == 1]

            n_major = len(df_major)
            n_minor = len(df_minor)

            if n_minor == 0 or n_major == 0:
                # Edge case: one class missing
                df_biased = df
            else:
                # compute how many minority to keep to hit target proportion
                keep_minority = int(np.floor((target_minority_pct * n_major) / (1 - target_minority_pct)))

                # cap to available samples
                keep_minority = min(n_minor, max(1, keep_minority))

                df_minor_biased = df_minor.sample(n=keep_minority, random_state=self.seed, replace=False)
                df_biased = pd.concat([df_major, df_minor_biased], axis=0) \
                            .sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            y_out = df_biased["__y__"].to_numpy(dtype=int)
            X_out = df_biased.drop(columns=["__y__"])
            return X_out, y_out

        # 3) Apply same bias in each split (no cross-split duplication, distributions aligned)
        target_minority_pct = bias_pct  # e.g., bias_pct=0.25 -> 25% of dataset is minority data
        X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, target_minority_pct)
        X_val_biased_df,   y_val_biased   = apply_bias(X_val_df,   y_val,   target_minority_pct)
        X_test_biased_df,  y_test_biased  = apply_bias(X_test_df,  y_test,  target_minority_pct)

        # Optional: subsample θ_train to fixed size *after* biasing (stratified)
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _ = train_test_split(
                X_train_biased_df, y_train_biased,
                train_size=train_size, random_state=self.seed, stratify=y_train_biased
            )

        # 4) Fit encoder + scaler on θ_train only; transform val/test (no leakage)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler  = StandardScaler()

        # Fit on train
        X_train_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_all = np.hstack([X_train_num, X_train_cat])

        # Transform val/test with the *fitted* encoder/scaler
        X_val_cat  = encoder.transform(X_val_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_num  = scaler.transform(X_val_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_all  = np.hstack([X_val_num, X_val_cat])

        X_test_cat = encoder.transform(X_test_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_num = scaler.transform(X_test_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_all = np.hstack([X_test_num, X_test_cat])

        # 5) Fit PCA on θ_train only; transform val/test (no leakage)
        pca = PCA(n_components=self.pca_components)
        X_train_pca = pca.fit_transform(X_train_all)
        X_val_pca   = pca.transform(X_val_all)
        X_test_pca  = pca.transform(X_test_all)

        # 6) Convert to torch tensors on device
        X_train_theta = torch.tensor(X_train_pca, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_pca,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_pca,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

        # ---- Sanity check logging ----
        def log_distribution(name, y_split):
            n_total = len(y_split)
            n_min = np.sum(y_split == 1)
            pct_min = 100.0 * n_min / n_total if n_total > 0 else 0.0
            print(f"[{name}] size={n_total}, minority={n_min} ({pct_min:.2f}%)")

        log_distribution("TRAIN", y_train_biased)
        log_distribution("VAL",   y_val_biased)
        log_distribution("TEST",  y_test_biased)
        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


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
        w_max_step: float = 0.05        # max |Δw| per episode
    ):
        # --- λ schedule (unchanged) ---
        lambda_start, lambda_end = lambda_schedule
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress

        with torch.no_grad():
            # ---- Global term on θ_val (unchanged) ----
            p1_alpha_val      = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val       = self._p1_from_agent(beta_model,  x_theta_val)
            p1_stale_beta_val = self._p1_from_agent(stale_beta_model, x_theta_val)

            f1_minority_alpha = self.f1_from_probs(y_theta_val, p1_alpha_val, f1_thresh)
            f1_minority_beta  = self.f1_from_probs(y_theta_val, p1_beta_val,  f1_thresh)
            f1_minority_beta_stale = self.f1_from_probs(y_theta_val, p1_stale_beta_val, f1_thresh)

            # macro-F1 for logging
            f1_neg_beta   = self.f1_from_probs(1 - y_theta_val, 1 - p1_beta_val, 1 - f1_thresh)
            f1_macro_beta = 0.5 * (f1_minority_beta + f1_neg_beta)

            delta_f1_val = (f1_minority_beta - f1_minority_alpha)

            # ---- Judge gating (EMA) (unchanged) ----
            # margin > 0 means stale-β is a better judge than α on θ_val
            margin_sa  = float(f1_minority_beta_stale - f1_minority_alpha)
            ema_val = ema.update(margin_sa)
            m       = ema_val - w_margin
            w_target = 1.0 / (1.0 + exp(-w_k * m))

            if progress < burnin_frac:
                w_target = 0.0

            delta = max(-w_max_step, min(w_max_step, w_target - w_current))
            new_w = float(min(1.0, max(0.0, w_current + delta)))

            # ---- Local term on synthetic (REDESIGNED) ----
            # Base probabilities
            p1_phi_alpha = self._p1_from_agent(alpha_model,      x_phi)   # [T]
            p1_phi_stale = self._p1_from_agent(stale_beta_model, x_phi)   # [T]
            p1_phi_blend = new_w * p1_phi_stale + (1.0 - new_w) * p1_phi_alpha

            # (a) Judge confidence/correctness: 1 - Brier
            brier_t    = self.brier_per_sample(y_phi, p1_phi_blend)       # [T]
            judge_conf = 1.0 - brier_t                                    # [T] in [0,1]

            if self.use_new_reward:
                # ================= NEW LOCAL REWARD: Boundary Helpfulness =================
                # y_phi == 1 for synthetic tuples; "wrong" means alpha predicts majority.

                # ---- Banding by Alpha's distance to boundary (0.5) ----
                margin = torch.abs(p1_phi_alpha - 0.5)   # [T]
                tau_boundary = 0.05
                tau_shoulder = 0.15

                near_mask     = (margin <= tau_boundary)
                shoulder_mask = (margin >  tau_boundary) & (margin <= tau_shoulder)
                far_mask      = (margin >  tau_shoulder)

                # ---- Realism gate with band-aware floor ----
                floor_near, floor_shoulder, floor_far = 0.35, 0.20, 0.05
                floor_per = torch.where(
                    near_mask, torch.tensor(floor_near, device=margin.device),
                    torch.where(shoulder_mask, torch.tensor(floor_shoulder, device=margin.device),
                                torch.tensor(floor_far, device=margin.device))
                )
                realism_gate = torch.maximum(judge_conf, floor_per)  # baseline, in [0,1]

                # ---- Boundary usefulness weights (driver) ----
                alpha_wrong = (p1_phi_alpha < 0.5).float()  # y=1 synthetic

                w_near_wrong, w_near_right         = 1.00, 0.40
                w_shoulder_wrong, w_shoulder_right = 0.70, 0.20
                w_far_any                           = 0.05

                weight_near = near_mask.float() * (alpha_wrong * w_near_wrong + (1.0 - alpha_wrong) * w_near_right)
                weight_shou = shoulder_mask.float() * (alpha_wrong * w_shoulder_wrong + (1.0 - alpha_wrong) * w_shoulder_right)
                weight_far  = far_mask.float() * w_far_any
                boundary_usefulness = (weight_near + weight_shou + weight_far).clamp(0.0, 1.0)

                # ---- New local score: additive bonus on top of judge_conf ----
                k_bonus = 0.6  # try 0.4–0.8 in small sweeps
                score_local_raw = (realism_gate + k_bonus * (1.0 - realism_gate) * boundary_usefulness).clamp(0.0, 1.0)

                LOCAL_CAP = 0.80
                cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
                over_mask = (score_local_raw > cap_t).float()
                score_local = torch.minimum(score_local_raw, cap_t)

                # Diagnostics (keep names your tracker expects)
                uncert_alpha = 1.0 - (2.0 * margin).clamp(0, 1)   # boundary proximity
                corr_factor  = boundary_usefulness                 # report the boundary weight
            else:
                uncert_alpha = torch.zeros_like(judge_conf)  # for diagnostics consistency
                alpha_wrong  = torch.zeros_like(judge_conf)
                corr_factor  = torch.ones_like(judge_conf) * 0.5
                score_local  = judge_conf

        # ---- Combine global + local (unchanged) ----
        reward = lambda_t * delta_f1_val + (1.0 - lambda_t) * score_local  # [T]
        mean_local = float(score_local.mean().item())
        local_cap_frac = float(over_mask.mean().item()) 

        diagnostics = {
            # Reward metrics
            "global_reward": f1_minority_beta,
            "local_reward": mean_local,
            "f1_macro_beta": f1_macro_beta,#For Human reading

            #EMA metrics
            "margin_stale_minus_alpha": margin_sa,
            "ema_margin": ema_val,
            "w_target": w_target,
            "w_used": new_w,
            "lambda_t": lambda_t,

            # new reward specific metrics
            "judge_conf_mean": float(judge_conf.mean()),
            "uncert_alpha_mean": float(uncert_alpha.mean()),
            "alpha_wrong_rate": float(alpha_wrong.mean()),
            "delta_f1_val": float(delta_f1_val),            # β − α on θ_val (per-episode scalar)
            "f1_minority_alpha": float(f1_minority_alpha),  # α's F1 on θ_val
            "f1_minority_beta_stale": float(f1_minority_beta_stale),  # stale-β's F1 on θ_val
            "local_cap_frac": local_cap_frac,

            # explicit corrective multiplier avg (can be derived, but nice to log)
            "corr_factor_mean": float(corr_factor.mean()),
        }

        return reward, diagnostics


    # ---------------- Training loop ----------------
    def __call__(self):
        start_time = time.time()

        run_stats = {
            "EPISODES": EPISODES,
            "TRAJ_LENGTH": TRAJ_LENGTH,
            "REAL_DATA_SIZE": REAL_DATA_SIZE,
            "BIAS_PCT": BIAS_PCT,
            "lambda_schedule": self.lambda_schedule,
            "seed": self.seed,                      # <-- important!
            "pca_components": self.pca_components,
            "use_new_reward": self.use_new_reward,
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
            save_dir="training_runs",
            compare_metric="average_reward",
            beta_factory=beta_factory
        ) as tracker:
            self.tracker = tracker  

        # Data
        x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = self.split_dataset(
            train_size=REAL_DATA_SIZE, bias_pct=BIAS_PCT
        )

        total_data = len(x_theta_train) + TRAJ_LENGTH
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (TRAJ_LENGTH / total_data) * 100
        print(f"Beta will train with: {real_percentage:.2f}% real data, {synthetic_percentage:.2f}% synthetic data: ")

        # Train α once on real-only train set
        self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)

        # Environment (state is 2D PCA coords; target is 1 class for synthetic generation)
        env = Environment(
            target=1,
            max_actions=TRAJ_LENGTH,
            device=self.device,
            seed=self.seed
        )

        for episode in range(EPISODES):
            # Pre-allocate (GPU tensors)
            D = x_theta_train.shape[1]  # equals pca_components
            states      = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            actions     = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            next_states = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            dones       = torch.zeros(TRAJ_LENGTH, dtype=torch.bool, device=self.device)

            x_syn_tensor = torch.zeros((TRAJ_LENGTH, D), dtype=torch.float32, device=self.device)
            y_syn_tensor = torch.zeros(TRAJ_LENGTH, dtype=torch.long, device=self.device)

            state = env.reset()

            # Reset beta to its initial snapshot each episode for a stable baseline
            stale_beta_model = copy.deepcopy(self.beta_model)
            self.beta_model.reset()

            for t in range(TRAJ_LENGTH):
                action = self.agent.predict(state)                      # [2]
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
            #print(f"[Judge] w={diagnostics['w_used']:.3f} ema={diagnostics['ema_margin']:.4f} margin={diagnostics['margin_stale_minus_alpha']:.4f}")

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
            f1_thresh=0.5,  # or a θ_val-fixed threshold
            x_train=x_theta_train, y_train=y_theta_train,
            jitter_n=TRAJ_LENGTH,                 # same count as your per-episode synthetic
            jitter_scale=0.20                     # tweak as desired (0.1–0.3 is typical)
            )


        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")


def run_one(device_str, w_burnin_val, seed=42, tau=0.1):
    print(f"\n=== Running on {device_str} with w_burnin={w_burnin_val} ===")
    train = Training(
        seed=seed,
        use_new_reward=True,
        tau=tau,
        device=device_str,
        w_burnin=w_burnin_val,   # <- EMA-only switch for alpha-only when >1
    )
    train()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # safe for CUDA

    seed = 42
    tau  = 0.1

    procs = []
    #procs.append(mp.Process(target=run_one, args=("cuda:0", 0.15, seed, tau)))  # CAP ONLY (blend)
    procs.append(mp.Process(target=run_one, args=("cuda:1", 2.0, seed, tau)))   # CAP + ALPHA-ONLY

    for p in procs: p.start()
    for p in procs: p.join()

