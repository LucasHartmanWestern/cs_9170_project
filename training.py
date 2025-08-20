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

# Local modules
from env import Environment
from agents.reinforce_agent import ReinforceAgent
# from agents.ppo_agent import PPOAgent  # optional
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker

# ---------------- Config ----------------
EPISODES        = 300
TRAJ_LENGTH     = 2000
REAL_DATA_SIZE  = 3000
BIAS_PCT        = 0.8


class Training:
    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.pca_components = 2
        self.lambda_ = 0.8

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # PPO config (unused here)
        ppo_config = {
            'state_size': 2,
            'action_size': self.pca_components,
            'hidden_size': 64,
            'lr': 3e-4,
            'gamma': 0.9,
            'clip_epsilon': 0.2,
            'update_epochs': 10,
            'batch_size': 64,
            'c1': 0.5,
            'c2': 0.01,
            'device': self.device,
            'seed': self.seed
        }

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

    # ---------------- Data prep ----------------
    # ---------------- Data prep ----------------
    def split_dataset(self, train_size=None, bias_pct=0.75,
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
        def apply_bias(df_split, y_split, frac_keep_minority):
            df = df_split.copy()
            df["__y__"] = y_split
            df_major = df[df["__y__"] == 0]
            df_minor = df[df["__y__"] == 1]
            if len(df_minor) > 0:
                keep_n = max(1, int(np.floor(len(df_minor) * frac_keep_minority)))
                df_minor_biased = df_minor.sample(n=keep_n, random_state=self.seed, replace=False)
            else:
                df_minor_biased = df_minor  # empty; edge case
            df_biased = pd.concat([df_major, df_minor_biased], axis=0)\
                        .sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            y_out = df_biased["__y__"].to_numpy(dtype=int)
            X_out = df_biased.drop(columns=["__y__"])
            return X_out, y_out

        # 3) Apply same bias in each split (no cross-split duplication, distributions aligned)
        frac_keep = max(0.0, 1.0 - bias_pct)  # e.g., bias_pct=0.75 -> keep 0.25 of minority
        X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, frac_keep)
        X_val_biased_df,   y_val_biased   = apply_bias(X_val_df,   y_val,   frac_keep)
        X_test_biased_df,  y_test_biased  = apply_bias(X_test_df,  y_test,  frac_keep)

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

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def train_predictor_model(self, model, x_train, y_train):
        train_dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=self.dl_generator)
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
    def compute_reward(self, alpha_model, beta_model, stale_beta_model,
                       x_theta_test, x_theta_val, y_theta_val, y_theta_test,
                       x_phi, y_phi,
                       lambda_=0.99, f1_thresh=0.5):
        """
        r_t = λ * (F1_pos^β(θ_test) − F1_pos^α(θ_test)) + (1 − λ) * (1 − Brier_t^β on Φ)
        Returns:
          reward[T], f1_pos_beta (scalar), mean(1−Brier) over Φ (scalar), f1_macro_beta (scalar)
        Also returns: mean brier scores for stale_beta_model and alpha_model on y_theta_test
        """
        with torch.no_grad():

            # alpha baseline on theta_val
            p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
            f1_minority_alpha  = self.f1_from_probs(y_theta_val, p1_alpha_val, f1_thresh)
            brier_alpha_val = self.brier_per_sample(y_theta_val, p1_alpha_val).mean()

            # beta performance on theta_val
            p1_beta_val  = self._p1_from_agent(beta_model, x_theta_val)
            f1_minority_beta   = self.f1_from_probs(y_theta_val, p1_beta_val, f1_thresh)

            # stale_beta_model mean brier score on y_theta_val
            p1_stale_beta_val = self._p1_from_agent(stale_beta_model, x_theta_val)
            brier_stale_beta_val = self.brier_per_sample(y_theta_val, p1_stale_beta_val).mean()
            # macro-F1 for logging
            f1_neg_beta   = self.f1_from_probs(1 - y_theta_val, 1 - p1_beta_val, 1 - f1_thresh)
            f1_macro_beta = 0.5 * (f1_minority_beta + f1_neg_beta)

            # Use brier scores to select between stale_beta_model and alpha_model for cost
            if brier_stale_beta_val >= brier_alpha_val:
                judge = alpha_model
                print("Using alpha for cost (lower brier)")
            else:
                judge = stale_beta_model
                print("Using stale_beta for cost (lower brier)")
                
            # per-step local term on Φ (β probs)
            p1_phi_judge   = self._p1_from_agent(judge, x_phi)    # [T]
            brier_t       = self.brier_per_sample(y_phi, p1_phi_judge) # [T]
            score_local   = 1.0 - brier_t                             # [T]

        global_delta = (f1_minority_beta - f1_minority_alpha)                   # scalar
        reward = lambda_ * global_delta + (1.0 - lambda_) * score_local
        return reward, f1_minority_beta, score_local.mean(), f1_macro_beta

    # ---------------- Training loop ----------------
    def __call__(self):
        start_time = time.time()

        self.tracker = EpisodeTracker(
            {
                "EPISODES": EPISODES,
                "TRAJ_LENGTH": TRAJ_LENGTH,
                "REAL_DATA_SIZE": REAL_DATA_SIZE,
                "BIAS_PCT": BIAS_PCT,
                "lambda_": self.lambda_,
                "seed": self.seed,
                "pca_components": self.pca_components,
            },
            save_dir="training_runs",
            compare_metric="average_reward",
            beta_factory=lambda: FFNNAgent(**self.ffnn_config)  # <- your class
        )

        # Data
        x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = self.split_dataset(
            train_size=REAL_DATA_SIZE, bias_pct=BIAS_PCT
        )

        print(f"Size of train set: {len(x_theta_train)}")
        total_data = len(x_theta_train) + TRAJ_LENGTH
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (TRAJ_LENGTH / total_data) * 100
        print(f"Real data: {real_percentage:.2f}% of total, Synthetic data: {synthetic_percentage:.2f}% of total")

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

            # Reset β to its initial snapshot each episode for a stable baseline
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
            x_phi_t = x_syn_tensor[:T].clone().detach()
            y_phi_t = y_syn_tensor[:T].clone().detach()

            # Train beta on hybrid (real + synthetic for this episode)
            x_hybrid = torch.cat([x_theta_train, x_phi_t])
            y_hybrid = torch.cat([y_theta_train, y_phi_t])
            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

            # Rewards (alpha baseline global + per-step local)
            rewards, f1_pos_beta, brier_synth_local_mean, f1_macro_beta = self.compute_reward(
                self.alpha_model, self.beta_model, stale_beta_model,
                x_theta_test, x_theta_val, y_theta_val, y_theta_test,
                x_phi_t, y_phi_t,
                lambda_=self.lambda_, f1_thresh=0.5
            )

            # Truncate episode tensors and learn
            states      = states[:T]
            actions     = actions[:T]
            next_states = next_states[:T]
            dones       = dones[:T]
            rewards     = rewards[:T]

            self.agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

            # Log
            avg_reward = torch.mean(rewards)
            self.tracker.log_episode(
                episode + 1,
                avg_reward,                     # Avg reward across trajectory
                f1_pos_beta,                    # F1 for positive class on θ_val (β)
                brier_synth_local_mean,         # Mean (1 - Brier) on synthetic Φ (β)
                f1_macro_beta                   # macro F1 on θ_val (β), for context
            )
            # Save periodic snapshot and best-so-far synthetic
            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_phi_t,
                y_syn=y_phi_t,
                avg_reward=float(avg_reward),
                obj1=float(f1_pos_beta),                 # use compare_metric="obj1" if you want best-on-val F1
                obj2_mean=float(brier_synth_local_mean),
                global_f1=float(f1_macro_beta),
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                beta_model=self.beta_model               # <-- NEW: saves best β when is_best=True
            )


        self.tracker.log_final_test(
            alpha_model=self.alpha_model,
            x_test=x_theta_test,
            y_test=y_theta_test,
            f1_thresh=0.5  # or a θ_val-fixed threshold
        )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")


if __name__ == "__main__":
    train = Training(seed=42)
    train()
