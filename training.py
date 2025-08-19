# --- Imports ---

# Standard library
import os
import time

# Third-party: numpy, pandas, sklearn, torch
import numpy as np
import pandas as pd
import torch

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
        self.lambda_ = 1

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
    def split_dataset(self, train_size=None, bias_pct=0.75):
        data_path = "census+income/adult.data"

        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        X_df = pd.read_csv(data_path, header=None, names=column_names, na_values="?", skipinitialspace=True)
        y_df = X_df[["income"]]
        X_df = X_df.drop(columns=["income"])

        X = X_df.values
        y = np.ravel(y_df.values)
        y = np.where(np.isin(y, ['>50K', '>50K.']), 1, 0)

        cat_cols = [i for i, dt in enumerate(X_df.dtypes) if dt.name in ['category', 'object', 'bool']]
        num_cols = [i for i, dt in enumerate(X_df.dtypes) if np.issubdtype(dt, np.number)]

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(X[:, cat_cols]) if cat_cols else np.empty((X.shape[0], 0))

        scaler = StandardScaler()
        X_num = scaler.fit_transform(X[:, num_cols]) if num_cols else np.empty((X.shape[0], 0))

        X_all = np.hstack([X_num, X_cat])
        y_all = y

        df_all = np.hstack([X_all, y_all.reshape(-1, 1)])
        target_col = -1
        majority_mask = df_all[:, target_col] == 0
        minority_mask = df_all[:, target_col] == 1

        df_majority = df_all[majority_mask]
        df_minority = df_all[minority_mask]

        n_majority = df_majority.shape[0]
        n_minority = df_minority.shape[0]
        indices = np.random.choice(n_minority, size=n_majority, replace=True)
        df_minority_upsampled = df_minority[indices]

        df_balanced = np.vstack([df_majority, df_minority_upsampled])
        np.random.shuffle(df_balanced)

        X_bal = df_balanced[:, :-1]
        y_bal = df_balanced[:, -1]

        df_balanced_pd = pd.DataFrame(X_bal, columns=[f'feat_{i}' for i in range(X_bal.shape[1])])
        df_balanced_pd['target'] = y_bal
        df_class_0 = df_balanced_pd[df_balanced_pd['target'] == 0]
        df_class_1 = df_balanced_pd[df_balanced_pd['target'] == 1]
        df_class_1_biased = df_class_1.sample(frac=1 - bias_pct, random_state=self.seed)
        df_biased = pd.concat([df_class_0, df_class_1_biased], axis=0).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        X_biased = df_biased.drop('target', axis=1).values
        y_biased = df_biased['target'].values

        X_train, X_test_theta, y_train, y_test_theta = train_test_split(
            X_biased, y_biased, test_size=0.2, random_state=self.seed, stratify=y_biased
        )

        if train_size is not None and train_size < len(X_train):
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=train_size, random_state=self.seed, stratify=y_train
            )

        pca = PCA(n_components=self.pca_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca  = pca.transform(X_test_theta)

        X_train_theta = torch.tensor(X_train_pca, dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_pca,  dtype=torch.float32, device=self.device)
        y_train_theta = torch.tensor(y_train, dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_theta, dtype=torch.long, device=self.device)

        return X_train_theta, X_test_theta, y_train_theta, y_test_theta

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

    def compute_reward(self, alpha_model, beta_model,
                       x_theta_test, y_theta_test,
                       x_phi, y_phi,
                       lambda_=0.99, f1_thresh=0.5):
        """
        r_t = λ * (F1_pos^β(θ_test) − F1_pos^α(θ_test)) + (1 − λ) * (1 − Brier_t^β on Φ)
        Returns:
          reward[T], f1_pos_beta (scalar), mean(1−Brier) over Φ (scalar), f1_macro_beta (scalar)
        """
        with torch.no_grad():
            # alpha baseline on theta_test
            p1_alpha_test = self._p1_from_agent(alpha_model, x_theta_test)
            f1_pos_alpha  = self.f1_from_probs(y_theta_test, p1_alpha_test, f1_thresh)

            # beta performance on theta_test
            p1_beta_test  = self._p1_from_agent(beta_model, x_theta_test)
            f1_pos_beta   = self.f1_from_probs(y_theta_test, p1_beta_test, f1_thresh)

            # macro-F1 for logging
            f1_neg_beta   = self.f1_from_probs(1 - y_theta_test, 1 - p1_beta_test, 1 - f1_thresh)
            f1_macro_beta = 0.5 * (f1_pos_beta + f1_neg_beta)

            if f1_pos_alpha > f1_pos_beta:#Make statement false to disable model cost switching
                model = alpha_model
                print("Using alpha for cost")
            else:
                model = beta_model
                print("Using beta for cost")
                
            # per-step local term on Φ (β probs)
            p1_phi_beta   = self._p1_from_agent(model, x_phi)    # [T]
            brier_t       = self.brier_per_sample(y_phi, p1_phi_beta) # [T]
            score_local   = 1.0 - brier_t                             # [T]

        global_delta = (f1_pos_beta - f1_pos_alpha)                   # scalar
        reward = lambda_ * global_delta + (1.0 - lambda_) * score_local
        return reward, f1_pos_beta, score_local.mean(), f1_macro_beta

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
            save_dir="training_runs"
        )

        # Data
        x_theta_train, x_theta_test, y_theta_train, y_theta_test = self.split_dataset(
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
                self.alpha_model, self.beta_model,
                x_theta_test, y_theta_test,
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
                f1_pos_beta,                    # F1 for positive class on θ_test (β)
                brier_synth_local_mean,         # Mean (1 - Brier) on synthetic Φ (β)
                f1_macro_beta                   # macro F1 on θ_test (β), for context
            )
            # Save periodic snapshot and best-so-far synthetic
            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_phi_t,
                y_syn=y_phi_t,
                avg_reward=float(avg_reward),
                obj1=float(f1_pos_beta),                # F1_pos on θ_test (β)
                obj2_mean=float(brier_synth_local_mean),# mean(1 - Brier) on Φ
                global_f1=float(f1_macro_beta),         # macro-F1 on θ_test (β)
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])]
            )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")


if __name__ == "__main__":
    train = Training(seed=42)
    train()
