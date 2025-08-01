# --- Imports ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from env import Environment
from agents.ppo_agent import PPOAgent
from fairlearn.datasets import fetch_acs_income
from sklearn.neural_network import MLPRegressor
import torch.nn.functional as F

class Training:
    def __init__(self, seed=1234, device='cpu'):
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        ppo_config = {
            'state_size': 2,  
            'action_size': 4,   
            'hidden_size': 64,
            'lr': 1e-2, 
            'gamma': 0.8,
            'clip_epsilon': 0.2,
            'update_epochs': 10,
            'batch_size': 32,
            'c1': 0.5,
            'c2': 0.01,
            'seed': self.seed
        }
        self.ppo_agent = PPOAgent(**ppo_config)

    # Given AGEP  COW  SCHL  WKHP  SEX we are predicting PINCP
    #Link to data documentation: https://github.com/fairlearn/fairlearn/blob/main/docs/user_guide/datasets/acs_income.rst
    #Can see plots for this dataset in new_biases.ipynb
    def split_dataset(self, seq_len=5, seed=42):
        #Fetch raw data
        data_bunch = fetch_acs_income(as_frame=True)

        #Reduce feature space to AGEP  COW  SCHL  WKHP  SEX
        X = data_bunch.data.iloc[:100000].drop(columns=['OCCP', 'POBP', 'RELP', 'RAC1P', 'MAR', ])
        y = data_bunch.target.iloc[:100000]

        df = pd.concat([X, y.rename("INCOME")], axis=1)

        #Threshold for "High_income"
        self.INCOME = 150_000

        # Identify all males and females earning >150K
        high_income_mask = df['INCOME'] > self.INCOME
        self.males_hi   = df[(df['SEX']==1.0) & high_income_mask]
        females_hi = df[(df['SEX']==2.0) & high_income_mask]

        # Find minimum count for balanced split
        n_hi = min(len(self.males_hi), len(females_hi))

        # Randomly sample n_hi from each group for test set
        males_hi_test   = self.males_hi.sample(n=n_hi, random_state=0)
        females_hi_test = females_hi.sample(n=n_hi, random_state=0)
        test_idx = males_hi_test.index.union(females_hi_test.index)

        # The rest is train set
        train_idx = df.index.difference(test_idx)

        # Split all arrays accordingly
        X_train, X_theta_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_theta_test = y.loc[train_idx], y.loc[test_idx]
        return X_train, X_theta_test, y_train, y_theta_test

    def train_alpha_agent(self, X_train, y_train, X_test, y_test, hidden_layer_sizes=(64, 32), max_iter=10, lr=1e-3, seed=0, **kwargs):

        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', max_iter=max_iter, learning_rate_init=lr, random_state=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        baseline_mse = mean_squared_error(y_test, y_pred)
        baseline_mae = mean_absolute_error(y_test, y_pred)
        return model, baseline_mse, baseline_mae



    def evaluate_cost(self, model, X, y, metric="mse", reduction="mean"):
        model.eval()
        with torch.no_grad():
            y_pred = model(X)

            if metric.lower() == "mse":
                return F.mse_loss(y_pred, y, reduction=reduction).item()
            # elif metric.lower() == "mae":
            #     return F.l1_loss(y_pred, y, reduction=reduction).item()
            # elif metric.lower() == "me":
            #     diff = y_pred - y
            #     return diff.mean().item() if reduction == "mean" else diff.sum().item()

    def global_error(self, beta_model, X_theta_test, y_theta_test, metric="mse"):
        return self.evaluate_cost(
            beta_model,
            X_theta_test, 
            y_theta_test,
            metric=metric,
            reduction="mean"
            )

    def independent_error(self, alpha_model, beta_model, X_phi, y_phi, X_theta_test, y_theta_test):

        alpha_mean = self.evaluate_cost(
            alpha_model, X_theta_test, y_theta_test,
            metric=self.metric, reduction="mean"
        )
        beta_mean = self.evaluate_cost(
            beta_model, X_theta_test, y_theta_test,
            metric=self.metric, reduction="mean"
        )

        # per sample error vectors for both models
        alpha_model.eval()
        beta_model.eval()
        with torch.no_grad():
            pred_a = alpha_model(X_phi)
            pred_b = beta_model(X_phi)
            diff_a = pred_a - y_phi
            diff_b = pred_b - y_phi

            if self.metric == "mse":
                err_a = diff_a.pow(2)
                err_b = diff_b.pow(2)
            elif self.metric == "mae":
                err_a = diff_a.abs()
                err_b = diff_b.abs()
            elif self.metric == "me":
                err_a = diff_a
                err_b = diff_b
            else:
                raise ValueError(f"Unknown metric '{self.metric}'.")

            # if multi‐dimensional, collapse to one scalar per sample
            if err_a.dim() > 1:
                err_a = err_a.mean(dim=1)
                err_b = err_b.mean(dim=1)

            err_a = err_a.cpu().tolist()
            err_b = err_b.cpu().tolist()

        if beta_mean >= alpha_mean:
            return err_a      # use alpha model’s per‐action cost
        else:
            return err_b      # use beta model’s per‐action cost

    def compute_reward(self, alpha_model, beta_model,X_theta_test, y_theta_test, X_phi, y_phi, lamb=1.0):
        #global term 
        g = self.global_error(beta_model, X_theta_test, y_theta_test)

        #individualized term
        ind = self.independent_error(alpha_model, beta_model, X_phi, y_phi)

        #combine to create reward vector (one element per action/synthetic_row)
        reward = []
        for d in ind:
            r = (lamb * g) + ((1 - lamb) * d)
            reward.append(r)
        return reward

    #Called automatically
    def __call__(self):
        print('Begin train loop')
        # Training hyperparameters
        EPISODES        = 20
        TRAJ_LENGTH     = 200
        MODEL_EPOCHS    = 5

        window_size = 5

        # Prepare data and baseline (Alpha model)
        X_theta_train, X_theta_test, y_theta_train, y_theta_test = self.split_dataset()
        alpha_model, baseline_mse, baseline_mae = self.train_alpha_agent(X_theta_train, y_theta_train, X_theta_test, y_theta_test, epochs=MODEL_EPOCHS)

        beta_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=10, learning_rate_init=1e-3, random_state=seed)

        # Environment and agent setup NOTE that sex is forced female in loop
        features = ['AGEP', 'COW', 'SCHL', 'WKHP']
        target = 'PINCP'
        threshold = 100.0
        seed = self.seed


        env = Environment(
            train_df=pd.concat([self.X_train, self.y_train.rename("PINCP")], axis=1),
            threshold=threshold,
            features=features,
            target=target,
            baseline_mse=baseline_mse,
            male_hi=self.males_hi,
            max_samples=TRAJ_LENGTH,
            window_size=window_size,
            seed=seed
        )

        for episode in range(EPISODES):
            # Buffers for PPO
            trajectory = np.zeros((TRAJ_LENGTH, len(features) + 2))  # +1 for SEX, +1 for target
            states, actions = [], []
            next_states, dones = [], []

            X_syn_list, y_syn_list = [], []

            # Reset env
            state = env.reset()
            #Generate a trajectory of length TRAJ_LENGTH
            for t in range(TRAJ_LENGTH):
                #Get action
                action = self.ppo_agent.predict(state)             
                next_state, _, done, info = env.step(action, (TRAJ_LENGTH + 1))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)

                # combine action with SEX and target, append to trajectory
                sampled_t = info['sampled_target']
                action_with_sex = np.concatenate([action, [2.0]])

                X_syn_list.append(action_with_sex)
                y_syn_list.append(sampled_t)

                state = next_state
                if done:
                    break
                print(f'Generated synthetic tuple {t}/{TRAJ_LENGTH}')

            T = len(X_syn_list)
            X_syn = np.stack(X_syn_list)    # shape [T, feature_dim]
            y_syn = np.array(y_syn_list)    # shape [T]

            X_theta_test_t = torch.tensor(X_theta_test, dtype=torch.float32, device=self.device)
            y_theta_test_t = torch.tensor(y_theta_test, dtype=torch.float32, device=self.device)
            X_phi_t       = torch.tensor(X_syn, dtype=torch.float32, device=self.device)
            y_phi_t       = torch.tensor(y_syn, dtype=torch.float32, device=self.device)    

            X_hybrid = np.vstack([X_theta_train, X_phi_t ])
            y_hybrid = np.concatenate([y_theta_train, y_phi_t ])

            beta_model.fit(X_hybrid, y_hybrid)
            rewards = self.compute_reward(alpha_model, beta_model, X_theta_test_t, y_theta_test_t, X_phi_t, y_phi_t)

            for s, a, r, s_next, d in zip(states, actions, rewards, next_states, dones):
                self.ppo_agent.learn(s, a, r, s_next, d)

            avg_reward = np.mean(rewards)
            print(f"Episode {episode+1}/{EPISODES} — Average reward: {avg_reward:.3f}")
    
if __name__ == "__main__":
    train = Training()
    train()