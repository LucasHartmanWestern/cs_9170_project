# --- Imports ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from env import Environment
from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from fairlearn.datasets import fetch_acs_income
from sklearn.neural_network import MLPRegressor
import torch.nn.functional as F
import math

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
        self.ffnn_config = {
            'input_size': 5,
            'hidden_sizes': [16, 16],
            'output_size': 1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'type': 'regression',
            'classes': None,
            'seed': self.seed
        }
        self.ppo_agent = PPOAgent(**ppo_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)

    # Given AGEP  COW  SCHL  WKHP  SEx we are predicting PINCP
    #Link to data documentation: https://github.com/fairlearn/fairlearn/blob/main/docs/user_guide/datasets/acs_income.rst
    #Can see plots for this dataset in new_biases.ipynb
    def split_dataset(self, seq_len=5, seed=42):
        #Fetch raw data
        data_bunch = fetch_acs_income(as_frame=True)

        #Reduce feature space to AGEP  COW  SCHL  WKHP  SEx
        x = data_bunch.data.iloc[:100000].drop(columns=['OCCP', 'POBP', 'RELP', 'RAC1P', 'MAR', ])
        y = data_bunch.target.iloc[:100000]

        df = pd.concat([x, y.rename("INCOME")], axis=1)

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
        x_train, x_theta_test = x.loc[train_idx], x.loc[test_idx]
        y_train, y_theta_test = y.loc[train_idx], y.loc[test_idx]

        x_train_np = x_train.to_numpy(dtype=np.float32)
        y_train_np = y_train.to_numpy(dtype=np.float32)
        x_test_np  = x_theta_test.to_numpy(dtype=np.float32)
        y_test_np  = y_theta_test.to_numpy(dtype=np.float32)

        return x_train_np, x_test_np, y_train_np, y_test_np

    def train_predictor_model(self, model, x_train, y_train):

        # Convert numpy arrays to torch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train(loader)

        return model



    def evaluate_cost(self, model, x, y, metric="mse", reduction="mean"):
        with torch.no_grad():
            y_pred = model.predict(x).squeeze(-1)   # → [N]
            y_true = y.squeeze(-1)                 # → [N]
            diff   = y_pred - y_true

        return float(diff.pow(2).mean().item())  # MSE

    def global_error(self, beta_model, x_theta_test, y_theta_test, metric="mse"):
        return self.evaluate_cost(
            beta_model,
            x_theta_test, 
            y_theta_test,
            metric=metric,
            reduction="mean"
            )

    def independent_error(self, alpha_model, beta_model, x_phi, y_phi, x_theta_test, y_theta_test):
        self.metric = 'mse'
        alpha_mean = self.evaluate_cost(
            alpha_model, x_theta_test, y_theta_test,
            metric="mse", reduction="mean"
        )
        beta_mean = self.evaluate_cost(
            beta_model, x_theta_test, y_theta_test,
            metric="mse", reduction="mean"
        )

        # per sample error vectors for both models
        with torch.no_grad():
            pred_a = alpha_model.predict(x_phi)
            pred_b = beta_model.predict(x_phi)
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

    def mean_error(self, target, pred):
        # return torch.mean((pred - target) ** 2)
        return torch.mean(torch.abs(pred-target))

    def error_vector(self, target, pred):
        # return (pred - target) ** 2
        return torch.abs(pred-target)

    def compute_reward(self, alpha_model, beta_model,x_theta_test, y_theta_test, x_phi, y_phi, lambda_=0.5):
        # #global term 
        # g = self.global_error(beta_model, x_theta_test, y_theta_test)

        # #individualized term
        # ind = self.independent_error(alpha_model, beta_model, x_phi, y_phi, x_theta_test, y_theta_test)

        # #combine to create reward vector (one element per action/synthetic_row)
        # # reward = []
        # # for d in ind:
        # #     r = ((lamb * g) + ((1 - lamb) * d) ) * -1
        # #     reward.append(r)
        with torch.no_grad():
            y_hat_theta_beta = beta_model.predict(x_theta_test).squeeze(-1)   # → [N]
            y_hat_theta_alpha= alpha_model.predict(x_theta_test).squeeze(-1)   # → [N]

        # Calculating objective 1, global error using Beta model on theta test set
        objective_global = self.mean_error(y_theta_test, y_hat_theta_beta)

        # Calculating objective 2, vector error using Alpha and Beta models on phi set
        beta_cost = self.mean_error(y_theta_test, y_hat_theta_beta)
        alpha_cost= self.mean_error(y_theta_test, y_hat_theta_alpha)


        with torch.no_grad():
            if alpha_cost < beta_cost:
                print(f'Working with alpha cost')
                y_hat_phi_alpha = alpha_model.predict(x_phi).squeeze(-1)   # → [N]
                objective_individual = self.error_vector(y_phi, y_hat_phi_alpha)
            else:
                print('Changed to work with beta cost')
                y_hat_phi_beta = beta_model.predict(x_phi).squeeze(-1)   # → [N]
                objective_individual = self.error_vector(y_phi, y_hat_phi_beta)
        # print(f'line 210 obj individual {objective_individual}')

       
        reward = -1*(lambda_*objective_global + (1.0-lambda_)*objective_individual)
        return reward

    #Called automatically
    def __call__(self):
        print('Begin train loop')
        # Training loop params
        EPISODES        = 20
        TRAJ_LENGTH     = 10

        # Prepare data and baseline (Alpha model)
        x_theta_train, x_theta_test, y_theta_train, y_theta_test = self.split_dataset()
        self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)

        # Environment and agent setup NOTE that sex is forced female in loop
        features = ['AGEP', 'COW', 'SCHL', 'WKHP']
        target = 'PINCP'
        seed = self.seed

        env = Environment(
            target=target,
            male_hi=self.males_hi,#index for sampling target
            max_actions=TRAJ_LENGTH,
            seed=seed
        )

        for episode in range(EPISODES):
            states, actions = [], []
            next_states, dones = [], []

            x_syn_list, y_syn_list = [], []

            # Reset env
            state = env.reset()
            #Generate a trajectory of length TRAJ_LENGTH
            for t in range(TRAJ_LENGTH):
                #Get action
                action = self.ppo_agent.predict(state)             
                next_state, done, info = env.step(action, (TRAJ_LENGTH + 1))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)

                # combine action with SEX and target, append to trajectory
                sampled_t = info['sampled_target']
                action_with_sex = np.concatenate([action, [2.0]])

                x_syn_list.append(action_with_sex)
                y_syn_list.append(sampled_t)

                state = next_state
                if done:
                    print(f'Generated synthetic tuple {t + 1}/{TRAJ_LENGTH}')
                    break


            T = len(x_syn_list)
            x_syn = np.stack(x_syn_list)    # shape [T, feature_dim]
            y_syn = np.array(y_syn_list)    # shape [T]

            x_theta_test_t = torch.tensor(x_theta_test, dtype=torch.float32, device=self.device)
            y_theta_test_t = torch.tensor(y_theta_test, dtype=torch.float32, device=self.device)
            x_phi_t       = torch.tensor(x_syn, dtype=torch.float32, device=self.device)
            y_phi_t       = torch.tensor(y_syn, dtype=torch.float32, device=self.device)    

            x_hybrid = np.vstack([x_theta_train, x_phi_t ])
            y_hybrid = np.concatenate([y_theta_train, y_phi_t ])

            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

            rewards = self.compute_reward(self.alpha_model, self.beta_model, x_theta_test_t, y_theta_test_t, x_phi_t, y_phi_t)

            for idx, (s, a, r, s_next, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
                # learn
                self.ppo_agent.learn(s, a, r, s_next, d)

            #self.ppo_agent.learn_trajectory(states, actions, rewards, next_states, dones)

            #Resets beta model
            self.beta_model = FFNNAgent(**self.ffnn_config)

            avg_reward = torch.mean(rewards)
            print(f"Episode {episode+1}/{EPISODES} — Average reward: {avg_reward:.3f}")
    
if __name__ == "__main__":
    train = Training(seed=42)
    train()