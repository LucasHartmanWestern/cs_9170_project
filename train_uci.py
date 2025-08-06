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
from sklearn.metrics import f1_score as _sk_f1
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
from agents.ffnn_agent2 import FFNNAgent
from torch.utils.data import TensorDataset, DataLoader
import torch

class Training:
    def __init__(self, seed=42, device='cpu'):
        self.device = device
        self.seed = seed
        self.pca_components = 2

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        ppo_config = {
            'state_size': 2,  
            'action_size': self.pca_components,   
            'hidden_size': 64,
            'lr': 3e-4, 
            'gamma': 0.9,
            'clip_epsilon': 0.2,
            'update_epochs': 5,
            'batch_size': 64,
            'c1': 0.5,
            'c2': 0.01,
            'seed': self.seed
        }
        self.ffnn_config = {
            'input_size': self.pca_components,
            'hidden_sizes': [32, 16],
            'output_size': 1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10,
            'type': 'classification',
            'classes': [0, 1],
            'seed': self.seed
        }
        self.ppo_agent = PPOAgent(**ppo_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)
        self.dl_generator = torch.Generator(device=self.device).manual_seed(self.seed)

    # Given AGEP  COW  SCHL  WKHP  SEx we are predicting PINCP
    #Link to data documentation: https://github.com/fairlearn/fairlearn/blob/main/docs/user_guide/datasets/acs_income.rst
    #Can see plots for this dataset in new_biases.ipynb
    def split_dataset(self, train_size=None):
        #Fetch dataset
        adult = fetch_ucirepo(id=2)
        X_df = adult.data.features  
        y_df = adult.data.targets   

        X = X_df.values
        y = np.ravel(y_df.values)

        # Map labels to 0/1
        y = np.where(np.isin(y, ['>50K', '>50K.']), 1, 0)

        # Identify categorical and numerical columns
        cat_cols = [i for i, dt in enumerate(X_df.dtypes) if dt.name in ['category', 'object', 'bool']]
        num_cols = [i for i, dt in enumerate(X_df.dtypes) if np.issubdtype(dt, np.number)]

        # One-hot encode categoricals, standardize numericals
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(X[:, cat_cols]) if cat_cols else np.empty((X.shape[0], 0))

        scaler = StandardScaler()
        X_num = scaler.fit_transform(X[:, num_cols]) if num_cols else np.empty((X.shape[0], 0))

        # stack back together
        X_all = np.hstack([X_num, X_cat])
        y_all = y

        # For balancing, target is last column
        df_all = np.hstack([X_all, y_all.reshape(-1, 1)])

        target_col = -1
        majority_mask = df_all[:, target_col] == 0
        minority_mask = df_all[:, target_col] == 1

        df_majority = df_all[majority_mask]
        df_minority = df_all[minority_mask]

        # Upsample minority class
        n_majority = df_majority.shape[0]
        n_minority = df_minority.shape[0]

        indices = np.random.choice(n_minority, size=n_majority, replace=True)
        df_minority_upsampled = df_minority[indices]

        # Concatenate and shuffle
        df_balanced = np.vstack([df_majority, df_minority_upsampled])
        np.random.shuffle(df_balanced)

        X_bal = df_balanced[:, :-1]
        y_bal = df_balanced[:, -1]

        #Set bias percentage
        remove_pct = 0.75

        # Bias the dataset: remove a percentage of Class 1 examples from the balanced set
        df_balanced_pd = pd.DataFrame(X_bal, columns=[f'feat_{i}' for i in range(X_bal.shape[1])])
        df_balanced_pd['target'] = y_bal

        # Separate class 0 and class 1
        df_class_0 = df_balanced_pd[df_balanced_pd['target'] == 0]
        df_class_1 = df_balanced_pd[df_balanced_pd['target'] == 1]

        # Remove a fraction of class 1
        df_class_1_biased = df_class_1.sample(frac=1 - remove_pct, random_state=self.seed)
        df_biased = pd.concat([df_class_0, df_class_1_biased], axis=0).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        X_biased = df_biased.drop('target', axis=1).values
        y_biased = df_biased['target'].values

        #PCA analysis 
        pca = PCA(n_components=self.pca_components)
        X_biased_pca = pca.fit_transform(X_biased)

        #Train-test split after biasing and PCA
        X_train, X_test, y_train, y_test = train_test_split(
            X_biased_pca, y_biased, test_size=0.2, random_state=self.seed, stratify=y_biased
        )

        # If train_size is specified, subsample the train set to the requested size
        # Bias in class distribution is maintained.
        if train_size is not None:
            if train_size < len(X_train):
                # Stratified subsample to preserve the biased class distribution
                X_train_sub, _, y_train_sub, _ = train_test_split(
                    X_train, y_train, train_size=train_size, random_state=self.seed, stratify=y_train
                )
                X_train = X_train_sub
                y_train = y_train_sub

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return X_train, X_test, y_train, y_test

    def train_predictor_model(self, model, x_train, y_train):

        # Convert numpy arrays to torch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long,   device=self.device)

        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=self.dl_generator)

        model.train(loader)

        return model


    def mean_error(self, target, pred):
        # return torch.mean((pred - target) ** 2)
        y_true = target.detach().cpu().numpy().ravel()
        y_pred = torch.round(pred).detach().cpu().numpy().ravel()
        f1 = _sk_f1(y_true, y_pred)
        #print(f"Mean error f1: {f1}")
        return torch.tensor(f1)

    def error_vector(self, target, pred):

        y_true = target.detach().cpu().numpy().ravel()
        y_pred = torch.round(pred).detach().cpu().numpy().ravel()

        # Compute the F1 score for each sample individually
        f1_scores = []
        for true_label, pred_label in zip(y_true, y_pred):
            score = _sk_f1([true_label], [pred_label], zero_division=0)
            f1_scores.append(score)
        f1_scores = np.array(f1_scores, dtype=np.float32)

        # tensor of F1 scores
        f1_tensor = torch.tensor(f1_scores, device=pred.device).view_as(target)
        #print(f'Error Vector: {f1_tensor}')
        return f1_tensor

    def compute_reward(self, alpha_model, beta_model,x_theta_test, y_theta_test, x_phi, y_phi, lambda_=0.5):
        with torch.no_grad():
            # θ–test predictions from β
            beta_out = beta_model.predict(x_theta_test)   # <-- this is a list
            y_hat_theta_beta = torch.tensor(
                beta_out,
                dtype=torch.float32,
                device=self.device
            ).squeeze(-1)

            # same for α
            alpha_out = alpha_model.predict(x_theta_test)
            y_hat_theta_alpha = torch.tensor(
                alpha_out,
                dtype=torch.float32,
                device=self.device
            ).squeeze(-1)

        # Calculating objective 1, global error using Beta model on theta test set
        objective_global = self.mean_error(y_theta_test, y_hat_theta_beta)

        # Calculating objective 2, vector error using Alpha and Beta models on phi set
        beta_cost = self.mean_error(y_theta_test, y_hat_theta_beta)
        alpha_cost= self.mean_error(y_theta_test, y_hat_theta_alpha)


        with torch.no_grad():
            if alpha_cost < beta_cost:
                print(f'Working with alpha cost')
                pred = alpha_model.predict(x_phi)
                y_hat_phi = torch.tensor(pred, dtype=torch.float32, device=self.device).squeeze(-1)
                objective_individual = self.error_vector(y_phi, y_hat_phi)
            else:
                print('Changed to work with beta cost')
                pred = beta_model.predict(x_phi)
                y_hat_phi = torch.tensor(pred, dtype=torch.float32, device=self.device).squeeze(-1)
                objective_individual = self.error_vector(y_phi, y_hat_phi)
        # print(f'line 210 obj individual {objective_individual}')

       
        #Maximized for f1 score
        reward = 1*(lambda_*objective_global + (1.0-lambda_)*objective_individual)
        return reward

    #Called automatically
    def __call__(self):
        print('Begin train loop')
        # Training loop params
        EPISODES        = 200
        TRAJ_LENGTH     = 500
        REAL_DATA_SIZE  = 3000

        # Prepare data and baseline (Alpha model)
        x_theta_train, x_theta_test, y_theta_train, y_theta_test = self.split_dataset(train_size=REAL_DATA_SIZE)

        print(f"Size of train set: {len(x_theta_train)}")
        total_data = len(x_theta_train) + TRAJ_LENGTH
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (TRAJ_LENGTH / total_data) * 100
        print(f"Real data: {real_percentage:.2f}% of total, Synthetic data: {synthetic_percentage:.2f}% of total")

        self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)

        # Environment and agent setup NOTE that sex is forced female in loop
        #features = ['AGEP', 'COW', 'SCHL', 'WKHP']
        #target = 'PINCP'
        seed = self.seed

        env = Environment(
            target=1,#Does nothing right now
            male_hi=0,#Does nothing right now
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

                next_state, done, info = env.step(action, (t + 1))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)

                #Always = 1 (Underrepresented class)
                sampled_t = info['sampled_target']
                #action_with_sex = np.concatenate([action, [2.0]])

                x_syn_list.append(action)
                y_syn_list.append(sampled_t)

                state = next_state
                if done:
                    print(f'Generated synthetic tuple {t + 1}/{TRAJ_LENGTH}')
                    break


            T = len(x_syn_list)
            x_syn = np.stack(x_syn_list)    # shape [T, feature_dim]
            y_syn = np.array(y_syn_list)    # shape [T]

            x_theta_test_t = torch.tensor(x_theta_test, dtype=torch.long, device=self.device)
            y_theta_test_t = torch.tensor(y_theta_test, dtype=torch.long, device=self.device)
            x_phi_t       = torch.tensor(x_syn, dtype=torch.long, device=self.device)
            y_phi_t       = torch.tensor(y_syn, dtype=torch.long, device=self.device)    

            x_hybrid = np.vstack([x_theta_train, x_phi_t ])
            y_hybrid = np.concatenate([y_theta_train, y_phi_t ])

            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

            rewards = self.compute_reward(self.alpha_model, self.beta_model, x_theta_test_t, y_theta_test_t, x_phi_t, y_phi_t)

            # for idx, (s, a, r, s_next, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
            #     # learn
            #     self.ppo_agent.learn(s, a, r, s_next, d)

            self.ppo_agent.learn_trajectory(states, actions, rewards, next_states, dones)

            #Resets beta model
            self.beta_model = FFNNAgent(**self.ffnn_config)

            avg_reward = torch.mean(rewards)
            print(f"Episode {episode+1}/{EPISODES} — Average reward: {avg_reward:.3f}")
    
if __name__ == "__main__":
    train = Training(seed=42)
    train()