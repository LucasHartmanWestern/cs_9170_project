# --- Imports ---
import numpy as np
import pandas as pd
import os
import shutil
import torch
from torch.utils.data import DataLoader, TensorDataset
from env import Environment
from agents.ppo_agent import PPOAgent
from agents.ffnn_agent2 import FFNNAgent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score as _sk_f1, accuracy_score
from sklearn.utils import resample
import copy
import time
from episode_tracker import EpisodeTracker

class Training:
    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        self.pca_components = 2
        self.lambda_ = 0.5

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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
        self.ffnn_config = {
            'input_size': self.pca_components,
            'hidden_sizes': [32, 16],
            'output_size': 1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20,
            'type': 'classification',
            'classes': [0, 1],#1 is >50k,
            'device': self.device,
            'seed': self.seed
        }

        self.ppo_agent = PPOAgent(**ppo_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.dl_generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self.beta_base_model = FFNNAgent(**self.ffnn_config)
        self.restart_beta()

    # Given AGEP  COW  SCHL  WKHP  SEx we are predicting PINCP
    #Link to data documentation: https://github.com/fairlearn/fairlearn/blob/main/docs/user_guide/datasets/acs_income.rst
    #Can see plots for this dataset in new_biases.ipynb

    def restart_beta(self):
        self.beta_model = None
        self.beta_model = copy.deepcopy(self.beta_base_model)

    def split_dataset(self, train_size=None, bias_pct=0.75):
        # Fetch dataset
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

        # Bias the dataset: remove a percentage of Class 1 examples from the balanced set
        df_balanced_pd = pd.DataFrame(X_bal, columns=[f'feat_{i}' for i in range(X_bal.shape[1])])
        df_balanced_pd['target'] = y_bal

        # Separate class 0 and class 1
        df_class_0 = df_balanced_pd[df_balanced_pd['target'] == 0]
        df_class_1 = df_balanced_pd[df_balanced_pd['target'] == 1]

        # Remove a fraction of class 1
        df_class_1_biased = df_class_1.sample(frac=1 - bias_pct, random_state=self.seed)
        df_biased = pd.concat([df_class_0, df_class_1_biased], axis=0).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        X_biased = df_biased.drop('target', axis=1).values
        y_biased = df_biased['target'].values

        # Train-test split after biasing, before PCA
        X_train, X_test_theta, y_train, y_test_theta = train_test_split(
            X_biased, y_biased, test_size=0.2, random_state=self.seed, stratify=y_biased
        )

        # If train_size is specified, subsample the train set to the requested size
        # Bias in class distribution is maintained.
        if train_size is not None and train_size < len(X_train):
            print('Reducing size of dataset')
            # Stratified subsample to preserve the biased class distribution
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=train_size, random_state=self.seed, stratify=y_train
            )

        # PCA analysis: fit on train, transform both train and test
        pca = PCA(n_components=self.pca_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test_theta)

        X_train_theta = torch.tensor(X_train_pca, dtype=torch.float32, device=self.device)
        X_test_theta = torch.tensor(X_test_pca, dtype=torch.float32, device=self.device)
        y_train_theta = torch.tensor(y_train, dtype=torch.long, device=self.device)
        y_test_theta = torch.tensor(y_test_theta, dtype=torch.long, device=self.device)
        print(f"Size of X_train_theta: {X_train_theta.shape}")

        return X_train_theta, X_test_theta, y_train_theta, y_test_theta

    def train_predictor_model(self, model, x_train, y_train):

        # # Convert numpy arrays to torch tensors
        # x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        # y_train_tensor = torch.tensor(y_train, dtype=torch.long,   device=self.device)

        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=self.dl_generator)

        model.train(loader)

        return model


    def mean_error(self, target, pred):
        return torch.mean((pred - target) ** 2)


    def error_vector(self, target, pred):
        return (pred - target) ** 2

    def f1_error(self, target, pred):
        y_true = target.detach().cpu().numpy().ravel()
        y_pred = torch.round(pred).detach().cpu().numpy().ravel()
        f1 = _sk_f1(y_true, y_pred)
        return torch.tensor(f1)


      
    def compute_reward(self, alpha_model, beta_model, x_theta_test, y_theta_test, x_phi, y_phi):
        with torch.no_grad():
            # θ–test predictions from β
            y_hat_theta_beta = beta_model.predict(x_theta_test)

            # beta on minority
            idx = y_theta_test == 1
            y_hat_theta_beta_min = beta_model.predict(x_theta_test[idx, :])

            # same for α
            y_hat_theta_alpha = alpha_model.predict(x_theta_test)

        # How well Beta performed on the real test set
        # Calculating objective 1, global error using Beta model on theta test set
        objective_global = self.f1_error(y_theta_test, y_hat_theta_beta)
        # Calculating objective 1, global error using Beta model on theta test set
        objective_1 = self.mean_error(y_theta_test[idx], y_hat_theta_beta_min)

        # How well the best performing model could predict synthetic data, learns that minority has the most absurd feature values?
        # Calculating objective 2, vector error using Alpha and Beta models on phi set
        beta_cost = self.mean_error(y_theta_test, y_hat_theta_beta)
        alpha_cost = self.mean_error(y_theta_test, y_hat_theta_alpha)

        with torch.no_grad():
            if False:  # alpha_cost < beta_cost:
                print(f'Working with alpha cost')
                y_hat_phi = alpha_model.predict(x_phi)
                objective_individual = self.error_vector(y_phi, y_hat_phi)
            else:
                print('Changed to work with beta cost')
                y_hat_phi = beta_model.predict(x_phi)
                objective_individual = self.error_vector(y_phi, y_hat_phi)

        # print(f'obj global {objective_global:.4f} individual {objective_individual.mean():.4f}')
        reward = 1 * (self.lambda_ * objective_1 + (1.0 - self.lambda_) * objective_individual)
        return reward, objective_1, objective_individual, objective_global

    #Called automatically
    def __call__(self):
        print('Begin train loop')
        # Training loop params
        start_time = time.time()

        EPISODES        = 2 #200
        TRAJ_LENGTH     = 3000 #1000
        REAL_DATA_SIZE  = 3000
        BIAS_PCT        = 0.9
        self.lambda_    = 0.8

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

        self.save_by_episode = 100
        data_exp_folder = 'data_exp/exp_001/'

        #SAVE_DATA      
        # Prepare data
        x_theta_train, x_theta_test, y_theta_train, y_theta_test = self.split_dataset(train_size=REAL_DATA_SIZE, bias_pct=BIAS_PCT)

        print(f"Size of train set: {len(x_theta_train)}")
        total_data = len(x_theta_train) + TRAJ_LENGTH
        real_percentage = (len(x_theta_train) / total_data) * 100
        synthetic_percentage = (TRAJ_LENGTH / total_data) * 100
        print(f"Real data: {real_percentage:.2f}% of total, Synthetic data: {synthetic_percentage:.2f}% of total")

        self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)

        # Environment and agent setup NOTE that sex is forced female in loop
        seed = self.seed

        env = Environment(
            target=1,#Does nothing right now
            max_actions=TRAJ_LENGTH,
            device=self.device,
            seed=seed
        )

        last_x_syn = None
        last_y_syn = None

        col_labels = [f"pca_{i}" for i in range(self.pca_components)]
        # Delete the folder if it exists
        if os.path.exists(data_exp_folder):
            shutil.rmtree(data_exp_folder)
        # Create the folder
        os.makedirs(data_exp_folder)

        for episode in range(EPISODES):
            # Pre-allocate arrays for performance (now as torch tensors)
            states = [None] * TRAJ_LENGTH
            actions = [None] * TRAJ_LENGTH
            next_states = [None] * TRAJ_LENGTH
            dones = [None] * TRAJ_LENGTH

            # Pre-allocate synthetic data as torch tensors
            x_syn_tensor = torch.zeros((TRAJ_LENGTH, x_theta_train.shape[1]), dtype=torch.float32, device=self.device)
            y_syn_tensor = torch.zeros(TRAJ_LENGTH, dtype=torch.long, device=self.device)

            # Reset env
            state = env.reset()
            # Generate a trajectory of length TRAJ_LENGTH
            for t in range(TRAJ_LENGTH):
                # Get action
                action = self.ppo_agent.predict(state)

                next_state, done, info = env.step(action, (t + 1))

                states[t] = state
                actions[t] = action
                next_states[t] = next_state
                dones[t] = done

                x_syn_tensor[t] = action
                y_syn_tensor[t] = info['sampled_target'] # Always = 1 (Underrepresented class)

                state = next_state
                if done:
                    print(f'Generated synthetic tuple {t + 1}/{TRAJ_LENGTH}')
                    break

            # Only keep the filled part if early break
            T = t + 1 if done else TRAJ_LENGTH
            x_syn = x_syn_tensor[:T].clone().detach()
            y_syn = y_syn_tensor[:T].clone().detach()
            x_phi_t = x_syn
            y_phi_t = y_syn

            # Save the last trajectory for later use (move to cpu/numpy for DataFrame)
            last_x_syn = x_syn.cpu().numpy()
            last_y_syn = y_syn.cpu().numpy()
            x_hybrid = torch.cat([x_theta_train, x_phi_t ])
            y_hybrid = torch.cat([y_theta_train, y_phi_t ])

            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)

            rewards, obj_1, obj_2, global_ = self.compute_reward(self.alpha_model, self.beta_model,\
                                                                 x_theta_test, y_theta_test, x_phi_t, y_phi_t)


            states      = states[:T]
            actions     = actions[:T]
            next_states = next_states[:T]
            dones       = dones[:T]
            rewards     = rewards[:T]
            self.ppo_agent.learn_trajectory(states, actions, rewards, next_states, dones)

            # Resets beta model
            self.restart_beta()

            avg_reward = torch.mean(rewards)
            self.tracker.log_episode(
                episode+1,
                avg_reward,
                obj_1,                # minority F1
                obj_2.mean(),         # mean individual metric
                global_,              # global F1
                self.lambda_
            )
            self.tracker.maybe_save_synthetic(
                episode_num=episode+1,
                x_syn=x_phi_t,                 # torch.Tensor or np.ndarray
                y_syn=y_phi_t,
                avg_reward=float(avg_reward),
                obj1=float(obj_1),
                obj2_mean=float(obj_2.mean()),
                global_f1=float(global_),
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])]
            )
            print(f"Episode {episode+1}/{EPISODES} — Average reward: {avg_reward:.4f}" +\
                 f"- Obj 1 {obj_1:.4f}, Obj 2 {obj_2.mean():.4f},"+\
                 f" f1 score  {global_:.4f}, lambda {self.lambda_:.2f}")


            if ((episode +1) % self.save_by_episode)==0:
                # Format as DataFrame
                df_syn = pd.DataFrame(last_x_syn, columns=col_labels)
                df_syn["target"] = last_y_syn
                df_syn.to_csv(f"{data_exp_folder}synthetic_trajectory_ep{episode+1}.csv", index=False)
                print(f"Saved synthetic trajectory at episode {episode + 1} to synthetic_trajectory.csv")
                

        print(f'Total time {time.time()-start_time}')

        # After training, save the last generated synthetic trajectory to a file
        if last_x_syn is not None and last_y_syn is not None:
            # Format as DataFrame
            df_syn = pd.DataFrame(last_x_syn, columns=col_labels)
            df_syn["target"] = last_y_syn

            df_syn.to_csv("synthetic_trajectory.csv", index=False)
            
            print(f"Saved last synthetic trajectory to synthetic_trajectory.csv")
            print(f"[Tracker] Finished. Run folder: {self.tracker.summary_path()}")

if __name__ == "__main__":
    train = Training(seed=42)
    train()