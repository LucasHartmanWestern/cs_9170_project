import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_processing import get_xy_from_data
from training import generate_state, evaluate_model

class Environment:
    def __init__(
        self,
        df_train,
        df_val,
        target_features,
        base_agent,
        action_range,
        action_dim,
        horizon,
        oneshot,
        accuracy_reward_multiplier,
        seed,
        device,
    ):
        self.device = torch.device(device)
        self.seed = seed
        torch.manual_seed(seed)

        # load real data
        xtr_df, ytr_df = get_xy_from_data(df_train, target_features)
        xval_df, yval_df = get_xy_from_data(df_val,   target_features)

        # convert to tensors
        self.x_train = torch.tensor(xtr_df.values, dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(ytr_df.values, dtype=torch.float32, device=self.device)
        self.x_val   = torch.tensor(xval_df.values, dtype=torch.float32, device=self.device)
        self.y_val   = torch.tensor(yval_df.values, dtype=torch.float32, device=self.device)

        # dimensions
        self.features_dim = self.x_train.shape[1]
        self.targets_dim = self.y_train.shape[1]
        #Generate one row of data per action
        self.T = 1

        # parameters
        self.action_dim = action_dim
        self.horizon = horizon
        self.base_agent = base_agent
        self.oneshot = oneshot
        self.acc_mul = accuracy_reward_multiplier
        self.action_range = action_range

        # compute normalization stats for actions (features + labels)
        mean_x = xtr_df.values.mean(axis=0)
        std_x  = xtr_df.values.std(axis=0) + 1e-6
        mean_y = ytr_df.values.mean(axis=0)
        std_y  = ytr_df.values.std(axis=0) + 1e-6
        action_mean = np.concatenate([mean_x, mean_y], axis=0).astype(np.float32)
        action_std  = np.concatenate([std_x,  std_y ], axis=0).astype(np.float32)

        self.action_mean = torch.from_numpy(action_mean).to(self.device)
        self.action_std  = torch.from_numpy(action_std).to(self.device)

        # feature column lookups
        cols = list(xtr_df.columns)
        self.timestamp_idx = cols.index("Timestamp")
        self.sex_idx       = cols.index("Sex - Female")


        self.state_shape  = (5,)
        self.action_shape = (self.features_dim + self.targets_dim,)

    def reset(self):
        self.synthetic_data   = torch.zeros((self.T, self.features_dim), device=self.device)
        self.synthetic_labels = torch.zeros((self.T, self.targets_dim), device=self.device)

        self.i = 0

        init_mf = self.x_train[:, self.sex_idx].mean().item()

        self.state = generate_state(
            self.x_train,
            self.timestamp_idx,
            init_mf,
            torch.tensor(0.0, device=self.device),
            torch.Generator(self.device).manual_seed(int(self.acc_mul * 1e3)),
            self.device,
        )
        return self.state

    def step(self, action):
        # action is normalized: invert normalization 
        a_norm = torch.tensor(action, dtype=torch.float32, device=self.device)
        a = a_norm.cpu() * self.action_std + self.action_mean

        #flatten and split
        a_vec = a.view(-1)
        timestamp = self.state[0].unsqueeze(0)
        age = self.state[1].unsqueeze(0)
        activity_id = self.state[2].unsqueeze(0)

        features = torch.cat([
                timestamp, #timestamp from state
                activity_id,
                a_vec[1:-5], #all sensor data
                a_vec[-3:-2] #"Resting HR", "Max HR",
        ])
        self.synthetic_data[self.i] = features

        labels = torch.cat([
            a_vec[-1],   # Sex,
            a_vec[1], #Heart Rate
            age,      # Age from state
            a_vec[-4],    # Weight
            a_vec[-5] #Height
        ], dim=0)

        self.synthetic_labels[self.i] = labels

        self.i += 1
        done_flag = (self.i == self.T)

        reward_value = 0.0

        #Generation finished
        if done_flag:
            print(f'Finished Data generation, evaluating baseline agent...')
            self.base_agent.reset()
            Xc = torch.cat([self.x_train, self.synthetic_data], dim=0)
            Yc = torch.cat([self.y_train, self.synthetic_labels], dim=0)
            ds = TensorDataset(Xc, Yc)
            loader = DataLoader(ds, batch_size=Xc.size(0), shuffle=False)
            with torch.enable_grad():
                _ = self.base_agent.train(loader)

            val_mse, _, _= evaluate_model(
                self.base_agent, self.x_val, self.y_val, sex_idx=self.sex_idx
            )
            reward_value = -self.acc_mul * val_mse

        reward = torch.tensor([reward_value], dtype=torch.float32, device=self.device)
        done   = torch.tensor([done_flag],    dtype=torch.bool,  device=self.device)

        #Provide new state
        if not done_flag:
            mf = self.synthetic_data[: self.i, self.sex_idx].mean().item()
            self.state = generate_state(
                self.x_train,
                self.timestamp_idx,
                mf,
                torch.tensor(float(self.i), device=self.device),
                torch.Generator(self.device).manual_seed(int(self.acc_mul * 1e3)),
                self.device,
            )
            state = self.state
        else:
            state = torch.zeros(self.state_shape, dtype=torch.float32, device=self.device)

        return state, reward, done, {"step": self.i}