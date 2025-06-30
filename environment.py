import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_processing import get_xy_from_data
from training       import generate_state, evaluate_model

class Environment:
    def __init__(self, df_train, df_val, target_features, base_agent, action_range, horizon, oneshot, accuracy_reward_multiplier, seed, device ):

        self.device = torch.device(device)
        torch.manual_seed(seed)

        # load real data
        xtr_df, ytr_df = get_xy_from_data(df_train, target_features)
        xval_df, yval_df = get_xy_from_data(df_val,   target_features)

        self.x_train = torch.tensor(xtr_df.values,  dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(ytr_df.values,  dtype=torch.float32, device=self.device)
        self.x_val   = torch.tensor(xval_df.values, dtype=torch.float32, device=self.device)
        self.y_val   = torch.tensor(yval_df.values, dtype=torch.float32, device=self.device)

        self.D = self.x_train.shape[1]
        self.L = self.y_train.shape[1]
        self.T = horizon

        self.horizon = horizon
        self.base_agent = base_agent
        self.oneshot = oneshot
        self.acc_mul  = accuracy_reward_multiplier
        self.base_agent = base_agent

        self.sex_idx = 34
        self.timestamp_idx = 0
        self.seed = seed

        self.action_range = action_range

        if self.oneshot:
            # one‐step: action must be shape (T, D+L)
            self.state_shape    = (1,)            # dummy scalar
            self.action_shape = (horizon, self.D + self.L)
        else:
            # multi‐step: action is (D+L,) each call
            self.state_shape    = (5,)            # (timestamp, mf_ratio, n_samples, age, activity_id)
            self.action_shape = (self.D + self.L,)
    
    def reset(self):
        if self.oneshot:
            return torch.empty(0)
        else:
            # stepwise mode: clear buffers + initial state
            self.synthetic_data   = torch.zeros((self.T, self.D),   device=self.device)
            self.synthetic_labels = torch.zeros((self.T, self.L),   device=self.device)
            self.sum_female = 0.0
            self.i          = 0

            init_mf = float(self.x_train[:, self.sex_idx].mean().item())
            self.state = generate_state(
                self.x_train, self.timestamp_idx,
                init_mf, torch.tensor(0.0, device=self.device),
                torch.Generator(self.device).manual_seed(int(self.acc_mul*1e3)),
                self.device
            )
            return self.state        

    #Generate a dataset
    def step(self, action):
        if self.oneshot:
            A = torch.tensor(action, device=self.device, dtype=torch.float32)
            Xs, Ys = A[:, :self.D], A[:, self.D:]
            # train+eval
            self.base_agent.reset()
            Xc = torch.cat([self.x_train, Xs], dim=0)
            Yc = torch.cat([self.y_train, Ys], dim=0)

            ds = TensorDataset(Xc, Yc)
            loader = DataLoader(ds, batch_size=Xc.size(0), shuffle=False)
            _ = self.base_agent.train(loader)

            val_mse, _ = evaluate_model(self.base_agent, self.x_val, self.y_val, sex_idx=32)
            reward = -self.acc_mul * float(val_mse)

            state = torch.empty(self.state_shape, dtype=torch.float32)
            done = True
            info = {"val_mse": val_mse}
            return state, reward, done, info
        else:
            a = torch.tensor(action, device=self.device, dtype=torch.float32)
            row   = a[:self.D]
            label = a[self.D:]
            idx = self.i
            self.synthetic_data[idx]   = row
            self.synthetic_labels[idx] = label

            self.i += 1
            done = (self.i == self.T)
            reward = 0.0

            if done:
                # train+eval at final step
                self.base_agent.reset()
                Xc = torch.cat([self.x_train,   self.synthetic_data],   dim=0)
                Yc = torch.cat([self.y_train, self.synthetic_labels], dim=0)
                ds = TensorDataset(Xc, Yc)
                loader = DataLoader(ds, batch_size=Xc.size(0), shuffle=False)
                _ = self.base_agent.train(loader)

                val_mse, _ = evaluate_model(self.base_agent, self.x_val, self.y_val, sex_idx=0)
                reward = -self.acc_mul * float(val_mse)

            # next state (agent sees it if not done)
            if not done:
                mf = float(self.synthetic_data[:self.i, self.sex_idx].mean().item())
                self.state = generate_state(
                    self.x_train, self.timestamp_idx,
                    mf, torch.tensor(float(self.i), device=self.device),
                    torch.Generator(self.device).manual_seed(int(self.acc_mul*1e3)),
                    self.device
                )
                state = self.state.cpu().numpy()
            else:
                state = np.zeros(self.state_shape, dtype=np.float32)  # or reuse last state

            info = {"step": self.i}
            return state, float(reward), done, info