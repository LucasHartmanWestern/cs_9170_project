import numpy as np
import pandas as pd
from collections import deque

class Environment:
    """
    Environment for generating high-usage synthetic data only (low_usage bias).
    State: sliding window of past observations + time embeddings + sampled target.
    Reward: per-step heterogenity shaping.
    """
    def __init__(self,
                 train_df: pd.DataFrame,
                 threshold: float,
                 features: list,
                 target: str,
                 baseline_mse: float,
                 male_hi: np.ndarray,
                 max_samples: int,
                 window_size: int,
                 seed: int = 42):
                 
        self.train_df      = train_df.reset_index(drop=True)
        self.threshold     = threshold
        self.features      = features
        self.target        = target
        self.baseline_mse  = baseline_mse
        self.male_hi       = np.array(male_hi, dtype=np.float32)
        self.max_samples   = max_samples
        self.window_size   = window_size

        np.random.seed(seed)

        # Buffers for warm-start and generation
        self.history_buffer   = deque(maxlen=self.window_size)
        self.generated_buffer = deque(maxlen=self.window_size)
        self.heterogenity_rewards = []
        self.step_count = 0

    def sample_target(self):
        #Random from male distribution for now
        if isinstance(self.male_hi, pd.DataFrame):
            vals = self.male_hi[self.target].values
        elif isinstance(self.male_hi, pd.Series):
            vals = self.male_hi.values
        else:
            vals = np.asarray(self.male_hi)
        if vals.ndim > 1:
            vals = vals.flatten()
        return np.random.choice(vals)
    def generate_state(self, buffer_length=0):
        # 1) Draw a target_income from the male high‑income distribution
        target_income = self.sample_target()
        
        # 2) Compute fraction of budget already used
        n_done    = buffer_length
        frac_done = n_done / self.max_samples
        
        # 3) Return the state vector
        return np.array([target_income, frac_done], dtype=np.float32)

    def reset(self):
        # Select high-income indices
        idxs = np.where(self.train_df[self.target] >= self.threshold)[0]
        if len(idxs) <= self.window_size:
            raise ValueError("Not enough high-income samples for warm start.")
        start = np.random.choice(idxs[self.window_size:]) - self.window_size
        seq = self.train_df.iloc[start:start + self.window_size]

        # Populate history buffer
        self.history_buffer.clear()
        for _, row in seq.iterrows():
            obs = row[self.features + [self.target]].values.astype(np.float32)
            self.history_buffer.append(obs)

        # Clear generation buffers and counters
        self.generated_buffer.clear()
        self.heterogenity_rewards.clear()
        self.step_count = 0

        # Initial state
        state_vec = self.generate_state()
        self.current_target = state_vec[0]
        return state_vec

    def step(self, action: np.ndarray, buffer_length):

        # Record generated sample (features + target)
        new_obs = np.concatenate([action, [self.current_target]])
        self.generated_buffer.append(new_obs)

        # Update history buffer
        self.history_buffer.append(new_obs)

        # Generate next state
        state_vec = self.generate_state(buffer_length)
        self.current_target = state_vec[0]

        self.step_count += 1
        done = False
        info = {'sampled_target': state_vec[0]}
        return state_vec, h_r, done, info

    def get_episode_data(self):
        return list(self.heterogenity_rewards), np.array(self.generated_buffer)
