import numpy as np
import pandas as pd
from collections import deque

class Environment:
    def __init__(self,
                 target: str,
                 male_hi: np.ndarray,
                 max_actions: int,
                 seed: int = 42):

        self.target        = target
        self.male_hi       = np.array(male_hi, dtype=np.float32)
        self.max_actions   = max_actions
        self.seed = seed

        np.random.seed(seed)

    def sample_target(self):
        #Sample From male distribution
        if isinstance(self.male_hi, pd.DataFrame):
            vals = self.male_hi[self.target].values
        elif isinstance(self.male_hi, pd.Series):
            vals = self.male_hi.values
        else:
            vals = np.asarray(self.male_hi)
        if vals.ndim > 1:
            vals = vals.flatten()
        return np.random.choice(vals)

    def generate_state(self, curr_length=0):
        #Draw a target_income from the male high‑income distribution
        #target_income = self.sample_target()

        #Compute distance to end of trajectory
        frac_done = curr_length / self.max_actions
        
        #Return the state vector, forced target to be minority
        return np.array([1, frac_done], dtype=np.float32)

    def reset(self):
        np.random.seed(self.seed)
        # Initial state
        state_vec = self.generate_state()

        return state_vec

    def step(self, action: np.ndarray, curr_length):
        # Generate next state
        state_vec = self.generate_state(curr_length)

        done = curr_length >= self.max_actions
        info = {'sampled_target': state_vec[0]}
        return state_vec, done, info

    def get_episode_data(self):
        return list(self.heterogenity_rewards), np.array(self.generated_buffer)
