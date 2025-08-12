import numpy as np
import pandas as pd
from collections import deque
import torch

class Environment:
    def __init__(self,
                 target: str,
                 max_actions: int,
                 device,
                 seed: int = 42):

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target = target
        self.max_actions = int(max_actions)
        self.seed = seed

        np.random.seed(seed)

    def generate_state(self, curr_length=0):
        #Compute distance to end of trajectory
        frac_done = float(curr_length) / float(self.max_actions)
        
        #Return the state vector, forced target to be minority
        return torch.tensor([1.0, frac_done], dtype=torch.float32, device=self.device)

    def reset(self):
        state_vec = self.generate_state()

        return state_vec

    def step(self, action, curr_length):
        # Generate next state
        state_vec = self.generate_state(curr_length)

        done = curr_length >= self.max_actions
        info = {'sampled_target': state_vec[0].to(torch.long)}
        return state_vec, done, info

    def get_episode_data(self):
        return list(self.heterogenity_rewards), np.array(self.generated_buffer)
