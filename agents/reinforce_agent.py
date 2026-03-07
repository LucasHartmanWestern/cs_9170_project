import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class PolicyNetwork(nn.Module):  # Policy network for continuous actions 
    def __init__(self, state_size, action_size, layers):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.ModuleList()

        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(state_size, layer_size)
            else:
                linear_layer = nn.Linear(layers[i-1], layer_size)

            self.layers.append(linear_layer)

        self.mean_layer = nn.Linear(layers[-1], action_size)
        self.log_std_layer = nn.Linear(layers[-1], action_size)


    def forward(self, state):
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class ReinforceAgent():
    def __init__(self, state_size, action_size, hidden_sizes, total_episodes, lr, gamma, entropy_start=1e-2, entropy_end=0.0, seed=42, device='cpu'):
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = device
        _dev = device if isinstance(device, torch.device) else torch.device(device if device else "cpu")
        self._action_gen = torch.Generator(device=_dev).manual_seed(seed)
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_sizes).to(self.device)
        try:
            self.policy_network = torch.compile(self.policy_network)
        except Exception:
            pass  # torch.compile requires PyTorch >= 2.0
        
        self.total_episodes = total_episodes
        self.gamma = gamma
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        
        

    def entropy_annealing(self, ep_idx, total_episodes, start=1e-2, end=0.0, schedule="linear"):
        """Return an annealed entropy coefficient for episode ep_idx in [0, total_episodes-1]."""
        t = min(max(ep_idx / max(total_episodes - 1, 1), 0.0), 1.0)  # normalized progress 0→1

        if schedule == "linear":
            return start + (end - start) * t
        elif schedule == "cosine":  # smooth start/end
            return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))
        elif schedule == "exp":
            # avoids zero; if end==0, set end to a tiny value like 1e-5
            e = end if end > 0 else 1e-5
            return start * (e / start) ** t
        else:
            return start


    def compute_loss(self, experiences, entropy_coef=0.0):
        states, actions, rewards, _, dones = experiences  # rewards,dones: [T]
        dones = dones.to(dtype=rewards.dtype)

        # 1) Discounted returns 
        T = rewards.shape[0]
        returns = torch.empty_like(rewards)              # [T]
        G = torch.zeros((), device=rewards.device)       # scalar on correct device
        for t in range(T - 1, -1, -1):                   # backward through time
            G = rewards[t] + self.gamma * G * (1.0 - dones[t])
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3) Policy + log-probs
        mean, log_std = self.policy_network(states)          # [T, A], [T, A]
        inv_std = torch.exp(-log_std)
        diff = (actions - mean) * inv_std               # (a - μ)/σ
        LOG_2PI = 1.8378770664093453                    # ln(2π)
        # Log-prob per-dim, then sum over action dims
        log_probs = (-0.5 * (diff * diff + 2.0 * log_std + LOG_2PI)).sum(dim=-1)  # [T]
        # Entropy per-dim: 0.5*(1+ln(2π)) + log σ
        entropy = (0.5 * (1.0 + LOG_2PI) + log_std).sum(dim=-1)  

        # 4) REINFORCE objective (+ optional entropy bonus)
        loss = -(log_probs * returns).mean() - entropy_coef * entropy.mean()
        return loss

    def learn_trajectory(self, states, actions, rewards, next_states, dones, episode_idx):
        experiences_torch = (states, actions, rewards, next_states, dones)
        entropy_coef = self.entropy_annealing(episode_idx, self.total_episodes, start=self.entropy_start, end=self.entropy_end)
        loss = self.compute_loss(experiences_torch, entropy_coef=entropy_coef)

        self.optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), entropy_coef


    def predict(self, state):
        policy = self.policy_network

        state_t = state.to(self.device).float()
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)  # [1, state_size]

        with torch.no_grad():
            mean, log_std = policy(state_t)                          # [1, action_size] each
            std = torch.exp(log_std)        # diagonal Gaussian
            noise = torch.randn(mean.shape, generator=self._action_gen, device=mean.device, dtype=mean.dtype)
            action = (mean + std * noise).squeeze(0)                   # [action_size] (e.g., [2])

        return action.detach()
