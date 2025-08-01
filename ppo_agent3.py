import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.body(x)
        return self.mean(x), self.log_std.expand_as(self.mean(x))

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
class PPOAgent:
    def __init__(self, state_size, action_size, lr, gamma, clip_epsilon, lam, seed, device="cpu"):
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.rng = torch.Generator().manual_seed(self.seed)
        self.device = device

        
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_epsilon

        self.memory = []
        self.pending_trajectory = []  # to hold intermediate steps

    def store(self, state, action, done, value, log_prob):
        # Store a step with zero reward (final reward added on terminal step)
        self.pending_trajectory.append((state, action, 0.0, done, value, log_prob))

    def finish_trajectory(self, final_reward):
        """
        Called at the end of an episode, adds the terminal reward to the last step.
        """
        # Replace the reward of the last transition
        if self.pending_trajectory:
            last_step = self.pending_trajectory[-1]
            updated_step = (last_step[0], last_step[1], final_reward, last_step[3], last_step[4], last_step[5])
            self.pending_trajectory[-1] = updated_step
            self.memory.extend(self.pending_trajectory)
            self.pending_trajectory = []

    def compute_gae(self):
        states, actions, rewards, dones, values, log_probs = zip(*self.memory)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(values + (0,), dtype=torch.float32, device=self.device)
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = values[:-1] + torch.tensor(advantages, device=self.device)
        return (torch.stack(states), torch.stack(actions), 
                torch.tensor(advantages, device=self.device), 
                returns.detach(), torch.stack(log_probs))

    def update(self, epochs=10, batch_size=64):
        states, actions, advs, returns, old_log_probs = self.compute_gae()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for s, a, logp_old, ret, adv in loader:
                mean, log_std = self.actor(s)
                std = log_std.exp().clamp(min=1e-6)
                dist = Normal(mean, std)
                logp = dist.log_prob(a).sum(-1)
                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = torch.min(surr1, surr2).mean()
                value = self.critic(s)
                critic_loss = F.mse_loss(value, ret)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.memory = []

    def save(self, path):
        """
        Save the actor and critic network parameters to the specified path
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
