import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

# torch.backends.cudnn.deterministic        = True
# torch.backends.cudnn.benchmark            = False
# torch.use_deterministic_algorithms(True)

# Actor network: outputs mean and log_std for continuous actions
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        return mean, self.log_std.expand_as(mean)

# Critic network: outputs a scalar state-value
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=3e-4,
                 gamma=0.99, clip_epsilon=0.2, update_epochs=4, batch_size=64,
                 c1=0.5, c2=0.01, action_std=0.5, device='cpu', seed=42):

        # Set random seed for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.rng = torch.Generator().manual_seed(self.seed)
        self.device = device

        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.c1 = c1
        self.c2 = c2
        self.action_std = action_std

        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.memory = []
    
    def predict(self, state):
        """
        Sample one action from the current policy.

        Returns:
            action   (Tensor [action_size])  — the delta to apply
            log_prob (Tensor scalar)         — log-prob under behavior policy,
                                               stored in the trajectory buffer
                                               for the PPO importance-ratio
        """
        state = state.clone().detach().float().to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            mean, log_std = self.actor(state)
            dist = Normal(mean, log_std.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu(), log_prob.squeeze(0).cpu()

    def learn_trajectory(self, states, actions, old_log_probs, rewards, next_states, dones,
                         gae_lambda: float = 0.95):
        """
        PPO update from a single trajectory.

        Args:
            states:        Tensor [T, state_size]
            actions:       Tensor [T, action_size]
            old_log_probs: Tensor [T] — log-probs recorded at rollout time under
                           the behavior policy.  Must NOT be recomputed here;
                           recomputing from the already-updated actor breaks the
                           importance ratio.
            rewards:       Tensor [T]
            next_states:   Tensor [T, state_size]
            dones:         Tensor [T]  (bool or float 0/1)
            gae_lambda:    GAE λ (default 0.95)
        """
        T = states.shape[0]

        # ---- Compute values and GAE advantages (bootstrapped) ----
        with torch.no_grad():
            values       = self.critic(states).squeeze(-1)                          # [T]
            next_values  = self.critic(next_states).squeeze(-1)                     # [T]
            deltas       = rewards + self.gamma * next_values * (1.0 - dones.float()) - values

            advantages = torch.zeros_like(deltas, device=self.device)
            gae = 0.0
            for t in reversed(range(T)):
                gae = deltas[t] + self.gamma * gae_lambda * (1.0 - dones[t].float()) * gae
                advantages[t] = gae

            returns = values + advantages                                            # [T]
            # old_log_probs is passed in from rollout — do NOT recompute here

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- Create mini-batches ----
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self.rng)

        # ---- PPO update ----
        for _ in range(self.update_epochs):
            for b_states, b_actions, b_old_logp, b_returns, b_advs in loader:
                mean, log_std = self.actor(b_states)
                dist = Normal(mean, log_std.exp())

                logp    = dist.log_prob(b_actions).sum(dim=-1)            # [B]
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (logp - b_old_logp).exp()
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                v_pred = self.critic(b_states).squeeze(-1)
                critic_loss = F.mse_loss(v_pred, b_returns)

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # Optional but recommended:
                # torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()

        return


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
        
    def load(self, path):
        """
        Load the actor and critic network parameters from the specified path
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Model loaded from {path}")