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
        Generate a single synthetic data sample
        
        Args:
            state: The current state
            
        Returns:
            Synthetic data sample (a torch.Tensor)
        """

        # already a Tensor—clone+detach to avoid in-place/grad issues
        state = state.clone().detach().float().to(self.device)

        # Ensure state is properly shaped for the network
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Set networks to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Get action mean and log_std from actor network
            mean, log_std = self.actor(state)
            
            # Create normal distribution
            std = log_std.exp()
            dist = Normal(mean, std)
            
            # Sample action from distribution
            action = dist.sample()
    
            # Calculate log probability of the action
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Get value estimate from critic
            value = self.critic(state)

        # Return the synthetic data (action) as a torch.Tensor
        return action.squeeze(0).cpu()

    def learn_trajectory(self, states, actions, rewards, next_states, dones, lam: float = 0.95):
        """
        PPO update from a single trajectory.

        Args:
            states:       list/array/tensor of shape [T, state_size]
            actions:      list/array/tensor of shape [T, action_size]
            rewards:      list/array/tensor of shape [T]
            next_states:  list/array/tensor of shape [T, state_size]  (s_{t+1} for each t)
            dones:        list/array/tensor of shape [T] with {0,1}
            lam:          GAE lambda (default 0.95)
        """

        T = states.shape[0]

        # ---- Compute values and GAE advantages (bootstrapped) ----
        with torch.no_grad():
            values       = self.critic(states).squeeze(-1)                 # [T]
            next_values  = self.critic(next_states).squeeze(-1)            # [T]
            deltas       = rewards + self.gamma * next_values * (1.0 - dones) - values  # [T]

            advantages = torch.zeros_like(deltas, device=self.device)      # [T]
            gae = 0.0
            for t in reversed(range(T)):
                gae = deltas[t] + self.gamma * lam * (1.0 - dones[t]) * gae
                advantages[t] = gae

            returns = values + advantages                                  # [T]

            # Old log probs under behavior policy (assumed current policy; for off-policy, store these at collection time)
            mean_old, log_std_old = self.actor(states)
            dist_old = Normal(mean_old, log_std_old.exp())
            old_log_probs = dist_old.log_prob(actions).sum(dim=-1)         # [T]

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