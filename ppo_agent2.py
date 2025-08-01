import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


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
                 c1=0.5, c2=0.01, action_std=0.5, lam= 0.95, device='cpu', seed=42):

        # Set random seed for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.rng = torch.Generator().manual_seed(self.seed)
        self.device = device

        self.gamma = gamma
        self.lam = lam
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
            Synthetic data sample (a list of floats)
        """

        if torch.is_tensor(state):
            # already a Tensor—clone+detach to avoid in-place/grad issues
            state = state.clone().detach().float().to(self.device)
        else:
            # numpy array or list
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure state is properly shaped for the network
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if needed
        
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
        
        # Return the synthetic data (action) as a list of floats
        return action.squeeze(0).cpu().tolist()
    
    # def learn(self, state, action, reward, next_state, done):
    #     """
    #     Update policy and value networks using PPO algorithm
        
    #     Args:
    #         state: Current state
    #         action: Action taken (synthetic data generated)
    #         reward: Reward received
    #         next_state: Next state
    #         done: Whether episode is done
    #         random_state: Random seed for reproducibility
    #     """
    #     torch.manual_seed(self.seed)
    #     torch.cuda.manual_seed_all(self.seed)
    #     self.rng.manual_seed(self.seed)

    #     # Convert inputs to tensors if they aren't already
    #     state      = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
    #     action     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
    #     reward     = torch.as_tensor(reward,   dtype=torch.float32, device=self.device)
    #     next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
    #     done       = torch.as_tensor([float(done)], dtype=torch.float32, device=self.device)
        
    #     # Ensure states have batch dimension
    #     if len(state.shape) == 1:
    #         state = state.unsqueeze(0)
    #     if len(next_state.shape) == 1:
    #         next_state = next_state.unsqueeze(0)
        
    #     # Set networks to evaluation mode for data collection
    #     self.actor.eval()
    #     self.critic.eval()
        
    #     with torch.no_grad():
    #         # Get action distribution parameters
    #         mean, log_std = self.actor(state)
    #         std = log_std.exp()
    #         dist = Normal(mean, std)
            
    #         # Calculate log probability of the action
    #         log_prob = dist.log_prob(action).sum(dim=-1)
            
    #         # Get value estimate
    #         value = self.critic(state).squeeze(-1) 
            
    #         # Get next state value (if not done)
    #         next_value = 0
    #         if not done.item():
    #             next_value = self.critic(next_state).item()
            
    #         # Calculate target value using TD(0)
    #         target_value = reward.item() + self.gamma * next_value * (1 - done.item())
            
    #         # Calculate advantage
    #         advantage = target_value - value.item()

    #     if action.dim() == 1:
    #         action = action.unsqueeze(0)
        
    #     # Store transition in memory
    #     self.memory.append((
    #         state,                    # [1, S]
    #         action,                   # [1, A]
    #         log_prob,                 # scalar tensor
    #         torch.as_tensor([reward.item()], dtype=torch.float32, device=self.device),  
    #         done,                     # [1]
    #         value       # [1]
    #     ))
        
    #     # If we have enough samples, perform PPO update
    #     if len(self.memory) >= self.batch_size:
    #         # Set networks to training mode
    #         self.actor.train()
    #         self.critic.train()
            
    #         # Prepare data from memory
    #         states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
    #         states         = torch.cat(states, dim=0)            # [N, S]
    #         actions        = torch.cat(actions, dim=0)           # [N, A]
    #         old_log_probs  = torch.stack(old_log_probs).squeeze()# [N]
    #         rewards        = torch.stack(rewards).squeeze()      # [N]
    #         dones          = torch.stack(dones).squeeze()        # [N]
    #         values         = torch.cat(values).squeeze()         # [N]

    #         # Calculate returns and advantages
    #         returns = []
    #         advantages = []
    #         for i in range(len(rewards)):
    #             if dones[i]:
    #                 ret = rewards[i]
    #                 adv = ret - values[i]
    #             else:
    #                 next_idx = min(i + 1, len(values) - 1)
    #                 ret = rewards[i] + self.gamma * values[next_idx]
    #                 adv = ret - values[i]
    #             returns.append(ret)
    #             advantages.append(adv)
            
    #         returns    = torch.stack(returns)
    #         advantages = (torch.stack(advantages) - torch.stack(advantages).mean()) \
    #                     / (torch.stack(advantages).std() + 1e-8)
            
    #         dataset = torch.utils.data.TensorDataset(
    #             states, actions, old_log_probs, returns, advantages
    #         )
    #         loader = torch.utils.data.DataLoader(
    #             dataset,
    #             batch_size=self.batch_size,
    #             shuffle=True,
    #             generator=self.rng
    #         )


    #         # PPO update for multiple epochs
    #         for _ in range(self.update_epochs):
    #             # Process in minibatches
    #             for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:

    #                 # Get current action distribution
    #                 means, log_stds = self.actor(batch_states)
    #                 stds = log_stds.exp()
    #                 dist = Normal(means, stds)
                    
    #                 # Calculate current log probabilities
    #                 curr_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
    #                 entropy = dist.entropy().mean()
                    
    #                 # Fix the dimension issue - instead of reshaping, ensure both tensors are 1D
    #                 if len(curr_log_probs.shape) > 1:
    #                     curr_log_probs = curr_log_probs.squeeze()
    #                 if len(batch_old_log_probs.shape) > 1:
    #                     batch_old_log_probs = batch_old_log_probs.squeeze()
                    
    #                 # Calculate ratio for PPO
    #                 ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                    
    #                 # Calculate surrogate losses
    #                 surr1 = ratio * batch_advantages
    #                 surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    
    #                 # Get value predictions
    #                 values = self.critic(batch_states).squeeze(-1)
                    
    #                 # Calculate losses
    #                 actor_loss = -torch.min(surr1, surr2).mean()
    #                 critic_loss = F.mse_loss(values, batch_returns)
    #                 total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                    
    #                 # Update networks
    #                 self.optimizer.zero_grad()
    #                 total_loss.backward()
    #                 self.optimizer.step()
            
    #         # Clear memory after update
    #         self.memory = []

    def learn(self, states, actions, rewards, next_states, dones):
        """
        Learn from a full trajectory of transitions using PPO.
    
        Args:
            states: List of state tensors (each [state_dim])
            actions: List of action tensors (each [action_dim])
            rewards: List of scalar rewards (float or 0-D tensor)
            next_states: List of next state tensors (each [state_dim])
            dones: List of done flags (bools or 0-D tensors)
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.rng.manual_seed(self.seed)
    
        self.actor.eval()
        self.critic.eval()
    
        # Lists for batched tensors
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        value_batch = []
        log_prob_batch = []
    
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            state = state.to(self.device).unsqueeze(0)
            action = action.to(self.device).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
    
            with torch.no_grad():
                mean, log_std = self.actor(state)
                std = log_std.exp()
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state).squeeze(-1)
    
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
            value_batch.append(value)
            log_prob_batch.append(log_prob)
    
        # Stack into single tensors
        states = torch.cat(state_batch, dim=0)
        actions = torch.cat(action_batch, dim=0)
        rewards = torch.cat(reward_batch).squeeze()
        dones = torch.cat(done_batch).squeeze()
        values = torch.stack(value_batch).squeeze()
        old_log_probs = torch.stack(log_prob_batch).squeeze()
    
        # Compute returns and advantages (GAE)
        returns, advantages = [], []
        gae = 0
        next_value = 0
    
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
    
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
        # Store trajectory
        self.memory.append((states, actions, old_log_probs, returns, advantages))
    
        if len(self.memory) >= self.batch_size:
            self.actor.train()
            self.critic.train()
    
            all_states, all_actions, all_log_probs, all_returns, all_advs = zip(*self.memory)
            states = torch.cat(all_states, dim=0)
            actions = torch.cat(all_actions, dim=0)
            old_log_probs = torch.cat(all_log_probs, dim=0)
            returns = torch.cat(all_returns, dim=0)
            advantages = torch.cat(all_advs, dim=0)
    
            dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self.rng)
    
            for _ in range(self.update_epochs):
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advs in loader:
                    mean, log_std = self.actor(batch_states)
                    eps = 1e-6  # small positive constant
                    std = torch.exp(log_std).clamp(min=eps)
                    # std = log_std.exp()
                    dist = Normal(mean, std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().mean()
    
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advs
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advs
    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_preds = self.critic(batch_states).squeeze(-1)
                    critic_loss = F.mse_loss(value_preds, batch_returns)
                    total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
    
                    self.optimizer.zero_grad()
                    total_loss.backward()
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
        
    def load(self, path):
        """
        Load the actor and critic network parameters from the specified path
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Model loaded from {path}")