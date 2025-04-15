import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal

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
                 c1=0.5, c2=0.01, action_std=0.5):
        """
        state_size: dimension of the state space
        action_size: dimension of continuous action space (synthetic data size)
        hidden_size: hidden layer size for both networks
        lr: learning rate
        gamma: discount factor
        clip_epsilon: clipping parameter for PPO
        update_epochs: number of epochs to update the policy per training iteration
        batch_size: minibatch size for PPO updates
        c1: coefficient for the value loss
        c2: coefficient for the entropy bonus
        action_std: initial standard deviation for continuous actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.c1 = c1
        self.c2 = c2
        self.action_std = action_std

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # Memory for storing trajectories (on-policy data)
        # Each entry is a tuple: (state, action, log_prob, reward, done, value)
        self.memory = []
    
    def predict(self, state):
        """
        Generate a single synthetic data sample
        
        Args:
            state: The current state
            
        Returns:
            Synthetic data sample (a list of floats)
        """
        # Convert state to tensor if it's not already
        if isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
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
        return action.cpu().numpy().flatten().tolist()
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """
        Helper method to store a transition in memory
        """
        self.memory.append((state, action, log_prob, reward, done, value))
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update policy and value networks using PPO algorithm
        
        Args:
            state: Current state
            action: Action taken (synthetic data generated)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert inputs to tensors if they aren't already
        if isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if isinstance(action, np.ndarray) or isinstance(action, list):
            action = torch.FloatTensor(action).to(self.device)
        
        if isinstance(reward, (int, float)):
            reward = torch.FloatTensor([reward]).to(self.device)
        
        if isinstance(next_state, list):
            next_state = torch.FloatTensor(next_state).to(self.device)
        elif isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        
        if isinstance(done, bool):
            done = torch.FloatTensor([float(done)]).to(self.device)
        
        # Ensure states have batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)
        
        # Set networks to evaluation mode for data collection
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Get action distribution parameters
            mean, log_std = self.actor(state)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            # Calculate log probability of the action
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Get value estimate
            value = self.critic(state)
            
            # Get next state value (if not done)
            next_value = 0
            if not done.item():
                next_value = self.critic(next_state).item()
            
            # Calculate target value using TD(0)
            target_value = reward.item() + self.gamma * next_value * (1 - done.item())
            
            # Calculate advantage
            advantage = target_value - value.item()
        
        # Store transition in memory
        self.store_transition(
            state.cpu().numpy(),
            action.cpu().numpy(),
            log_prob.item(),
            reward.item(),
            done.item(),
            value.item()
        )
        
        # If we have enough samples, perform PPO update
        if len(self.memory) >= self.batch_size:
            # Set networks to training mode
            self.actor.train()
            self.critic.train()
            
            # Prepare data from memory
            states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            
            # Calculate returns and advantages
            returns = []
            advantages = []
            for i in range(len(rewards)):
                if dones[i]:
                    ret = rewards[i]
                    adv = ret - values[i]
                else:
                    next_idx = min(i + 1, len(values) - 1)
                    ret = rewards[i] + self.gamma * values[next_idx]
                    adv = ret - values[i]
                returns.append(ret)
                advantages.append(adv)
            
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update for multiple epochs
            for _ in range(self.update_epochs):
                # Process in minibatches
                for start_idx in range(0, len(self.memory), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(self.memory))
                    batch_indices = slice(start_idx, end_idx)
                    
                    # Get batch data
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    # Get current action distribution
                    means, log_stds = self.actor(batch_states)
                    stds = log_stds.exp()
                    dist = Normal(means, stds)
                    
                    # Calculate current log probabilities
                    curr_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    
                    # Fix the dimension issue - instead of reshaping, ensure both tensors are 1D
                    if len(curr_log_probs.shape) > 1:
                        curr_log_probs = curr_log_probs.squeeze()
                    if len(batch_old_log_probs.shape) > 1:
                        batch_old_log_probs = batch_old_log_probs.squeeze()
                    
                    # Calculate entropy
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratio for PPO
                    ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    
                    # Get value predictions
                    values = self.critic(batch_states).squeeze()
                    
                    # Calculate losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, batch_returns)
                    total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                    
                    # Update networks
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            
            # Clear memory after update
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
