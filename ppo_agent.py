import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

# Actor network: outputs logits for each action
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

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
                 c1=0.5, c2=0.01):
        """
        state_size: dimension of the state space
        action_size: number of possible actions
        hidden_size: hidden layer size for both networks
        lr: learning rate
        gamma: discount factor
        clip_epsilon: clipping parameter for PPO
        update_epochs: number of epochs to update the policy per training iteration
        batch_size: minibatch size for PPO updates
        c1: coefficient for the value loss
        c2: coefficient for the entropy bonus
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.c1 = c1
        self.c2 = c2

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
        Given a state, select an action using the current policy
        Returns the chosen action
        
        Note: This method also computes the log probability and state value,
        which you should store along with the reward and done signal after taking the action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor)

        self.last_log_prob = log_prob.item()
        self.last_value = value.item()
        self.last_action = action.item()
        
        return action.item()
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """
        Helper method to store a transition in memory
        """
        self.memory.append((state, action, log_prob, reward, done, value))
    
    def train(self):
        """
        Train the PPO agent using the collected trajectories in memory
        This method computes the returns and advantages, then performs several epochs
        of mini-batch updates using the PPO clipped objective
        
        After training, the memory is cleared
        """
        if len(self.memory) == 0:
            return
        
        # Unpack memory
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)
        values = torch.FloatTensor(values).to(self.device)
        
        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Compute advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        for epoch in range(self.update_epochs):
            # Shuffle indices for mini-batch sampling
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mini_idx = indices[start:end]
                
                mini_states = states[mini_idx]
                mini_actions = actions[mini_idx]
                mini_old_log_probs = old_log_probs[mini_idx]
                mini_returns = returns[mini_idx]
                mini_advantages = advantages[mini_idx]
                
                # Recompute log probabilities and state values under the current policy
                logits = self.actor(mini_states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mini_actions)
                entropy = dist.entropy().mean()
                new_values = self.critic(mini_states).squeeze()
                
                # Compute ratio (new probabilities divided by old probabilities)
                ratio = torch.exp(new_log_probs - mini_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (value function loss)
                critic_loss = F.mse_loss(new_values, mini_returns)
                
                # Total loss with entropy bonus
                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                
                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear memory after training
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
