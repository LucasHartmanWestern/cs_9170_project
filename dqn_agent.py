import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# NN for approximating Q-values
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3,
                 gamma=0.99, batch_size=32, memory_size=10000,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_size: Number of neurons in hidden layers
        lr: Learning rate for optimizer
        gamma: Discount factor
        batch_size: Number of experiences to sample for training
        memory_size: Maximum size of the replay buffer
        epsilon_*: Parameters for the epsilon-greedy policy
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Q-Network
        self.model = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Epsilon parameters for ε-greedy action selection
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def predict(self, state):
        """
        Given a state, select an action using an epsilon-greedy policy.
        state: A numpy array representing the current state.
        Returns an integer representing the chosen action.
        """
        # With probability epsilon, choose a random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Otherwise, choose the action with the highest predicted Q-value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition (experience) in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        Sample a batch from memory and update the Q-network
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train
        
        # Sample a random batch from the memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.stack([np.array(s) for s in states])).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([np.array(ns) for ns in next_states])).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute the Q-values for the current states
        q_values = self.model(states).gather(1, actions)
        
        # Compute Q-values for the next states
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-values using the Bellman equation
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculate the loss
        loss = self.criterion(q_values, expected_q_values)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save(self, path):
        """
        Save the Q-network weights to the specified path
        """
        torch.save(self.model.state_dict(), path)