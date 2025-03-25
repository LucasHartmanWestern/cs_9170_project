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
        Generate a single synthetic data sample
        
        Args:
            state: The current state
            
        Returns:
            Synthetic data sample (a list of integers)
        """
        # Convert state to tensor if it's not already
        if isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Ensure state is properly shaped for the network
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if needed
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            # Random action: generate a list of random integers
            synthetic_data = [random.randint(0, 1) for _ in range(self.action_size)]
        else:
            # Get Q-values from the model
            with torch.no_grad():
                q_values = self.model(state)
            
            # Convert Q-values to binary synthetic data (0 or 1)
            # Values above 0 become 1, below 0 become 0
            synthetic_data = [1 if q > 0 else 0 for q in q_values.cpu().numpy().flatten()]
        
        return synthetic_data

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition (experience) in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, state, action, reward, next_state, done):
        """
        Update policy and value networks using DQN algorithm
        
        Args:
            state: Current state
            action: Action taken (synthetic data generated)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store the experience in memory
        self.remember(state, action, reward, next_state, done)
        
        # Only start learning if we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Set model to training mode
        self.model.train()
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)
            
            # Ensure states have batch dimension
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
            
            # Get current Q-values
            current_q = self.model(state)
            
            # Get next Q-values
            with torch.no_grad():
                next_q = self.model(next_state)
                max_next_q = torch.max(next_q)
            
            # Calculate target Q-value
            target_q = reward + (1 - done) * self.gamma * max_next_q
            
            # Calculate loss
            # We need to reshape current_q to match the action shape for element-wise multiplication
            q_values = torch.sum(current_q * action)
            loss = self.criterion(q_values, target_q)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """
        Save the Q-network weights to the specified path
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def load(self, path):
        """
        Load the Q-network weights from the specified path
        """
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
