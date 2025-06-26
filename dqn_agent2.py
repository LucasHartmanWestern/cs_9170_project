import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device

        # Pre-allocate buffers
        self.states      = torch.zeros((capacity, state_dim),   device=device)
        self.actions     = torch.zeros((capacity, action_dim),  device=device)
        self.rewards     = torch.zeros((capacity, 1),           device=device)
        self.next_states = torch.zeros((capacity, state_dim),   device=device)
        self.dones       = torch.zeros((capacity, 1),           device=device)

        self.ptr   = 0
        self.size  = 0

    def push(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


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
        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.net(x)
    
    def get_binary_output(self, x):
        # Get continuous values from the network
        logits = self.net(x)
        # Apply sigmoid to get values between 0 and 1
        probs = self.sigmoid(logits)
        # Convert to binary (0 or 1) based on threshold of 0.5
        return probs

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99, batch_size=32, memory_size=10000, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, device='cpu', seed=42):

        self.seed = seed
        torch.manual_seed(seed)
        self.device = device
        self.rng = torch.Generator(self.device).manual_seed(self.seed)

        
        self.gamma = gamma
        self.state_size  = state_size
        self.action_size = action_size
        self.batch_size = batch_size


        # ε-greedy params
        self.epsilon     = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # model, opt, loss
        self.model     = QNetwork(state_size, action_size, hidden_size).to(self.device)
        #replay buffer
        self.memory = ReplayBuffer(state_size, action_size, memory_size, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def predict(self, state):
        """
        Generate a single synthetic data sample
        
        Args:
            state: The current state
            random_state: Random seed for reproducibility
        Returns:
            Synthetic data sample (a list of integers)
        """
        # Convert state to tensor if it's not already
        if torch.is_tensor(state):
            # already a Tensor—clone+detach to avoid in-place/grad issues
            state = state.clone().detach().float().to(self.device)
        else:
            # numpy array or list
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure state is properly shaped for the network
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if needed
        
        self.model.eval()

        # Epsilon-greedy action selection
        if torch.rand(1, generator=self.rng, device=self.device).item() <= self.epsilon:
            # Single call returns a tensor of shape (action_size,)
            synthetic_data = torch.randint(0, 2, (self.action_size,),generator=self.rng,device=self.device).tolist()
        else:
            with torch.no_grad():
                probs = self.model.get_binary_output(state)
            
            # Convert probabilities to binary (0 or 1) with threshold 0.5
            synthetic_data = (probs >= 0.5).int().squeeze(0).tolist()
        
        return synthetic_data

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition (experience) in the replay memory.
        """
        s  = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        r  = torch.as_tensor(reward,   dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(done,     dtype=torch.float32, device=self.device)

        self.memory.push(s, a, r, ns, d)

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
        if self.memory.size < self.batch_size:
            return
        
        # Sample batch
        idx = torch.randperm(self.memory.size, generator=self.rng, device=self.device)
        idx = idx[: self.batch_size]  

        states      = self.memory.states[idx]
        actions     = self.memory.actions[idx]
        rewards     = self.memory.rewards[idx]
        next_states = self.memory.next_states[idx]
        dones       = self.memory.dones[idx]
        
        self.model.train()
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            # ensure batch dim of 1
            if state.ndim == 1:
                state      = state.unsqueeze(0)
                next_state = next_state.unsqueeze(0)

            # current Q(s,·)
            current_q = self.model(state)

            #next-Q vector and its max
            with torch.no_grad():
                next_q     = self.model(next_state)   # shape (1, action_size)
                max_next_q = next_q.max()             

            # Bellman target
            target_q = reward + (1 - done) * self.gamma * max_next_q
            q_values = (current_q * action).sum().unsqueeze(0)
        
        
            loss = self.criterion(q_values, target_q)
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
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
