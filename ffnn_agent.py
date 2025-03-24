import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNNModel, self).__init__()
        
        # Create layers dynamically based on the provided sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FFNNAgent:
    def __init__(self, input_size, hidden_sizes=[64, 64], output_size=1, 
                 learning_rate=0.001, batch_size=32, epochs=100, type="regression", device=None):
        """
        Initialize the Feed-Forward Neural Network agent.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output values
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to run the model on (cpu or cuda)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
            
        # Initialize model
        self.model = FFNNModel(input_size, hidden_sizes, output_size).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function - MSE for regression, CrossEntropy for classification
        if type == "regression":
            self.criterion = nn.MSELoss()
        elif type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid type. Must be 'regression' or 'classification'.")
    
    def predict(self, features):
        """
        Make predictions using the trained model.
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            Predictions as numpy array
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(features)
        
        # Return as numpy array
        return predictions.cpu().numpy()
    
    def train(self, features, targets):
        """
        Train the model on the provided dataset.
        
        Args:
            features: Input features (numpy array or torch tensor)
            targets: Target values (numpy array or torch tensor)
            
        Returns:
            List of training losses
        """
        # Convert to tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_features, batch_targets in dataloader:
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Average loss for the epoch
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
        
        return losses

    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate the model with the same architecture
        self.input_size = checkpoint['input_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.output_size = checkpoint['output_size']
        self.learning_rate = checkpoint['learning_rate']
        
        # Initialize model with loaded architecture
        self.model = FFNNModel(self.input_size, self.hidden_sizes, self.output_size).to(self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer and load its state
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {path}")