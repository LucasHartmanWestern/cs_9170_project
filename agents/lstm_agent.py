import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int, device: torch.device = None):
    torch.manual_seed(seed)
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, bidirectional: bool = False):
        """
        A simple LSTM regression model.

        Args:
            input_dim: Number of input features per time step.
            hidden_dim: Hidden state dimension.
            output_dim: Number of outputs.
            num_layers: Number of LSTM layers.
            bidirectional: Use bidirectional LSTM if True.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        # take last time step
        last = out[:, -1, :]
        return self.fc(last)


class LSTMAgent:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        seed: int = 42,
        device: str = None
    ):
        # Device setup
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Seeding
        set_seed(seed, self.device)
        self.seed = seed

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs

        # Model, loss, optimizer
        self.model = LSTMModel(input_dim, hidden_dim, output_dim,
                               num_layers, bidirectional).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def reset(self) -> None:
        """Reinitialize model weights and optimizer state."""
        set_seed(self.seed, self.device)
        self.model = LSTMModel(
            self.model.lstm.input_size,
            self.model.hidden_dim,
            self.model.fc.out_features,
            self.model.num_layers,
            self.model.bidirectional
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.optimizer.param_groups[0]['lr'])

    def train(self, loader: DataLoader) -> list[float]:
        set_seed(self.seed, self.device)
        self.model.train()
        losses = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                # Awkward find better fix
                if xb.dim() == 2:
                    xb = xb.unsqueeze(1)
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                preds = self.model(xb)
                loss = self.criterion(preds, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # Ensure input is 3D
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = x.to(self.device)
            preds = self.model(x)
        return preds.cpu()

    def save(self, path: str) -> None:
        """Save model state and hyperparameters."""
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'input_dim': self.model.lstm.input_size,
            'hidden_dim': self.model.hidden_dim,
            'output_dim': self.model.fc.out_features,
            'num_layers': self.model.num_layers,
            'bidirectional': self.model.bidirectional,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'seed': self.seed
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model state and hyperparameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.epochs = checkpoint['epochs']
        self.batch_size = checkpoint['batch_size']
        self.seed = checkpoint['seed']
        self.model = LSTMModel(
            checkpoint['input_dim'],
            checkpoint['hidden_dim'],
            checkpoint['output_dim'],
            checkpoint['num_layers'],
            checkpoint['bidirectional']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer'])
