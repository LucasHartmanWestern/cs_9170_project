import torch
import torch.nn as nn
import torch.optim as optim

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True) 

class FFNNModel(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: list[int], 
        output_size: int
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            prev_size = hidden
        
        # final output layer (no activation here)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.network:
            if isinstance(m, nn.Linear):
                #This is new, more ideal for relu activation
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FFNNAgent:
    def __init__(self, input_size, hidden_sizes=[64, 64], output_size=1, 
                 learning_rate=0.001, batch_size=32, epochs=100, type="regression", 
                 classes=None, device='cpu', seed=42):
        super(FFNNAgent, self).__init__()
        self.device = device 
        self.seed   = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(self.seed)
        print(f"FFNN Using device: {self.device}")
        self.dl_generator = torch.Generator().manual_seed(self.seed)

        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.learning_rate = learning_rate
        self.type = type


        if self.type == "classification":
            if classes is not None:
                self.classes = classes 
            else:
                self.classes = list(range(output_size))

            self.output_size = len(self.classes)

        elif self.type == "regression":
            self.classes = None
            self.output_size = output_size
        else:
            raise ValueError("Type must be 'regression' or 'classification'")
        

        # Initialize model
        self.model = FFNNModel(input_size, hidden_sizes, self.output_size).to(self.device) 
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = (
            nn.MSELoss()
            if self.type == "regression"
            else nn.CrossEntropyLoss()
        )
        with torch.no_grad():
            # Keep snapshot on the same device;
            self._init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        # Keep optimizer settings to build fresh optimizer each reset
        self._optim_cfg = dict(
            lr=self.learning_rate,
            betas=self.optimizer.defaults.get('betas', (0.9, 0.999)),
            eps=self.optimizer.defaults.get('eps', 1e-8),
            weight_decay=self.optimizer.defaults.get('weight_decay', 0.0),
            amsgrad=self.optimizer.defaults.get('amsgrad', False),
        )
    
    def reset(self):
        """
        Restore model weights to the exact initial snapshot and
        re-create a fresh optimizer (no momentum carryover).
        """
        with torch.no_grad():
            self.model.load_state_dict(self._init_state, strict=True)

        # Fresh optimizer so Adam/SGD momentum/EMA buffers are reset too
        self.optimizer = optim.Adam(self.model.parameters(), **self._optim_cfg)

        # Criterion doesn’t need re-instantiation unless you actually change task type,
        # but keep it if you prefer explicitness:
        if self.type == "regression":
            self.criterion = nn.MSELoss()
        elif self.type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        

    def predict(self, features):
        """
        Make predictions using the trained model.

        Args:
            features: Input features (numpy array or torch tensor)

        Returns:
            Tensor of predictions (floats for regression, class indices for classification)
        """

        if not torch.is_tensor(features):
            features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        else:
            features = features.to(self.device, dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features) # shape (batch_size, output_size)

            if self.type == "classification":
                preds = torch.argmax(outputs, dim=1)
                return preds
            else:
                return outputs
    def train(self, loader: torch.utils.data.DataLoader) -> list[float]:
        """
        Train the model on the provided DataLoader.

        Args:
            loader: DataLoader yielding (features, targets) batches

        Returns:
            List of average training loss per epoch
        """
        # switch to train mode
        self.model.train()
        losses: list[float] = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_features, batch_targets in loader:
                # move to device
                batch_features = batch_features.to(self.device)
                batch_targets  = batch_targets .to(self.device)

                # forward + loss
                outputs = self.model(batch_features).squeeze(-1)
                loss = self.criterion(outputs, batch_targets)

                # backward + step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            # record epoch’s mean loss
            losses.append(epoch_loss / batch_count)

        return losses

    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size':           self.input_size,
            'hidden_sizes':         self.hidden_sizes,
            'output_size':          self.output_size,
            'learning_rate':        self.learning_rate,
            'batch_size':           self.batch_size,
            'epochs':               self.epochs,
            'type':                 self.type,
            'classes':              self.classes,
            'seed':                 self.seed,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")


    def load(self, path):
        """
        Load the model and optimizer state from the specified path.

        Args:
            path: Path to load the checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore hyper-parameters
        self.seed         = checkpoint.get('seed', 42)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.input_size   = checkpoint['input_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.output_size  = checkpoint['output_size']
        self.learning_rate= checkpoint['learning_rate']
        self.batch_size   = checkpoint['batch_size']
        self.epochs       = checkpoint['epochs']
        self.type         = checkpoint.get('type', 'regression')
        self.classes      = checkpoint.get('classes', None)
        self.dl_generator = torch.Generator().manual_seed(self.seed)


        # Rebuild the model and optimizer
        self.model = FFNNModel(
            self.input_size,
            self.hidden_sizes,
            self.output_size
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Make sure optimizer tensors are on the right device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print(f"Model loaded from {path}")