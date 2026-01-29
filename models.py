import torch
import torch.nn as nn

class LinearModel(nn.Module):
    """Simple linear regression model."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SimpleMLP(nn.Module):
    """
    Minimal nonlinear model for CTL experiments.
    Same model class across tasks.
    """

    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def num_parameters(self):
        """
        Useful for FLOP or cost normalization.
        """
        return sum(p.numel() for p in self.parameters())
    
class ResidualMLP(nn.Module):
    def __init__(self, d, hidden=32, eps=0.1):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        self.residual = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.eps = eps

    def forward(self, x):
        return self.linear(x).squeeze(-1) + self.eps * self.residual(x).squeeze(-1)
        