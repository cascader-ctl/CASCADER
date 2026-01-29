from contextlib import contextmanager, nullcontext
from models import CascadeFM, SimpleMLP, LinearModel, ResidualMLP
import numpy as np 
import time
import torch
import torch.nn as nn
from typing import Optional, Tuple

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

class CostTracker:
    """
    Tracks optimization cost in a model-agnostic way.
    Primary metric: number of update steps.
    Secondary metric: wall-clock time.
    """

    def __init__(self):
        self.steps = 0
        self.time = 0.0

    def reset(self):
        self.steps = 0
        self.time = 0.0

    def add_step(self, duration):
        self.steps += 1
        self.time += duration

    @contextmanager
    def track_step(self):
        """
        Context manager to wrap a single optimization step.
        """
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.add_step(end - start)

    def summary(self):
        return {
            "total_steps": self.steps,
            "total_time_sec": self.time,
            "time_per_step_sec": self.time / max(self.steps, 1),
        }

    def __repr__(self):
        return (
            f"CostTracker(steps={self.steps}, "
            f"time={self.time:.4f}s, "
            f"time/step={self.time / max(self.steps, 1):.6f}s)"
        )
    
def train_linear_head_sgd(
    head: nn.Module,
    Z: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    eta: float,
):
    """
    Train only a linear head on cached features.
    Z: (n, d)
    y: (n,) or (n,1)
    """
    head.train()

    if y.ndim == 1:
        y = y.unsqueeze(1)

    criterion = nn.MSELoss()

    for _ in range(steps):
        pred = head(Z)
        loss = criterion(pred, y)
        head.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in head.parameters():
                p -= eta * p.grad

    return head
    
def train_model_sgd(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    steps: int,
    eta: float,
    device: torch.device,
    cost_tracker: Optional["CostTracker"] = None,
) -> nn.Module:
    """
    Train a PyTorch model using full-batch gradient descent
    with numerical safeguards and CostTracker support.
    """

    model = model.to(device)
    model.train()

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(1)

    criterion = nn.MSELoss()

    for _ in range(steps):

        context = (
            cost_tracker.track_step()
            if cost_tracker is not None
            else nullcontext()
        )

        with context:
            pred = model(X_t)
            loss = criterion(pred, y_t)

            if not torch.isfinite(loss):
                raise RuntimeError("NaN or Inf detected in loss")

            model.zero_grad(set_to_none=True)
            loss.backward()

            # 🔒 stability: avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-eta)

    return model


def clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of a model."""
    # reads the class of the model (either LinearModel, SimpleMLP or ResidualMLP) and creates a new instance
    if isinstance(model, SimpleMLP):
        new_model = SimpleMLP(model.net[0].in_features, model.net[0].out_features)
    elif isinstance(model, LinearModel):
        new_model = LinearModel(model.linear.in_features)
    elif isinstance(model, ResidualMLP):
        new_model = ResidualMLP(model.linear.in_features, model.residual[0].out_features, model.eps)
    elif isinstance(model, CascadeFM):
        new_model = CascadeFM(
            backbone_name=model.backbone_name,
            output_dim=model.head.out_features,
            freeze_backbone=True,
            device=model.device,
        )
        new_model.head.load_state_dict(model.head.state_dict())
    else:
        raise ValueError("Unsupported model type for cloning.")

    new_model.load_state_dict(model.state_dict())
    return new_model

def predict_node(model: nn.Module, X: np.ndarray, device: torch.device = DEVICE) -> np.ndarray:
    """Generate predictions from model."""
    model = model.to(device)
    model.eval()
    
    X_t = torch.from_numpy(X).float().to(device)
    
    with torch.no_grad():
        pred = model(X_t)
    
    return pred.cpu().numpy()

def compute_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and MAPE."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return float(mse), float(mape)

def gd_train_logistic(theta_init, X, y, steps=100, lr=0.1, reg=1e-4, device=None):
    """Train logistic regression with gradient descent on specified device."""
    if device is None:
        device = torch.device('cpu')
    
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    theta_t = torch.from_numpy(theta_init.copy()).float().to(device)
    
    for _ in range(steps):
        z = X_t @ theta_t
        z = torch.clamp(z, -500, 500)
        prob = 1 / (1 + torch.exp(-z))
        grad = X_t.T @ (prob - y_t) / len(y_t)
        grad += reg * theta_t
        theta_t -= lr * grad
    
    return theta_t.cpu().numpy()

def compute_accuracy(theta, X, y, device=None):
    """Compute classification accuracy on specified device."""
    if device is None:
        device = torch.device('cpu')
    
    X_t = torch.from_numpy(X).float().to(device)
    theta_t = torch.from_numpy(theta).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    
    z = X_t @ theta_t
    z = torch.clamp(z, -500, 500)
    prob = 1 / (1 + torch.exp(-z))
    preds = (prob >= 0.5).float()
    
    return float((preds == y_t).float().mean().cpu().item())
