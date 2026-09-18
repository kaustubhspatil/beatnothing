"""
Model architectures of the reference contestants, so their frozen weights can be
loaded and scored on any universe without the capstone repository. Torch is imported
lazily: the rest of the package does not need it.
"""
from __future__ import annotations


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def stock_predictor(input_dim: int, hidden_dims=(128, 64, 32), dropout: float = 0.3):
    """Feedforward network: Linear, BatchNorm, ReLU, Dropout blocks and a linear head."""
    torch, nn = _torch()
    layers, prev = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class _StockPredictor:
    """State dict compatible wrapper (the capstone saved the network under the attribute `net`)."""

    def __new__(cls, input_dim, hidden_dims=(128, 64, 32), dropout=0.3):
        torch, nn = _torch()

        class StockPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = stock_predictor(input_dim, hidden_dims, dropout)

            def forward(self, x):
                return self.net(x)

        return StockPredictor()


def StockPredictor(input_dim, hidden_dims=(128, 64, 32), dropout=0.3):  # noqa: N802
    return _StockPredictor(input_dim, hidden_dims, dropout)


def FinancialLSTM(input_dim, hidden_dim=64, num_layers=2, dropout=0.2):  # noqa: N802
    torch, nn = _torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return Net()


def FinancialCNN1D(input_channels, dropout=0.3):  # noqa: N802
    torch, nn = _torch()

    def block(c_in, c_out, k):
        return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(block(input_channels, 32, 7), block(32, 64, 5), block(64, 128, 3))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, x):
            x = x.permute(0, 2, 1)
            return self.head(self.pool(self.features(x)))

    return Net()


def CostAwareNet(n_features, hidden=(128, 64, 32), dropout=0.3):  # noqa: N802
    """Outputs weights directly: sigmoid(f(x)) / N per name, so a book of N names holds at most 1."""
    torch, nn = _torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, p = [], n_features
            for h in hidden:
                layers += [nn.Linear(p, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
                p = h
            layers.append(nn.Linear(p, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x, mask):
            s = torch.sigmoid(self.net(x).squeeze(-1)) * mask
            return s / mask.sum(-1, keepdim=True).clamp(min=1)

    return Net()
