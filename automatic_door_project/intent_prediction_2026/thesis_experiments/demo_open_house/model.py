"""
GRU intent-prediction model — matches the architecture in
utils/models.py (IntentGRU).

Architecture:
    GRU(input_dim, hidden=16, layers=1, batch_first=True)
    → take final hidden state
    → Dropout(0.1)
    → Linear(16, 1)   (raw logit, apply sigmoid for probability)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class IntentGRU(nn.Module):
    """Lightweight GRU for binary intent prediction (Enter vs Pass)."""

    def __init__(self, input_dim: int = 7, hidden_size: int = 16,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        logits : (batch,)
        """
        _, h_n = self.gru(x)              # h_n: (num_layers, batch, hidden)
        h = h_n[-1]                        # (batch, hidden)
        h = self.dropout(h)
        return self.classifier(h).squeeze(1)  # (batch,)

    def predict_proba(self, x: torch.Tensor) -> float:
        """Return P(enter) as a Python float."""
        with torch.no_grad():
            logit = self.forward(x)
            return torch.sigmoid(logit).item()


def load_model(
    path: Optional[str] = None,
    input_dim: int = 7,
    device: str = "cpu",
) -> IntentGRU:
    """
    Load a trained model from a state_dict file.
    If path is None, returns an untrained model (for pipeline testing).
    """
    model = IntentGRU(input_dim=input_dim)
    if path is not None:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[MODEL] Loaded weights from {path}")
    else:
        print("[MODEL] No weights provided — using random initialisation (demo mode)")
    model.eval()
    model.to(device)
    return model


def load_normalisation_stats(path: str):
    """
    Load z-score mean/std from a .npz file.
    Expected keys: 'mean' and 'std', each shape (7,).
    """
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)
