#!/usr/bin/env python3
"""
models.py

Three models for binary pedestrian intention prediction (enter vs pass):

    IntentMLP  –  single-frame baseline. Uses ONLY the last frame of the
                  sequence. If this matches or beats the GRU, it means
                  temporal history adds no value beyond the current state.

    IntentGRU  –  recurrent sequence model. Processes all T frames and builds
                  a compressed representation of the full trajectory history
                  before classifying.

    IntentCNN  –  temporal 1D-CNN. Uses convolutional filters over time as a
                  lightweight, non-recurrent alternative to the GRU.

All three models:
  - Accept the same input shape: (batch_size, seq_len, input_dim)
  - Are instantiated with the same signature:
        Model(input_dim, hidden_size=H, dropout=D)
  - Output a single raw logit per sample (no sigmoid inside forward)
  - Use BCEWithLogitsLoss during training
  - Support ablation study by changing only input_dim:

        Trajectory only           input_dim = 6
        Trajectory + Body Pose    input_dim = 8
        Trajectory + Pose + Head  input_dim = 9
"""

import torch
import torch.nn as nn


class IntentMLP(nn.Module):
    """
    Single-frame MLP baseline for pedestrian intention prediction.

    Extracts the LAST frame of the input sequence and classifies from that
    alone. This is the simplest possible model and sets a lower bound:
    if the GRU cannot beat this, the sequence history contains no useful
    information that the current state does not already provide.

    Architecture:
        Last frame  →  Linear + ReLU  →  Dropout  →  Linear(1 logit)

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (same as for IntentGRU).
    hidden_size : int
        Width of the single hidden layer. Default: 32.
    dropout : float
        Dropout probability after the hidden layer. Default: 0.3.
    """

    def __init__(self, input_dim: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Take the input_dim features of the last frame
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Classify
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, input_dim)
            Full input sequence – only the last frame is used.

        Returns
        -------
        torch.Tensor, shape (batch_size,)
            One raw logit per sample.
        """
        # Discard all but the last timestep: (batch, seq_len, dim) → (batch, dim)
        last_frame = x[:, -1, :]

        # (batch, dim) → (batch, 1) → (batch,)
        return self.net(last_frame).squeeze(1)


class IntentGRU(nn.Module):
    """
    Single-stream GRU classifier for pedestrian intention prediction.

    Processes the full input sequence so the model can learn from how
    features evolve over time, not just from their current values.

    Architecture:
        GRU  →  final hidden state  →  Dropout  →  Linear(1 logit)

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
        Set to 6 for the trajectory-only baseline.
    hidden_size : int
        Number of units in the GRU hidden state. Default: 32.
    dropout : float
        Dropout probability applied to the final hidden state before
        the linear layer. Default: 0.3.
        (Set higher than 0.1 because the dataset is small, ~500 samples.)
    """

    def __init__(self, input_dim: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size

        # Processes the full input sequence (batch, seq_len, input_dim).
        # batch_first=True matches the (batch, seq, feature) convention.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Applied to the final hidden state before the classifier head.
        # Regularises the bottleneck representation on small datasets.
        self.dropout = nn.Dropout(p=dropout)

        # Maps the GRU's hidden state to a single logit.
        # No sigmoid – BCEWithLogitsLoss handles it more stably.
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, input_dim)
            Input sequence of per-frame feature vectors.

        Returns
        -------
        torch.Tensor, shape (batch_size,)
            One raw logit per sample. Positive → "enter", negative → "pass".
        """
        # _all_hidden  : (batch, seq_len, hidden_size) – one per timestep
        # final_hidden : (num_layers=1, batch, hidden_size)
        _all_hidden, final_hidden = self.gru(x)

        # Squeeze the num_layers dim: (1, batch, hidden) → (batch, hidden)
        h = final_hidden.squeeze(0)

        h = self.dropout(h)

        # (batch, hidden) → (batch, 1) → (batch,)
        return self.classifier(h).squeeze(1)


class IntentCNN(nn.Module):
    """
    Temporal 1D-CNN for pedestrian intention prediction.

    A lightweight alternative to the GRU that detects local temporal patterns
    using convolutional filters instead of recurrence. Two stacked Conv1d
    layers are followed by global average pooling over the time axis, which
    summarises the sequence into a fixed-size vector regardless of T.

    Advantages over GRU on small datasets:
      - No hidden state to forget/saturate on short sequences
      - More parallelisable (no sequential dependency)
      - Fewer parameters for the same hidden_size

    Architecture:
        (B, T, D)  →  transpose  →  (B, D, T)
        Conv1d(D, hidden_size, k=3) + ReLU + Dropout
        Conv1d(hidden_size, hidden_size, k=3) + ReLU
        Global average pool over T  →  (B, hidden_size)
        Linear(hidden_size, 1)  →  one logit per sample

    Parameters
    ----------
    input_dim   : number of features per timestep (same as other models)
    hidden_size : number of convolutional channels (analogous to GRU hidden_size)
    dropout     : dropout probability after the first conv block
    """

    def __init__(self, input_dim: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()

        # kernel_size=3 with padding=1 preserves the time dimension length
        self.conv1   = nn.Conv1d(input_dim,   hidden_size, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.relu    = nn.ReLU()

        # Maps the pooled representation to a single logit
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)
            Input sequence of per-frame feature vectors.

        Returns
        -------
        torch.Tensor, shape (batch,)
            One raw logit per sample. Positive → "enter", negative → "pass".
        """
        # Conv1d expects (batch, channels, length) – so swap T and D
        x = x.permute(0, 2, 1)          # (B, T, D) -> (B, D, T)

        x = self.relu(self.conv1(x))    # (B, hidden, T)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))    # (B, hidden, T)

        # Global average pooling: summarise the whole sequence into one vector
        x = x.mean(dim=2)               # (B, hidden)

        return self.classifier(x).squeeze(1)  # (B,)
