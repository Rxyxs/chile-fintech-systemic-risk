"""Tests for ml_predictions/train_lstm_equity.py — sequence building and the LSTM head."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_predictions"))
import train_lstm_equity as lstm  # noqa: E402


def test_build_sequences_shapes():
    arr = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    targets = np.arange(30, dtype=np.float32)

    X, y = lstm.build_sequences(arr, targets, seq_len=20)

    assert X.shape == (10, 20, 3)
    assert y.shape == (10,)


def test_build_sequences_aligns_target_to_the_step_after_the_window():
    arr = np.arange(25).reshape(25, 1).astype(np.float32)
    targets = np.arange(25).astype(np.float32)

    X, y = lstm.build_sequences(arr, targets, seq_len=5)

    # window i covers arr[i:i+5]; its target should be targets[i+5], one step ahead
    assert y[0] == 5
    assert X[0][-1][0] == 4  # last element of the first window is arr[4]


def test_directional_lstm_output_shape():
    model = lstm.DirectionalLSTM(n_features=3, hidden=8)
    batch = torch.randn(4, 20, 3)

    out = model(batch)

    assert out.shape == (4,)  # one logit per sequence, squeezed


def test_directional_lstm_is_trainable_single_step():
    """A single gradient step should reduce loss on a batch — catches a broken forward/backward wiring."""
    model = lstm.DirectionalLSTM(n_features=3, hidden=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X = torch.randn(16, 20, 3)
    y = torch.randint(0, 2, (16,)).float()

    logits_before = model(X)
    loss_before = loss_fn(logits_before, y).item()

    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    loss_after = loss_fn(model(X), y).item()
    assert loss_after < loss_before
