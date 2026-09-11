#!/usr/bin/env python3
"""
Step 2 — Retrain RQ2 finalist models on full data.

Cross-validation (RQ1) was used for model *selection* and performance
*estimation*.  Here we retrain the chosen GRU configurations on all
available data so the deployed model has seen every sample.

An 85/15 stratified split is used solely for early stopping.  No evaluation
metrics are reported from this split; the CV numbers from RQ1 remain the
official performance estimate.

Each .pt checkpoint stores:
    model_state_dict  — trained weights
    mean              — per-feature normalisation mean  (numpy)
    std               — per-feature normalisation std   (numpy)
    metadata          — dict with all hyperparameters, feature names, etc.

Run:
    python step2_train_finalists.py
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT, CHECKPOINT_DIR, RESULTS_DIR,
    TTE_VALUES, WINDOW_S,
    HIDDEN_SIZE, DROPOUT, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, VAL_FRACTION, RANDOM_STATE,
)

from utils.dataset_utils import load_all_samples
from utils.models import IntentGRU
from utils.sequence_dataset import SequenceDataset, normalise

# Import builders from train_ablation (the structured ablation script)
from train_ablation import build_experiment_data, EXPERIMENTS, WINDOW_SECONDS

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def train_final_model(model, X_all, y_all, seed):
    """Train on X_all with an internal val split used only for early stopping."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VAL_FRACTION,
        stratify=y_all,
        random_state=seed,
    )

    X_train_n, X_val_n, mean, std = normalise(X_train, X_val)

    shuffle_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SequenceDataset(X_train_n, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_gen,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val_n, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32
    ).to(DEVICE)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(
                    model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                ).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_weights)
    return model, mean, std, best_epoch


def main():
    print("=" * 72)
    print("  STEP 2: TRAIN FINALISTS on full data")
    print(f"  Device       : {DEVICE}")
    print(f"  Window       : {WINDOW_S}s")
    print(f"  TTE values   : {TTE_VALUES}")
    print(f"  Val fraction : {VAL_FRACTION} (early stopping only)")
    print(f"  Epochs       : {EPOCHS}  |  hidden={HIDDEN_SIZE}  dropout={DROPOUT}")
    print(f"  Patience     : {EARLY_STOPPING_PATIENCE}")
    print("=" * 72)

    # Load finalist keys
    finalists_csv = RESULTS_DIR / "finalists.csv"
    if not finalists_csv.exists():
        print(f"\n  ERROR: {finalists_csv} not found.")
        print(f"  Run step1_select_finalists.py first.")
        return

    fin_df = pd.read_csv(finalists_csv)
    finalist_keys = fin_df["experiment_key"].tolist()

    # Validate
    for key in finalist_keys:
        if key not in EXPERIMENTS:
            print(f"  ERROR: experiment key '{key}' not in EXPERIMENTS dict.")
            print(f"  Available: {sorted(EXPERIMENTS.keys())}")
            return

    print(f"\n  {len(finalist_keys)} finalists to train:")
    for key in finalist_keys:
        exp = EXPERIMENTS[key]
        print(f"    {key:30s}  {exp['label']}")

    # Load data
    samples = load_all_samples(DATA_ROOT)
    n_enter = sum(1 for s in samples if s.get("label") == "enter")
    n_pass = sum(1 for s in samples if s.get("label") == "pass")
    print(f"\n  Data: {len(samples)} samples (enter={n_enter}, pass={n_pass})")

    if not samples:
        print("  No samples found. Check DATA_ROOT.")
        return

    saved = 0

    for exp_key in finalist_keys:
        exp = EXPERIMENTS[exp_key]
        label = exp["label"]

        print(f"\n{'=' * 72}")
        print(f"  {label}  [{exp_key}]")
        print(f"{'=' * 72}")

        for tte in TTE_VALUES:
            print(f"\n  TTE = {tte:.1f}s ...")

            X, y, input_dim, feat_names = build_experiment_data(
                exp, samples, tte
            )

            if len(y) == 0 or len(np.unique(y)) < 2:
                print(f"    Skipped: insufficient data (n={len(y)})")
                continue

            n_e = int((y == 1).sum())
            n_p = int((y == 0).sum())
            seq_len = X.shape[1]
            print(f"    Samples: {len(y)} (enter={n_e}, pass={n_p})  "
                  f"shape=({len(y)}, {seq_len}, {input_dim})")

            seed = RANDOM_STATE + int(tte * 10)
            torch.manual_seed(seed)
            model = IntentGRU(input_dim, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            trained, mean, std, best_epoch = train_final_model(model, X, y, seed)
            print(f"    Best epoch: {best_epoch}  |  params: {n_params}")

            filename = f"gru_{exp_key}_tte{tte:.1f}.pt"
            save_path = CHECKPOINT_DIR / filename

            torch.save({
                "model_state_dict": trained.cpu().state_dict(),
                "mean": mean,
                "std": std,
                "metadata": {
                    "model_class": "IntentGRU",
                    "experiment_key": exp_key,
                    "label": label,
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                    "tte_s": tte,
                    "window_s": WINDOW_S,
                    "epochs_trained": best_epoch,
                    "max_epochs": EPOCHS,
                    "patience": EARLY_STOPPING_PATIENCE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "val_fraction": VAL_FRACTION,
                    "n_samples": len(y),
                    "n_enter": n_e,
                    "n_pass": n_p,
                    "n_params": n_params,
                    "feature_names": list(feat_names) if not isinstance(feat_names, list) else feat_names,
                },
            }, save_path)
            print(f"    Saved: {filename}")
            saved += 1

    print(f"\n{'=' * 72}")
    print(f"  Done. {saved} checkpoints saved to {CHECKPOINT_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
