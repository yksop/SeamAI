#!/usr/bin/env python3
"""
Step 8 — Benchmark GRU hidden size to show diminishing returns.

Trains GRU models with hidden_size = {4, 8, 16, 32, 64, 128} for a
representative feature configuration (A2_traj6, dim=6) and measures
inference latency.

Purpose: Demonstrate that (a) the GRU is already so small (~1 ms) that
varying hidden size has no measurable effect on latency, and (b) the
current hidden_size=16 strikes the right balance between capacity and
efficiency (larger sizes don't improve accuracy on this dataset).

Output:
    outputs/results/gru_size_benchmark.csv — latency per hidden size
    outputs/figures/rq2_gru_size_comparison.pdf — latency + accuracy vs hidden size

Run:
    python step8_benchmark_gru_sizes.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    RESULTS_DIR, FIG_DIR, TARGET_FPS, DATA_ROOT,
    TTE_VALUES, WINDOW_S,
    DROPOUT, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, VAL_FRACTION, RANDOM_STATE,
    CHECKPOINT_DIR, style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.models import IntentGRU

# ---------------------------------------------------------------------------
# SETTINGS — edit these to change the run
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [4, 8, 16, 32, 64, 128]

# Use Traj-6 (A2_traj6) as representative config — same for all hidden sizes
EXPERIMENT_KEY = "A2_traj6"
INPUT_DIM = 6
REFERENCE_TTE = 1.0  # single TTE for training (latency is TTE-independent)

# Benchmark settings
WARMUP_RUNS = 200
TIMED_RUNS = 2000   # more runs because GRU is so fast, need stable measurements
SEQ_LEN = 5         # 0.5s window at 10 fps

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_quick(hidden_size, X_train, y_train, X_val, y_val):
    """Train a GRU with given hidden size, return (model, val_accuracy)."""
    model = IntentGRU(
        input_dim=INPUT_DIM,
        hidden_size=hidden_size,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        # Simple full-batch training (small dataset)
        idx = torch.randperm(len(X_train))
        for start in range(0, len(X_train), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            xb = X_train[batch_idx].to(DEVICE)
            yb = y_train[batch_idx].to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE)).squeeze(-1)
            val_preds = (val_logits > 0).float()
            val_acc = (val_preds == y_val.to(DEVICE)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_acc


def benchmark_inference(model, input_dim, seq_len):
    """Measure inference latency for a single model."""
    model.eval()
    model.to(DEVICE)

    dummy = torch.randn(1, seq_len, input_dim).to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

    timings = np.array(timings)
    return {
        "mean_ms": timings.mean(),
        "std_ms": timings.std(),
        "p50_ms": np.percentile(timings, 50),
        "p95_ms": np.percentile(timings, 95),
        "p99_ms": np.percentile(timings, 99),
        "min_ms": timings.min(),
        "max_ms": timings.max(),
    }


def main():
    print(f"\n{'='*60}")
    print(f"  STEP 8: GRU Hidden Size Benchmark")
    print(f"  Device: {DEVICE}")
    print(f"  Config: {EXPERIMENT_KEY} (dim={INPUT_DIM})")
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"  Timed runs: {TIMED_RUNS}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # Load data for training (use the RQ1 dataset)
    # ---------------------------------------------------------------
    print("\n  Loading training data ...")
    try:
        from utils.dataset_utils import load_all_samples
        from train_ablation import EXPERIMENTS, build_experiment_data

        samples = load_all_samples(DATA_ROOT)
        experiment = EXPERIMENTS[EXPERIMENT_KEY]
        X, y, input_dim, feature_names = build_experiment_data(
            experiment, samples, REFERENCE_TTE,
        )
        print(f"  Loaded {len(X)} samples, dim={input_dim}, "
              f"features={feature_names}")
        # Train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_FRACTION, stratify=y,
            random_state=RANDOM_STATE,
        )
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        has_data = True
    except Exception as e:
        print(f"  Could not load training data: {e}")
        print(f"  Will benchmark latency only (no accuracy)")
        has_data = False

    # ---------------------------------------------------------------
    # Train and benchmark each hidden size
    # ---------------------------------------------------------------
    results = []

    for hs in HIDDEN_SIZES:
        print(f"\n  --- hidden_size = {hs} ---")

        if has_data:
            model, val_acc = train_quick(hs, X_train, y_train, X_val, y_val)
            print(f"    Val accuracy: {val_acc:.4f}")
        else:
            model = IntentGRU(
                input_dim=INPUT_DIM, hidden_size=hs, dropout=DROPOUT,
            ).to(DEVICE)
            model.eval()
            val_acc = None

        n_params = count_parameters(model)
        print(f"    Parameters: {n_params}")

        timing = benchmark_inference(model, INPUT_DIM, SEQ_LEN)
        print(f"    Latency: {timing['mean_ms']:.3f} ms (p95: {timing['p95_ms']:.3f})")

        result = {
            "hidden_size": hs,
            "n_params": n_params,
            "val_acc": val_acc,
            **timing,
        }
        results.append(result)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "gru_size_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: GRU Hidden Size vs Latency")
    print(f"{'='*70}")
    print(f"  {'H':>5s}  {'Params':>8s}  {'Mean ms':>8s}  {'p95 ms':>8s}  "
          f"{'p99 ms':>8s}  {'Val acc':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, r in df.iterrows():
        acc_str = f"{r['val_acc']:.4f}" if r['val_acc'] is not None else "N/A"
        print(f"  {r['hidden_size']:>5.0f}  {r['n_params']:>8.0f}  "
              f"{r['mean_ms']:>8.3f}  {r['p95_ms']:>8.3f}  "
              f"{r['p99_ms']:>8.3f}  {acc_str:>8s}")


if __name__ == "__main__":
    main()
