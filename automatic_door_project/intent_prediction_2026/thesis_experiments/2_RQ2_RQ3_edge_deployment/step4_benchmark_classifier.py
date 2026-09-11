#!/usr/bin/env python3
"""
Step 4 — RQ2 classifier inference latency benchmark

Loads the five GRU finalist checkpoints (produced by train_finalists.py),
exports them to ONNX, and measures inference latency in three modes:

  1. PyTorch (FP32)          — baseline
  2. ONNX Runtime (FP32)     — optimised graph execution
  3. ONNX Runtime (INT8)     — dynamic quantisation

For each mode the script runs N warm-up passes followed by N timed passes
and records:
  - mean latency (ms)
  - std latency (ms)
  - p95 latency (ms)
  - throughput (inferences per second)
  - model size on disk (bytes)

The benchmark uses batch_size=1 to simulate real-time single-sample
inference as it would run on edge hardware.

Output
------
  outputs/results/benchmark_latency.csv    — one row per (model × mode)
  outputs/results/benchmark_summary.png    — grouped bar chart: latency vs model
  outputs/results/benchmark_fps.png        — FPS vs input dimension

Run
---
    cd 2_RQ2_RQ3_edge_deployment
    python step4_benchmark_classifier.py

Notes
-----
  - Run this on the Jetson. Step 2 already wrote the checkpoints to
    checkpoints/, so no files are copied between machines.
  - For GPU benchmarking, CUDA synchronisation is inserted before timing.
  - Install onnxruntime:  pip install onnxruntime
  - Install onnx:         pip install onnx
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths via config.py
# ---------------------------------------------------------------------------
from config import (
    CHECKPOINT_DIR, ONNX_DIR, RESULTS_DIR, FIG_DIR,
    WARMUP_RUNS, TIMED_RUNS, TARGET_FPS, TARGET_LATENCY_MS,
)

from utils.models import IntentGRU

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BATCH_SIZE = 1  # real-time single-sample inference

# Default TTE to benchmark (one is enough since model weights differ but
# architecture/input_dim are the same across TTE)
DEFAULT_TTE = 1.0

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helper: load checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path):
    """Load checkpoint and return (model, metadata, mean, std)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]

    model = IntentGRU(
        input_dim=meta["input_dim"],
        hidden_size=meta["hidden_size"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, meta, ckpt["mean"], ckpt["std"]


# ---------------------------------------------------------------------------
# Helper: export to ONNX
# ---------------------------------------------------------------------------
def export_to_onnx(model, meta, onnx_path):
    """Export PyTorch model to ONNX format."""
    seq_len = meta["seq_len"]
    input_dim = meta["input_dim"]

    dummy_input = torch.randn(BATCH_SIZE, seq_len, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=17,
    )
    return onnx_path


# ---------------------------------------------------------------------------
# Helper: quantise ONNX model (dynamic INT8)
# ---------------------------------------------------------------------------
def quantise_onnx(onnx_path, quant_path):
    """Apply dynamic quantisation to an ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("    WARNING: onnxruntime.quantization not available, skipping INT8")
        return None

    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
    )
    return quant_path


# ---------------------------------------------------------------------------
# Benchmark: PyTorch
# ---------------------------------------------------------------------------
def benchmark_pytorch(model, seq_len, input_dim, device,
                      warmup_runs=WARMUP_RUNS, timed_runs=TIMED_RUNS):
    """Benchmark PyTorch inference latency."""
    model = model.to(device)
    model.eval()

    dummy = torch.randn(BATCH_SIZE, seq_len, input_dim, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    return np.array(latencies)


# ---------------------------------------------------------------------------
# Benchmark: ONNX Runtime
# ---------------------------------------------------------------------------
def benchmark_onnx(onnx_path, seq_len, input_dim,
                   warmup_runs=WARMUP_RUNS, timed_runs=TIMED_RUNS):
    """Benchmark ONNX Runtime inference latency (CPU)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("    WARNING: onnxruntime not installed, skipping ONNX benchmark")
        return None

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1  # single-threaded for latency
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(str(onnx_path), sess_options,
                                   providers=["CPUExecutionProvider"])

    dummy = np.random.randn(BATCH_SIZE, seq_len, input_dim).astype(np.float32)

    # Warm-up
    for _ in range(warmup_runs):
        session.run(None, {"input": dummy})

    # Timed runs
    latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return np.array(latencies)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RQ2 inference latency benchmark"
    )
    parser.add_argument(
        "--tte", type=float, default=DEFAULT_TTE,
        help=f"TTE to benchmark (default: {DEFAULT_TTE})",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Warm-up iterations (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--runs", type=int, default=TIMED_RUNS,
        help=f"Timed iterations (default: {TIMED_RUNS})",
    )
    parser.add_argument(
        "--skip-quantize", action="store_true",
        help="Skip INT8 quantisation benchmark",
    )
    args = parser.parse_args()

    warmup_runs = args.warmup
    timed_runs = args.runs

    print("=" * 72)
    print("  RQ2 INFERENCE LATENCY BENCHMARK")
    print(f"  Device       : {DEVICE}")
    print(f"  TTE          : {args.tte}s")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Warm-up      : {warmup_runs}")
    print(f"  Timed runs   : {timed_runs}")
    print(f"  Quantize     : {'no' if args.skip_quantize else 'yes'}")
    print("=" * 72)

    # Find checkpoints for the chosen TTE
    tte_str = f"tte{args.tte:.1f}"
    checkpoints = sorted(CHECKPOINT_DIR.glob(f"gru_*_{tte_str}.pt"))

    if not checkpoints:
        print(f"\n  ERROR: No checkpoints found matching *_{tte_str}.pt")
        print(f"  Run train_finalists.py first.")
        print(f"  Looking in: {CHECKPOINT_DIR}")
        return

    print(f"\n  Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"    {cp.name}")

    # Results accumulator
    all_results = []

    for ckpt_path in checkpoints:
        model, meta, mean, std = load_checkpoint(ckpt_path)
        exp_key = meta["experiment_key"]
        label = meta["label"]
        input_dim = meta["input_dim"]
        seq_len = meta["seq_len"]
        n_params = meta["n_params"]

        print(f"\n{'─' * 72}")
        print(f"  {label}  [{exp_key}]")
        print(f"  input_dim={input_dim}  seq_len={seq_len}  params={n_params}")
        print(f"{'─' * 72}")

        # --- 1. PyTorch benchmark -----------------------------------------
        print(f"\n  [1/3] PyTorch ({DEVICE}) ...")
        pt_latencies = benchmark_pytorch(
            model, seq_len, input_dim, DEVICE,
            warmup_runs=warmup_runs, timed_runs=timed_runs,
        )

        pt_result = {
            "experiment_key": exp_key,
            "label": label,
            "input_dim": input_dim,
            "seq_len": seq_len,
            "n_params": n_params,
            "mode": f"PyTorch ({DEVICE})",
            "mean_ms": pt_latencies.mean(),
            "std_ms": pt_latencies.std(),
            "p95_ms": np.percentile(pt_latencies, 95),
            "min_ms": pt_latencies.min(),
            "max_ms": pt_latencies.max(),
            "fps": 1000.0 / pt_latencies.mean(),
            "model_size_bytes": os.path.getsize(ckpt_path),
        }
        all_results.append(pt_result)
        print(f"    mean={pt_result['mean_ms']:.3f}ms  "
              f"std={pt_result['std_ms']:.3f}ms  "
              f"p95={pt_result['p95_ms']:.3f}ms  "
              f"FPS={pt_result['fps']:.0f}")

        # --- 2. ONNX FP32 benchmark --------------------------------------
        onnx_path = ONNX_DIR / f"{exp_key}_tte{args.tte:.1f}.onnx"
        print(f"\n  [2/3] ONNX FP32 (CPU) ...")

        try:
            export_to_onnx(model, meta, onnx_path)
            onnx_latencies = benchmark_onnx(
                onnx_path, seq_len, input_dim,
                warmup_runs=warmup_runs, timed_runs=timed_runs,
            )

            if onnx_latencies is not None:
                onnx_result = {
                    "experiment_key": exp_key,
                    "label": label,
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "n_params": n_params,
                    "mode": "ONNX FP32 (CPU)",
                    "mean_ms": onnx_latencies.mean(),
                    "std_ms": onnx_latencies.std(),
                    "p95_ms": np.percentile(onnx_latencies, 95),
                    "min_ms": onnx_latencies.min(),
                    "max_ms": onnx_latencies.max(),
                    "fps": 1000.0 / onnx_latencies.mean(),
                    "model_size_bytes": os.path.getsize(onnx_path),
                }
                all_results.append(onnx_result)
                print(f"    mean={onnx_result['mean_ms']:.3f}ms  "
                      f"std={onnx_result['std_ms']:.3f}ms  "
                      f"p95={onnx_result['p95_ms']:.3f}ms  "
                      f"FPS={onnx_result['fps']:.0f}")
        except Exception as e:
            print(f"    ONNX export failed: {e}")

        # --- 3. ONNX INT8 quantised benchmark -----------------------------
        if not args.skip_quantize:
            quant_path = ONNX_DIR / f"{exp_key}_tte{args.tte:.1f}_int8.onnx"
            print(f"\n  [3/3] ONNX INT8 quantised (CPU) ...")

            quant_result_path = quantise_onnx(onnx_path, quant_path)
            if quant_result_path is not None:
                quant_latencies = benchmark_onnx(
                    quant_path, seq_len, input_dim,
                    warmup_runs=warmup_runs, timed_runs=timed_runs,
                )

                if quant_latencies is not None:
                    q_result = {
                        "experiment_key": exp_key,
                        "label": label,
                        "input_dim": input_dim,
                        "seq_len": seq_len,
                        "n_params": n_params,
                        "mode": "ONNX INT8 (CPU)",
                        "mean_ms": quant_latencies.mean(),
                        "std_ms": quant_latencies.std(),
                        "p95_ms": np.percentile(quant_latencies, 95),
                        "min_ms": quant_latencies.min(),
                        "max_ms": quant_latencies.max(),
                        "fps": 1000.0 / quant_latencies.mean(),
                        "model_size_bytes": os.path.getsize(quant_path),
                    }
                    all_results.append(q_result)
                    print(f"    mean={q_result['mean_ms']:.3f}ms  "
                          f"std={q_result['std_ms']:.3f}ms  "
                          f"p95={q_result['p95_ms']:.3f}ms  "
                          f"FPS={q_result['fps']:.0f}")
        else:
            print(f"\n  [3/3] Skipped (--skip-quantize)")

    # --- Save results -----------------------------------------------------
    if not all_results:
        print("\n  No results to save.")
        return

    import pandas as pd

    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "benchmark_latency.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    # --- Plot: latency comparison -----------------------------------------
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
            "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
            "axes.grid": True, "grid.alpha": 0.3,
            "axes.spines.top": False, "axes.spines.right": False,
        })

        modes = results_df["mode"].unique()
        mode_colors = {
            m: c for m, c in zip(
                modes, ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
            )
        }

        # --- Grouped bar chart: latency ------------------------------------
        labels_unique = results_df["label"].unique()
        n_models = len(labels_unique)
        n_modes = len(modes)
        x = np.arange(n_models)
        width = 0.8 / n_modes

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, mode in enumerate(modes):
            sub = results_df[results_df["mode"] == mode]
            # Align by label order
            vals = []
            errs = []
            for lbl in labels_unique:
                row = sub[sub["label"] == lbl]
                if not row.empty:
                    vals.append(row["mean_ms"].values[0])
                    errs.append(row["std_ms"].values[0])
                else:
                    vals.append(0)
                    errs.append(0)

            ax.bar(
                x + i * width, vals, width, yerr=errs,
                label=mode, color=mode_colors.get(mode, "#999"),
                capsize=3, edgecolor="white", linewidth=0.5,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Inference Latency (ms)")
        ax.set_title(f"RQ2 — Inference Latency (batch=1, TTE={args.tte}s)")
        ax.set_xticks(x + width * (n_modes - 1) / 2)
        dims = [
            results_df[results_df["label"] == lbl]["input_dim"].values[0]
            for lbl in labels_unique
        ]
        ax.set_xticklabels(
            [f"{lbl}\n(dim={d})" for lbl, d in zip(labels_unique, dims)],
            fontsize=9,
        )
        ax.legend()

        # Add 30 FPS target line (33.3ms)
        ax.axhline(y=33.3, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(
            n_models - 0.5, 34, "30 FPS target",
            color="red", fontsize=9, ha="right",
        )

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "benchmark_latency.png")
        plt.close()

        # --- FPS vs input dimension ----------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        for mode in modes:
            sub = results_df[results_df["mode"] == mode].sort_values("input_dim")
            ax.plot(
                sub["input_dim"], sub["fps"],
                marker="o", lw=2, label=mode,
                color=mode_colors.get(mode, "#999"),
            )

        ax.axhline(y=30, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(
            results_df["input_dim"].max() - 1, 32, "30 FPS target",
            color="red", fontsize=9, ha="right",
        )

        ax.set_xlabel("Input Dimension (number of features)")
        ax.set_ylabel("Throughput (inferences / second)")
        ax.set_title(f"RQ2 — Throughput vs Model Complexity (TTE={args.tte}s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "benchmark_fps.png")
        plt.close()

        print(f"  Plots saved to {RESULTS_DIR}/")

    except ImportError:
        print("  matplotlib not available — skipping plots")

    # --- Console summary ---------------------------------------------------
    print(f"\n{'=' * 72}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 72}")

    for _, r in results_df.iterrows():
        fps_ok = "OK" if r["fps"] >= 30 else "BELOW"
        print(
            f"  {r['label']:30s}  {r['mode']:25s}  "
            f"mean={r['mean_ms']:7.3f}ms  "
            f"p95={r['p95_ms']:7.3f}ms  "
            f"FPS={r['fps']:8.0f}  [{fps_ok}]  "
            f"size={r['model_size_bytes']/1024:.1f}KB"
        )

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
