# 2 — RQ2 / RQ3 Edge Deployment

Takes the models from RQ1 and asks whether they run on the target edge device (an
NVIDIA Jetson Orin Nano Super) in real time.

- **RQ2 — can it run on the edge?** Benchmark the GRU classifier and the camera
  perception frontends (YOLO detection vs. pose, PyTorch FP32 vs. TensorRT FP16)
  against a 30 FPS / 33 ms-per-frame budget. Steps 1–8.
- **RQ3 — what is the end-to-end latency?** Measure full system latency from a
  recorded clip (or live camera): capture → perception → features → GRU. Step 9.

## Everything runs on the Jetson

The whole pipeline runs on one machine. Steps 1–3 train the GRU finalists and
write them to `outputs/checkpoints/`; steps 4–9 benchmark on that same Jetson and load
the checkpoints straight from `outputs/checkpoints/`. Nothing is copied between machines.

Training the GRU is small and hardware-agnostic, so running it on the Jetson is
fine and keeps the pipeline a single command. The numbers that matter are the
*inference* latencies measured in steps 4–9 on the Jetson GPU.

## Before you start (on the Jetson)

- The repo is on the Jetson with the dependencies installed (see the top-level
  README; TensorRT ships with JetPack).
- The cleaned dataset is in `MasterData/` — steps 1–3 and 8 train on it.
- **RQ1 has been run.** This stage reads two RQ1 outputs and will tell you to run
  RQ1 first if they are missing:
  `1_RQ1_feature_importance/neural_network_ablation/results/results_all.csv`
  and `.../neural_network_ablation/results/gru_finalists.csv`.
- Video for the benchmarks lives in `benchmark_videos/` — clips for steps 5–7 in
  `rq2_perception_clips/` and one recorded clip for step 9 in `rq3_system_clip/`.
  See that folder's README; `run_all` auto-detects both.

## Run it

```bash
cd 2_RQ2_RQ3_edge_deployment

# 1. prepare the two video inputs (details in benchmark_videos/)
python benchmark_videos/rq2_perception_clips/select_clips.py            # clips for steps 5-7
python benchmark_videos/rq3_system_clip/record_clip.py --duration 90    # clip for step 9 (RQ3)

# 2. run everything — run_all auto-detects benchmark_videos/
python run_all.py
```

`run_all.py` runs the steps in order and stops if one fails. Useful flags:
`--from N` (resume at step N), `--only 5 6`, `--dry-run` (list without running),
and `--synthetic` (a quick check of steps 5–7 on random frames, no video needed).

## The steps

| # | Script | Does | Needs |
|---|--------|------|-------|
| 1 | `step1_select_finalists.py` | Load/confirm the GRU finalists from RQ1 | RQ1 outputs |
| 2 | `step2_train_finalists.py` | Train the 5 finalists → `outputs/checkpoints/gru_*.pt` | `MasterData/` |
| 3 | `step3_validate_hyperparams.py` | Hyperparameter sensitivity check | `MasterData/` |
| 4 | `step4_benchmark_classifier.py` | GRU inference latency (PyTorch / ONNX / INT8) | `outputs/checkpoints/` |
| 5 | `step5_benchmark_perception.py` | YOLO frontend latency | clips |
| 6 | `step6_benchmark_pipeline.py` | End-to-end perception + features + GRU (main RQ2 figure) | clips + `outputs/checkpoints/` |
| 7 | `step7_benchmark_model_sizes.py` | YOLO size × precision (FP32 / FP16 / INT8) sweep | clips |
| 8 | `step8_benchmark_gru_sizes.py` | GRU hidden-size sweep (a pruning proxy) | `MasterData/` |
| 9 | `step9_benchmark_live_camera.py` | RQ3 end-to-end system latency | recorded clip or camera |

The checkpoint flow is automatic: step 2 writes `outputs/checkpoints/gru_<config>_tte<TTE>.pt`,
and steps 4 and 6 read them from there. Run a benchmark before training and the
script prints "run step2 first" rather than failing cryptically.

## Configuration

`config.py` holds everything the steps share: paths, the TTE values and fixed
hyperparameters (identical to the RQ1 ablation), the perception frontends to
benchmark, the complexity tiers, and the 30 FPS target. The benchmark protocol
matches the thesis — 50 warmup + 500 timed frames for the perception and pipeline
steps, and 200 + 2000 for the sub-millisecond GRU sweep.

## Helper scripts

- `benchmark_videos/` — the video inputs and the scripts that make them:
  `rq2_perception_clips/select_clips.py` (clips for steps 5-7) and
  `rq3_system_clip/record_clip.py` + `verify_clip.py` (the RQ3 clip). See its README.
- `plot_rq2_results.py` / `generate_rq3_figures.py` — turn the saved results into
  the RQ2/RQ3 figures.

## Models, engines, and outputs

YOLO weights download automatically on first use, so they are not committed.
TensorRT `.engine` files are built on the Jetson the first time a `[TRT]`
configuration runs (they are tied to the device's GPU and TensorRT version), so
they are not committed either. Everything the scripts generate lands under `outputs/` (`checkpoints/`, `onnx/`,
`results/`, `figures/`) — regenerated, not shipped. The auto-downloaded YOLO
weights and `.engine` files sit in the folder itself and are gitignored.
