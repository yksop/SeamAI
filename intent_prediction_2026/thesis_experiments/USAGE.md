# Setup & usage

How to set up the environment, install dependencies, and run each script — and
when you pass `--flags` versus editing settings in a script.

For a straight top-to-bottom run, see `RUN_ON_JETSON.md`. This file is the
reference: setup once, then look up whichever script you want to run.

---

## 1. Virtual environment & install

**Laptop (macOS / x86 Linux):**

```bash
cd thesis_experiments
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_env.py                  # reports which packages are present
```

`torch`/`torchvision` are commented out in `requirements.txt` — on a laptop,
uncomment them (or `pip install torch torchvision`) so PyTorch comes from PyPI.

**Jetson:**

```bash
cd thesis_experiments
python3 -m venv .venv --system-site-packages   # reuse the global CUDA torch / cv2 / numpy
source .venv/bin/activate
pip install -r requirements.txt
python check_env.py
```

On the Jetson, PyTorch/OpenCV/NumPy are the global CUDA builds (don't reinstall
them from PyPI). If `check_env.py` says `torch` is missing, install NVIDIA's
Jetson wheel first — see the notes at the top of `requirements.txt`.

`check_env.py` lists each package as `ok` / `MISS` and notes whether CUDA is
available. Run it whenever something import-fails.

---

## 2. How settings work — two patterns

This is the key thing to know before running anything:

1. **Command-line `--flags`.** The data and benchmark/utility scripts take flags
   for paths and quick options. They all support `--help`:

   ```bash
   python 0_data_cleaning/build_masterdata.py --help
   ```

2. **Edit the SETTINGS block at the top of the script.** The core experiment
   scripts (the RQ1 logistic-regression / random-forest / ablation scripts, and
   a few RQ2 steps) have **no** flags. To change the experiment — prediction
   horizons (TTE), observation window, hyperparameters, which experiments to run
   — open the script and edit the clearly marked settings block near the top,
   then run it plain (`python <script>`).

   The shared RQ2/RQ3 settings live in one place: `2_RQ2_RQ3_edge_deployment/config.py`.

When in doubt: if `python <script> --help` shows options, it takes flags;
otherwise it's a SETTINGS-block script.

---

## 3. Per-script reference

Run everything from the `thesis_experiments/` folder with the venv active.
"edit SETTINGS" = no flags, change the settings block at the top of the file.

### 0_data_cleaning  (build the dataset)

| Script | What it does | Settings |
|--------|--------------|----------|
| `build_masterdata.py` | `DataOriginal/` → cleaned `MasterData/` | flags: `--src`, `--dst`, `--summary-only` (count only, write nothing), `--min-duration-s`, `--ellipse-a/-b`, `--duplicate-report-csv` |
| `extract_pose.py` | adds `*_pose.json` pose sidecars to `MasterData/` | flag: `--overwrite` (redo finished files). Needs `pose_landmarker_lite.task` in `0_data_cleaning/` (one-time download) |
| `data_coverage_tte_window.py` | reports usable samples per TTE/window | edit SETTINGS (`TTE_VALUES`, `WINDOW_VALUES`) |

```bash
python 0_data_cleaning/build_masterdata.py                       # full DataOriginal
python 0_data_cleaning/build_masterdata.py --src DataOriginal/live_output_10_1   # one session
python 0_data_cleaning/build_masterdata.py --summary-only        # dry count, writes nothing
python 0_data_cleaning/extract_pose.py
```

### utils/  (library — nothing to run)

Imported by the experiments (`from utils.X import ...`): feature extraction,
dataset loading, model definitions. You don't run these directly.

### 1_RQ1_feature_importance  (all SETTINGS-block scripts — no flags)

| Script | What it does | Settings |
|--------|--------------|----------|
| `baseline_logreg/train_logreg.py` | logistic-regression baseline | edit SETTINGS (TTE values, window) |
| `baseline_rf_shap/rf_feature_importance.py` | random forest + SHAP, "core" feature sets | edit SETTINGS |
| `neural_network_ablation/train_ablation.py` | MLP/GRU/CNN ablation → `results/results_all.csv` | edit the SETTINGS block (TTE, window, hyperparameters, experiments) |
| `neural_network_ablation/select_gru_finalists.py` | picks GRU finalists → `results/gru_finalists.csv` | edit SETTINGS (tiers) |
| `neural_network_ablation/plot_ablation_figures.py` | makes the ablation figures | none |

```bash
python 1_RQ1_feature_importance/baseline_logreg/train_logreg.py
python 1_RQ1_feature_importance/baseline_rf_shap/rf_feature_importance.py
python 1_RQ1_feature_importance/neural_network_ablation/train_ablation.py
python 1_RQ1_feature_importance/neural_network_ablation/select_gru_finalists.py
```

### 2_RQ2_RQ3_edge_deployment  (edge benchmarks)

`config.py` holds the shared settings (TTE values, hyperparameters, perception
frontends, 30 FPS target) — edit it to change all steps at once. Everything the
steps generate (checkpoints, onnx, results, figures) goes under `outputs/`.

The easy path is the orchestrator:

```bash
cd 2_RQ2_RQ3_edge_deployment
python run_all.py --dry-run        # list the 9 steps without running
python run_all.py                  # run them in order (auto-detects benchmark_videos/)
python run_all.py --from 4         # resume at step 4
python run_all.py --only 5 6       # just these steps
```

`run_all.py` flags: `--from`, `--only`, `--dry-run`, `--synthetic` (steps 5–7 on
random frames, no video), `--input-dir` (clip folder for 5–7), `--rq3-video`
(clip for step 9). It auto-uses `benchmark_videos/` if those are populated.

Individual steps (run any directly; `--help` shows the full list):

| Step | Settings worth knowing |
|------|------------------------|
| `step1_select_finalists.py`, `step2_train_finalists.py`, `step8_benchmark_gru_sizes.py` | no flags (use `config.py`) |
| `step3_validate_hyperparams.py` | `--quick` (smaller sweep) |
| `step4_benchmark_classifier.py` | `--tte`, `--warmup`, `--runs`, `--skip-quantize` |
| `step5_benchmark_perception.py` | `--input-dir` / `--input` / `--synthetic`, `--device`, `--skip-trt` / `--only-trt`, `--max-clips`, `--warmup` / `--runs` |
| `step6_benchmark_pipeline.py` | `--input-dir` / `--synthetic`, `--tte`, `--device`, `--skip-trt` |
| `step7_benchmark_model_sizes.py` | `--input-dir` / `--synthetic`, `--device`, `--skip-trt` / `--skip-int8` |
| `step9_benchmark_live_camera.py` | `--video` / `--camera`, `--mode {quick,nano,standard,full}`, `--tte`, `--device` |

Helper scripts for the video inputs (in `benchmark_videos/`):

```bash
python benchmark_videos/rq2_perception_clips/select_clips.py        # clips for steps 5-7
python benchmark_videos/rq2_perception_clips/select_clips.py --n 50 # how many (default 30); also --seed, --output
python benchmark_videos/rq3_system_clip/record_clip.py --duration 90 # record the RQ3 clip; also --camera, --preview, --output
python benchmark_videos/rq3_system_clip/verify_clip.py               # check a clip; --video PATH
```

### demo_open_house  (the open-house demo)

```bash
cd demo_open_house
python demo_live.py --model weights/gru_core3head.pt --norm weights/norm_stats.npz
```

Main flags: `--model`, `--norm`, `--source` (camera index or video), `--yolo`,
`--threshold`, `--door-x/--door-y`, `--fullscreen`, `--no-features/--no-skeleton`.
`train_demo_model.py` (re)trains the demo model: `--data`, `--tte`, `--output`.

---

## 4. Settings you'll most likely change

- **Prediction horizon (TTE) and observation window** — for RQ1, edit the SETTINGS
  block of each script; for RQ2/RQ3, edit `2_RQ2_RQ3_edge_deployment/config.py`.
  Keep them consistent across RQ1 and RQ2 (they're meant to match).
- **Where the data is** — `build_masterdata.py --src/--dst`. Everything else reads
  `MasterData/` at the pipeline root automatically.
- **Quick vs full benchmark runs** — `--synthetic` (no video), `--quick`,
  `--max-clips`, `--runs` on the relevant steps for a fast smoke test.
