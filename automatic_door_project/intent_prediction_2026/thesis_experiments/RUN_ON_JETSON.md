# Running the pipeline on the Jetson

End-to-end walkthrough: from raw recordings to RQ1/RQ2/RQ3 results on the
NVIDIA Jetson Orin Nano. Run every command from this folder
(`thesis_experiments/`) with the virtual environment active.

The flow is: `DataOriginal/` (Hanna's format) → `0_data_cleaning` builds
`MasterData/` → RQ1 → RQ2/RQ3.

For setup details and every script's options/flags, see `USAGE.md`.

## 1. Setup (once)

```bash
cd <repo>/SeamAI/intent_prediction_2026/thesis_experiments
python3 -m venv .venv --system-site-packages   # reuse the Jetson's global torch / cv2 / numpy
source .venv/bin/activate
pip install -r requirements.txt
python check_env.py                              # everything "ok"?
```

If `check_env.py` reports `torch` missing, install NVIDIA's Jetson wheel first —
see the notes at the top of `requirements.txt`.

## 2. One-time download — the pose model

`extract_pose.py` needs the MediaPipe pose model next to it, in
`0_data_cleaning/`. It is found automatically from anywhere once it is there:

```bash
curl -L -o 0_data_cleaning/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## 3. Put the data in place

- **Recordings** in `DataOriginal/`, in the original format produced by
  `automatic_door/automatic_labeling/data_collection.py` (Norberg) — per-session
  folders with `enter/` and `pass/`, each clip a matching `.json` + `.mp4`. See
  `DataOriginal/README.md`.
- **RQ3 reference clip** at
  `2_RQ2_RQ3_edge_deployment/benchmark_videos/rq3_system_clip/rq3_clip.mp4`
  (a ~90 s clip of someone walking toward/away from the camera). If you already
  have one, just copy it there; otherwise record one with
  `benchmark_videos/rq3_system_clip/record_clip.py`.

## 4. Step 0 — clean + pose

```bash
python 0_data_cleaning/build_masterdata.py        # DataOriginal/ -> MasterData/
python 0_data_cleaning/extract_pose.py            # adds *_pose.json sidecars
python 0_data_cleaning/data_coverage_tte_window.py   # optional: how much usable data
```

## 5. Step 1 — RQ1 (feature importance), in order

```bash
python 1_RQ1_feature_importance/baseline_logreg/train_logreg.py
python 1_RQ1_feature_importance/baseline_rf_shap/rf_feature_importance.py
python 1_RQ1_feature_importance/neural_network_ablation/train_ablation.py
python 1_RQ1_feature_importance/neural_network_ablation/select_gru_finalists.py
```

The last script writes `results/gru_finalists.csv`, which RQ2 reads.

## 6. Step 2 — RQ2 / RQ3 (edge benchmarks)

```bash
cd 2_RQ2_RQ3_edge_deployment

# perception clips for steps 5-7 (a stratified sample pulled from MasterData):
python benchmark_videos/rq2_perception_clips/select_clips.py

# run all 9 steps; run_all auto-detects the clips and the RQ3 clip:
python run_all.py
```

`run_all.py` prints an "Inputs" check first — confirm it lists the steps 5-7
clips as found and the step 9 (RQ3) clip as found. If step 9 reports no clip,
check that `rq3_clip.mp4` is in `benchmark_videos/rq3_system_clip/`.

## Quick test on a subset

To smoke-test the whole chain fast, clean just one recording session and run as
above — the results are only a sanity check, not full numbers:

```bash
python 0_data_cleaning/build_masterdata.py --src DataOriginal/live_output_10_1
```

## Where the outputs land

Each step writes its output when it runs: RQ1 into a `results/`/`figures/` folder
(e.g. `1_RQ1_feature_importance/neural_network_ablation/results/`), RQ2/RQ3 into
`2_RQ2_RQ3_edge_deployment/outputs/`. These are regenerated, not committed.
