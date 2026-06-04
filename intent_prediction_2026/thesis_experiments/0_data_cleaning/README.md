# 0 — Data Cleaning

Builds the cleaned dataset that every experiment uses, starting from the raw
recordings.

## What it does

- Reads raw recordings from `DataOriginal/` (at the pipeline root).
- Filters out low-quality and problematic samples (too short, corrupted video,
  likely "exit" behavior, people only partially visible at the camera bottom,
  and global duplicates).
- Writes a unified dataset to `MasterData/`, keeping the source data unchanged.
- Adds pose landmarks for each sample as `*_pose.json` sidecar files.

## Scripts

- `build_masterdata.py` — cleans the raw recordings and builds `MasterData/`.
  Also writes `MasterData/duplicate_report.csv`, listing the clip pairs it
  dropped as duplicates.
- `extract_pose.py` — runs MediaPipe Pose on each clip and saves the landmarks
  as `*_pose.json`. This is the prerequisite for the pose-based ablation
  conditions in RQ1.
- `data_coverage_tte_window.py` — reports how many usable enter/pass samples
  remain at each time-to-event (TTE) window. Useful as a sanity check on the
  cleaned dataset before training.

## One-time setup — the pose model

`extract_pose.py` needs the MediaPipe pose model in this folder
(`0_data_cleaning/`); it is found automatically from anywhere once it is here.
Download it once:

```bash
curl -L -o 0_data_cleaning/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Run order

```bash
python 0_data_cleaning/build_masterdata.py   # DataOriginal/ -> MasterData/
python 0_data_cleaning/extract_pose.py        # add *_pose.json sidecars

# optional: check how much usable data you have
python 0_data_cleaning/data_coverage_tte_window.py
```

`build_masterdata.py` reads `DataOriginal/` by default. Point it elsewhere with
`--src /path/to/raw/recordings`. `extract_pose.py` takes ~2–5 minutes on CPU and
skips files it has already processed (use `--overwrite` to redo them).

## Input / output

- Input: `DataOriginal/` — see `../DataOriginal/README.md` for the expected layout.
- Output: `MasterData/` — see `../MasterData/README.md` for the layout.

## Next step

Continue with `1_RQ1_feature_importance/`.
