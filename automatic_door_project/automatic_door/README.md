# automatic_door

The original smart-door scripts (by Hanna Norberg). The part most people will
run is the **data collection** in `automatic_labeling/`, which records people
walking toward the door and auto-labels each clip as **enter** or **pass**.

> The commands below are written relative to the **repository root** (the top
> folder of this checkout). `cd` into it first, then copy the commands as-is.

## Folder overview

- `automatic_labeling/` — the live collection scripts. **Start here.**
  - `data_collection.py` — records + auto-labels enter/pass. This is the one to run.
  - `always_on.py` — same tracking, but a "view only" mode that saves nothing.
- `trajectory_prediction/`, `intention_prediction/`, `data_enrichment/`,
  `camera_calibration/` — offline analysis/experiments (need extra packages,
  see the bottom of this file).

## 1. Set up the environment (once)

Creates a `.venv` inside this folder. On the Jetson, `--system-site-packages`
reuses the global CUDA `torch` / `OpenCV` / `NumPy`, and pip adds `ultralytics`
on top.

```bash
cd automatic_door
python3 -m venv .venv --system-site-packages
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

No `activate` needed — the venv's `pip` and `python` are called by path.

**Shortcut:** the same steps are wrapped in a script (run it from the repo root):

```bash
bash automatic_door/setup_venv.sh
```

## 2. Run the data collection

The script writes its output to the folder you run it from, so `cd` into
`automatic_labeling/` first — the dataset then lands in
`automatic_labeling/live_output/`.

```bash
cd automatic_door/automatic_labeling
../.venv/bin/python data_collection.py
```

- A live window ("Live Tracking") opens — run this on the Jetson **with a
  monitor connected** (not over plain SSH). Press **`q`** to stop.
- The camera is read from index `0` (`/dev/video0`); the Jetson lens calibration
  is built in (`computer="Jetson"`).
- First run downloads the YOLO weights (`yolo11s.pt`) automatically (needs
  internet once).

### Output

```
automatic_labeling/live_output/
  enter/   ... <clip>.json + <clip>.mp4   (walked through the door)
  pass/    ... <clip>.json + <clip>.mp4   (walked past)
  temp/    ... work-in-progress clips
```

### Changing settings

`data_collection.py` has no `--flags`; edit the `main()` block at the bottom of
the file. Useful ones: `test_rec=True` (writes to `live_output_test/` for a
throwaway test), `camera_index=...`, and the door zone `ellipse_axes=...`.

## 3. Feed it into the thesis pipeline

From the repository root, copy the collected `live_output` folder into the
cleaning pipeline as one session, then clean it:

```bash
cp -r automatic_door/automatic_labeling/live_output \
      intent_prediction_2026/thesis_experiments/DataOriginal/live_output_NEW

cd intent_prediction_2026/thesis_experiments
.venv/bin/python 0_data_cleaning/build_masterdata.py
```

(Rename `live_output_NEW` to any session name you like.) The enter/pass +
paired `.json`/`.mp4` layout is exactly what `build_masterdata.py` expects.

## Note on the other subfolders

The analysis folders (`trajectory_prediction/`, `intention_prediction/`,
`data_enrichment/`) also use `scikit-learn`, `matplotlib`, `joblib`,
`mediapipe` and `tensorflow`. Those are **not** needed just to collect data —
uncomment the relevant lines in `requirements.txt` and reinstall only if you
run them. (`tensorflow` on the Jetson needs NVIDIA's special build, not PyPI.)
