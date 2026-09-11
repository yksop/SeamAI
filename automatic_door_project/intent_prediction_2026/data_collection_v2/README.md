# data_collection_v2

The start of a new data-collection pipeline for the smart-door pedestrian
dataset. It records people approaching the door, labels each clip
`enter` / `pass` / `exit` while recording, and applies the thesis cleaning
heuristics on the fly, so the dataset comes out already labeled, sorted, and
filtered in a single pass.

It builds on the original data collection and automatic labeling in
`automatic_door/automatic_labeling/` (Norberg) and folds in the automatic
cleaning developed for the thesis — instead of collecting first and cleaning as a
separate step afterward.

## How it builds on the original

Compared to `automatic_door/automatic_labeling/data_collection.py`, the headline
differences are:

1. **One pose model does everything.** A single `yolo11n-pose` pass yields the
   bounding box **and** the 17 COCO keypoints per person, instead of detecting
   only the box and adding a skeleton afterward with a separate MediaPipe step.
2. **`exit` is a real class.** People walking away from the door are detected and
   labeled at collection time, alongside `enter` and `pass`.
3. **Cleaning happens at capture.** The thesis cleaning heuristics run live:
   clips that are too short, have too few detections, show only a partial body at
   the frame bottom, or are corrupt are sorted into `removed/<reason>/` rather
   than mixed into the dataset. Nothing is deleted, and each clip records its own
   `cleaning_result` / `cleaning_reason`.

(Plus: robust handling of people leaving and re-entering the frame, the real
measured fps with per-frame timestamps, and an optional `_pose.json` sidecar with
the skeleton as normalized named landmarks. See `data_collection/README.md` for
the full list and the JSON schema.)

## Structure

```
data_collection_v2/
  requirements.txt     Python dependencies (see Setup)
  check_env.py         run first: checks numpy / opencv / torch / ultralytics
  data_collection/     START HERE — the live collection script + its README
    pose_data_collection.py
    README.md
  data/                where the collection script writes the dataset (default output)
```

The collection script is **self-contained** — it imports no other project code
and writes its dataset to `data/` by default. Turning the collected clips into
model features is a separate, later concern and is not part of collection.

## Setup (Jetson)

**Prerequisite:** CUDA-enabled **PyTorch + torchvision**, **OpenCV** and **NumPy**
are already installed *globally* on the Jetson (as they are on the lab Jetson).
The per-user virtual environment reuses them via `--system-site-packages`, so you
only add `ultralytics` on top.

```bash
python3 check_env.py                            # confirms torch / cuda / cv2 / numpy are present
python3 -m venv .venv --system-site-packages    # reuse the global torch / cv2 / numpy
source .venv/bin/activate
pip install -r requirements.txt                 # adds ultralytics
python data_collection/pose_data_collection.py
```

For real-time speed on the Jetson, export the pose model to a TensorRT engine
once and point the collector at the `.engine` (TensorRT ships with JetPack):

```bash
yolo export model=yolo11n-pose.pt format=engine half=True
```

## Privacy

This records identifiable video and pose data of people; handle it accordingly.
