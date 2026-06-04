# DataOriginal/ — raw recordings (not included)

This folder is where the **raw** pedestrian recordings go. The data-cleaning
step (`0_data_cleaning/build_masterdata.py`) reads from here.

It is intentionally empty in the repository. The recordings are large and
contain identifiable video of people, so they are not committed to git. Get
them from RISE and drop them in here (or point the cleaning script somewhere
else with `--src`).

## Expected layout

The cleaning script walks this folder recursively, so the exact nesting is
flexible. The recordings are grouped into per-session folders, each split into
`enter/` and `pass/`. Every pedestrian is a pair of files that share a name:

```
DataOriginal/
  live_output_10_1/              # one recording session
    enter/
      20251001_125855_id3_enter.json   # per-frame trajectory + metadata
      20251001_125855_id3_enter.mp4    # matching video clip
    pass/
      20251001_130204_id5_pass.json
      20251001_130204_id5_pass.mp4
  live_output_10_2/
    ...
```

Each `.json` holds the frame-level tracking data plus metadata
(`id`, `label`, `frames`, `fps`, `frame_width`, `frame_height`, `door_center`,
`ellipse_axes`).

## Using a different location

```bash
python 0_data_cleaning/build_masterdata.py --src /path/to/raw/recordings
```

These recordings are in the format produced by the **original** collection
script, `automatic_door/automatic_labeling/data_collection.py` (Norberg) —
bounding-box tracking saved as the per-clip JSON + MP4 pairs shown above. To
collect more data for these experiments, run that script and drop its output
here.

Note: the newer `data_collection_v2/` pipeline records a *different* format
(single-pass YOLO pose with COCO-17 keypoints), so its output is **not** a
drop-in source for `build_masterdata.py`.
