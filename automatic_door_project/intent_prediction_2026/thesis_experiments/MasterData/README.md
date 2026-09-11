# MasterData/ — cleaned dataset (generated, not included)

This is the **cleaned** dataset that every experiment reads from
(`1_RQ1_feature_importance/` and `2_RQ2_RQ3_edge_deployment/`). It is generated
from `DataOriginal/` by step 0, so it is not committed to git (it is large and
derived from identifiable video).

## How to generate it

```bash
python 0_data_cleaning/build_masterdata.py   # DataOriginal/  ->  MasterData/
python 0_data_cleaning/extract_pose.py        # adds *_pose.json pose sidecars
```

## Layout

```
MasterData/
  enter/                 # pedestrians who entered the door
    *.json               # one trajectory sample per pedestrian
    *.mp4                # matching clip
    *_pose.json          # pose sidecar (added by extract_pose.py)
  pass/                  # pedestrians who walked past
    ...
  Corrup/                # clips rejected during cleaning (kept for reference)
```

Samples that cleaning flags as `exit` or `removed` are moved into the
`enter/exit`, `enter/removed`, `pass/exit`, and `pass/remove` subfolders. The
loaders in `utils/dataset_utils.py` skip those automatically, so only valid
`enter`/`pass` samples reach the experiments.
