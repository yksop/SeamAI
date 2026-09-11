# utils

Common code imported by the experiments in `1_RQ1_feature_importance/` and
`2_RQ2_RQ3_edge_deployment/`. Keeping it here means every experiment loads data
and builds features the same way.

Scripts add the pipeline root to `sys.path` and import these as
`from utils.<module> import ...`, so run the experiment scripts from within the
`thesis_experiments/` tree and the imports resolve on their own.

## Modules

- `dataset_utils.py` — loads valid `enter`/`pass` samples from `MasterData/`
  (skipping anything cleaning flagged as exit/removed/corrupt) and builds the
  feature matrix used by the baselines.
- `feature_extractor.py` — turns one trajectory sample into the 30 handcrafted
  features: 10 per-frame signals, each summarized over the observation window as
  `[mean, variance, latest]`. Also decides whether a sample is usable for a
  given time-to-event and window.
- `sequence_dataset.py` — builds the per-frame sequences for the neural
  networks, with optional body-pose channels, plus the normalization helpers.
- `models.py` — the three model definitions, all with the same constructor
  signature and input shape:
  - `IntentMLP` — single-frame baseline (uses only the last frame).
  - `IntentGRU` — recurrent model over the full sequence.
  - `IntentCNN` — temporal 1D-CNN as a non-recurrent alternative.

## Key conventions

- A "sample" is one pedestrian: a trajectory JSON, its video clip, and an
  optional `*_pose.json` pose sidecar.
- Labels: `enter` = 1, `pass` = 0.
- Time-to-event (TTE) is how far before the event the prediction is made; the
  observation window is the half-second of frames just before that point.
