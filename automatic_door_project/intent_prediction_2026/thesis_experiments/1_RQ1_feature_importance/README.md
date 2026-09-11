# 1 — RQ1 Feature Importance

Experiments for the first research question:
**which visual features are most predictive of intent to enter?**

The features come in two forms, both built in `utils/`:

- **30 handcrafted features** — 10 per-frame trajectory signals (distance to
  door, closure rate, velocity, angle to door, bounding-box height, etc.), each
  summarized over the observation window as `[mean, variance, latest]`.
- **Per-frame sequences** — the same signals frame by frame, optionally extended
  with body-pose landmarks, fed to the sequence models.

## Subfolders

- `baseline_logreg/` — logistic regression on the 30 handcrafted features. Fast
  and interpretable; the reference point for everything else.
- `baseline_rf_shap/` — random forest with SHAP. A non-linear view of which
  features matter; its findings define the "core" feature sets used below.
- `neural_network_ablation/` — the main MLP / GRU / CNN ablation across
  trajectory and pose features. See its own README.

## Shared data and code

- All scripts read the cleaned dataset from `MasterData/`.
- The feature, dataset, and model code lives in `utils/`.

## Recommended run order

```bash
python baseline_logreg/train_logreg.py
python baseline_rf_shap/rf_feature_importance.py
python neural_network_ablation/train_ablation.py
python neural_network_ablation/select_gru_finalists.py
```

## Feeds into RQ2

The neural-network ablation produces the inputs that RQ2 builds on:
`neural_network_ablation/results/results_all.csv` (all ablation results) and
`neural_network_ablation/results/gru_finalists.csv`
(the GRU finalists per complexity tier). Run this folder before
`2_RQ2_RQ3_edge_deployment/`.
