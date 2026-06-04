# neural_network_ablation

The main RQ1 experiment: a structured ablation of neural networks (MLP, GRU, CNN)
over trajectory and body-pose features. It answers whether neural networks beat
the logistic-regression baseline, whether temporal modeling helps, and which pose
information is worth its cost.

Three stages, run in order — the file names match the order:

```
neural_network_ablation/
  train_ablation.py         # 1. run the ablation         -> results/results_all.csv
  select_gru_finalists.py   # 2. pick the GRU finalists    -> results/gru_finalists.csv
  plot_ablation_figures.py  # 3. make the comparison figs  -> results/figures/
  ablation_config.py        #    shared paths + tiers (used by stages 2-3)
  results/                  #    everything generated lands here (gitignored)
```

## The three stages

1. **`train_ablation.py`** — runs the full ablation. Experiments are grouped:
   - **Group A** anchors the neural networks against logistic regression on the
     same features, and checks whether the GRU/CNN beat the single-frame MLP.
   - **Group B** adds body pose progressively (head → torso+head → full body) in
     both angle and raw-landmark form.
   - **Group C** combines the small "core" feature sets from the random forest
     with targeted pose features.

   Each cell is evaluated with 5-fold cross-validation × 3 repeats at five TTE
   values, and one row per (experiment, TTE, model) is written to
   `results/results_all.csv`.

2. **`select_gru_finalists.py`** — reads `results/results_all.csv`, scores each
   GRU configuration, and picks the best one per complexity tier. Writes
   `results/gru_finalists.csv` (plus `gru_all_scored.csv` and a per-TTE table)
   and an audit figure. This is the handoff to RQ2.

3. **`plot_ablation_figures.py`** — turns `results/results_all.csv` into the RQ1
   comparison figures (groups A/B/C, model comparison, etc.) under
   `results/figures/`.

`ablation_config.py` holds the paths and complexity-tier definitions shared by
stages 2 and 3.

## Run order

```bash
python train_ablation.py          # 1.  -> results/results_all.csv
python select_gru_finalists.py    # 2.  -> results/gru_finalists.csv
python plot_ablation_figures.py   # 3.  -> results/figures/
```

## Outputs that RQ2 consumes

`2_RQ2_RQ3_edge_deployment/` starts from two files produced here:

- `results/results_all.csv` — every ablation result.
- `results/gru_finalists.csv` — the GRU finalists, one per complexity tier.

Everything under `results/` is created when the scripts run and is not committed
(regenerated, gitignored). Run this experiment before RQ2.
