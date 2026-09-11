# baseline_rf_shap

Random Forest feature-importance analysis with focus on SHAP.

## Contents
- `rf_feature_importance.py` — trains RF, computes importance, and runs ablation checks
- `results/` — figures and result tables (created when you run the script; not shipped)

## Why this analysis
- Complements logistic regression with a non-linear model
- SHAP provides robust global feature-importance insight

## Run
```bash
python 1_RQ1_feature_importance/baseline_rf_shap/rf_feature_importance.py
```
