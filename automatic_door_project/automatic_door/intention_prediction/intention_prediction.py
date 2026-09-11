"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import os
import json
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split

from config import CONFIG, get_scenario_flags
from data_processing import load_master_json, build_intention_dataset, prepare_classification_data
from models import IntentModels, compute_classification_metrics, find_best_classification_models

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

def run_single_scenario(master_json, past_seconds, results_dir, scenario_name, distance_from_end=None):
    """Run training for a single scenario."""
    if distance_from_end is not None:
        print(f"\n=== {scenario_name}: window={past_seconds}s, distance={distance_from_end}s from end ===")
    else:
        print(f"\n=== {scenario_name} ===")

    os.makedirs(results_dir, exist_ok=True)
    results = {}

    for feature_config in CONFIG["feature_configs"]:
        print(f"\n--- Processing {feature_config} ---")

        scenario_flags = get_scenario_flags(feature_config)
        X_seq, labels = build_intention_dataset(
            master_json, scenario_flags, past_seconds, distance_from_end
        )

        if len(X_seq) == 0:
            print("No data found for this scenario.")
            continue

        X_train, X_test, y_train, y_test, feat_scaler = prepare_classification_data(
            X_seq, labels, CONFIG["test_size"]
        )

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        feature_results = {}
        models = IntentModels()
        num_classes = 2

        for model_name in CONFIG["models"]:
            print(f"    Training {model_name}")

            try:
                if model_name == "lstm":
                    model, history = models.train_lstm(
                        X_train, y_train, num_classes
                    )
                elif model_name == "cnn":
                    model, history = models.train_cnn(
                        X_train, y_train, num_classes
                    )
                elif model_name == "knn":
                    model = models.train_knn(X_train, y_train)
                elif model_name == "logistic":
                    model = models.train_logistic(X_train, y_train)

                y_pred = models.predict(model_name, X_test)
                metrics = compute_classification_metrics(y_test, y_pred)

                print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

                feature_results[model_name] = {
                    **metrics,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "past_seconds": past_seconds,
                    "feature_config": feature_config,
                    "model_name": model_name
                }

                # Save model and scaler
                model_dir = os.path.join(results_dir, "models", feature_config)
                os.makedirs(model_dir, exist_ok=True)

                if model_name in ["lstm", "cnn"]:
                    model.save(os.path.join(model_dir, f"{model_name}.h5"))
                else:
                    dump(model, os.path.join(model_dir, f"{model_name}.pkl"))

                dump(feat_scaler, os.path.join(model_dir, "scaler.pkl"))
            
            except Exception as e:
                print(f"    Error during training: {e}")
                feature_results[model_name] = {"error": str(e)}

        results[feature_config] = feature_results

        # Find best model for this feature config
        feature_best_models, _ = find_best_classification_models({feature_config: feature_results})
        if feature_config in feature_best_models:
            best_model_for_config = feature_best_models[feature_config]["model"]
            best_metrics_for_config = feature_best_models[feature_config]["metrics"]

            results[feature_config]["best_model"] = {
                "model": best_model_for_config,
                "metrics": best_metrics_for_config
            }

            print(f"Best model for {feature_config}: {best_model_for_config} with metrics: {best_metrics_for_config}")
            print(f"Accuracy: {best_metrics_for_config['accuracy']:.4f}, F1: {best_metrics_for_config['f1']:.4f}")
    
    # Find overall best
    filtered_results = {config: {k: v for k,v in config_results.items() if k != "best_model"}
                        for config, config_results in results.items()}
    best_by_config, overall_best = find_best_classification_models(filtered_results)

    results["best_models"] = {
        "by_config": best_by_config,
        "overall_best": overall_best
    }

    if overall_best["model"]:
        print(f"\nBest overall for {scenario_name}: {overall_best['model']} ({overall_best['config']}) - Accuracy: {overall_best['accuracy']:.3f}")

    # Save results
    with open(os.path.join(results_dir, f"summary_results_{scenario_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def train_and_test_models():
    """Run all test scenarios."""
    print("\n=== Running all intention prediction scenarios ===")

    data_path = os.path.join(CONFIG["data_dir"], CONFIG["data_source"])
    master_json = load_master_json(data_path)

    if CONFIG.get("quick_test", False):
        master_json = dict(list(master_json.items())[:10])

    print(f"Loaded {len(master_json)} videos.")

    base_results_dir = CONFIG["results_dir"]
    all_results = {}

    for scenario_name, scenario_config in CONFIG["test_scenarios"].items():
        # Normalize to lists
        past_values = scenario_config["past"] if isinstance(scenario_config["past"], list) else [scenario_config["past"]]
        future_values = scenario_config["future"] if isinstance(scenario_config["future"], list) else [scenario_config["future"]]
        
        scenario_results = {}
        
        # Iterate over all combinations
        for past_val in past_values:
            for future_val in future_values:
                results_dir = os.path.join(base_results_dir, scenario_name, f"past_{past_val}_future_{future_val}")
                results = run_single_scenario(
                    master_json, past_val, results_dir,
                    f"{scenario_name}_past_{past_val}_future_{future_val}", future_val
                )
                scenario_results[f"past_{past_val}_future_{future_val}"] = results
        
        all_results[scenario_name] = scenario_results
    
    # Save all results
    with open(os.path.join(base_results_dir, "all_scenario_results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== All scenarios completed ===")
    print(f"Results saved in {base_results_dir}")

if __name__ == "__main__":
    train_and_test_models()