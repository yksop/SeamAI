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
from data_processing import load_master_json, build_trajectory_dataset, prepare_data
from models import TrajectoryModels, compute_accuracy, compute_metrics, find_best_models, plot_trajectories

def run_single_scenario(master_json, past_seconds, future_seconds, results_dir, scenario_name):
    """Run training for a single scenario."""
    print(f"\n=== {scenario_name}: past={past_seconds}s, future={future_seconds}s ===")

    os.makedirs(results_dir, exist_ok=True)
    results = {}

    for feature_config in CONFIG["feature_configs"]:
        print(f"\n--- Processing {feature_config} ---")

        scenario_flags = get_scenario_flags(feature_config)
        X_seq, y_seq, labels, frame_info = build_trajectory_dataset(
            master_json, scenario_flags, past_seconds, future_seconds
        )

        if len(X_seq) == 0:
            print("No data found for this scenario.")
            continue

        X_train, X_test, y_train, y_test, feat_scaler, cent_scaler = prepare_data(
            X_seq, y_seq, CONFIG["test_size"]
        )

        _, labels_test = train_test_split(labels, test_size=CONFIG["test_size"], random_state=42)

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        feature_results = {}
        models = TrajectoryModels()

        for model_name in CONFIG["models"]:
            print(f"    Training {model_name}")

            try:
                if model_name == "lstm":
                    model, history = models.train_lstm(X_train, y_train)
                elif model_name == "cnn":
                    model, history = models.train_cnn(X_train, y_train)
                elif model_name == "knn":
                    model = models.train_knn(X_train, y_train)
                elif model_name == "linear":
                    model = models.train_linear(X_train, y_train)

                y_pred_scaled = models.predict(model_name, X_test, y_test.shape[1])
                y_pred = cent_scaler.inverse_transform(y_pred_scaled.reshape(-1,2)).reshape(y_pred_scaled.shape)
                y_test_unscaled = cent_scaler.inverse_transform(y_test.reshape(-1, 2)).reshape(y_test.shape)

                ade, fde = compute_metrics(y_test_unscaled, y_pred)
                acc = compute_accuracy(y_test_unscaled, y_pred, labels_test)

                print(f"    ADE: {ade:.2f}, FDE: {fde:.2f}, ACC: {acc:.2f}")

                feature_results[model_name] = {
                    "ade": float(ade),
                    "fde": float(fde),
                    "accuracy": float(acc),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "past_horison": past_seconds,
                    "future_horizon": future_seconds,
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
                
                # Save scalers
                dump(feat_scaler, os.path.join(model_dir, "feature_scaler.pkl"))
                dump(cent_scaler, os.path.join(model_dir, "centroid_scaler.pkl"))

                # Save predictions for plotting
                pred_dir = os.path.join(results_dir, "predictions", feature_config)
                os.makedirs(pred_dir, exist_ok=True)
                np.save(os.path.join(pred_dir, f"{model_name}_y_test.npy"), y_test_unscaled)
                np.save(os.path.join(pred_dir, f"{model_name}_y_pred.npy"), y_pred)
                np.save(os.path.join(pred_dir, f"{model_name}_labels.npy"), labels_test)
            
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                feature_results[model_name] = {"error": str(e)}
                continue
        
        results[feature_config] = feature_results
        
        # Find best model for this specific feature config
        feature_best_models, _ = find_best_models({feature_config: feature_results})
        best_model_for_config = feature_best_models[feature_config]["model"]
        best_metrics_for_config = feature_best_models[feature_config]["metrics"]
        
        # Store best model info in results
        results[feature_config]["best_model"] = {
            "model": best_model_for_config,
            "metrics": best_metrics_for_config
        }
        
        print(f"Best model for {feature_config}: {best_model_for_config} - ADE: {best_metrics_for_config['ade']:.2f}")
        
        # Plot best model for this feature config
        if best_model_for_config:
            plot_dir = os.path.join(results_dir, "plots", feature_config)
            pred_dir = os.path.join(results_dir, "predictions", feature_config)
            
            # Load predictions for best model
            y_test = np.load(os.path.join(pred_dir, f"{best_model_for_config}_y_test.npy"))
            y_pred = np.load(os.path.join(pred_dir, f"{best_model_for_config}_y_pred.npy"))
            labels = np.load(os.path.join(pred_dir, f"{best_model_for_config}_labels.npy"))
            
            plot_trajectories(y_test, y_pred, labels, best_model_for_config, feature_config, plot_dir)

    filtered_results = {}
    for config, config_results in results.items():
        filtered_results[config] = {k: v for k,v in config_results.items() if k!= "best_model"}
    # Find overall best across all feature configs for this scenario
    best_by_config, overall_best = find_best_models(filtered_results)
    
    # Add to results
    results["best_models"] = {
        "by_config": best_by_config,
        "overall_best": overall_best
    }

    print(f"\nBest overall for {scenario_name}: {overall_best['model']} ({overall_best['config']}) - ADE: {overall_best['ade']:.2f}")
    
    # Save results
    with open(os.path.join(results_dir, f"summary_results_{scenario_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def train_and_test_models():
    """Run all test scenarios."""
    print("\n=== Running all test scenarios ===")

    data_path = os.path.join(CONFIG["data_dir"], CONFIG["data_source"])
    master_json = load_master_json(data_path)

    if CONFIG.get("quick_test", False):
        print("Quick test mode enabled, using limited data.")
        master_json = dict(list(master_json.items())[:10])
    
    print(f"Loaded {len(master_json)} videos.")

    base_results_dir = CONFIG["results_dir"]
    all_results = {}

    for scenario_name, scenario_config in CONFIG["test_scenarios"].items():
        if isinstance(scenario_config["past"], list):
            # Vary past scenario
            scenario_results = {}
            for past_val in scenario_config["past"]:
                results_dir = os.path.join(base_results_dir, scenario_name, f"past_{past_val}")
                results = run_single_scenario(
                    master_json, past_val, scenario_config["future"],
                    results_dir, f"{scenario_name}_past_{past_val}"
                )
                scenario_results[f"past_{past_val}"] = results
            all_results[scenario_name] = scenario_results
        
        elif isinstance(scenario_config["future"], list):
            # Vary future scenario
            scenario_results = {}
            for future_val in scenario_config["future"]:
                results_dir = os.path.join(base_results_dir, scenario_name, f"future_{future_val}")
                results = run_single_scenario(
                    master_json, scenario_config["past"], future_val,
                    results_dir, f"{scenario_name}_future_{future_val}"
                )
                scenario_results[f"future_{future_val}"] = results
            all_results[scenario_name] = scenario_results

        else:
            # Single scenario
            results_dir = os.path.join(base_results_dir, scenario_name)
            results = run_single_scenario(
                master_json, scenario_config["past"], scenario_config["future"],
                results_dir, scenario_name
            )
            all_results[scenario_name] = results
    
    # Save all results
    with open(os.path.join(base_results_dir, "all_scenario_results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== All scenarios completed ===")
    print(f"Results saved in {base_results_dir}")

if __name__ == "__main__":
    train_and_test_models()
