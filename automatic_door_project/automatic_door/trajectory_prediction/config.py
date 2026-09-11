"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import os

CONFIG = {
    "data_source": "master_json.json", # master json file with all data instances
    "data_dir": "path/to/json_data", # directory where the data is stored, e.g. the master_json.json is stored
    "results_dir": "path/to/results", # where to save results

    "models": ["lstm", "cnn", "knn", "linear"],

    "feature_configs": [
        "bbox_only",
        "skeleton_only",
        "angles_only",
        "bbox_skeleton",
        "bbox_angles",
        "skeleton_angles",
        "bbox_skeleton_angles"
    ],

    "test_scenarios": {
        "baseline": {"past": 2.0, "future": [1.0, 5.0]},
        "TTE": {"past": 1.0, "future": [1.0, 2.0, 3.0, 4.0, 5.0]}
    },

    "test_size": 0.2,
    "random_state": 42,
    "plot_samples": 5,
    "quick_test": False
}

def get_scenario_flags(scenario):
    """Convert scenario name to feature flags."""
    return {
        "use_bbox": "bbox" in scenario,
        "use_skeleton": "skeleton" in scenario,
        "use_angles": "angles" in scenario
    }