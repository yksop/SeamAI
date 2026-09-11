"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os


class TrajectoryModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model for trajectory prediction."""
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape

        model = Sequential([
            LSTM(128, input_shape=(T_x, F)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(T_y * 2)
        ])

        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train.reshape(N, -1),
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2, callbacks=[es], verbose=0
        )

        self.models['lstm'] = model
        return model, history
    
    def train_cnn(self, X_train, y_train, epochs=50, batch_size=32):
        """Train CNN model for trajectory prediction."""
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape

        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(T_x, F)),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(T_y * 2)
        ])

        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train.reshape(N, -1),
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2, callbacks=[es], verbose=0
        )

        self.models['cnn'] = model
        return model, history
    
    def train_knn(self, X_train, y_train, n_neighbors=5):
        """Train KNN model for trajectory prediction."""
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape

        X_flat = X_train.reshape(N, -1)
        y_flat = y_train.reshape(N, -1)

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X_flat, y_flat)

        self.models['knn'] = model
        return model
    
    def train_linear(self, X_train, y_train):
        """Train Linear Regression model for trajectory prediction."""
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape

        X_flat = X_train.reshape(N, -1)
        y_flat = y_train.reshape(N, -1)

        model = LinearRegression()
        model.fit(X_flat, y_flat)

        self.models['linear'] = model
        return model
    
    def predict(self, model_name, X_test, future_length):
        """Make predictions with specified model."""
        model = self.models[model_name]

        if model_name in ['lstm', 'cnn']:
            preds_flat = model.predict(X_test, verbose=0)
            return preds_flat.reshape(len(X_test), future_length, 2)
        else:
            X_flat = X_test.reshape(len(X_test), -1)
            preds_flat = model.predict(X_flat)
            return preds_flat.reshape(len(X_test), future_length, 2)
        
def compute_metrics(y_true, y_pred):
    """Compute ADE and FDE metrics."""
    ade_list, fde_list = [], []

    for true_seq, pred_seq in zip(y_true, y_pred):
        # Compute distances for each time step
        distances = np.linalg.norm(true_seq - pred_seq, axis=1)

        # ADE: Average of all distances
        ade_list.append(np.mean(distances))
        
        # FDE: Final distance
        fde_list.append(distances[-1])

    return np.mean(ade_list), np.mean(fde_list)

def compute_accuracy(y_true, y_pred, labels):
    def is_in_door_area(x, y, frame_width=640, frame_height=480):
        # Door center
        door_center_x, door_center_y = frame_width // 2, frame_height

        # Ellipse parameters (half ellipse: width: 250, height=120)
        ellipse_a, ellipse_b = 250, 120

        if y > door_center_y:
            return False
        return ((x - door_center_x)**2) / (ellipse_a**2) + ((y - door_center_y)**2) / (ellipse_b**2) <= 1
    
    def predict_label(final_pos):
        x, y, = final_pos
        return "enter" if is_in_door_area(x, y) else "pass"
    
    correct = 0
    for i in range(len(y_true)):
        final_pred = y_pred[i][-1]
        predicted_label = predict_label(final_pred)
        true_label = labels[i]
        if predicted_label == true_label:
            correct += 1
    
    return correct / len(y_true) if len(y_true) > 0 else 0.0

def find_best_models(results):
    """Find best performing model for each feature config and overall."""
    best_models = {}
    overall_best = {"model": None, "config": None, "ade": float('inf')}
    
    for config, models in results.items():
        valid_models = {k: v for k, v in models.items() if "error" not in v}
        if not valid_models:
            continue
            
        best_model = min(valid_models.items(), key=lambda x: x[1]["ade"])
        best_models[config] = {
            "model": best_model[0],
            "metrics": best_model[1]
        }
        
        # Check for overall best
        if best_model[1]["ade"] < overall_best["ade"]:
            overall_best = {
                "model": best_model[0],
                "config": config,
                "ade": best_model[1]["ade"],
                "fde": best_model[1]["fde"],
                "accuracy": best_model[1]["accuracy"]
            }
    
    return best_models, overall_best

def plot_trajectories(y_true, y_pred, labels, model_name, feature_config, save_dir, n_samples=5):
    """Plot predicted vs true trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    n_samples = min(n_samples, len(y_true))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Plot true trajectory
        ax.plot(y_true[i][:, 0], y_true[i][:, 1], 'o-', 
                label='True', color='blue', linewidth=2, markersize=4)
        
        # Plot predicted trajectory
        ax.plot(y_pred[i][:, 0], y_pred[i][:, 1], 'x--', 
                label='Predicted', color='red', linewidth=2, markersize=6)
        
        # Draw door area (half ellipse)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((320, 480), 500, 240, fill=False, color='green', linestyle='--')
        ax.add_patch(ellipse)
        ax.set_ylim(480, 300)  # Show only upper part
        
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Invert y-axis for image coordinates
        ax.set_title(f'Sample {i+1} ({labels[i]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name.upper()} - {feature_config}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature_config}_{model_name}_trajectories.png'), dpi=150)
    plt.close()