"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump

class IntentModels:
    def __init__(self):
        self.models = {}

    def train_lstm(self, X_train, y_train, num_classes, epochs=50):
        N, T_x, F = X_train.shape
        
        model = Sequential([
            LSTM(128, input_shape=(T_x, F)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                          validation_split=0.2, callbacks=[es], verbose=0)
        
        self.models['lstm'] = model
        return model, history

    def train_cnn(self, X_train, y_train, num_classes, epochs=50):
        N, T_x, F = X_train.shape
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(T_x, F)),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                          validation_split=0.2, callbacks=[es], verbose=0)
        
        self.models['cnn'] = model
        return model, history

    def train_knn(self, X_train, y_train, n_neighbors=5):
        X_flat = X_train.reshape(len(X_train), -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_flat, y_train)
        self.models['knn'] = model
        return model

    def train_logistic(self, X_train, y_train):
        X_flat = X_train.reshape(len(X_train), -1)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_flat, y_train)
        self.models['logistic'] = model
        return model

    def predict(self, model_name, X_test):
        model = self.models[model_name]
        if model_name in ['lstm', 'cnn']:
            return model.predict(X_test, verbose=0).argmax(axis=1)
        else:
            X_flat = X_test.reshape(len(X_test), -1)
            return model.predict(X_flat)

def compute_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def find_best_classification_models(results):
    best_models = {}
    overall_best = {"model": None, "config": None, "accuracy": 0.0}
    
    for config, models in results.items():
        valid_models = {k: v for k, v in models.items() if "error" not in v}
        if not valid_models:
            continue
            
        best_model = max(valid_models.items(), key=lambda x: x[1]["accuracy"])
        best_models[config] = {
            "model": best_model[0],
            "metrics": best_model[1]
        }
        
        if best_model[1]["accuracy"] > overall_best["accuracy"]:
            overall_best = {
                "model": best_model[0],
                "config": config,
                **best_model[1]
            }
    
    return best_models, overall_best
