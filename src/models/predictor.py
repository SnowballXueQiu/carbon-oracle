import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple
from ..core.types import ExtractedFeatures, PredictionResult
from ..mock.generator import MockBatchGenerator
from ..features.extract import FeatureExtractor
from ..core.config_loader import config

from ..core.database import ExperimentDatabase

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")

class Predictor:
    def __init__(self):
        self.model = None
        self.extractor = FeatureExtractor()
        self.db = ExperimentDatabase()
        self._load_or_train_model()

    def fit_on_history(self, db: ExperimentDatabase):
        """Public method to force retraining on specific DB history"""
        self.db = db
        X_hist, y_hist = self.db.get_training_data()
        if len(X_hist) > 0:
            print(f"[Model] Force retraining on {len(X_hist)} records from history...")
            self._train_on_data(X_hist, y_hist)
            # Save the updated model
            joblib.dump(self.model, MODEL_PATH)

    def _load_or_train_model(self):
        # 1. Check if we have enough history to retrain/fine-tune
        X_hist, y_hist = self.db.get_training_data()
        
        if len(X_hist) >= 5:
            print(f"[Model] Found {len(X_hist)} real experiments in DB. Retraining on Real Data...")
            self._train_on_data(X_hist, y_hist)
        elif os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("Loaded existing model.")
            except:
                print("Model load failed. Retraining bootstrap...")
                self._train_bootstrap_model()
        else:
            print("No model found. Training bootstrap model...")
            self._train_bootstrap_model()

    def _train_on_data(self, X, y):
        # If small data, mix with some synthetic to avoid overfitting?
        # For PoC, just train on what we have + maybe some synthetic if very small.
        # Let's simple: If < 20 samples, add 20 synthetic samples.
        if len(X) < 20:
             print("[Model] Data scarce (<20), augmenting with synthetic data...")
             X_syn, y_syn = self._generate_synthetic_data(count=20)
             X.extend(X_syn)
             y.extend(y_syn)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Model retrained on {len(X)} samples and saved.")

    def _generate_synthetic_data(self, count=50) -> Tuple[List, List]:
        X_train = []
        y_train = []
        for i in range(count):
            gen = MockBatchGenerator(f"TRAIN_{i}")
            records = []
            while True:
                rec = gen.step()
                if rec is None: break
                records.append(rec)
            
            feats = self.extractor.extract(records)
            ground_truth = gen.calculate_ground_truth_capacity()
            
            feat_vector = [
                feats.ph_final, feats.ph_slope, feats.temp_mean, 
                feats.temp_std, feats.color_peak, feats.weight_loss
            ]
            X_train.append(feat_vector)
            y_train.append(ground_truth)
        return X_train, y_train

    def _train_bootstrap_model(self):
        """
        Generate mock data and train a model on fly.
        """
        print("Generating synthetic training data (50 batches)...")
        X_train, y_train = self._generate_synthetic_data(50)
        self._train_on_data(X_train, y_train)


    def predict(self, features: ExtractedFeatures) -> PredictionResult:
        if not self.model:
            return PredictionResult(capacity=0.0, confidence=0.0)

        # Prepare input
        X = [[
            features.ph_final, features.ph_slope, features.temp_mean,
            features.temp_std, features.color_peak, features.weight_loss
        ]]
        
        # Predict
        pred_capacity = self.model.predict(X)[0]
        
        # Estimate confidence (heuristic based on tree variance if available, or just dummy)
        # Using standard deviation of trees in forest
        preds = [tree.predict(X)[0] for tree in self.model.estimators_]
        std_dev = np.std(preds)
        
        # Invert std_dev to get confidence (0-1). 
        # Assuming std_dev > 1.0 is bad.
        confidence = max(0.0, 1.0 - std_dev) 
        
        return PredictionResult(
            capacity=float(pred_capacity),
            confidence=float(confidence)
        )
