import sqlite3
import os
import datetime
from typing import List, Tuple
from .types import ExtractedFeatures

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "experiments.db")

class ExperimentDatabase:
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                timestamp DATETIME,
                ph_final REAL,
                ph_slope REAL,
                temp_mean REAL,
                temp_std REAL,
                color_peak REAL,
                weight_loss REAL,
                ground_truth REAL,
                pred_capacity REAL
            )
        ''')
        conn.commit()
        conn.close()

    def get_total_experiments(self) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

    def save_experiment(self, batch_id: str, features: ExtractedFeatures, ground_truth: float, pred_capacity: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (
                batch_id, timestamp, ph_final, ph_slope, temp_mean, temp_std, 
                color_peak, weight_loss, ground_truth, pred_capacity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id,
            datetime.datetime.now(),
            features.ph_final,
            features.ph_slope,
            features.temp_mean,
            features.temp_std,
            features.color_peak,
            features.weight_loss,
            ground_truth,
            pred_capacity
        ))
        
        conn.commit()
        conn.close()
        print(f"[DB] Saved experiment {batch_id} to history.")

    def get_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """
        Returns (X, y) for model training
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ph_final, ph_slope, temp_mean, temp_std, color_peak, weight_loss, ground_truth
            FROM experiments
            WHERE ground_truth > 0
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return [], []
            
        X = []
        y = []
        for row in rows:
            # First 6 columns are features
            X.append(list(row[0:6]))
            # Last column is target
            y.append(row[6])
            
        return X, y
