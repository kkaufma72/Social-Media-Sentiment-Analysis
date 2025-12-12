"""
Performance metrics logging utilities
"""

import json
import csv
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Log and track model performance metrics"""
    
    def __init__(self, log_dir='results'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / 'metrics_log.csv'
        
        # Create log file with headers if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'model_type', 'accuracy', 
                    'precision', 'recall', 'f1_score', 'notes'
                ])
    
    def log_metrics(self, model_type, metrics, notes=''):
        """
        Log performance metrics
        
        Args:
            model_type: Type of model
            metrics: Dict with performance metrics
            notes: Optional notes
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                model_type,
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                notes
            ])
        
        print(f"✅ Metrics logged to {self.log_file}")
    
    def get_best_model(self, metric='accuracy'):
        """
        Get best performing model
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Dict with best model info
        """
        if not self.log_file.exists():
            return None
        
        import pandas as pd
        df = pd.read_csv(self.log_file)
        
        if len(df) == 0:
            return None
        
        best_idx = df[metric].idxmax()
        best = df.loc[best_idx].to_dict()
        
        return best
    
    def print_history(self, n=10):
        """Print last n experiments"""
        if not self.log_file.exists():
            print("No metrics logged yet")
            return
        
        import pandas as pd
        df = pd.read_csv(self.log_file)
        
        print("="*80)
        print(f"LAST {min(n, len(df))} EXPERIMENTS")
        print("="*80)
        
        for _, row in df.tail(n).iterrows():
            print(f"\n{row['timestamp']} - {row['model_type']}")
            print(f"  Accuracy: {row['accuracy']:.4f}")
            if row['notes']:
                print(f"  Notes: {row['notes']}")
