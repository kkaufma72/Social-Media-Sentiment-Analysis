"""
Model utility functions for saving and loading
"""

import joblib
import json
from pathlib import Path


def save_model_with_metadata(model, filepath, metadata=None):
    """
    Save model with metadata
    
    Args:
        model: Trained model
        filepath: Path to save model
        metadata: Dict with model info (accuracy, date, etc)
    """
    model_path = Path(filepath)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    
    # Save metadata
    if metadata:
        meta_path = model_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to {filepath}")


def load_model_with_metadata(filepath):
    """
    Load model with metadata
    
    Args:
        filepath: Path to model file
        
    Returns:
        tuple: (model, metadata)
    """
    model = joblib.load(filepath)
    
    # Load metadata if exists
    meta_path = Path(filepath).with_suffix('.json')
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata
