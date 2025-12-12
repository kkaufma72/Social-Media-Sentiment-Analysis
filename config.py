"""
Configuration settings for sentiment analysis project
"""

import os
from pathlib import Path


class Config:
    """Configuration class for project settings"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Feature extraction
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 2
    MAX_DF = 0.8
    
    # Text preprocessing
    REMOVE_STOPWORDS = True
    LEMMATIZE = True
    MIN_TOKEN_LENGTH = 2
    
    # Logistic Regression
    LR_MAX_ITER = 1000
    LR_C = 1.0
    LR_SOLVER = 'lbfgs'
    
    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_METRICS_LOGGING = True
    
    # Visualization
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, 
                        cls.PROCESSED_DATA_DIR, cls.MODELS_DIR, 
                        cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("CONFIGURATION SETTINGS")
        print("="*60)
        
        print("\nPaths:")
        print(f"  Data directory: {cls.DATA_DIR}")
        print(f"  Models directory: {cls.MODELS_DIR}")
        print(f"  Results directory: {cls.RESULTS_DIR}")
        
        print("\nModel Parameters:")
        print(f"  Random state: {cls.RANDOM_STATE}")
        print(f"  Test size: {cls.TEST_SIZE}")
        print(f"  Max features: {cls.MAX_FEATURES}")
        
        print("\nPreprocessing:")
        print(f"  Remove stopwords: {cls.REMOVE_STOPWORDS}")
        print(f"  Lemmatize: {cls.LEMMATIZE}")
        
        print("="*60)


if __name__ == "__main__":
    Config.create_dirs()
    Config.print_config()
