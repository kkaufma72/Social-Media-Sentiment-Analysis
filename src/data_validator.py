"""
Data validation utilities for sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class DataValidator:
    """Validate input data for sentiment analysis"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          text_column: str = 'text',
                          target_column: str = 'sentiment') -> Tuple[bool, List[str]]:
        """
        Validate input dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            target_column: Name of target column
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if dataframe is empty
        if df is None or len(df) == 0:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        if text_column not in df.columns:
            errors.append(f"Missing required column: '{text_column}'")
        
        if target_column not in df.columns:
            errors.append(f"Missing required column: '{target_column}'")
        
        if errors:
            return False, errors
        
        # Check for null values
        null_texts = df[text_column].isnull().sum()
        if null_texts > 0:
            errors.append(f"Found {null_texts} null values in text column")
        
        null_labels = df[target_column].isnull().sum()
        if null_labels > 0:
            errors.append(f"Found {null_labels} null values in target column")
        
        # Check minimum samples
        if len(df) < 10:
            errors.append(f"Insufficient samples: {len(df)}. Need at least 10.")
        
        # Check label distribution
        label_counts = df[target_column].value_counts()
        if len(label_counts) < 2:
            errors.append("Need at least 2 different classes")
        
        min_class_samples = label_counts.min()
        if min_class_samples < 2:
            errors.append(f"Class imbalance: smallest class has only {min_class_samples} samples")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def check_text_quality(texts: pd.Series) -> dict:
        """
        Check quality metrics of text data
        
        Args:
            texts: Series of text data
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'total_samples': len(texts),
            'empty_texts': (texts.str.len() == 0).sum(),
            'avg_length': texts.str.len().mean(),
            'min_length': texts.str.len().min(),
            'max_length': texts.str.len().max(),
            'very_short': (texts.str.len() < 10).sum()  # Less than 10 chars
        }
        
        return metrics
    
    @staticmethod
    def print_validation_report(df: pd.DataFrame,
                               text_column: str = 'text',
                               target_column: str = 'sentiment'):
        """Print comprehensive validation report"""
        
        print("="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        is_valid, errors = DataValidator.validate_dataframe(
            df, text_column, target_column
        )
        
        if is_valid:
            print("✅ Data validation PASSED")
        else:
            print("❌ Data validation FAILED")
            print("\nErrors found:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        print("\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        if target_column in df.columns:
            print(f"\nClass Distribution:")
            class_dist = df[target_column].value_counts()
            for label, count in class_dist.items():
                pct = (count / len(df)) * 100
                print(f"  Class {label}: {count} ({pct:.1f}%)")
        
        if text_column in df.columns:
            print(f"\nText Quality Metrics:")
            metrics = DataValidator.check_text_quality(df[text_column])
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        print("="*60)


if __name__ == "__main__":
    # Test validation
    sample_df = pd.DataFrame({
        'text': ['Good product', 'Bad service', 'Okay'],
        'sentiment': [1, 0, 2]
    })
    
    DataValidator.print_validation_report(sample_df)
