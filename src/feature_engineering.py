"""
Feature Engineering Module for Sentiment Analysis
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    """Feature extraction class for text data"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True
        )
        
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        print(f"Extracting features using TF-IDF...")
        
        if len(texts) == 0:
            raise ValueError("Cannot extract features from empty text list")
        
        features = self.vectorizer.fit_transform(texts)
        print(f"Feature matrix shape: {features.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return features.toarray()
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        if len(texts) == 0:
            raise ValueError("Cannot transform empty text list")
            
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """Get top n features"""
        feature_names = self.get_feature_names()
        return list(feature_names[:min(n, len(feature_names))])


def prepare_train_test_split(df, text_column='processed_text', 
                             target_column='sentiment', test_size=0.2,
                             random_state=42):
    """Prepare train-test split with feature extraction"""
    print("Preparing train-test split...")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")
    
    X = df[text_column]
    y = df[target_column]
    
    # Check if we have enough samples
    if len(X) < 10:
        raise ValueError(f"Not enough samples: {len(X)}. Need at least 10 samples.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    feature_extractor = FeatureExtractor(max_features=5000)
    X_train_features = feature_extractor.fit_transform(X_train)
    X_test_features = feature_extractor.transform(X_test)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train_features, X_test_features, y_train, y_test, feature_extractor
