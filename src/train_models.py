"""
Model Training Module for Sentiment Analysis
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class SentimentModelTrainer:
    """Model trainer for sentiment analysis"""
    
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        
    def build_logistic_regression(self):
        """Build Logistic Regression model"""
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1
        )
    
    def build_random_forest(self):
        """Build Random Forest model"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'logistic_regression':
            self.model = self.build_logistic_regression()
        elif self.model_type == 'random_forest':
            self.model = self.build_random_forest()
        
        self.model.fit(X_train, y_train)
        print("Training complete!")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{self.model_type.upper()} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def save_model(self, filepath):
        """Save model to file"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
