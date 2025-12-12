"""
Demo Script for Sentiment Analysis
"""

import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append('src')

from data_preprocessing import TextPreprocessor
from feature_engineering import prepare_train_test_split
from train_models import SentimentModelTrainer
from evaluate import ModelEvaluator


def create_demo_dataset(size='medium'):
    """Create a demo dataset"""
    
    sizes = {
        'small': 20,
        'medium': 40,
        'large': 80
    }
    
    multiplier = sizes.get(size, 40)
    
    positive_texts = [
        "I absolutely love this product! It's amazing!",
        "Outstanding quality and fantastic service!",
        "Best purchase I've ever made!",
        "Incredible! Will definitely buy again!",
        "Perfect! Exceeded all expectations!",
    ] * multiplier
    
    negative_texts = [
        "Worst product ever. Complete waste of money.",
        "Terrible quality and horrible service.",
        "Very disappointed. Would not recommend.",
        "Poor quality. Broke after one use.",
        "Awful experience. Save your money!",
    ] * multiplier
    
    neutral_texts = [
        "It's okay, nothing special.",
        "Average product, met basic expectations.",
        "Decent quality, works as described.",
        "Fair product for the price.",
        "Acceptable, nothing to complain about.",
    ] * multiplier
    
    texts = positive_texts + negative_texts + neutral_texts
    labels = ([1] * len(positive_texts) + 
             [0] * len(negative_texts) + 
             [2] * len(neutral_texts))
    
    df = pd.DataFrame({'text': texts, 'sentiment': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main(args):
    """Run the complete demo"""
    
    print("="*80)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*80)
    
    # Create dataset
    print(f"\n[1/5] Creating {args.size} demo dataset...")
    df = create_demo_dataset(args.size)
    print(f"Dataset created with {len(df)} samples")
    
    # Preprocess
    print("\n[2/5] Preprocessing text...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df = preprocessor.preprocess_dataframe(df)
    
    # Feature extraction
    print("\n[3/5] Extracting features...")
    X_train, X_test, y_train, y_test, feature_extractor = prepare_train_test_split(df)
    
    # Train models
    print("\n[4/5] Training models...")
    results = {}
    
    if args.model in ['all', 'lr']:
        lr_trainer = SentimentModelTrainer('logistic_regression')
        lr_trainer.train(X_train, y_train)
        results['Logistic Regression'] = lr_trainer.evaluate(X_test, y_test)
        best_trainer = lr_trainer
    
    if args.model in ['all', 'rf']:
        rf_trainer = SentimentModelTrainer('random_forest')
        rf_trainer.train(X_train, y_train)
        results['Random Forest'] = rf_trainer.evaluate(X_test, y_test)
        if 'best_trainer' not in locals():
            best_trainer = rf_trainer
    
    # Compare
    print("\n[5/5] Model Comparison:")
    print("="*80)
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.4f}")
    
    # Test predictions
    if args.test:
        print("\n" + "="*80)
        print("TESTING ON NEW EXAMPLES")
        print("="*80)
        
        test_texts = [
            "This is absolutely fantastic! I love it!",
            "Terrible experience, would not recommend.",
            "It's fine, nothing extraordinary.",
        ]
        
        sentiment_map = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'NEUTRAL'}
        
        for text in test_texts:
            processed = preprocessor.preprocess(text)
            features = feature_extractor.transform([processed])
            prediction = best_trainer.predict(features)[0]
            
            print(f"\nText: {text}")
            print(f"Predicted: {sentiment_map[prediction]}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis Demo')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                       default='medium', help='Dataset size')
    parser.add_argument('--model', choices=['lr', 'rf', 'all'], 
                       default='all', help='Model to train')
    parser.add_argument('--test', action='store_true', 
                       help='Run predictions on test examples')
    
    args = parser.parse_args()
    main(args)
