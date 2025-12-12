"""
Model Evaluation Module for Sentiment Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, y_true, y_pred, class_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names or [str(i) for i in range(len(np.unique(y_true)))]
        
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
    def print_detailed_metrics(self):
        """Print detailed evaluation metrics"""
        print("="*60)
        print("DETAILED EVALUATION METRICS")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred,
                                   target_names=self.class_names))


def compare_models(results_dict, metric='accuracy', figsize=(10, 6), save_path=None):
    """Compare multiple models"""
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
