"""Model evaluation module with comprehensive metrics."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


class ModelEvaluator:
    """Comprehensive model evaluation with focus on high-risk recall."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_labels = self.config['risk_labels']
        self.class_names = [self.risk_labels[i] for i in sorted(self.risk_labels.keys())]
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        for i, label in enumerate(self.class_names):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            
            results[f'precision_{label.lower()}'] = precision_score(binary_true, binary_pred, zero_division=0)
            results[f'recall_{label.lower()}'] = recall_score(binary_true, binary_pred, zero_division=0)
            results[f'f1_{label.lower()}'] = f1_score(binary_true, binary_pred, zero_division=0)
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                results['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                results['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            except ValueError:
                pass
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )
        
        return results
    
    def evaluate_high_risk_focus(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate with focus on high-risk class (class 2)."""
        high_risk_class = 2
        
        # Binary metrics for high-risk
        binary_true = (y_true == high_risk_class).astype(int)
        binary_pred = (y_pred == high_risk_class).astype(int)
        
        results = {
            'high_risk_precision': precision_score(binary_true, binary_pred, zero_division=0),
            'high_risk_recall': recall_score(binary_true, binary_pred, zero_division=0),
            'high_risk_f1': f1_score(binary_true, binary_pred, zero_division=0),
            'high_risk_support': int(binary_true.sum()),
            'high_risk_predicted': int(binary_pred.sum()),
        }
        
        # False negative analysis (missed high-risk cases)
        false_negatives = ((y_true == high_risk_class) & (y_pred != high_risk_class)).sum()
        results['high_risk_false_negatives'] = int(false_negatives)
        results['high_risk_miss_rate'] = false_negatives / max(binary_true.sum(), 1)
        
        # Cost-sensitive metrics
        if y_proba is not None:
            high_risk_proba = y_proba[:, high_risk_class]
            results['high_risk_avg_precision'] = average_precision_score(binary_true, high_risk_proba)
        
        return results
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                               target_recall: float = 0.9) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold for high-risk class to achieve target recall."""
        high_risk_class = 2
        binary_true = (y_true == high_risk_class).astype(int)
        high_risk_proba = y_proba[:, high_risk_class]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(binary_true, high_risk_proba)
        
        # Find threshold that achieves target recall
        valid_idx = recalls[:-1] >= target_recall
        if valid_idx.any():
            # Get highest precision at target recall
            best_idx = np.where(valid_idx)[0][np.argmax(precisions[:-1][valid_idx])]
            optimal_threshold = thresholds[best_idx]
        else:
            # Use lowest threshold if target recall not achievable
            optimal_threshold = thresholds[0]
        
        # Calculate metrics at optimal threshold
        binary_pred = (high_risk_proba >= optimal_threshold).astype(int)
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision_score(binary_true, binary_pred, zero_division=0),
            'recall': recall_score(binary_true, binary_pred, zero_division=0),
            'f1': f1_score(binary_true, binary_pred, zero_division=0)
        }
        
        return optimal_threshold, metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curves for each class."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (ax, label) in enumerate(zip(axes, self.class_names)):
            binary_true = (y_true == i).astype(int)
            class_proba = y_proba[:, i]
            
            precision, recall, _ = precision_recall_curve(binary_true, class_proba)
            ap = average_precision_score(binary_true, class_proba)
            
            ax.plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{label} Risk - Precision-Recall Curve')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for each class."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, label in enumerate(self.class_names):
            binary_true = (y_true == i).astype(int)
            class_proba = y_proba[:, i]
            
            fpr, tpr, _ = roc_curve(binary_true, class_proba)
            auc = roc_auc_score(binary_true, class_proba)
            
            ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves by Risk Level')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> str:
        """Generate comprehensive text report."""
        results = self.evaluate(y_true, y_pred, y_proba)
        high_risk_results = self.evaluate_high_risk_focus(y_true, y_pred, y_proba)
        
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\n## Overall Metrics")
        report.append(f"Accuracy: {results['accuracy']:.4f}")
        report.append(f"Macro Precision: {results['precision_macro']:.4f}")
        report.append(f"Macro Recall: {results['recall_macro']:.4f}")
        report.append(f"Macro F1: {results['f1_macro']:.4f}")
        
        if 'roc_auc_ovr' in results:
            report.append(f"ROC-AUC (OvR): {results['roc_auc_ovr']:.4f}")
        
        report.append("\n## High-Risk Class Focus")
        report.append(f"High-Risk Precision: {high_risk_results['high_risk_precision']:.4f}")
        report.append(f"High-Risk Recall: {high_risk_results['high_risk_recall']:.4f}")
        report.append(f"High-Risk F1: {high_risk_results['high_risk_f1']:.4f}")
        report.append(f"High-Risk False Negatives: {high_risk_results['high_risk_false_negatives']}")
        report.append(f"High-Risk Miss Rate: {high_risk_results['high_risk_miss_rate']:.4f}")
        
        report.append("\n## Classification Report")
        report.append(results['classification_report'])
        
        report.append("\n## Confusion Matrix")
        cm = results['confusion_matrix']
        report.append(f"{'':>10} {'Low':>8} {'Medium':>8} {'High':>8}")
        for i, label in enumerate(self.class_names):
            report.append(f"{label:>10} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")
        
        return '\n'.join(report)
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models side by side."""
        comparison = []
        
        for model_name, results in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision (Macro)': results.get('precision_macro', 0),
                'Recall (Macro)': results.get('recall_macro', 0),
                'F1 (Macro)': results.get('f1_macro', 0),
                'High-Risk Recall': results.get('recall_high', 0),
                'ROC-AUC': results.get('roc_auc_ovr', 0)
            }
            comparison.append(row)
        
        return pd.DataFrame(comparison).sort_values('High-Risk Recall', ascending=False)
