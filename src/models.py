"""Model training module with multiple classifiers."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import csr_matrix
import yaml
import joblib


class ModelTrainer:
    """Handles training and optimization of multiple classifiers."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
    
    def _get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def _initialize_models(self, y: np.ndarray) -> Dict[str, Any]:
        """Initialize all model configurations."""
        class_weights = self._get_class_weights(y)
        
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    class_weight=class_weights,
                    max_iter=1000,
                    random_state=42,
                    solver='saga',
                    n_jobs=-1
                ),
                'params': {
                    'C': self.config['models']['logistic_regression']['C']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    class_weight=class_weights,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': self.config['models']['random_forest']['n_estimators'],
                    'max_depth': self.config['models']['random_forest']['max_depth']
                }
            },
            'svm': {
                'model': LinearSVC(
                    class_weight=class_weights,
                    random_state=42,
                    max_iter=2000
                ),
                'params': {
                    'C': self.config['models']['svm']['C']
                }
            }
        }
        
        # Add XGBoost if available
        try:
            from xgboost import XGBClassifier
            
            # Calculate scale_pos_weight for multi-class
            n_samples = len(y)
            n_classes = len(np.unique(y))
            
            models['xgboost'] = {
                'model': XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ),
                'params': {
                    'n_estimators': self.config['models']['xgboost']['n_estimators'],
                    'max_depth': self.config['models']['xgboost']['max_depth'],
                    'learning_rate': self.config['models']['xgboost']['learning_rate']
                }
            }
        except ImportError:
            pass
        
        return models
    
    def train_all_models(self, X: csr_matrix, y: np.ndarray) -> Dict[str, Any]:
        """Train all models with hyperparameter tuning."""
        model_configs = self._initialize_models(y)
        cv = StratifiedKFold(n_splits=self.config['evaluation']['cv_folds'], shuffle=True, random_state=42)
        
        results = {}
        
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='recall_macro',  # Prioritize recall for high-risk
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            # Store results
            self.models[name] = grid_search.best_estimator_
            results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV recall: {grid_search.best_score_:.4f}")
        
        self.cv_results = results
        
        # Select best model based on recall
        best_name = max(results, key=lambda x: results[x]['best_score'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name} (recall: {results[best_name]['best_score']:.4f})")
        
        return results
    
    def train_single_model(self, X: csr_matrix, y: np.ndarray, 
                          model_name: str = 'logistic_regression') -> Any:
        """Train a single model."""
        model_configs = self._initialize_models(y)
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = model_configs[model_name]
        cv = StratifiedKFold(n_splits=self.config['evaluation']['cv_folds'], shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv,
            scoring='recall_macro',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def predict(self, X: csr_matrix, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using specified or best model."""
        model = self.models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No trained model available")
        return model.predict(X)
    
    def predict_proba(self, X: csr_matrix, model_name: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities."""
        model = self.models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No trained model available")
        
        # Handle models without predict_proba (like LinearSVC)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # Use decision function for SVM
            decision = model.decision_function(X)
            # Convert to pseudo-probabilities using softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def get_high_risk_predictions(self, X: csr_matrix, threshold: float = 0.3,
                                  model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions with custom threshold for high-risk class."""
        proba = self.predict_proba(X, model_name)
        
        # High risk is class 2
        high_risk_proba = proba[:, 2]
        
        # Adjust predictions based on threshold
        predictions = np.argmax(proba, axis=1)
        predictions[high_risk_proba >= threshold] = 2
        
        return predictions, high_risk_proba
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save model to disk."""
        model = self.models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No model to save")
        joblib.dump(model, filepath)
    
    def load_model(self, filepath: str, model_name: str = 'loaded'):
        """Load model from disk."""
        model = joblib.load(filepath)
        self.models[model_name] = model
        return model
    
    def get_feature_importance(self, feature_names: List[str], 
                               model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance from model."""
        model = self.models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No trained model available")
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models - use absolute mean across classes
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        # Handle sparse matrices
        if hasattr(importance, 'toarray'):
            importance = importance.toarray().flatten()
        
        df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
