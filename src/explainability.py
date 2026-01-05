"""Model explainability module for risk predictions."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import yaml


class RiskExplainer:
    """Provides explanations for risk predictions."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_labels = self.config['risk_labels']
        self.class_names = [self.risk_labels[i] for i in sorted(self.risk_labels.keys())]
        
        self.lime_explainer = None
        self.shap_explainer = None
    
    def setup_lime(self, feature_names: List[str], class_names: Optional[List[str]] = None):
        """Initialize LIME explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            self.lime_explainer = LimeTabularExplainer(
                training_data=np.zeros((1, len(feature_names))),  # Placeholder
                feature_names=feature_names,
                class_names=class_names or self.class_names,
                mode='classification'
            )
        except ImportError:
            print("LIME not available. Install with: pip install lime")
    
    def setup_shap(self, model, X_train):
        """Initialize SHAP explainer."""
        try:
            import shap
            
            # Use appropriate explainer based on model type
            model_type = type(model).__name__
            
            if 'Forest' in model_type or 'XGB' in model_type or 'Tree' in model_type:
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models (slower but universal)
                # Sample background data for efficiency
                if hasattr(X_train, 'toarray'):
                    background = shap.sample(X_train.toarray(), min(100, X_train.shape[0]))
                else:
                    background = shap.sample(X_train, min(100, len(X_train)))
                
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
    
    def explain_prediction_lime(self, model, instance, feature_names: List[str],
                                num_features: int = 10) -> Dict[str, Any]:
        """Explain a single prediction using LIME."""
        if self.lime_explainer is None:
            self.setup_lime(feature_names)
        
        # Convert sparse matrix to dense if needed
        if hasattr(instance, 'toarray'):
            instance = instance.toarray().flatten()
        
        explanation = self.lime_explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features,
            top_labels=3
        )
        
        # Extract explanation for each class
        result = {
            'predicted_class': model.predict(instance.reshape(1, -1))[0],
            'probabilities': model.predict_proba(instance.reshape(1, -1))[0].tolist(),
            'explanations': {}
        }
        
        for label in explanation.available_labels():
            result['explanations'][self.class_names[label]] = {
                'features': explanation.as_list(label),
                'score': explanation.score
            }
        
        return result
    
    def explain_prediction_shap(self, instance, feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction using SHAP."""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call setup_shap first.")
        
        # Convert sparse matrix to dense if needed
        if hasattr(instance, 'toarray'):
            instance = instance.toarray()
        
        shap_values = self.shap_explainer.shap_values(instance)
        
        result = {
            'shap_values': {},
            'base_value': self.shap_explainer.expected_value
        }
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            for i, class_name in enumerate(self.class_names):
                values = shap_values[i].flatten()
                top_indices = np.argsort(np.abs(values))[-10:][::-1]
                result['shap_values'][class_name] = [
                    (feature_names[idx], float(values[idx])) 
                    for idx in top_indices if idx < len(feature_names)
                ]
        else:
            values = shap_values.flatten()
            top_indices = np.argsort(np.abs(values))[-10:][::-1]
            result['shap_values']['overall'] = [
                (feature_names[idx], float(values[idx])) 
                for idx in top_indices if idx < len(feature_names)
            ]
        
        return result
    
    def get_feature_importance_explanation(self, model, feature_names: List[str],
                                           top_n: int = 20) -> pd.DataFrame:
        """Get global feature importance with interpretable explanations."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For multi-class, get importance per class
            importance = np.abs(model.coef_).mean(axis=0)
            if hasattr(importance, 'toarray'):
                importance = importance.toarray().flatten()
        else:
            return pd.DataFrame()
        
        # Create dataframe
        df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Add interpretations
        df['interpretation'] = df['feature'].apply(self._interpret_feature)
        
        return df
    
    def _interpret_feature(self, feature_name: str) -> str:
        """Provide human-readable interpretation of feature."""
        interpretations = {
            'vader_neg': 'Negative sentiment intensity',
            'vader_pos': 'Positive sentiment intensity',
            'vader_compound': 'Overall sentiment score',
            'textblob_polarity': 'Text polarity (-1 to 1)',
            'textblob_subjectivity': 'Subjectivity level',
            'text_length': 'Length of complaint text',
            'word_count': 'Number of words',
            'exclamation_count': 'Number of exclamation marks (urgency indicator)',
            'question_count': 'Number of questions asked',
            'caps_ratio': 'Proportion of capital letters (emphasis)',
            'urgency_word_count': 'Count of urgency-related words',
            'negative_word_count': 'Count of negative sentiment words',
            'escalation_word_count': 'Count of escalation-related words'
        }
        
        if feature_name in interpretations:
            return interpretations[feature_name]
        elif feature_name.startswith('tfidf_'):
            return f'TF-IDF weight for "{feature_name[6:]}"'
        else:
            return f'Text feature: {feature_name}'
    
    def explain_high_risk_case(self, model, instance, original_text: str,
                               feature_names: List[str]) -> Dict[str, Any]:
        """Provide comprehensive explanation for a high-risk prediction."""
        # Get prediction
        if hasattr(instance, 'toarray'):
            instance_dense = instance.toarray()
        else:
            instance_dense = instance.reshape(1, -1)
        
        prediction = model.predict(instance_dense)[0]
        probabilities = model.predict_proba(instance_dense)[0]
        
        explanation = {
            'original_text': original_text,
            'predicted_risk': self.class_names[prediction],
            'confidence': float(probabilities[prediction]),
            'risk_probabilities': {
                name: float(prob) for name, prob in zip(self.class_names, probabilities)
            },
            'risk_indicators': self._extract_risk_indicators(original_text),
            'recommended_actions': self._get_recommended_actions(prediction, probabilities)
        }
        
        return explanation
    
    def _extract_risk_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract specific risk indicators from text."""
        text_lower = text.lower()
        
        indicators = {
            'urgency_signals': [],
            'negative_sentiment': [],
            'escalation_threats': [],
            'churn_indicators': []
        }
        
        # Urgency signals
        urgency_words = ['urgent', 'immediately', 'asap', 'emergency', 'now', 'today']
        indicators['urgency_signals'] = [w for w in urgency_words if w in text_lower]
        
        # Negative sentiment
        negative_words = ['terrible', 'horrible', 'awful', 'worst', 'hate', 'angry',
                         'frustrated', 'disappointed', 'unacceptable', 'ridiculous']
        indicators['negative_sentiment'] = [w for w in negative_words if w in text_lower]
        
        # Escalation threats
        escalation_words = ['lawyer', 'attorney', 'sue', 'lawsuit', 'legal', 'court',
                           'regulator', 'bbb', 'cfpb', 'media', 'news', 'supervisor']
        indicators['escalation_threats'] = [w for w in escalation_words if w in text_lower]
        
        # Churn indicators
        churn_words = ['cancel', 'close', 'leave', 'switch', 'competitor', 'done', 'quit']
        indicators['churn_indicators'] = [w for w in churn_words if w in text_lower]
        
        return indicators
    
    def _get_recommended_actions(self, prediction: int, probabilities: np.ndarray) -> List[str]:
        """Get recommended actions based on risk level."""
        actions = {
            0: [  # Low risk
                "Standard response within SLA",
                "Route to general support queue",
                "Send automated acknowledgment"
            ],
            1: [  # Medium risk
                "Prioritize for same-day response",
                "Assign to experienced agent",
                "Review account history before contact",
                "Prepare retention offer if needed"
            ],
            2: [  # High risk
                "IMMEDIATE escalation to senior support",
                "Notify customer success manager",
                "Prepare executive-level response",
                "Review for regulatory compliance",
                "Document all interactions thoroughly",
                "Consider proactive outreach within 1 hour"
            ]
        }
        
        base_actions = actions.get(prediction, [])
        
        # Add probability-based recommendations
        if probabilities[2] > 0.3 and prediction != 2:
            base_actions.insert(0, "⚠️ Elevated high-risk probability - monitor closely")
        
        return base_actions
    
    def plot_feature_importance(self, model, feature_names: List[str],
                                top_n: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance visualization."""
        df = self.get_feature_importance_explanation(model, feature_names, top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
        
        ax.barh(range(len(df)), df['importance'].values, color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top Feature Importance for Risk Classification')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_explanation_report(self, model, X_test, y_test, y_pred,
                                    feature_names: List[str], 
                                    original_texts: pd.Series) -> str:
        """Generate comprehensive explanation report."""
        report = []
        report.append("=" * 60)
        report.append("RISK PREDICTION EXPLAINABILITY REPORT")
        report.append("=" * 60)
        
        # Feature importance
        report.append("\n## Top Features Driving Risk Classification")
        importance_df = self.get_feature_importance_explanation(model, feature_names, 15)
        for _, row in importance_df.iterrows():
            report.append(f"  - {row['feature']}: {row['importance']:.4f}")
            report.append(f"    ({row['interpretation']})")
        
        # Sample high-risk explanations
        report.append("\n## Sample High-Risk Case Explanations")
        high_risk_indices = np.where(y_pred == 2)[0][:3]
        
        for idx in high_risk_indices:
            if idx < len(original_texts):
                text = original_texts.iloc[idx]
                indicators = self._extract_risk_indicators(text)
                
                report.append(f"\n### Case {idx}")
                report.append(f"Text: {text[:200]}...")
                report.append("Risk Indicators Found:")
                for category, words in indicators.items():
                    if words:
                        report.append(f"  - {category}: {', '.join(words)}")
        
        return '\n'.join(report)
