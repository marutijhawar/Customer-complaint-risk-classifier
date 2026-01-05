"""Lightweight version of main pipeline optimized for MacBook Air M3 16GB RAM."""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.explainability import RiskExplainer


def main():
    """Run the lightweight NLP risk classification pipeline."""
    print("=" * 60)
    print("CUSTOMER COMPLAINT RISK CLASSIFICATION SYSTEM")
    print("(Lightweight Version for MacBook Air M3)")
    print("=" * 60)

    # Create output directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    print("\n[1/6] Loading Data...")
    loader = DataLoader()

    # Check if real data exists, otherwise create synthetic
    raw_path = Path("data/raw/complaints.csv")
    if raw_path.exists():
        df = loader.load_data(str(raw_path))
        print(f"  Loaded {len(df)} complaints from file")
    else:
        print("  No data file found. Creating synthetic data for demonstration...")
        df = loader.create_synthetic_data(n_samples=300)
        df.to_csv(raw_path, index=False)
        print(f"  Created {len(df)} synthetic complaints")

    print(f"\n  Class Distribution:")
    for risk, count in df['risk_level'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        risk_name = {0: 'Low', 1: 'Medium', 2: 'High'}[risk]
        print(f"    {risk_name}: {count} ({pct:.1f}%)")

    # 2. Preprocess Text
    print("\n[2/6] Preprocessing Text...")
    print("  (Lightweight mode: lemmatization disabled)")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, 'complaint_narrative')
    print(f"  Preprocessed {len(df)} complaints")

    # Save processed data
    df.to_csv("data/processed/complaints_processed.csv", index=False)

    # 3. Split Data
    print("\n[3/6] Splitting Data...")
    train_df, test_df = loader.split_data(df)
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Test set: {len(test_df)} samples")

    # 4. Feature Engineering
    print("\n[4/6] Engineering Features...")
    print("  (Optimized: 200 TF-IDF features, unigrams only)")
    feature_engineer = FeatureEngineer()

    X_train, feature_names = feature_engineer.fit_transform(train_df, 'cleaned_text')
    X_test, _ = feature_engineer.transform(test_df, 'cleaned_text')

    y_train = train_df['risk_level'].values
    y_test = test_df['risk_level'].values

    print(f"  Total features: {len(feature_names)}")
    tfidf_count = len([f for f in feature_names if not f.startswith(('vader', 'textblob', 'text_', 'word_', 'avg_', 'sentence_', 'exclamation_', 'question_', 'caps_', 'urgency_', 'negative_', 'escalation_'))])
    sentiment_count = len([f for f in feature_names if f.startswith(('vader', 'textblob'))])
    stats_count = len([f for f in feature_names if f.startswith(('text_', 'word_', 'avg_', 'sentence_', 'exclamation_', 'question_', 'caps_', 'urgency_', 'negative_', 'escalation_'))])

    print(f"    TF-IDF: {tfidf_count}, Sentiment: {sentiment_count}, Stats: {stats_count}")

    # 5. Train Models
    print("\n[5/6] Training Models...")
    print("  (Optimized: reduced trees, 3-fold CV)")
    trainer = ModelTrainer()
    cv_results = trainer.train_all_models(X_train, y_train)

    # 6. Evaluate Models
    print("\n[6/6] Evaluating Models...")
    evaluator = ModelEvaluator()

    # Evaluate best model
    y_pred = trainer.predict(X_test)
    y_proba = trainer.predict_proba(X_test)

    # Generate evaluation report
    report = evaluator.generate_report(y_test, y_pred, y_proba)
    print(report)

    # High-risk focus evaluation
    high_risk_results = evaluator.evaluate_high_risk_focus(y_test, y_pred, y_proba)

    print("\n" + "=" * 60)
    print("HIGH-RISK DETECTION PERFORMANCE")
    print("=" * 60)
    print(f"High-Risk Recall: {high_risk_results['high_risk_recall']:.4f}")
    print(f"High-Risk Precision: {high_risk_results['high_risk_precision']:.4f}")
    print(f"Missed High-Risk Cases: {high_risk_results['high_risk_false_negatives']}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(y_test, y_pred, "reports/figures/confusion_matrix.png")
    evaluator.plot_precision_recall_curve(y_test, y_proba, "reports/figures/precision_recall.png")
    evaluator.plot_roc_curves(y_test, y_proba, "reports/figures/roc_curves.png")

    # Feature importance
    print("\nGenerating feature importance...")
    explainer = RiskExplainer()

    importance_df = explainer.get_feature_importance_explanation(
        trainer.best_model, feature_names, top_n=15
    )
    print("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    explainer.plot_feature_importance(
        trainer.best_model, feature_names, top_n=15,
        save_path="reports/figures/feature_importance.png"
    )

    # Save model
    trainer.save_model("models/best_model.joblib")
    print("\nModel saved to models/best_model.joblib")

    # Business Impact Summary
    print("\n" + "=" * 60)
    print("BUSINESS IMPACT SUMMARY")
    print("=" * 60)

    business_config = trainer.config['business']

    # Calculate potential impact
    total_high_risk = (y_test == 2).sum()
    detected_high_risk = ((y_pred == 2) & (y_test == 2)).sum()
    missed_high_risk = high_risk_results['high_risk_false_negatives']

    avg_customer_value = business_config['avg_customer_value']
    churn_cost = avg_customer_value * business_config['churn_cost_multiplier']

    potential_churn_cost_saved = detected_high_risk * churn_cost * 0.3  # Assume 30% save rate
    missed_opportunity_cost = missed_high_risk * churn_cost * 0.3

    print(f"Total High-Risk Cases in Test Set: {total_high_risk}")
    print(f"Successfully Identified: {detected_high_risk} ({detected_high_risk/total_high_risk*100:.1f}%)")
    print(f"Missed Cases: {missed_high_risk}")
    print(f"\nEstimated Financial Impact:")
    print(f"  Potential Churn Cost Saved: ${potential_churn_cost_saved:,.0f}")
    print(f"  Missed Opportunity Cost: ${missed_opportunity_cost:,.0f}")
    print(f"  Net Benefit: ${potential_churn_cost_saved - missed_opportunity_cost:,.0f}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Memory-optimized for MacBook Air M3")
    print("=" * 60)

    return {
        'model': trainer.best_model,
        'feature_names': feature_names,
        'evaluator': evaluator,
        'results': {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    }


if __name__ == "__main__":
    results = main()
