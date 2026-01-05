"""Data loading and initial validation module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import yaml


class DataLoader:
    """Handles loading and initial validation of complaint data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.text_column = self.config['data']['text_column']
        self.target_column = self.config['data']['target_column']
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load complaint data from CSV file."""
        if filepath is None:
            filepath = self.config['data']['raw_path']
        
        df = pd.read_csv(filepath)
        self._validate_data(df)
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required_cols = [self.text_column]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create synthetic complaint data for demonstration."""
        np.random.seed(42)
        
        # Sample complaint templates by risk level
        low_risk_templates = [
            "I have a question about my account balance.",
            "Could you help me understand my statement?",
            "I'd like to update my contact information.",
            "When will my new card arrive?",
            "I need help setting up online banking.",
        ]
        
        medium_risk_templates = [
            "I've been waiting for a refund for two weeks now.",
            "This is the second time I'm calling about this issue.",
            "I'm not satisfied with the response I received.",
            "The fees on my account seem incorrect.",
            "I was promised a callback but never received one.",
        ]
        
        high_risk_templates = [
            "I'm extremely frustrated and considering closing my account!",
            "This is unacceptable! I've been a customer for 10 years!",
            "I will be filing a complaint with the regulatory authority.",
            "I'm going to share my terrible experience on social media.",
            "I demand to speak with a supervisor immediately! This is fraud!",
        ]
        
        # Generate samples with class imbalance (realistic scenario)
        risk_distribution = [0.6, 0.25, 0.15]  # Low, Medium, High
        
        data = []
        for _ in range(n_samples):
            risk_level = np.random.choice([0, 1, 2], p=risk_distribution)
            
            if risk_level == 0:
                template = np.random.choice(low_risk_templates)
            elif risk_level == 1:
                template = np.random.choice(medium_risk_templates)
            else:
                template = np.random.choice(high_risk_templates)
            
            # Add some variation
            variations = [
                f"{template} Please help.",
                f"{template} Thank you.",
                f"Hello, {template}",
                template,
            ]
            complaint = np.random.choice(variations)
            
            data.append({
                'complaint_id': f"CMP_{len(data):05d}",
                'complaint_narrative': complaint,
                'risk_level': risk_level,
                'product': np.random.choice(['Credit Card', 'Mortgage', 'Checking', 'Savings']),
                'date_received': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })
        
        return pd.DataFrame(data)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets with stratification."""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=df[self.target_column]
        )
        
        return train_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Get distribution of risk classes."""
        return df[self.target_column].value_counts(normalize=True)
