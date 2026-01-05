"""Feature engineering module for text-based features."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
import yaml


class FeatureEngineer:
    """Handles feature extraction from preprocessed complaint text."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['features']
        
        self.tfidf_vectorizer = None
        self.feature_names = []
        self._fitted = False
    
    def fit(self, texts: pd.Series) -> 'FeatureEngineer':
        """Fit feature extractors on training data."""
        # Initialize TF-IDF vectorizer
        tfidf_config = self.config['tfidf']
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            sublinear_tf=True
        )
        
        self.tfidf_vectorizer.fit(texts)
        self.feature_names = list(self.tfidf_vectorizer.get_feature_names_out())
        self._fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> Tuple[csr_matrix, List[str]]:
        """Transform text data into feature matrix."""
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(df[text_column])
        
        # Sentiment features
        sentiment_features, sentiment_names = self._extract_sentiment_features(df['original_text'])
        
        # Text statistics features
        stats_features, stats_names = self._extract_text_stats(df['original_text'])
        
        # Combine all features
        all_features = hstack([
            tfidf_features,
            csr_matrix(sentiment_features),
            csr_matrix(stats_features)
        ])
        
        all_feature_names = self.feature_names + sentiment_names + stats_names
        
        return all_features, all_feature_names
    
    def fit_transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> Tuple[csr_matrix, List[str]]:
        """Fit and transform in one step."""
        self.fit(df[text_column])
        return self.transform(df, text_column)
    
    def _extract_sentiment_features(self, texts: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract sentiment-based features."""
        from textblob import TextBlob
        
        features = []
        feature_names = []
        
        if self.config['sentiment']['include_vader']:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                
                vader_scores = texts.apply(lambda x: sia.polarity_scores(str(x)) if pd.notna(x) else 
                                          {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
                
                features.append(vader_scores.apply(lambda x: x['neg']).values.reshape(-1, 1))
                features.append(vader_scores.apply(lambda x: x['neu']).values.reshape(-1, 1))
                features.append(vader_scores.apply(lambda x: x['pos']).values.reshape(-1, 1))
                features.append(vader_scores.apply(lambda x: x['compound']).values.reshape(-1, 1))
                
                feature_names.extend(['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'])
            except Exception:
                pass
        
        if self.config['sentiment']['include_textblob']:
            tb_polarity = texts.apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
            tb_subjectivity = texts.apply(lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0)
            
            features.append(tb_polarity.values.reshape(-1, 1))
            features.append(tb_subjectivity.values.reshape(-1, 1))
            feature_names.extend(['textblob_polarity', 'textblob_subjectivity'])
        
        if features:
            return np.hstack(features), feature_names
        return np.zeros((len(texts), 1)), ['placeholder']
    
    def _extract_text_stats(self, texts: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract text statistics features."""
        stats_config = self.config['text_stats']
        features = []
        feature_names = []
        
        if stats_config['include_length']:
            features.append(texts.apply(lambda x: len(str(x)) if pd.notna(x) else 0).values.reshape(-1, 1))
            feature_names.append('text_length')
        
        if stats_config['include_word_count']:
            features.append(texts.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0).values.reshape(-1, 1))
            feature_names.append('word_count')
        
        if stats_config['include_avg_word_length']:
            features.append(texts.apply(
                lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and str(x).split() else 0
            ).values.reshape(-1, 1))
            feature_names.append('avg_word_length')
        
        if stats_config['include_sentence_count']:
            features.append(texts.apply(
                lambda x: str(x).count('.') + str(x).count('!') + str(x).count('?') if pd.notna(x) else 0
            ).values.reshape(-1, 1))
            feature_names.append('sentence_count')
        
        if stats_config['include_exclamation_count']:
            features.append(texts.apply(lambda x: str(x).count('!') if pd.notna(x) else 0).values.reshape(-1, 1))
            feature_names.append('exclamation_count')
        
        if stats_config['include_question_count']:
            features.append(texts.apply(lambda x: str(x).count('?') if pd.notna(x) else 0).values.reshape(-1, 1))
            feature_names.append('question_count')
        
        if stats_config['include_caps_ratio']:
            features.append(texts.apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1) if pd.notna(x) else 0
            ).values.reshape(-1, 1))
            feature_names.append('caps_ratio')
        
        # Additional risk-indicative features
        features.append(texts.apply(self._count_urgency_words).values.reshape(-1, 1))
        feature_names.append('urgency_word_count')
        
        features.append(texts.apply(self._count_negative_words).values.reshape(-1, 1))
        feature_names.append('negative_word_count')
        
        features.append(texts.apply(self._count_escalation_words).values.reshape(-1, 1))
        feature_names.append('escalation_word_count')
        
        return np.hstack(features), feature_names
    
    def _count_urgency_words(self, text: str) -> int:
        """Count urgency-indicating words."""
        urgency_words = {'urgent', 'immediately', 'asap', 'emergency', 'critical', 
                        'now', 'today', 'deadline', 'hurry', 'quick'}
        if pd.isna(text):
            return 0
        words = str(text).lower().split()
        return sum(1 for w in words if w in urgency_words)
    
    def _count_negative_words(self, text: str) -> int:
        """Count negative sentiment words."""
        negative_words = {'terrible', 'horrible', 'awful', 'worst', 'hate', 'angry',
                         'frustrated', 'disappointed', 'unacceptable', 'ridiculous',
                         'incompetent', 'useless', 'pathetic', 'disgusting', 'outrageous'}
        if pd.isna(text):
            return 0
        words = str(text).lower().split()
        return sum(1 for w in words if w in negative_words)
    
    def _count_escalation_words(self, text: str) -> int:
        """Count escalation-indicating words."""
        escalation_words = {'lawyer', 'attorney', 'sue', 'lawsuit', 'legal', 'court',
                           'regulator', 'complaint', 'bbb', 'cfpb', 'media', 'news',
                           'supervisor', 'manager', 'cancel', 'close', 'leave'}
        if pd.isna(text):
            return 0
        words = str(text).lower().split()
        return sum(1 for w in words if w in escalation_words)
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N TF-IDF features by importance."""
        if not self._fitted:
            return []
        return self.feature_names[:n]
