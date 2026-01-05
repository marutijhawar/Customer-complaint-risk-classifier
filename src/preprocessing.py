"""Text preprocessing module for complaint narratives."""

import re
import string
from typing import List, Optional
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import yaml


class TextPreprocessor:
    """Handles text cleaning and preprocessing for complaint narratives."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['preprocessing']
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stopwords to keep (important for sentiment)
        self.keep_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                          'nowhere', 'hardly', 'scarcely', 'barely', 'very',
                          'extremely', 'terrible', 'horrible', 'awful', 'worst'}
        self.stop_words -= self.keep_words
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                              else f'corpora/{resource}' if resource in ['stopwords', 'wordnet']
                              else f'taggers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.config['remove_urls']:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove emails
        if self.config['remove_emails']:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        if self.config['remove_phone_numbers']:
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.config['lowercase']:
            text = text.lower()
        
        tokens = word_tokenize(text)
        
        # Remove punctuation tokens
        if self.config['remove_punctuation']:
            tokens = [t for t in tokens if t not in string.punctuation]
        
        # Remove numbers (optional)
        if self.config['remove_numbers']:
            tokens = [t for t in tokens if not t.isdigit()]
        
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        if not self.config['lemmatize']:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while keeping sentiment-important words."""
        if not self.config['remove_stopwords']:
            return tokens
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Clean
        text = self.clean_text(text)
        
        # Check length constraints
        if len(text) < self.config['min_text_length']:
            return ""
        if len(text) > self.config['max_text_length']:
            text = text[:self.config['max_text_length']]
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocess all text in a dataframe."""
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.preprocess)
        df['original_text'] = df[text_column]
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def get_text_statistics(self, text: str) -> dict:
        """Extract text statistics before preprocessing."""
        if pd.isna(text) or not isinstance(text, str):
            return self._empty_stats()
        
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'unique_word_ratio': len(set(text.lower().split())) / len(text.split()) if text.split() else 0
        }
    
    def _empty_stats(self) -> dict:
        """Return empty statistics dictionary."""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'exclamation_count': 0, 'question_count': 0,
            'caps_ratio': 0, 'unique_word_ratio': 0
        }
