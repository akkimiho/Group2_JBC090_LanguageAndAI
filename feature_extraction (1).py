"""Feature extraction module for personality prediction.

This module implements both content-based and stylometric features
for MBTI personality trait classification.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re
from typing import Dict, List, Tuple
import nltk
from nltk import pos_tag, word_tokenize
import textstat


class ContentFeatures:
    """Extract content-based features using TF-IDF and BoW."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """Initialize content feature extractors.
        
        Parameters
        ----------
        max_features : int
            Maximum number of features to extract
        ngram_range : tuple
            N-gram range for feature extraction
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.bow = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        
    def fit_transform(self, texts):
        """Fit and transform texts to TF-IDF features."""
        return self.tfidf.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        return self.tfidf.transform(texts)
    
    def get_feature_names(self):
        """Get feature names."""
        return self.tfidf.get_feature_names_out()


class StylometricFeatures:
    """Extract stylometric (style-based) features from text."""
    
    def __init__(self):
        """Initialize stylometric feature extractor."""
        # Function words
        self.function_words = {
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
            'articles': ['a', 'an', 'the'],
            'prepositions': ['in', 'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through'],
            'conjunctions': ['and', 'but', 'or', 'nor', 'for', 'yet', 'so'],
            'auxiliaries': ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did']
        }
        
        # Pragmatic markers
        self.hedges = ['maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'seem', 'appears']
        self.intensifiers = ['very', 'really', 'extremely', 'absolutely', 'completely', 'totally']
        self.negations = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'none']
        
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all stylometric features from a single text.
        
        Parameters
        ----------
        text : str
            Input text to extract features from
            
        Returns
        -------
        dict
            Dictionary of feature names and values
        """
        features = {}
        
        # Tokenize
        tokens = text.split()
        if len(tokens) == 0:
            return self._get_empty_features()
        
        # Function word frequencies
        features.update(self._function_word_freq(tokens))
        
        # POS tag ratios
        features.update(self._pos_ratios(text))
        
        # Lexical richness
        features.update(self._lexical_richness(tokens))
        
        # Syntactic features
        features.update(self._syntactic_features(text, tokens))
        
        # Pragmatic markers
        features.update(self._pragmatic_markers(tokens))
        
        # Readability metrics
        features.update(self._readability_metrics(text))
        
        return features
    
    def _function_word_freq(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate function word frequencies."""
        features = {}
        total = len(tokens)
        
        for category, words in self.function_words.items():
            count = sum(1 for t in tokens if t in words)
            features[f'fw_{category}_freq'] = count / total if total > 0 else 0
        
        return features
    
    def _pos_ratios(self, text: str) -> Dict[str, float]:
        """Calculate POS tag ratios."""
        try:
            tagged = pos_tag(word_tokenize(text))
            total = len(tagged)
            
            pos_counts = Counter(tag[:2] for _, tag in tagged)
            
            return {
                'pos_noun_ratio': (pos_counts['NN'] + pos_counts['NP']) / total if total > 0 else 0,
                'pos_verb_ratio': (pos_counts['VB'] + pos_counts['VD'] + pos_counts['VG']) / total if total > 0 else 0,
                'pos_adj_ratio': pos_counts['JJ'] / total if total > 0 else 0,
                'pos_adv_ratio': pos_counts['RB'] / total if total > 0 else 0
            }
        except:
            return {
                'pos_noun_ratio': 0,
                'pos_verb_ratio': 0,
                'pos_adj_ratio': 0,
                'pos_adv_ratio': 0
            }
    
    def _lexical_richness(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate lexical richness measures."""
        total = len(tokens)
        unique = len(set(tokens))
        
        # Type-Token Ratio
        ttr = unique / total if total > 0 else 0
        
        # Hapax Legomena (words appearing once)
        word_counts = Counter(tokens)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / total if total > 0 else 0
        
        return {
            'lex_ttr': ttr,
            'lex_hapax_ratio': hapax_ratio,
            'lex_avg_word_length': np.mean([len(t) for t in tokens]) if tokens else 0
        }
    
    def _syntactic_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """Calculate syntactic features."""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'syn_avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'syn_sentence_count': len(sentences),
            'syn_comma_count': text.count(',') / len(tokens) if tokens else 0,
            'syn_question_marks': text.count('?') / len(tokens) if tokens else 0,
            'syn_exclamation_marks': text.count('!') / len(tokens) if tokens else 0
        }
    
    def _pragmatic_markers(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate pragmatic marker frequencies."""
        total = len(tokens)
        
        return {
            'prag_hedges': sum(1 for t in tokens if t in self.hedges) / total if total > 0 else 0,
            'prag_intensifiers': sum(1 for t in tokens if t in self.intensifiers) / total if total > 0 else 0,
            'prag_negations': sum(1 for t in tokens if t in self.negations) / total if total > 0 else 0
        }
    
    def _readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        try:
            return {
                'read_flesch_kincaid': textstat.flesch_kincaid_grade(text),
                'read_flesch_reading_ease': textstat.flesch_reading_ease(text),
                'read_smog': textstat.smog_index(text)
            }
        except:
            return {
                'read_flesch_kincaid': 0,
                'read_flesch_reading_ease': 0,
                'read_smog': 0
            }
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {key: 0 for key in [
            'fw_pronouns_freq', 'fw_articles_freq', 'fw_prepositions_freq',
            'fw_conjunctions_freq', 'fw_auxiliaries_freq',
            'pos_noun_ratio', 'pos_verb_ratio', 'pos_adj_ratio', 'pos_adv_ratio',
            'lex_ttr', 'lex_hapax_ratio', 'lex_avg_word_length',
            'syn_avg_sentence_length', 'syn_sentence_count', 'syn_comma_count',
            'syn_question_marks', 'syn_exclamation_marks',
            'prag_hedges', 'prag_intensifiers', 'prag_negations',
            'read_flesch_kincaid', 'read_flesch_reading_ease', 'read_smog'
        ]}
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extract features from multiple texts.
        
        Parameters
        ----------
        texts : list
            List of text strings
            
        Returns
        -------
        pd.DataFrame
            DataFrame with extracted features
        """
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)


class FeatureExtractor:
    """Main feature extraction class combining all feature types."""
    
    def __init__(self, feature_type='stylometric', max_features=5000):
        """Initialize feature extractor.
        
        Parameters
        ----------
        feature_type : str
            Type of features: 'content', 'stylometric', or 'combined'
        max_features : int
            Maximum features for content-based extraction
        """
        self.feature_type = feature_type
        self.content_extractor = ContentFeatures(max_features=max_features)
        self.stylo_extractor = StylometricFeatures()
        
    def fit_transform(self, texts: List[str], labels=None):
        """Fit and transform texts to features.
        
        Parameters
        ----------
        texts : list
            List of text strings
        labels : array-like, optional
            Target labels
            
        Returns
        -------
        Features in appropriate format (array or DataFrame)
        """
        if self.feature_type == 'content':
            return self.content_extractor.fit_transform(texts)
        elif self.feature_type == 'stylometric':
            return self.stylo_extractor.extract_batch(texts)
        elif self.feature_type == 'combined':
            content_features = self.content_extractor.fit_transform(texts)
            stylo_features = self.stylo_extractor.extract_batch(texts)
            # Combine sparse and dense features
            import scipy.sparse as sp
            return sp.hstack([content_features, stylo_features.values])
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def transform(self, texts: List[str]):
        """Transform texts using fitted extractors."""
        if self.feature_type == 'content':
            return self.content_extractor.transform(texts)
        elif self.feature_type == 'stylometric':
            return self.stylo_extractor.extract_batch(texts)
        elif self.feature_type == 'combined':
            content_features = self.content_extractor.transform(texts)
            stylo_features = self.stylo_extractor.extract_batch(texts)
            import scipy.sparse as sp
            return sp.hstack([content_features, stylo_features.values])


if __name__ == "__main__":
    # Test the feature extractors
    sample_texts = [
        "i really love this! it's amazing.",
        "the weather is nice today, but maybe it will rain later."
    ]
    
    # Test stylometric features
    stylo = StylometricFeatures()
    features = stylo.extract_batch(sample_texts)
    print("Stylometric Features:")
    print(features)
    print(f"\nFeature count: {len(features.columns)}")
