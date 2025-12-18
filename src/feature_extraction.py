"""
Feature Extraction Module - TF-IDF Vectorizer
Module dùng chung cho cả Machine Learning và Deep Learning
Đảm bảo công bằng khi so sánh các mô hình
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple, Optional, Union
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TFIDF_CONFIG, TFIDF_VECTORIZER_PATH, MODELS_DIR


class UnifiedVectorizer:
    """
    Unified TF-IDF Vectorizer cho cả ML và DL models
    Đảm bảo cùng một cách biểu diễn text cho việc so sánh công bằng
    """
    
    def __init__(
        self,
        max_features: int = TFIDF_CONFIG['max_features'],
        ngram_range: tuple = TFIDF_CONFIG['ngram_range'],
        min_df: int = TFIDF_CONFIG['min_df'],
        max_df: float = TFIDF_CONFIG['max_df'],
        sublinear_tf: bool = TFIDF_CONFIG['sublinear_tf']
    ):
        """
        Khởi tạo TF-IDF Vectorizer
        
        Args:
            max_features: Số lượng features tối đa
            ngram_range: Phạm vi n-gram (unigram, bigram, ...)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Sử dụng sublinear tf scaling (1 + log(tf))
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents=None,  # Giữ nguyên dấu tiếng Việt
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'  # Bao gồm cả từ tiếng Việt
        )
        
        self.is_fitted = False
        self.feature_names = None
        self.vocab_size = None
        
    def fit(self, texts: Union[list, pd.Series]) -> 'UnifiedVectorizer':
        """
        Fit vectorizer trên dữ liệu training
        
        Args:
            texts: Danh sách văn bản training
            
        Returns:
            self
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocab_size = len(self.feature_names)
        
        print(f"TF-IDF Vectorizer fitted with {self.vocab_size} features")
        return self
    
    def transform(self, texts: Union[list, pd.Series]) -> csr_matrix:
        """
        Transform texts thành TF-IDF vectors
        
        Args:
            texts: Danh sách văn bản cần transform
            
        Returns:
            Sparse matrix của TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer chưa được fit. Gọi fit() trước.")
            
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: Union[list, pd.Series]) -> csr_matrix:
        """
        Fit và transform cùng lúc
        
        Args:
            texts: Danh sách văn bản
            
        Returns:
            Sparse matrix của TF-IDF features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def to_dense(self, sparse_matrix: csr_matrix) -> np.ndarray:
        """Convert sparse matrix to dense array"""
        return sparse_matrix.toarray()
    
    def save(self, filepath: str = TFIDF_VECTORIZER_PATH) -> None:
        """
        Lưu vectorizer
        
        Args:
            filepath: Đường dẫn lưu file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'vocab_size': self.vocab_size
        }, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = TFIDF_VECTORIZER_PATH) -> 'UnifiedVectorizer':
        """
        Load vectorizer đã lưu
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            UnifiedVectorizer instance
        """
        data = joblib.load(filepath)
        
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.is_fitted = data['is_fitted']
        instance.feature_names = data['feature_names']
        instance.vocab_size = data['vocab_size']
        
        print(f"Vectorizer loaded from {filepath}")
        return instance
    
    def get_top_features_per_class(
        self,
        X: csr_matrix,
        y: np.ndarray,
        class_names: list,
        top_n: int = 20
    ) -> dict:
        """
        Lấy top features quan trọng nhất cho mỗi class
        
        Args:
            X: TF-IDF matrix
            y: Labels
            class_names: Tên các class
            top_n: Số features cần lấy
            
        Returns:
            Dictionary mapping class -> list of top features
        """
        top_features = {}
        
        for idx, class_name in enumerate(class_names):
            class_mask = y == idx
            class_tfidf = X[class_mask].mean(axis=0).A1
            
            top_indices = class_tfidf.argsort()[-top_n:][::-1]
            top_words = [(self.feature_names[i], class_tfidf[i]) 
                        for i in top_indices]
            
            top_features[class_name] = top_words
            
        return top_features


def prepare_features(
    train_texts: Union[list, pd.Series],
    val_texts: Optional[Union[list, pd.Series]] = None,
    test_texts: Optional[Union[list, pd.Series]] = None,
    save_vectorizer: bool = True
) -> Tuple[csr_matrix, Optional[csr_matrix], Optional[csr_matrix], UnifiedVectorizer]:
    """
    Chuẩn bị TF-IDF features cho tất cả các tập dữ liệu
    
    Args:
        train_texts: Text training
        val_texts: Text validation (optional)
        test_texts: Text test (optional)
        save_vectorizer: Có lưu vectorizer không
        
    Returns:
        Tuple (X_train, X_val, X_test, vectorizer)
    """
    # Initialize and fit on training data
    vectorizer = UnifiedVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    
    # Transform validation and test if provided
    X_val = vectorizer.transform(val_texts) if val_texts is not None else None
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None
    
    # Save vectorizer
    if save_vectorizer:
        vectorizer.save()
    
    print(f"\nFeature shapes:")
    print(f"  Train: {X_train.shape}")
    if X_val is not None:
        print(f"  Val: {X_val.shape}")
    if X_test is not None:
        print(f"  Test: {X_test.shape}")
    
    return X_train, X_val, X_test, vectorizer


def load_vectorizer() -> UnifiedVectorizer:
    """Load vectorizer đã train"""
    return UnifiedVectorizer.load()


if __name__ == "__main__":
    # Test module
    sample_texts = [
        "áo váy nữ thiết kế cao cấp chất liệu lụa mềm mại",
        "điện thoại samsung galaxy pin trâu camera đẹp",
        "laptop gaming msi rtx card đồ họa mạnh"
    ]
    
    vectorizer = UnifiedVectorizer(max_features=100)
    X = vectorizer.fit_transform(sample_texts)
    
    print(f"Shape: {X.shape}")
    print(f"Vocabulary size: {vectorizer.vocab_size}")
    print(f"Sample features: {vectorizer.feature_names[:10]}")
