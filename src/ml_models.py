"""
Machine Learning Models Module
Chứa 3 mô hình ML: Logistic Regression, SVM, Random Forest
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional, Any
import joblib
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ML_MODELS_CONFIG, ML_MODELS_DIR, RANDOM_SEED


class BaseMLModel(ABC):
    """Abstract base class cho các ML models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build(self, **kwargs) -> None:
        """Khởi tạo model"""
        pass
    
    def fit(self, X_train: csr_matrix, y_train: np.ndarray) -> 'BaseMLModel':
        """
        Train model
        
        Args:
            X_train: TF-IDF features
            y_train: Labels
            
        Returns:
            self
        """
        print(f"\n{'='*50}")
        print(f"Training {self.name}...")
        print(f"{'='*50}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """Dự đoán labels"""
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        return self.model.predict(X)
    
    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Dự đoán xác suất cho mỗi class"""
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: csr_matrix,
        y: np.ndarray,
        class_names: Optional[list] = None,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Đánh giá model
        
        Args:
            X: Features
            y: True labels
            class_names: Tên các class
            dataset_name: Tên dataset để hiển thị
            
        Returns:
            Dictionary chứa các metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred
        }
        
        print(f"\n{dataset_name} Results for {self.name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        if class_names:
            print(f"\nClassification Report:")
            print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
        
        return metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Lưu model
        
        Args:
            filepath: Đường dẫn lưu (optional)
            
        Returns:
            Đường dẫn file đã lưu
        """
        if filepath is None:
            os.makedirs(ML_MODELS_DIR, exist_ok=True)
            filepath = os.path.join(ML_MODELS_DIR, f"{self.name.lower().replace(' ', '_')}.pkl")
        
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'is_trained': self.is_trained
        }, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseMLModel':
        """Load model từ file"""
        data = joblib.load(filepath)
        
        instance = cls.__new__(cls)
        instance.model = data['model']
        instance.name = data['name']
        instance.is_trained = data['is_trained']
        
        print(f"Model loaded from {filepath}")
        return instance


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier"""
    
    def __init__(self):
        super().__init__("Logistic Regression")
        
    def build(self, **kwargs) -> None:
        config = ML_MODELS_CONFIG['logistic_regression'].copy()
        config.update(kwargs)
        self.model = LogisticRegression(**config)
        print(f"{self.name} initialized with config: {config}")


class SVMModel(BaseMLModel):
    """Support Vector Machine classifier"""
    
    def __init__(self):
        super().__init__("SVM")
        
    def build(self, **kwargs) -> None:
        config = ML_MODELS_CONFIG['svm'].copy()
        config.update(kwargs)
        self.model = SVC(**config)
        print(f"{self.name} initialized with config: {config}")


class RandomForestModel(BaseMLModel):
    """Random Forest classifier"""
    
    def __init__(self):
        super().__init__("Random Forest")
        
    def build(self, **kwargs) -> None:
        config = ML_MODELS_CONFIG['random_forest'].copy()
        config.update(kwargs)
        self.model = RandomForestClassifier(**config)
        print(f"{self.name} initialized with config: {config}")
    
    def get_feature_importance(
        self,
        feature_names: np.ndarray,
        top_n: int = 20
    ) -> list:
        """
        Lấy top features quan trọng nhất
        
        Args:
            feature_names: Tên các features
            top_n: Số features cần lấy
            
        Returns:
            List of (feature_name, importance)
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        return [(feature_names[i], importances[i]) for i in indices]


def create_ml_models() -> Dict[str, BaseMLModel]:
    """
    Factory function tạo tất cả ML models
    
    Returns:
        Dictionary mapping model_name -> model_instance
    """
    models = {
        'logistic_regression': LogisticRegressionModel(),
        'svm': SVMModel(),
        'random_forest': RandomForestModel()
    }
    
    for model in models.values():
        model.build()
    
    return models


def train_all_ml_models(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: Optional[csr_matrix] = None,
    y_val: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
    save_models: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train và evaluate tất cả ML models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        class_names: Tên các class
        save_models: Có lưu models không
        
    Returns:
        Dictionary chứa results cho mỗi model
    """
    models = create_ml_models()
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_metrics = model.evaluate(X_train, y_train, class_names, "Training")
        
        # Evaluate on validation data
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = model.evaluate(X_val, y_val, class_names, "Validation")
        
        # Save model
        if save_models:
            model.save()
        
        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    return results


def load_ml_model(model_name: str) -> BaseMLModel:
    """
    Load một ML model cụ thể
    
    Args:
        model_name: Tên model ('logistic_regression', 'svm', 'random_forest')
        
    Returns:
        Model instance
    """
    filepath = os.path.join(ML_MODELS_DIR, f"{model_name}.pkl")
    
    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'random_forest': RandomForestModel
    }
    
    return model_classes[model_name].load(filepath)


if __name__ == "__main__":
    # Test với dữ liệu giả
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=5,
                               n_informative=50, random_state=42)
    X = csr_matrix(X)
    
    models = create_ml_models()
    for name, model in models.items():
        model.fit(X, y)
        metrics = model.evaluate(X, y, dataset_name="Test")
        print(f"\n{name}: Accuracy = {metrics['accuracy']:.4f}")
