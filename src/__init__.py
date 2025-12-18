"""
Source modules for NLP Product Classification Project
"""

from .preprocessing import VietnameseTextPreprocessor, create_preprocessor, combine_text_features
from .data_utils import (
    load_data, prepare_text_data, encode_labels, split_data,
    load_split_data, load_label_encoder, get_class_weights
)
from .feature_extraction import UnifiedVectorizer, prepare_features, load_vectorizer
from .ml_models import (
    LogisticRegressionModel, SVMModel, RandomForestModel,
    create_ml_models, train_all_ml_models, load_ml_model
)
from .dl_models import LSTMClassifier, PhoBERTClassifier, create_dl_models
from .evaluation import (
    calculate_metrics, plot_confusion_matrix, plot_model_comparison,
    plot_per_class_metrics, plot_training_history,
    create_results_summary, save_all_visualizations, print_final_report
)

__all__ = [
    # Preprocessing
    'VietnameseTextPreprocessor',
    'create_preprocessor',
    'combine_text_features',
    
    # Data utilities
    'load_data',
    'prepare_text_data',
    'encode_labels',
    'split_data',
    'load_split_data',
    'load_label_encoder',
    'get_class_weights',
    
    # Feature extraction
    'UnifiedVectorizer',
    'prepare_features',
    'load_vectorizer',
    
    # ML Models
    'LogisticRegressionModel',
    'SVMModel',
    'RandomForestModel',
    'create_ml_models',
    'train_all_ml_models',
    'load_ml_model',
    
    # DL Models
    'LSTMClassifier',
    'PhoBERTClassifier',
    'create_dl_models',
    
    # Evaluation
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_per_class_metrics',
    'plot_training_history',
    'create_results_summary',
    'save_all_visualizations',
    'print_final_report'
]
