"""
Configuration file for the Product Category Classification Project
"""
import os

# ==================== PATH CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
ML_MODELS_DIR = os.path.join(MODELS_DIR, "ml")
DL_MODELS_DIR = os.path.join(MODELS_DIR, "dl")

# Visualization paths
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

# Data files
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "data_cleaned.csv")
TRAIN_DATA_PATH = os.path.join(SPLIT_DATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(SPLIT_DATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(SPLIT_DATA_DIR, "test.csv")

# ==================== MODEL CONFIGURATION ====================

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TF-IDF Configuration (UNIFIED for all models)
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),  # Unigrams and bigrams
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True
}

# Label Encoder path
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# ==================== ML MODEL CONFIGURATION ====================
ML_MODELS_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "random_state": RANDOM_SEED
    },
    "svm": {
        "C": 1.0,
        "kernel": "linear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "probability": True
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 50,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
}

# ==================== DL MODEL CONFIGURATION ====================

# LSTM Configuration
LSTM_CONFIG = {
    "vocab_size": 10000,
    "embedding_dim": 128,
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "max_sequence_length": 256,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "early_stopping_patience": 3
}

# PhoBERT Configuration
PHOBERT_CONFIG = {
    "model_name": "vinai/phobert-base",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2
}

# ==================== CATEGORY LABELS ====================
CATEGORIES = [
    "Thời trang Nữ",
    "Thời trang Nam", 
    "Điện thoại & Phụ kiện",
    "Laptop & Máy tính",
    "Đồ gia dụng",
    "Thực phẩm & Đồ uống",
    "Mỹ phẩm & Làm đẹp",
    "Thể thao & Du lịch",
    "Giày dép",
    "Nhà cửa & Đời sống",
    "Đồ chơi trẻ em",
    "Sách & Văn phòng phẩm"
]

NUM_CLASSES = len(CATEGORIES)

# ==================== VISUALIZATION CONFIGURATION ====================
PLOT_STYLE = {
    "figure_size": (12, 8),
    "font_size": 12,
    "title_size": 14,
    "cmap": "Blues"
}
