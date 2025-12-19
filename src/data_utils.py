"""
Data Utilities Module
Các hàm tiện ích để load, split và xử lý dữ liệu
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLEANED_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SPLIT_DATA_DIR, LABEL_ENCODER_PATH, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from src.preprocessing import VietnameseTextPreprocessor, combine_text_features


def load_data(filepath: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Load dữ liệu từ file CSV
    
    Args:
        filepath: Đường dẫn đến file CSV
        
    Returns:
        DataFrame chứa dữ liệu
    """
    # Đọc file với delimiter ; và xử lý multiline text
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', 
                     on_bad_lines='skip', engine='python')
    
    print(f"Loaded {len(df)} samples from {filepath}")
    print(f"Columns: {df.columns.tolist()}")
    if 'category' in df.columns:
        print(f"Categories: {df['category'].nunique()}")
        print(f"Category distribution:\n{df['category'].value_counts()}")
    return df


def prepare_text_data(
    df: pd.DataFrame,
    text_columns: list = ['product_name', 'description', 'brand'],
    preprocess: bool = True,
    use_word_segmentation: bool = False,
    remove_stopwords: bool = True
) -> pd.DataFrame:
    """
    Chuẩn bị dữ liệu text: kết hợp các cột và tiền xử lý
    
    Args:
        df: DataFrame gốc
        text_columns: Các cột text cần kết hợp
        preprocess: Có tiền xử lý không
        use_word_segmentation: Có tách từ tiếng Việt không
        remove_stopwords: Có loại bỏ stopwords không
        
    Returns:
        DataFrame với cột 'text' đã được xử lý
    """
    df = df.copy()
    
    # Kết hợp các cột text
    df['text'] = combine_text_features(df, columns=text_columns, separator=' ')
    
    # Tiền xử lý nếu cần
    if preprocess:
        preprocessor = VietnameseTextPreprocessor(
            use_word_segmentation=use_word_segmentation,
            remove_stopwords=remove_stopwords
        )
        print("Preprocessing text data...")
        df['text'] = preprocessor.preprocess_batch(df['text'].tolist(), show_progress=True)
    
    # Loại bỏ các dòng có text rỗng sau xử lý
    empty_mask = df['text'].str.strip() == ''
    if empty_mask.sum() > 0:
        print(f"Removing {empty_mask.sum()} empty text rows")
        df = df[~empty_mask].reset_index(drop=True)
    
    return df


def encode_labels(
    df: pd.DataFrame,
    label_column: str = 'category',
    encoder: Optional[LabelEncoder] = None,
    save_encoder: bool = True
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Mã hóa nhãn thành số
    
    Args:
        df: DataFrame chứa dữ liệu
        label_column: Tên cột nhãn
        encoder: LabelEncoder đã fit (nếu có)
        save_encoder: Có lưu encoder không
        
    Returns:
        Tuple (labels được mã hóa, encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(df[label_column])
        
        if save_encoder:
            os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
            joblib.dump(encoder, LABEL_ENCODER_PATH)
            print(f"Label encoder saved to {LABEL_ENCODER_PATH}")
    else:
        labels = encoder.transform(df[label_column])
    
    print(f"Number of classes: {len(encoder.classes_)}")
    print(f"Classes: {list(encoder.classes_)}")
    
    return labels, encoder


def load_label_encoder() -> LabelEncoder:
    """Load label encoder đã lưu"""
    return joblib.load(LABEL_ENCODER_PATH)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratify_column: str = 'category',
    random_state: int = RANDOM_SEED,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu thành train/val/test với stratified sampling
    
    Args:
        df: DataFrame cần chia
        train_ratio: Tỷ lệ tập train
        val_ratio: Tỷ lệ tập validation  
        test_ratio: Tỷ lệ tập test
        stratify_column: Cột dùng để stratify
        random_state: Random seed
        save: Có lưu các tập không
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Split train và temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[stratify_column],
        random_state=random_state
    )
    
    # Split temp thành val và test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df[stratify_column],
        random_state=random_state
    )
    
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save nếu cần
    if save:
        os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
        train_df.to_csv(TRAIN_DATA_PATH, index=False, encoding='utf-8')
        val_df.to_csv(VAL_DATA_PATH, index=False, encoding='utf-8')
        test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8')
        print(f"Split data saved to {SPLIT_DATA_DIR}")
    
    return train_df, val_df, test_df


def load_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dữ liệu đã split"""
    train_df = pd.read_csv(TRAIN_DATA_PATH, encoding='utf-8')
    val_df = pd.read_csv(VAL_DATA_PATH, encoding='utf-8')
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='utf-8')
    
    print(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Tính class weights cho imbalanced data
    
    Args:
        labels: Mảng nhãn
        
    Returns:
        Dictionary mapping class -> weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return dict(zip(classes, weights))


if __name__ == "__main__":
    # Test data utilities
    df = load_data()
    df = prepare_text_data(df, preprocess=True, use_word_segmentation=False)
    labels, encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
