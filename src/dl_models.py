"""
Deep Learning Models Module
Chứa 2 mô hình DL: LSTM và PhoBERT
Sử dụng TF-IDF features (cho LSTM) hoặc tokenizer riêng (cho PhoBERT)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_CONFIG, PHOBERT_CONFIG, DL_MODELS_DIR, NUM_CLASSES, RANDOM_SEED


class BaseDLModel(ABC):
    """Abstract base class cho các DL models"""
    
    def __init__(self, name: str, num_classes: int = NUM_CLASSES):
        self.name = name
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.is_trained = False
        
    @abstractmethod
    def build(self, **kwargs) -> None:
        """Khởi tạo model architecture"""
        pass
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> 'BaseDLModel':
        """Train model"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Dự đoán labels"""
        pass
    
    @abstractmethod
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Dự đoán xác suất"""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Lưu model"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseDLModel':
        """Load model"""
        pass


class LSTMClassifier(BaseDLModel):
    """
    LSTM-based text classifier
    Sử dụng TF-IDF vectors làm input để đảm bảo công bằng khi so sánh với ML models
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__("LSTM", num_classes)
        self.config = LSTM_CONFIG.copy()
        
    def build(
        self,
        input_dim: int,
        **kwargs
    ) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_dim: Số chiều input (TF-IDF features)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Dense, LSTM, Dropout, BatchNormalization,
                Reshape, Bidirectional, Input
            )
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
        
        # Update config với kwargs
        self.config.update(kwargs)
        
        # Set random seed
        tf.random.set_seed(RANDOM_SEED)
        
        # Build model - Dense layers cho TF-IDF input
        self.model = Sequential([
            Input(shape=(input_dim,)),
            
            # Dense layer để giảm chiều và học features
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Reshape để đưa vào LSTM
            Reshape((1, 512)),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=0.1
            )),
            Bidirectional(LSTM(
                self.config['lstm_units'] // 2,
                return_sequences=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=0.1
            )),
            
            # Classification layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{self.name} Model Architecture:")
        self.model.summary()
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'LSTMClassifier':
        """
        Train LSTM model
        
        Args:
            X_train: Training TF-IDF features (dense array)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        # Update config
        self.config.update(kwargs)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        print(f"\n{'='*50}")
        print(f"Training {self.name}...")
        print(f"{'='*50}")
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán labels"""
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất"""
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        return self.model.predict(X, verbose=0)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[list] = None,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """Đánh giá model"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
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
        """Lưu model"""
        if filepath is None:
            os.makedirs(DL_MODELS_DIR, exist_ok=True)
            filepath = os.path.join(DL_MODELS_DIR, "lstm_model")
        
        self.model.save(filepath)
        print(f"LSTM model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMClassifier':
        """Load model"""
        import tensorflow as tf
        
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        instance.is_trained = True
        print(f"LSTM model loaded from {filepath}")
        return instance


class PhoBERTClassifier(BaseDLModel):
    """
    PhoBERT-based text classifier
    Sử dụng PhoBERT pretrained model cho tiếng Việt
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__("PhoBERT", num_classes)
        self.config = PHOBERT_CONFIG.copy()
        self.tokenizer = None
        self.device = None
        
    def build(self, **kwargs) -> None:
        """Build PhoBERT model"""
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                AutoConfig
            )
        except ImportError:
            raise ImportError(
                "transformers and torch are required for PhoBERT. "
                "Install with: pip install transformers torch"
            )
        
        self.config.update(kwargs)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Load model config và model
        model_config = AutoConfig.from_pretrained(
            self.config['model_name'],
            num_labels=self.num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            config=model_config
        )
        
        self.model.to(self.device)
        
        print(f"\n{self.name} Model loaded: {self.config['model_name']}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _tokenize(self, texts: List[str]) -> Dict:
        """Tokenize texts"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
    
    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'PhoBERTClassifier':
        """
        Train PhoBERT model
        
        Args:
            train_texts: List of training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AdamW, get_linear_schedule_with_warmup
        from tqdm import tqdm
        
        self.config.update(kwargs)
        
        print(f"\n{'='*50}")
        print(f"Training {self.name}...")
        print(f"{'='*50}")
        
        # Tokenize
        train_encodings = self._tokenize(train_texts)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels, dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Validation data
        val_loader = None
        if val_texts is not None:
            val_encodings = self._tokenize(val_texts)
            val_dataset = TensorDataset(
                val_encodings['input_ids'],
                val_encodings['attention_mask'],
                torch.tensor(val_labels, dtype=torch.long)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = len(train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch in progress_bar:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        preds = torch.argmax(outputs.logits, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = correct / total
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        self._load_checkpoint()
                        break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        self.history = history
        self.is_trained = True
        return self
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        os.makedirs(DL_MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(DL_MODELS_DIR, "phobert_checkpoint")
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
    
    def _load_checkpoint(self):
        """Load model checkpoint"""
        from transformers import AutoModelForSequenceClassification
        checkpoint_path = os.path.join(DL_MODELS_DIR, "phobert_checkpoint")
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        self.model.to(self.device)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Dự đoán labels"""
        import torch
        
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        
        self.model.eval()
        predictions = []
        
        # Process in batches
        batch_size = self.config['batch_size']
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = self._tokenize(batch_texts)
            
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất"""
        import torch
        import torch.nn.functional as F
        
        if not self.is_trained:
            raise ValueError(f"{self.name} chưa được train!")
        
        self.model.eval()
        probabilities = []
        
        batch_size = self.config['batch_size']
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = self._tokenize(batch_texts)
            
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                proba = F.softmax(outputs.logits, dim=1).cpu().numpy()
                probabilities.extend(proba)
        
        return np.array(probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: np.ndarray,
        class_names: Optional[list] = None,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """Đánh giá model"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        
        y_pred = self.predict(texts)
        
        metrics = {
            'accuracy': accuracy_score(labels, y_pred),
            'precision_macro': precision_score(labels, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(labels, y_pred),
            'predictions': y_pred
        }
        
        print(f"\n{dataset_name} Results for {self.name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        if class_names:
            print(f"\nClassification Report:")
            print(classification_report(labels, y_pred, target_names=class_names, zero_division=0))
        
        return metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Lưu model"""
        if filepath is None:
            os.makedirs(DL_MODELS_DIR, exist_ok=True)
            filepath = os.path.join(DL_MODELS_DIR, "phobert_model")
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        print(f"PhoBERT model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'PhoBERTClassifier':
        """Load model"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        instance = cls()
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        instance.tokenizer = AutoTokenizer.from_pretrained(filepath)
        instance.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        instance.model.to(instance.device)
        instance.is_trained = True
        
        print(f"PhoBERT model loaded from {filepath}")
        return instance


def create_dl_models(
    input_dim: Optional[int] = None,
    num_classes: int = NUM_CLASSES
) -> Dict[str, BaseDLModel]:
    """
    Factory function tạo tất cả DL models
    
    Args:
        input_dim: Số chiều input cho LSTM (TF-IDF features)
        num_classes: Số classes
        
    Returns:
        Dictionary mapping model_name -> model_instance
    """
    models = {}
    
    # LSTM model (cần input_dim)
    if input_dim is not None:
        lstm = LSTMClassifier(num_classes)
        lstm.build(input_dim=input_dim)
        models['lstm'] = lstm
    
    # PhoBERT model
    phobert = PhoBERTClassifier(num_classes)
    phobert.build()
    models['phobert'] = phobert
    
    return models


if __name__ == "__main__":
    # Test LSTM với dữ liệu giả
    print("Testing LSTM...")
    lstm = LSTMClassifier(num_classes=5)
    lstm.build(input_dim=1000)
    
    # Fake data
    X_train = np.random.random((100, 1000)).astype(np.float32)
    y_train = np.random.randint(0, 5, 100)
    
    lstm.fit(X_train, y_train, epochs=2)
    preds = lstm.predict(X_train[:10])
    print(f"LSTM predictions: {preds}")
