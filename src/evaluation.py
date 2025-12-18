"""
Evaluation and Visualization Module
Đánh giá và visualize kết quả của các models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIS_DIR, PLOT_STYLE


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tính các metrics đánh giá
    
    Args:
        y_true: Labels thực
        y_pred: Labels dự đoán
        class_names: Tên các class
        
    Returns:
        Dictionary chứa các metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Vẽ confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Tên các class
        title: Tiêu đề
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        normalize: Chuẩn hóa theo hàng
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro'],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    So sánh performance của các models
    
    Args:
        results: Dictionary chứa metrics của mỗi model
        metrics: Danh sách metrics cần so sánh
        title: Tiêu đề
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        
    Returns:
        Figure object
    """
    # Prepare data
    model_names = list(results.keys())
    data = {metric: [] for metric in metrics}
    
    for model_name in model_names:
        for metric in metrics:
            value = results[model_name].get(metric, 0)
            data[metric].append(value)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=model_names)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2) * width
        bars = ax1.bar(x + offset, df[metric], width, label=metric.replace('_', ' ').title())
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'{title} - Bar Chart', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Heatmap
    ax2 = axes[1]
    sns.heatmap(
        df.T,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        ax=ax2,
        cbar_kws={'shrink': 0.8},
        linewidths=0.5
    )
    ax2.set_title(f'{title} - Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    return fig


def plot_per_class_metrics(
    results: Dict[str, Dict[str, Any]],
    class_names: List[str],
    metric: str = 'f1_per_class',
    title: str = "F1 Score per Class",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vẽ metrics theo từng class cho các models
    
    Args:
        results: Dictionary chứa metrics
        class_names: Tên các class
        metric: Metric cần vẽ ('f1_per_class', 'precision_per_class', 'recall_per_class')
        title: Tiêu đề
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    model_names = list(results.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        values = results[model_name].get(metric, np.zeros(len(class_names)))
        offset = (i - len(model_names) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics chart saved to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vẽ training history cho DL models
    
    Args:
        history: Dictionary chứa loss và accuracy history
        title: Tiêu đề
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    ax1 = axes[0]
    if 'loss' in history:
        ax1.plot(history['loss'], label='Training Loss', marker='o')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Training Loss', marker='o')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = axes[1]
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy', marker='o')
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    return fig


def create_results_summary(
    results: Dict[str, Dict[str, Any]],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Tạo bảng tổng hợp kết quả
    
    Args:
        results: Dictionary chứa metrics của các models
        class_names: Tên các class
        
    Returns:
        DataFrame tổng hợp
    """
    summary_data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'F1 (Macro)': metrics.get('f1_macro', 0),
            'F1 (Weighted)': metrics.get('f1_weighted', 0),
            'Precision (Macro)': metrics.get('precision_macro', 0),
            'Recall (Macro)': metrics.get('recall_macro', 0),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('F1 (Macro)', ascending=False).reset_index(drop=True)
    
    return df


def save_all_visualizations(
    results: Dict[str, Dict[str, Any]],
    class_names: List[str],
    output_dir: str = VIS_DIR
) -> None:
    """
    Lưu tất cả visualizations
    
    Args:
        results: Dictionary chứa metrics
        class_names: Tên các class
        output_dir: Thư mục lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Model comparison
    plot_model_comparison(
        results,
        title="Model Performance Comparison",
        save_path=os.path.join(output_dir, "model_comparison.png")
    )
    plt.close()
    
    # 2. Per-class F1 scores
    plot_per_class_metrics(
        results,
        class_names,
        metric='f1_per_class',
        title="F1 Score per Category",
        save_path=os.path.join(output_dir, "f1_per_class.png")
    )
    plt.close()
    
    # 3. Confusion matrices for each model
    for model_name, metrics in results.items():
        if 'confusion_matrix' in metrics:
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                class_names,
                title=f"Confusion Matrix - {model_name}",
                save_path=os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
            )
            plt.close()
    
    # 4. Summary table
    summary_df = create_results_summary(results, class_names)
    summary_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    print(f"\nResults summary saved to {os.path.join(output_dir, 'results_summary.csv')}")
    
    print(f"\nAll visualizations saved to {output_dir}")


def print_final_report(
    results: Dict[str, Dict[str, Any]],
    class_names: List[str]
) -> None:
    """
    In báo cáo tổng hợp cuối cùng
    
    Args:
        results: Dictionary chứa metrics
        class_names: Tên các class
    """
    print("\n" + "="*80)
    print("                        FINAL EVALUATION REPORT")
    print("="*80)
    
    # Summary table
    summary_df = create_results_summary(results, class_names)
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    # Best model
    best_model = summary_df.iloc[0]['Model']
    best_f1 = summary_df.iloc[0]['F1 (Macro)']
    
    print("\n" + "-"*80)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   F1 Score (Macro): {best_f1:.4f}")
    print(f"   Accuracy: {summary_df.iloc[0]['Accuracy']:.4f}")
    
    # Per-class analysis
    print("\n📈 PER-CLASS F1 SCORES (Best Model):")
    print("-" * 80)
    
    if 'f1_per_class' in results[best_model]:
        f1_scores = results[best_model]['f1_per_class']
        for i, (class_name, f1) in enumerate(zip(class_names, f1_scores)):
            bar = "█" * int(f1 * 20)
            print(f"  {class_name:<25} | {bar:<20} | {f1:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test với dữ liệu giả
    class_names = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    # Fake results
    results = {
        'Logistic Regression': {
            'accuracy': 0.85,
            'f1_macro': 0.84,
            'f1_weighted': 0.85,
            'precision_macro': 0.83,
            'recall_macro': 0.85,
            'f1_per_class': np.random.random(5) * 0.3 + 0.7,
            'confusion_matrix': np.random.randint(0, 100, (5, 5))
        },
        'SVM': {
            'accuracy': 0.87,
            'f1_macro': 0.86,
            'f1_weighted': 0.87,
            'precision_macro': 0.85,
            'recall_macro': 0.86,
            'f1_per_class': np.random.random(5) * 0.3 + 0.7,
            'confusion_matrix': np.random.randint(0, 100, (5, 5))
        },
        'Random Forest': {
            'accuracy': 0.82,
            'f1_macro': 0.81,
            'f1_weighted': 0.82,
            'precision_macro': 0.80,
            'recall_macro': 0.81,
            'f1_per_class': np.random.random(5) * 0.3 + 0.7,
            'confusion_matrix': np.random.randint(0, 100, (5, 5))
        }
    }
    
    print_final_report(results, class_names)
