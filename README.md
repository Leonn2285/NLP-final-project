# 🛒 Product Category Classification

## Đề tài: Phân loại sản phẩm theo danh mục (Text Classification)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

## Tổng quan

Dự án xây dựng hệ thống phân loại tự động sản phẩm vào 12 danh mục dựa trên tên, mô tả và thương hiệu sản phẩm. Sử dụng kết hợp các phương pháp Machine Learning và Deep Learning.

### Mục tiêu
- Phân loại chính xác sản phẩm vào 12 danh mục
- So sánh hiệu quả giữa ML và DL models
- Xây dựng ứng dụng demo thực tế

### Danh mục sản phẩm (12 categories)
1. Thời trang Nữ
2. Thời trang Nam
3. Điện thoại & Phụ kiện
4. Laptop & Máy tính
5. Đồ gia dụng
6. Thực phẩm & Đồ uống
7. Mỹ phẩm & Làm đẹp
8. Thể thao & Du lịch
9. Giày dép
10. Nhà cửa & Đời sống
11. Đồ chơi trẻ em
12. Sách & Văn phòng phẩm

## Cấu trúc thư mục

```
NLP/
├── app/
│   └── streamlit_app.py      # Ứng dụng web demo
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   ├── processed/            # Dữ liệu đã xử lý
│   └── split/                # Dữ liệu train/val/test
├── models/
│   ├── ml/                   # ML models (LR, SVM, RF)
│   └── dl/                   # DL models (LSTM, PhoBERT)
├── notebooks/
│   ├── data_cleaning.ipynb   # Notebook làm sạch dữ liệu
│   ├── EDA.ipynb            # Phân tích khám phá dữ liệu
│   └── model_training.ipynb # Training và so sánh models
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Tiền xử lý văn bản tiếng Việt
│   ├── data_utils.py        # Utilities xử lý dữ liệu
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── ml_models.py         # Machine Learning models
│   ├── dl_models.py         # Deep Learning models
│   └── evaluation.py        # Đánh giá và visualization
├── visualizations/           # Biểu đồ và kết quả
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── README.md
```

## Cài đặt

### 1. Clone repository
```bash
cd /Users/leonnn/Downloads/NLP
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc: venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cài đặt underthesea (Vietnamese NLP - optional)
```bash
pip install underthesea
```

## Hướng dẫn sử dụng

### 1. Chạy Data Cleaning và EDA (đã hoàn thành)
Mở và chạy các notebooks trong thư mục `notebooks/`:
- `data_cleaning.ipynb` - Làm sạch dữ liệu
- `EDA.ipynb` - Phân tích khám phá

### 2. Training Models
Mở và chạy notebook `notebooks/model_training.ipynb`:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### 3. Chạy Ứng dụng Demo
```bash
streamlit run app/streamlit_app.py
```
Truy cập: http://localhost:8501

## Models

### Machine Learning (3 models)
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier với regularization |
| **SVM** | Support Vector Machine với kernel linear |
| **Random Forest** | Ensemble của 200 decision trees |

### Deep Learning (2 models)
| Model | Description |
|-------|-------------|
| **LSTM** | Bidirectional LSTM với TF-IDF input |
| **PhoBERT** | Vietnamese BERT pretrained model |

### Vectorization
- **TF-IDF** với 10,000 features
- N-gram range: (1, 2) - unigrams và bigrams
- Áp dụng thống nhất cho tất cả models

## Kết quả dự kiến

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| Logistic Regression | ~85% | ~84% | ~85% |
| SVM | ~87% | ~86% | ~87% |
| Random Forest | ~82% | ~81% | ~82% |
| LSTM | ~83% | ~82% | ~83% |
| PhoBERT | ~90% | ~89% | ~90% |

*Kết quả thực tế sẽ được cập nhật sau khi train*

## Configuration

Các thông số cấu hình trong `config.py`:

```python
# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TF-IDF
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95
}

# LSTM
LSTM_CONFIG = {
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "epochs": 20,
    "batch_size": 32
}
```

## Visualizations

Sau khi train, các biểu đồ sẽ được lưu trong `visualizations/`:
- `model_comparison.png` - So sánh performance các models
- `f1_per_class.png` - F1 score theo từng category
- `confusion_matrix_*.png` - Ma trận nhầm lẫn

## Demo App

Ứng dụng Streamlit cho phép:
- Nhập thông tin sản phẩm (tên, mô tả, thương hiệu)
- Chọn model để phân loại
- Xem kết quả dự đoán với confidence score
- Hiển thị top 5 categories có khả năng cao nhất

## Quy trình thực hiện

1. **Thu thập dữ liệu** 
   - Crawl từ Tiki
   - 4 files dữ liệu gốc

2. **Phân tích EDA** 
   - Phân bố categories
   - Độ dài text
   - Word frequency

3. **Xử lý dữ liệu** 
   - Làm sạch text tiếng Việt
   - Loại bỏ stopwords
   - Chuẩn hóa Unicode

4. **Feature Engineering** 
   - TF-IDF vectorization
   - Text combination

5. **Model Training** 
   - 3 ML models
   - 2 DL models

6. **Đánh giá** 
   - Accuracy, F1, Precision, Recall
   - Confusion matrix
   - Per-class analysis

7. **Ứng dụng** 
   - Streamlit web app
   - Real-time prediction

## Thành viên nhóm
- Bảo Châu
- Duy Thái
- Minh Huy  
- Quốc Trung

## License
MIT License
---
**Lưu ý:** Chạy `model_training.ipynb` trước khi sử dụng ứng dụng demo!
