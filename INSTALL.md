# 🚀 Hướng dẫn cài đặt nhanh

## Đã cài đặt thành công! ✅

### Packages đã cài (Essential):
- ✅ pandas, numpy - Data processing
- ✅ scikit-learn - Machine Learning
- ✅ matplotlib, seaborn, plotly - Visualization
- ✅ streamlit - Web UI
- ✅ beautifulsoup4, lxml - HTML parsing
- ✅ jupyter, ipykernel - Notebooks
- ✅ underthesea - Vietnamese NLP

### Packages Optional (chưa cài):
- ⏸️ TensorFlow (cho LSTM) - Nặng, cài khi cần
- ⏸️ PyTorch + Transformers (cho PhoBERT) - Nặng, cài khi cần

## Tiếp theo làm gì?

### 1. Chạy notebook training (Cách 1 - Chỉ ML Models)
```bash
cd /Users/leonnn/Downloads/NLP
jupyter notebook notebooks/model_training.ipynb
```
**Trong notebook:**
- Chạy tất cả cells TRỪ phần LSTM và PhoBERT
- Models sẽ được train: Logistic Regression, SVM, Random Forest
- Thời gian: ~5-10 phút

### 2. Sau khi train xong, chạy Streamlit app
```bash
streamlit run app/streamlit_app.py
```
Mở browser tại: http://localhost:8501

### 3. Cài TensorFlow/PyTorch (Optional - nếu muốn train DL models)

**Cài TensorFlow (cho LSTM):**
```bash
pip install tensorflow
```

**Cài PyTorch + Transformers (cho PhoBERT):**
```bash
pip install torch transformers accelerate
```

## Lưu ý quan trọng

⚠️ **Phải chạy notebook training trước** để tạo các file models, nếu không Streamlit app sẽ báo lỗi!

✅ **Chỉ cần ML models** (LR, SVM, RF) là đủ để demo, không nhất thiết phải có DL models.

## Troubleshooting

### Lỗi khi import module
```bash
# Kiểm tra Python version
python --version  # Nên là 3.8 - 3.12

# Cài lại package bị lỗi
pip install --upgrade <package_name>
```

### Streamlit không chạy
```bash
# Kiểm tra Streamlit đã cài chưa
streamlit version

# Cài lại Streamlit
pip install --upgrade streamlit
```

### underthesea lỗi
```bash
# Có thể skip underthesea nếu lỗi
# Trong code, set use_word_segmentation=False
```

## Demo nhanh không cần training

Nếu muốn test code mà chưa train models, chạy Python script đơn giản:

```python
# test_preprocessing.py
from src.preprocessing import create_preprocessor

preprocessor = create_preprocessor(use_word_segmentation=False)
text = "Áo dài truyền thống màu đỏ, chất liệu lụa cao cấp"
processed = preprocessor.preprocess(text)
print(f"Original: {text}")
print(f"Processed: {processed}")
```

Chúc bạn thành công! 🎉
