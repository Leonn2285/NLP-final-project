"""
🛒 Product Category Classification App
Ứng dụng phân loại sản phẩm theo danh mục sử dụng Streamlit

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Page config
st.set_page_config(
    page_title="Product Category Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with warm light theme
st.markdown("""
<style>
    /* ============ MAIN BACKGROUND ============ */
    .stApp {
        background-color: #e8dcc8 !important;
    }
    
    /* Header bar (thanh trên cùng) */
    header[data-testid="stHeader"] {
        background-color: #d4c4b0 !important;
        border-bottom: 2px solid #c4b39a !important;
    }
    
    /* Toolbar (deploy button, settings) */
    [data-testid="stToolbar"] {
        background-color: #d4c4b0 !important;
    }
    [data-testid="stToolbar"] button {
        color: #3d3229 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #d4c4b0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #3d3229 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #2c241c !important;
    }
    
    /* ============ TEXT COLORS ============ */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #3d3229 !important;
    }
    
    /* ============ CUSTOM CLASSES ============ */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem;
        color: #2c241c !important;
        background: linear-gradient(135deg, #8b6914 0%, #c4a35a 50%, #8b6914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #5a4a3a !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #c4a35a 0%, #8b6914 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(139, 105, 20, 0.3);
        border: 2px solid #d4b86a;
    }
    .prediction-box h2 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .prediction-box p {
        color: #fff8e7 !important;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .confidence-bar {
        background: #d4c4b0;
        border-radius: 15px;
        padding: 4px;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-fill {
        background: linear-gradient(90deg, #c4a35a 0%, #8b6914 100%);
        border-radius: 12px;
        height: 24px;
        transition: width 0.5s ease;
    }
    
    .category-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    .category-tag:hover {
        transform: scale(1.05);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #f5ede2, #e8dcc8);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(90,74,58,0.15);
        text-align: center;
        border: 1px solid #d4c4b0;
    }
    
    /* ============ STREAMLIT COMPONENTS ============ */
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #f5ede2 !important;
        color: #3d3229 !important;
        border: 2px solid #c4b39a !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #c4a35a !important;
        box-shadow: 0 0 0 2px rgba(196, 163, 90, 0.3) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #8a7a6a !important;
    }
    /* Disabled textarea - fix màu text */
    .stTextArea > div > div > textarea:disabled {
        color: #3d3229 !important;
        background-color: #ebe3d5 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #3d3229 !important;
    }
    
    /* Labels */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #3d3229 !important;
        font-weight: 600 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #f5ede2 !important;
        border: 2px solid #c4b39a !important;
        border-radius: 10px !important;
    }
    .stSelectbox > div > div > div {
        color: #3d3229 !important;
    }
    
    /* Dropdown menu options (khi mở ra) */
    [data-baseweb="popover"] {
        background-color: transparent !important;
    }
    [data-baseweb="popover"] > div {
        background-color: #f5ede2 !important;
        border: 2px solid #c4b39a !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(90, 74, 58, 0.2) !important;
    }
    [role="listbox"] {
        background-color: #f5ede2 !important;
    }
    [role="option"] {
        background-color: #f5ede2 !important;
        color: #3d3229 !important;
        padding: 0.75rem 1rem !important;
    }
    [role="option"]:hover {
        background-color: #e8dcc8 !important;
    }
    [role="option"][aria-selected="true"] {
        background-color: #c4a35a !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #c4a35a 0%, #8b6914 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(139, 105, 20, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 105, 20, 0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #d4c4b0 !important;
        border-radius: 10px !important;
        padding: 5px !important;
        justify-content: center !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5a4a3a !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.75rem 2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c4a35a !important;
        color: white !important;
        font-weight: 800 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f5ede2 !important;
        color: #3d3229 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #f5ede2 !important;
        border: 1px solid #d4c4b0 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #f5ede2 !important;
        border-radius: 10px !important;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #f5ede2 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #c4a35a !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: #f5ede2 !important;
        color: #3d3229 !important;
        border-radius: 10px !important;
    }
    
    /* Markdown */
    .stMarkdown {
        color: #3d3229 !important;
    }
    
    /* Headers in main content */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3 {
        color: #2c241c !important;
        font-weight: 700 !important;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #5a4a3a !important;
        padding: 1.5rem;
        background: linear-gradient(145deg, #d4c4b0, #e8dcc8);
        border-radius: 15px;
        margin-top: 2rem;
    }
    .footer-text p {
        color: #5a4a3a !important;
        margin: 0.3rem 0;
    }
    .footer-text strong {
        color: #8b6914 !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly .bg {
        fill: #f5ede2 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        from src.preprocessing import create_preprocessor
        from src.feature_extraction import load_vectorizer
        from src.ml_models import load_ml_model
        from src.data_utils import load_label_encoder
        from config import DL_MODELS_DIR
        import os
        
        preprocessor = create_preprocessor(use_word_segmentation=False, remove_stopwords=True)
        vectorizer = load_vectorizer()
        label_encoder = load_label_encoder()
        
        models = {}
        # Load ML models
        for model_name in ['logistic_regression', 'svm', 'random_forest']:
            try:
                models[model_name] = load_ml_model(model_name)
            except:
                pass
        
        # LSTM model
        try:
            from src.dl_models import LSTMClassifier
            lstm_path = os.path.join(DL_MODELS_DIR, "lstm_model.keras")
            if os.path.exists(lstm_path):
                models['lstm'] = LSTMClassifier.load(lstm_path)
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
        
        # Load PhoBERT model
        try:
            from src.dl_models import PhoBERTClassifier
            phobert_path = os.path.join(DL_MODELS_DIR, "phobert_model")
            if os.path.exists(phobert_path):
                models['phobert'] = PhoBERTClassifier.load(phobert_path)
        except Exception as e:
            print(f"Could not load PhoBERT model: {e}")
        
        return {
            'preprocessor': preprocessor,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'models': models
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def predict_category(text, model_name, resources):
    """Predict category for input text"""
    # Preprocess
    processed_text = resources['preprocessor'].preprocess(text)
    
    model = resources['models'][model_name]
    
    # Handle different model types
    if model_name == 'lstm':
        # LSTM cần TF-IDF features dạng dense array
        # Suppress TensorFlow verbose output
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        X = resources['vectorizer'].transform([processed_text])
        X_dense = X.toarray().astype('float32')
        # LSTM prediction - có thể chậm lần đầu do TensorFlow warmup
        prediction = model.predict(X_dense)[0]
        probas = model.predict_proba(X_dense)[0]
    elif model_name == 'phobert':
        # PhoBERT cần raw text
        prediction = model.predict([processed_text])[0]
        probas = model.predict_proba([processed_text])[0]
    else:
        # ML models cần TF-IDF sparse matrix
        X = resources['vectorizer'].transform([processed_text])
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
    
    # Get category name
    category = resources['label_encoder'].inverse_transform([prediction])[0]
    
    # All predictions with probabilities
    all_predictions = []
    for i, prob in enumerate(probas):
        cat_name = resources['label_encoder'].classes_[i]
        all_predictions.append({'category': cat_name, 'probability': prob})
    
    all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
    
    return {
        'predicted_category': category,
        'confidence': probas[prediction],
        'all_predictions': all_predictions
    }


def predict_all_models(text, resources):
    """Predict with all available models for comparison"""
    results = {}
    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'svm': 'SVM',
        'random_forest': 'Random Forest',
        'lstm': 'LSTM',
        'phobert': 'PhoBERT'
    }
    
    for model_name in resources['models'].keys():
        try:
            result = predict_category(text, model_name, resources)
            results[model_display_names.get(model_name, model_name)] = result
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
    
    return results


def main():
    # Header
    st.markdown('<h1 class="main-header">Product Category Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân loại sản phẩm tự động theo danh mục sử dụng AI/ML</p>', unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("Đang tải models..."):
        resources = load_models()
    
    if resources is None:
        st.error("⚠️ Không thể tải models. Vui lòng chạy notebook training trước!")
        st.info("""
        **Hướng dẫn:**
        1. Mở notebook `notebooks/model_training.ipynb`
        2. Chạy tất cả các cells để train models
        3. Quay lại đây và refresh trang
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Model selection
        available_models = list(resources['models'].keys())
        model_display_names = {
            'logistic_regression': 'Logistic Regression',
            'svm': 'Support Vector Machine (SVM)',
            'random_forest': 'Random Forest',
            'lstm': 'LSTM (Deep Learning)',
            'phobert': 'PhoBERT (Transformer)'
        }
        
        # Compare mode toggle
        compare_mode = st.checkbox("So sánh tất cả models", value=False)
        
        if not compare_mode:
            selected_model = st.selectbox(
                "Chọn Model:",
                available_models,
                format_func=lambda x: model_display_names.get(x, x)
            )
        else:
            selected_model = None
            st.info(f"Sẽ dự đoán với {len(available_models)} models")
        
        st.markdown("---")
        
        # Categories info
        st.header("Danh mục")
        categories = resources['label_encoder'].classes_
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                  '#F8B500', '#82E0AA']
        
        for i, cat in enumerate(categories):
            color = colors[i % len(colors)]
            st.markdown(f'<span class="category-tag" style="background-color: {color}; color: white;">{cat}</span>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        **Dự án:**
        - 12 danh mục sản phẩm
        - ~8000 sản phẩm training
        - TF-IDF + ML/DL models
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Nhập thông tin sản phẩm")
        
        # Input tabs
        tab1, tab2 = st.tabs(["Nhập text", "Mẫu có sẵn"])
        
        with tab1:
            product_name = st.text_input("Tên sản phẩm:", placeholder="VD: Áo dài truyền thống màu đỏ")
            product_desc = st.text_area(
                "Mô tả sản phẩm:", 
                height=150,
                placeholder="VD: Áo dài truyền thống thiết kế cao cấp, chất liệu lụa mềm mại, phù hợp mặc dịp lễ tết..."
            )
            brand = st.text_input("Thương hiệu:", placeholder="VD: Ngado")
            
            input_text = f"{product_name} {product_desc} {brand}"
        
        with tab2:
            sample_products = {
                "🎀 Thời trang Nữ": "Áo dài truyền thống màu đỏ thêu hoa, chất liệu lụa cao cấp, form dáng chuẩn, phù hợp mặc dịp lễ tết",
                "👔 Thời trang Nam": "Áo sơ mi nam công sở slim fit, vải cotton cao cấp, nhiều màu sắc, form chuẩn Hàn Quốc",
                "📱 Điện thoại": "Điện thoại Samsung Galaxy S24 Ultra, chip Snapdragon mới nhất, camera 200MP, pin 5000mAh",
                "💻 Laptop": "Laptop gaming ASUS ROG Strix, RTX 4090, màn hình 144Hz, RAM 32GB DDR5",
                "💄 Mỹ phẩm": "Son môi MAC chính hãng màu đỏ cherry, lâu trôi cả ngày, dưỡng môi",
                "🏠 Đồ gia dụng": "Nồi chiên không dầu Philips dung tích 6L, công nghệ RapidAir, điều khiển cảm ứng",
                "📚 Sách": "Sách Đắc Nhân Tâm - Dale Carnegie, phiên bản tiếng Việt, bìa cứng cao cấp",
                "🏃 Thể thao": "Giày chạy bộ Nike Air Zoom Pegasus, đệm khí, nhẹ êm, chống trơn trượt"
            }
            
            selected_sample = st.selectbox("Chọn mẫu:", list(sample_products.keys()))
            input_text = sample_products[selected_sample]
            st.text_area("Nội dung mẫu:", input_text, height=100, disabled=True)
        
        # Predict button
        predict_btn = st.button("Phân loại sản phẩm", type="primary", use_container_width=True)
    
    with col2:
        st.header("Kết quả dự đoán")
        
        if predict_btn and input_text.strip():
            if compare_mode:
                # Compare all models
                with st.spinner("Đang phân tích với tất cả models..."):
                    all_results = predict_all_models(input_text, resources)
                
                if all_results:
                    # Summary comparison table
                    st.subheader("So sánh kết quả các Models")
                    
                    comparison_data = []
                    for model_name, result in all_results.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Dự đoán': result['predicted_category'],
                            'Độ tin cậy': result['confidence']
                        })
                    
                    df_compare = pd.DataFrame(comparison_data)
                    df_compare = df_compare.sort_values('Độ tin cậy', ascending=False)
                    
                    # Display as styled cards
                    for idx, row in df_compare.iterrows():
                        conf_color = '#28a745' if row['Độ tin cậy'] > 0.8 else '#ffc107' if row['Độ tin cậy'] > 0.5 else '#dc3545'
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #f5ede2, #e8dcc8); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border-left: 4px solid {conf_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: #2c241c; font-size: 1.1rem;">{row['Model']}</strong>
                                    <p style="margin: 0.3rem 0 0 0; color: #5a4a3a;">🏷️ {row['Dự đoán']}</p>
                                </div>
                                <div style="text-align: right;">
                                    <span style="background: {conf_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: bold;">
                                        {row['Độ tin cậy']:.1%}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Bar chart comparison
                    st.subheader("📈 Biểu đồ so sánh độ tin cậy")
                    
                    fig = go.Figure(go.Bar(
                        x=df_compare['Model'],
                        y=df_compare['Độ tin cậy'],
                        text=[f"{v:.1%}" for v in df_compare['Độ tin cậy']],
                        textposition='auto',
                        marker=dict(
                            color=df_compare['Độ tin cậy'],
                            colorscale='Viridis',
                            showscale=False
                        )
                    ))
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="Model",
                        yaxis_title="Độ tin cậy",
                        yaxis=dict(range=[0, 1], tickformat='.0%')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Check if all models agree
                    predictions = [r['predicted_category'] for r in all_results.values()]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ **Tất cả {len(all_results)} models đều đồng ý:** {predictions[0]}")
                    else:
                        unique_preds = list(set(predictions))
                        st.warning(f"⚠️ **Các models dự đoán khác nhau:** {', '.join(unique_preds)}")
                    
                    # Detailed results for each model
                    with st.expander("📋 Chi tiết từng model"):
                        for model_name, result in all_results.items():
                            st.markdown(f"**{model_name}**")
                            top_3 = result['all_predictions'][:3]
                            for i, pred in enumerate(top_3):
                                st.write(f"  {i+1}. {pred['category']}: {pred['probability']:.2%}")
                            st.markdown("---")
                
            else:
                # Single model prediction
                with st.spinner("Đang phân tích..."):
                    result = predict_category(input_text, selected_model, resources)
                
                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="margin: 0;">🏷️ {result['predicted_category']}</h2>
                    <p style="margin: 0.5rem 0;">Độ tin cậy: <strong>{result['confidence']:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {result['confidence']*100}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Top 5 predictions chart
                st.subheader("Top 5 danh mục có khả năng cao nhất")
                
                top_5 = result['all_predictions'][:5]
                
                fig = go.Figure(go.Bar(
                    x=[p['probability'] for p in top_5],
                    y=[p['category'] for p in top_5],
                    orientation='h',
                    marker=dict(
                        color=[p['probability'] for p in top_5],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"{p['probability']:.1%}" for p in top_5],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Xác suất",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1], tickformat='.0%'),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # All predictions table
                with st.expander("📋 Xem tất cả danh mục"):
                    df = pd.DataFrame(result['all_predictions'])
                    df['probability'] = df['probability'].apply(lambda x: f"{x:.2%}")
                    df.columns = ['Danh mục', 'Xác suất']
                    st.dataframe(df, use_container_width=True)
        
        elif predict_btn:
            st.warning("⚠️ Vui lòng nhập thông tin sản phẩm!")
        else:
            st.info("Nhập thông tin sản phẩm và nhấn 'Phân loại' để xem kết quả")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        <p><strong>NLP Project</strong> - Phân loại sản phẩm theo danh mục</p>
        <p>Models: Logistic Regression, SVM, Random Forest, LSTM, PhoBERT</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
