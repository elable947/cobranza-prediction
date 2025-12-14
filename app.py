import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Cobranza",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main Backgrounds */
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    
    /* Global Text - White/Light for Dark Mode */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, .stHtml, div[data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
    }
    
    h1 {
        font-weight: 700;
        text-align: center;
        padding-bottom: 2rem;
        background: -webkit-linear-gradient(45deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9 !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #6366f1; /* Indigo */
        color: white !important;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
    }
    
    /* Inputs & Cards */
    .css-1r6slb0, div[data-testid="stVerticalBlock"] > div[style*="background-color: white"] {
        background-color: #1e293b !important; /* Slate 800 */
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        border: 1px solid #30363d;
    }
    
    /* Targets our custom inline styled container */
    div[style*="background-color: white"] {
        background-color: #1e293b !important;
        color: #e0e0e0 !important;
    }
    
    /* Slider Customization */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Slider track/tick colors handled by Streamlit theme mostly, but ensuring labels are light */
    .stSlider label {
        color: #e0e0e0 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-card h3, .metric-card p {
        color: #94a3b8 !important; /* Slate 400 */
    }
    .metric-card h1 {
        color: #818cf8 !important; /* Indigo 400 */
    }
    
    /* Results */
    .prediction-success {
        background-color: rgba(6, 95, 70, 0.3);
        border: 1px solid #059669;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .prediction-success h2, .prediction-success p {
        color: #34d399 !important;
    }
    
    .prediction-danger {
        background-color: rgba(153, 27, 27, 0.3);
        border: 1px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .prediction-danger h2, .prediction-danger p {
        color: #fca5a5 !important;
    }
    
    /* Expander/Alerts */
    .stAlert {
        background-color: #1e293b !important;
        color: #e0e0e0 !important;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2704/2704332.png", width=100)
    st.title("Panel de Control")
    st.markdown("---")
    st.markdown("""
    **Modelo:** CNN (1D)
    \n**Versión:** 1.0.0
    \n**Estado:** Producción
    """)
    st.info("Este sistema utiliza Deep Learning para estimar la probabilidad de recuperación de deuda.")

# Main Layout
st.title("🔮 Sistema de Predicción de Cobranza")

# Rutas
DATA_FILE = 'bc_final.csv'
MODEL_FILE = 'best_model_tuned.keras'
SCALER_FILE = 'scaler.pkl'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

df = load_data()

# Validation
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error("⚠️ Archivos de modelo faltantes. Contacte al administrador.")
    st.stop()

try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()

# Feature Extraction
feature_columns = []
if df is not None:
    if 'TARGET_COBRANZA' in df.columns:
        feature_columns = df.drop('TARGET_COBRANZA', axis=1).columns
    else:
        feature_columns = df.columns
else:
    st.warning("⚠️ Dataset base no encontrado. Usando configuración por defecto.")
    st.stop()

# Input Section with Card-like container
st.subheader("📝 Datos del Cliente")
st.markdown("Ingrese las variables para calcular el score de cobranza.")

with st.container():
    st.markdown('<div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
    
    input_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(feature_columns):
        with cols[i % 3]:
            if df is not None:
                # Análisis de tipo de variable
                unique_vals = sorted(df[col].dropna().unique())
                n_unique = len(unique_vals)
                is_int = np.issubdtype(df[col].dtype, np.integer) or (df[col] % 1 == 0).all()
                
                # REGLA 1: Variables Binarias o Categoricas con pocos valores (<= 10)
                if n_unique <= 10:
                    # Si son floats que parecen enteros (ej: 0.0, 1.0), los convertimos a int para visualización
                    display_vals = [int(x) if x.is_integer() else x for x in unique_vals]
                    
                    input_data[col] = st.selectbox(
                        label=f"**{col}**",
                        options=display_vals,
                        index=display_vals.index(int(df[col].mean())) if int(df[col].mean()) in display_vals else 0,
                        help=f"Valores permitidos: {display_vals}"
                    )
                
                # REGLA 2: Variables Discretas (Enteros) con más opciones
                elif is_int:
                    min_val = int(df[col].min())
                    max_val = int(df[col].max())
                    mean_val = int(df[col].mean())
                    
                    input_data[col] = st.slider(
                        label=f"**{col}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=1, # Forzar pasos enteros
                        help=f"Rango entero: [{min_val}, {max_val}]"
                    )
                    
                # REGLA 3: Variables Continuas (Floats)
                else:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    
                    input_data[col] = st.slider(
                        label=f"**{col}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Rango continuo: [{min_val:.2f}, {max_val:.2f}]"
                    )
            
            else:
                # Fallback por defecto si no carga el CSV
                input_data[col] = st.slider(f"**{col}**", 0.0, 1.0, 0.5)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button
col_centered = st.columns([1, 2, 1])
with col_centered[1]:
    predict_btn = st.button("CALCULAR PROBABILIDAD", type="primary")

if predict_btn:
    input_df = pd.DataFrame([input_data])
    
    try:
        with st.spinner('Analizando perfil de riesgo...'):
            input_scaled = scaler.transform(input_df)
            input_cnn = input_scaled.reshape(1, input_scaled.shape[1], 1)
            prediction_proba = model.predict(input_cnn)[0][0]
            prediction_class = int(prediction_proba > 0.5)

        st.markdown("---")
        st.subheader("📊 Resultados del Análisis")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-bottom: 0;">Probabilidad</h3>
                <h1 style="color: #4F46E5; font-size: 3rem;">{prediction_proba:.1%}</h1>
                <p style="color: gray;">Score de Modelo</p>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            if prediction_class == 1:
                st.markdown("""
                <div class="prediction-success">
                    <h2>✅ BUEN PAGADOR</h2>
                    <p>Cliente con alta probabilidad de cumplimiento</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-danger">
                    <h2>⚠️ MAL PAGADOR</h2>
                    <p>Cliente con alto riesgo de incumplimiento</p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")
