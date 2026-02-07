import streamlit as st
import pickle
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="IA | Planificación Hospitalaria",
    page_icon="🏥",
    layout="wide"
)

# --- CSS PERSONALIZADO (Para darle estilo) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    with open('models/lasso_hospitales.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler_hospitales.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    cols = [
        "0-9 y/o % of total pop", "10-19 y/o % of total pop", "20-29 y/o % of total pop",
        "30-39 y/o % of total pop", "40-49 y/o % of total pop", "50-59 y/o % of total pop",
        "% White-alone", "% Black-alone", "% NA/AI-alone", "% Asian-alone",
        "% Hawaiian/PI-alone", "% Two or more races", "R_death_2018",
        "Percent of adults with less than a high school diploma 2014-18",
        "Percent of adults with a high school diploma only 2014-18",
        "Percent of adults completing some college or associate's degree 2014-18",
        "Percent of adults with a bachelor's degree or higher 2014-18",
        "MEDHHINC_2018", "Employed_2018", "Unemployment_rate_2018",
        "Percent of Population Aged 60+", "anycondition_prevalence", "Urban_rural_code"
    ]
    return model, scaler, cols

model, scaler, model_cols = load_assets()

# --- ENCABEZADO ---
st.title("🏥 Sistema Inteligente de Planificación Hospitalaria")
st.markdown("""
    Este panel utiliza un modelo de regresión **Lasso** para estimar la infraestructura sanitaria necesaria 
    basada en el perfil socio-demográfico de la región.
    ---
""")

# --- INTERFAZ DE USUARIO ---
col_inputs, col_results = st.columns([2, 1], gap="large")

with col_inputs:
    st.subheader("📋 Datos de la Región")
    
    tab1, tab2, tab3 = st.tabs(["👥 Demografía", "🧬 Perfil Social", "🎓 Educación y Empleo"])
    
    inputs = {}
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            inputs["0-9 y/o % of total pop"] = st.slider("% Niños (0-9)", 0.0, 100.0, 12.0)
            inputs["10-19 y/o % of total pop"] = st.slider("% Adolescentes (10-19)", 0.0, 100.0, 13.0)
            inputs["20-29 y/o % of total pop"] = st.slider("% Jóvenes (20-29)", 0.0, 100.0, 14.0)
            inputs["Percent of Population Aged 60+"] = st.slider("% Adulto Mayor (60+)", 0.0, 100.0, 20.0)
        with c2:
            inputs["30-39 y/o % of total pop"] = st.slider("% Adultos (30-39)", 0.0, 100.0, 12.0)
            inputs["40-49 y/o % of total pop"] = st.slider("% Adultos (40-49)", 0.0, 100.0, 12.0)
            inputs["50-59 y/o % of total pop"] = st.slider("% Adultos (50-59)", 0.0, 100.0, 13.0)
            inputs["Urban_rural_code"] = st.selectbox("Tipo de Zona", options=[1,2,3,4,5,6,7,8,9], help="1: Gran Metrópoli, 9: Rural")

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            inputs["% White-alone"] = st.number_input("% White", 0.0, 100.0, 70.0)
            inputs["% Black-alone"] = st.number_input("% Black", 0.0, 100.0, 15.0)
            inputs["% Asian-alone"] = st.number_input("% Asian", 0.0, 100.0, 5.0)
            # Variables fijas para simplificar
            inputs["% NA/AI-alone"] = 1.0
            inputs["% Hawaiian/PI-alone"] = 0.0
            inputs["% Two or more races"] = 9.0
        with c4:
            inputs["anycondition_prevalence"] = st.number_input("Prevalencia Enfermedades (%)", 0.0, 100.0, 30.0)
            inputs["R_death_2018"] = st.number_input("Tasa Mortalidad", 0.0, 50.0, 10.0)

    with tab3:
        inputs["MEDHHINC_2018"] = st.number_input("Ingreso Medio Hogar (USD)", value=50000)
        inputs["Unemployment_rate_2018"] = st.slider("% Desempleo", 0.0, 50.0, 5.0)
        inputs["Employed_2018"] = st.number_input("Total Empleados", value=10000)
        
        # Educación (simplificado)
        inputs["Percent of adults with less than a high school diploma 2014-18"] = st.number_input("% Sin Diploma HS", 0.0, 100.0, 12.0)
        inputs["Percent of adults with a high school diploma only 2014-18"] = 30.0
        inputs["Percent of adults completing some college or associate's degree 2014-18"] = 30.0
        inputs["Percent of adults with a bachelor's degree or higher 2014-18"] = 28.0

# --- LÓGICA DE PREDICCIÓN ---
with col_results:
    st.subheader("🚀 Predicción")
    if st.button("Calcular Necesidad"):
        # Crear DataFrame
        df_input = pd.DataFrame([inputs])[model_cols]
        # Escalar
        df_scaled = scaler.transform(df_input)
        # Predecir
        pred = model.predict(df_scaled)[0]
        # Asegurar que no sea negativo
        final_val = max(0, round(pred, 2))
        
        st.metric(label="Hospitales Recomendados", value=final_val)
        
        if final_val > 5:
            st.warning("⚠️ Alta demanda detectada para esta zona.")
        else:
            st.success("✅ Infraestructura adecuada para el perfil demográfico.")
        
        st.info("💡 Consejo: Los cambios en la población mayor (60+) suelen tener el mayor impacto en este modelo.")