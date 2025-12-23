import os
from typing import List, Optional, Dict, Any
from datetime import date

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib


st.set_page_config(page_title="Dashboard + Recomendador (Streamlit)", layout="wide")

# ======================================================
# HELPERS
# ======================================================
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def read_file_to_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Solo se permiten archivos .csv o .xlsx")


def plot_pie_from_series(title: str, s: pd.Series):
    fig, ax = plt.subplots()
    counts = s.value_counts(dropna=False)
    labels = ["N/A" if pd.isna(i) else str(i) for i in counts.index]
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)


def plot_bar_from_series(title: str, s: pd.Series, top_n: int = 20):
    fig, ax = plt.subplots()
    counts = s.value_counts(dropna=False).head(top_n)
    labels = ["N/A" if pd.isna(i) else str(i) for i in counts.index]
    ax.bar(labels, counts.values)
    ax.set_title(title)
    ax.set_ylabel("Cantidad")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    st.pyplot(fig)


@st.cache_resource
def load_artifacts(base_path: str):
    model_path = os.path.join(base_path, "modelo_final.joblib")
    enc_path = os.path.join(base_path, "label_encoder.joblib")
    feat_path = os.path.join(base_path, "features_modelo.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No encuentro: {model_path}")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"No encuentro: {enc_path}")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"No encuentro: {feat_path}")

    best = joblib.load(model_path)
    le = joblib.load(enc_path)
    features = joblib.load(feat_path)

    if not isinstance(best, dict) or "model" not in best or "model_type" not in best:
        raise ValueError("modelo_final.joblib debe ser un dict con keys: 'model' y 'model_type'")

    return best, le, features


def align_and_fill_user_df(user_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Alinea columnas al esquema del modelo y evita error np.isnan en OneHotEncoder
    forzando strings en categóricas.
    """
    # 1) agregar faltantes
    for c in features:
        if c not in user_df.columns:
            user_df[c] = np.nan

    # 2) ordenar/recortar
    user_df = user_df[features].copy()

    # 3) limpiar strings vacíos a NaN
    for c in user_df.columns:
        if user_df[c].dtype == "object":
            user_df[c] = user_df[c].replace("", np.nan)

    # 4) cast robusto
    numeric_hints = ("anio", "año", "edad", "score", "nota", "promedio", "ingreso", "salario", "monto", "precio")

    for c in user_df.columns:
        cl = c.lower()
        if any(h in cl for h in numeric_hints):
            user_df[c] = pd.to_numeric(user_df[c], errors="coerce").fillna(0)
        else:
            user_df[c] = user_df[c].fillna("N/A").astype(str)

    return user_df


def predict_topk(best: Dict[str, Any], le, user_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    model = best["model"]
    proba = model.predict_proba(user_df)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    top_labels = le.inverse_transform(top_idx)

    return pd.DataFrame({
        "Carrera": top_labels,
        "Probabilidad (%)": (proba[top_idx] * 100).round(1)
    })


# ======================================================
# STATE
# ======================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "view" not in st.session_state:
    st.session_state.view = "data"


# ======================================================
# PATHS (FIJOS)
# ======================================================
BASE = r"C:\Users\dimas\OneDrive\Escritorio\Streamlit practicas"
LOGO_PATH = os.path.join(BASE, "assets", "logo.jpg")


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Menú")

if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=220)
else:
    st.sidebar.caption("ℹ️ (Opcional) Pon un logo en assets/logo.jpg")

st.sidebar.divider()

data_file = st.sidebar.file_uploader("📄 Drag and drop file here", type=["xlsx", "csv"])

if st.sidebar.button("📄 Ver datos"):
    st.session_state.view = "data"
if st.sidebar.button("📊 Mostrar gráficas"):
    st.session_state.view = "plots"
if st.sidebar.button("🎓 Recomendador (Top-3)"):
    st.session_state.view = "reco"

st.sidebar.divider()
st.sidebar.caption("Modo recomendado: carga un archivo y usa el recomendador.")


# ======================================================
# MAIN
# ======================================================
st.title("App Streamlit: Datos, Gráficas y Recomendador de Carrera")

# --------- CARGA DEL ARCHIVO ----------
if data_file is not None:
    try:
        df_loaded = read_file_to_df(data_file)
        if df_loaded.empty:
            st.error("El archivo está vacío.")
        else:
            st.session_state.df = df_loaded
            st.session_state.file_name = data_file.name
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")

df = st.session_state.df


# ======================================================
# VIEW: DATOS
# ======================================================
if st.session_state.view == "data":
    if df is None:
        st.info("⬅️ Arrastra un archivo .csv o .xlsx para comenzar.")
    else:
        st.subheader("📌 Datos cargados (vista previa)")
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Columnas", f"{df.shape[1]:,}")
        c3.markdown(f"**Archivo:** {st.session_state.file_name}")
        st.dataframe(df.head(10))


# ======================================================
# VIEW: GRÁFICAS
# ======================================================
if st.session_state.view == "plots":
    if df is None:
        st.warning("Primero carga un archivo.")
    else:
        st.subheader("📊 Gráficas")

        genero_col = find_col(df, ["Genero", "Género", "Sexo", "genero", "sexo"])
        estado_civil_col = find_col(df, ["Estado_Civil", "Estado_civil", "estado civil", "estado_civil"])
        residencia_col = find_col(df, ["Residencia", "Ciudad_Residencia", "ciudad_residencia"])
        canal_col = find_col(df, ["Canal", "Canal_Captacion", "canal_captacion"])
        area_pref_col = find_col(df, ["Area_Preferida", "Area_preferida", "area_preferida"])
        modalidad_col = find_col(df, ["Modalidad", "modalidad"])
        carrera1_col = find_col(df, ["Primer_Carrera", "carrera_interes"])
        graduacion_col = find_col(df, ["Anio_Graduacion", "Año_Graduacion", "Graduacion", "anio_graduacion"])
        educacion_col = find_col(df, ["Educacion_Completada", "Nivel_Educativo", "Educacion_completada", "nivel_educativo"])

        colA, colB = st.columns(2)
        with colA:
            if genero_col:
                plot_pie_from_series("Género", df[genero_col])
        with colB:
            if estado_civil_col:
                plot_pie_from_series("Estado civil", df[estado_civil_col])

        st.divider()

        bars = [
            ("Residencia", residencia_col),
            ("Canal", canal_col),
            ("Área preferida", area_pref_col),
            ("Modalidad", modalidad_col),
            ("Carrera", carrera1_col),
            ("Año de graduación", graduacion_col),
            ("Educación completada", educacion_col),
        ]

        left, right = st.columns(2)
        for i, (title, colname) in enumerate(bars):
            with (left if i % 2 == 0 else right):
                if colname:
                    plot_bar_from_series(title, df[colname])


# ======================================================
# VIEW: RECOMENDADOR (TOP-3)
# ======================================================
if st.session_state.view == "reco":
    st.subheader("🎓 Recomendador de Carrera (Top-3)")

    try:
        best, le, features = load_artifacts(BASE)
    except Exception as e:
        st.error(f"No puedo cargar el modelo/artefactos: {e}")
        st.info(
            "Asegúrate de haber generado estos archivos en la carpeta BASE:\n"
            "- modelo_final.joblib\n"
            "- label_encoder.joblib\n"
            "- features_modelo.joblib"
        )
        st.stop()

    st.caption(f"Modelo activo: **{best['model_type']}**")

    tab1, tab2 = st.tabs(["🧾 Formulario (1 persona)", "📄 Predecir para archivo"])

    # =========================
    # TAB 1: FORMULARIO CONTROLADO
    # =========================
    with tab1:
        st.markdown("Completa campos clave (los demás puedes dejarlos en blanco).")

        CANAL_CAPTACION_OPTS = [
            "",
            "Facebook",
            "Referido familiar",
            "Recomendación amigo",
            "Búsqueda web",
            "Instagram",
            "Evento educativo",
            "WhatsApp de asesor",
            "Sitio web institucional",
            "Visita presencial",
        ]

        SITUACION_LABORAL_OPTS = [
            "",
            "Desempleada",
            "Empleado",
            "Desempleado",
            "Estudiante",
            "Empleada",
        ]

        NIVEL_EDUCATIVO_OPTS = [
            "",
            "Bachillerato",
            "Bachillerato Técnico",
            "Licenciatura",
        ]

        GENERO_OPTS = ["", "M", "F"]
        current_year = date.today().year

        HIDE_COLS = {"estado_lead", "asesor_asignado"}

        inputs = {}
        cols = st.columns(3)

        visible_features = [c for c in features if c not in HIDE_COLS]

        for i, c in enumerate(visible_features):
            with cols[i % 3]:
                cl = c.lower()

                if cl == "genero":
                    inputs[c] = st.selectbox(c, GENERO_OPTS, index=0)

                elif cl in ("anio_graduacion", "año_graduacion", "aniograduacion"):
                    inputs[c] = st.number_input(
                        c,
                        min_value=1970,
                        max_value=current_year,
                        value=current_year,
                        step=1,
                        format="%d"
                    )

                elif cl == "canal_captacion":
                    inputs[c] = st.selectbox(c, CANAL_CAPTACION_OPTS, index=0)

                elif cl == "situacion_laboral":
                    inputs[c] = st.selectbox(c, SITUACION_LABORAL_OPTS, index=0)

                elif cl == "nivel_educativo":
                    inputs[c] = st.selectbox(c, NIVEL_EDUCATIVO_OPTS, index=0)

                else:
                    # Default: texto
                    inputs[c] = st.text_input(c, value="")

        if st.button("🔍 Recomendar (Top-3)"):
            user = pd.DataFrame([inputs])
            user = align_and_fill_user_df(user, features)

            out = predict_topk(best, le, user, k=3)
            st.success("✅ Recomendación generada")
            st.dataframe(out)

    # =========================
    # TAB 2: PREDICCIÓN POR ARCHIVO
    # =========================
    with tab2:
        if df is None:
            st.info("Primero carga un archivo en el sidebar para predecir por lote.")
        else:
            st.markdown("Esto predice Top-1 para cada fila del archivo cargado (sin entrenar, solo inferencia).")

            df_work = df.copy()
            df_work = align_and_fill_user_df(df_work, features)

            if st.button("⚡ Predecir Top-1 para todo el archivo"):
                model = best["model"]
                proba = model.predict_proba(df_work)
                pred_idx = np.argmax(proba, axis=1)
                pred_label = le.inverse_transform(pred_idx)

                results = df.copy()
                results["Prediccion_Carrera"] = pred_label
                results["Confianza_%"] = (np.max(proba, axis=1) * 100).round(1)

                st.success("✅ Predicciones listas")
                st.dataframe(results.head(50))
                st.caption("Mostrando primeras 50 filas (puedes descargar si quieres).")

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar predicciones (CSV)",
                    data=csv,
                    file_name="predicciones_carreras.csv",
                    mime="text/csv",
                )
