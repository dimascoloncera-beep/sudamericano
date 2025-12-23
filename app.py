import os
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Dashboard + Modelo (Streamlit)", layout="wide")


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


def model_predict(model, X_one_row: pd.DataFrame):
    out = {}
    if hasattr(model, "predict"):
        out["predict"] = model.predict(X_one_row)
    if hasattr(model, "predict_proba"):
        out["predict_proba"] = model.predict_proba(X_one_row)
    return out


# ======================================================
# SESSION STATE
# ======================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "model" not in st.session_state:
    st.session_state.model = None
if "view" not in st.session_state:
    st.session_state.view = "data"  # data | plots | train


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Menú")

# Logo fijo
LOGO_PATH = os.path.join("assets", "logo.jpg")
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.warning("No se encontró assets/logo.jpg")

st.sidebar.divider()

# Drag & drop
data_file = st.sidebar.file_uploader("📄 Drag and drop file here", type=["xlsx", "csv"])

# Navegación por vistas
if st.sidebar.button("📄 Ver datos", use_container_width=True):
    st.session_state.view = "data"
if st.sidebar.button("📊 Mostrar gráficas", use_container_width=True):
    st.session_state.view = "plots"
if st.sidebar.button("🤖 Entrenar modelo", use_container_width=True):
    st.session_state.view = "train"

st.sidebar.divider()

# Modelo joblib (opcional)
model_file = st.sidebar.file_uploader("🧠 Cargar modelo (.joblib)", type=["joblib"])
if model_file is not None and joblib is not None:
    try:
        st.session_state.model = joblib.load(model_file)
        st.sidebar.success("✅ Modelo cargado")
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando modelo: {e}")


# ======================================================
# MAIN
# ======================================================
st.title("App Streamlit: Datos, Gráficas y Modelo")

# Cargar archivo (y guardarlo en session_state)
if data_file is not None:
    try:
        df_loaded = read_file_to_df(data_file)
        if df_loaded.empty:
            st.error("❌ El archivo está vacío.")
        else:
            st.session_state.df = df_loaded
            st.session_state.file_name = data_file.name
    except Exception as e:
        st.error(f"❌ Error al cargar archivo: {e}")

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

        st.dataframe(df.head(10), use_container_width=True)


# ======================================================
# VIEW: GRÁFICAS
# ======================================================
if st.session_state.view == "plots":
    if df is None:
        st.warning("Primero carga un archivo.")
    else:
        st.subheader("📊 Gráficas (todas)")

        genero_col = find_col(df, ["Genero", "Género", "Sexo", "genero", "género", "sexo"])
        residencia_col = find_col(df, ["Residencia", "Ciudad_Residencia", "Residencias", "ciudad"])
        canal_col = find_col(df, ["Canal", "Canal_Captacion", "canal"])
        area_pref_col = find_col(df, ["Area_preferida", "Área_preferida", "area_preferida", "área_preferida"])
        modalidad_col = find_col(df, ["Modalidad", "modalidad"])
        carrera1_col = find_col(df, ["Primer_Carrera", "primera_carrera", "primer_carrera"])
        estado_civil_col = find_col(df, ["Estado_civil", "Estado_Civil", "estado civil", "estado_civil"])
        educacion_col = find_col(df, ["Educacion_completada", "Educación_completada", "educacion_completada", "nivel_educativo"])

        colA, colB = st.columns(2)
        with colA:
            if genero_col:
                plot_pie_from_series(f"Género ({genero_col})", df[genero_col])
            else:
                st.info("No encontré columna de género.")
        with colB:
            if estado_civil_col:
                plot_pie_from_series(f"Estado civil ({estado_civil_col})", df[estado_civil_col])
            else:
                st.info("No encontré columna de estado civil.")

        st.divider()

        bars = [
            ("Residencia", residencia_col),
            ("Canal", canal_col),
            ("Área preferida", area_pref_col),
            ("Modalidad", modalidad_col),
            ("Educación completada", educacion_col),
            ("Primer Carrera (conteo)", carrera1_col),
        ]

        left, right = st.columns(2)
        for i, (title, colname) in enumerate(bars):
            with (left if i % 2 == 0 else right):
                if colname:
                    plot_bar_from_series(f"{title} ({colname})", df[colname], top_n=20)
                else:
                    st.info(f"No encontré columna para: {title}")


# ======================================================
# VIEW: ENTRENAR MODELO (FORMULARIO)
# ======================================================
if st.session_state.view == "train":
    if df is None:
        st.warning("Primero carga un archivo.")
    else:
        st.subheader("🤖 Formulario (solo aquí, no muestra tabla)")
        st.caption("Mientras estés aquí, solo verás el formulario y los resultados.")

        with st.expander("⚙️ Columnas de entrada del modelo", expanded=True):
            cols = df.columns.tolist()
            cols_selected = st.multiselect(
                "Selecciona columnas de entrada (NO incluyas el target)",
                options=cols,
                default=cols,
                key="cols_model"
            )

        CURRENT_YEAR = datetime.now().year

        with st.form("form_pred"):
            st.write("### 🧾 Ingreso de datos")
            inputs = {}

            for c in cols_selected:
                s = df[c]
                nunique = s.nunique(dropna=True)

                # =========================
                # NUMÉRICOS
                # =========================
                if pd.api.types.is_numeric_dtype(s):

                    is_year = (
                        "anio" in c.lower()
                        or "año" in c.lower()
                        or "year" in c.lower()
                        or "graduacion" in c.lower()
                        or "graduación" in c.lower()
                    )

                    if is_year:
                        # ✅ RANGO FIJO: 1970 .. AÑO ACTUAL (ENTERO)
                        inputs[c] = st.number_input(
                            c,
                            min_value=1970,
                            max_value=CURRENT_YEAR,
                            value=min(2000, CURRENT_YEAR),
                            step=1,
                            format="%d",
                            key=f"num_{c}"
                        )
                    else:
                        # Otros numéricos: rango basado en datos
                        vals = pd.to_numeric(s, errors="coerce").dropna()
                        vmin = float(vals.min()) if not vals.empty else 0.0
                        vmax = float(vals.max()) if not vals.empty else 1.0
                        default = float(vals.median()) if not vals.empty else 0.0

                        inputs[c] = st.number_input(
                            c,
                            value=default,
                            min_value=vmin,
                            max_value=vmax,
                            step=0.01,
                            key=f"num_{c}"
                        )

                # =========================
                # CATEGÓRICOS
                # =========================
                else:
                    if nunique <= 50:
                        options = ["(vacío)"] + sorted(s.dropna().astype(str).unique().tolist())
                        sel = st.selectbox(c, options=options, key=f"sel_{c}")
                        inputs[c] = None if sel == "(vacío)" else sel
                    else:
                        inputs[c] = st.text_input(c, value="", key=f"txt_{c}")

            submit = st.form_submit_button("🔎 Ejecutar modelo")

        if submit:
            X_one = pd.DataFrame([inputs])
            st.write("### ✅ Datos enviados al modelo")
            st.dataframe(X_one, use_container_width=True)

            if st.session_state.model is None:
                st.warning("Carga primero un modelo .joblib en el sidebar.")
            else:
                try:
                    result = model_predict(st.session_state.model, X_one)
                    st.write("### 📌 Resultados")
                    st.write(result)
                except Exception as e:
                    st.error(f"❌ Error usando el modelo: {e}")
                    st.info("El modelo necesita las mismas columnas/preprocesamiento con las que fue entrenado.")
