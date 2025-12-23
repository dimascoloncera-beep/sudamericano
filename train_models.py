import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# =====================================================
# CONFIG
# =====================================================
BASE_PATH = r"C:\Users\dimas\OneDrive\Escritorio\Streamlit practicas"
FILE = os.path.join(BASE_PATH, "test.csv")

TARGET = "carrera_interes"
DROP_COLS = ["id_registro", "nombre_completo", "fecha_registro"]  # opcional

MODEL_OUTPUT = os.path.join(BASE_PATH, "modelo_final.joblib")
ENCODER_OUTPUT = os.path.join(BASE_PATH, "label_encoder.joblib")
FEATURES_OUTPUT = os.path.join(BASE_PATH, "features_modelo.joblib")

RANDOM_STATE = 42

# =====================================================
# 1) LOAD
# =====================================================
df = pd.read_csv(FILE)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y_text = df[TARGET].astype(str)

# =====================================================
# 2) RELLENAR NULOS (evita errores)
# =====================================================
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

for c in cat_cols:
    X[c] = X[c].fillna("N/A").astype(str)
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

# =====================================================
# 3) ENCODE TARGET
# =====================================================
le = LabelEncoder()
y = le.fit_transform(y_text)

joblib.dump(le, ENCODER_OUTPUT)
joblib.dump(X.columns.tolist(), FEATURES_OUTPUT)

num_class = len(le.classes_)

print("✅ Dataset:", X.shape)
print("✅ Clases:", num_class)
print("Top clases:\n", y_text.value_counts().head(10))

# =====================================================
# 4) SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================================
# 5) PIPELINE (OneHot + XGBoost)
# =====================================================
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=num_class,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("pre", pre),
    ("model", xgb)
])

pipe.fit(X_train, y_train)

pred = pipe.predict(X_test).astype(int)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

print("\n================ XGBOOST ================")
print("Accuracy:", round(acc, 4))
print("F1 macro:", round(f1, 4))
print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))

# =====================================================
# 6) SAVE
# =====================================================
joblib.dump({"model": pipe, "model_type": "xgboost"}, MODEL_OUTPUT)

print("\n✅ Guardado:")
print(" -", MODEL_OUTPUT)
print(" -", ENCODER_OUTPUT)
print(" -", FEATURES_OUTPUT)
