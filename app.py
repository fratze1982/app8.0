import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen mit Fehlerbehandlung ---
try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", on_bad_lines='skip')
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("🧾 Gefundene Spalten:", df.columns.tolist())

# --- Flexible Auswahl der Rohstoff- und Zielspalten ---
alle_spalten = df.columns.tolist()
vorgeschlagene_rohstoffe = alle_spalten[:6]
vorgeschlagene_zielgroessen = alle_spalten[6:]

st.subheader("🔧 Spaltenauswahl")
rohstoff_spalten = st.multiselect(
    "🧪 Wähle die Rohstoffspalten (Einflussgrößen)", 
    options=alle_spalten,
    default=vorgeschlagene_rohstoffe
)

zielspalten_options = [s for s in alle_spalten if s not in rohstoff_spalten]
default_zielspalten = [s for s in vorgeschlagene_zielgroessen if s in zielspalten_options]

zielspalten = st.multiselect(
    "🎯 Wähle die Zielgrößen (Kennwerte)", 
    options=zielspalten_options,
    default=default_zielspalten
)

if not rohstoff_spalten or not zielspalten:
    st.error("Bitte sowohl Rohstoff- als auch Zielspalten auswählen.")
    st.stop()

# --- Eingabe- und Zielvariablen trennen ---
X = df[rohstoff_spalten].copy()
y = df[zielspalten].copy()

# Spaltentypen bestimmen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Fehlende Werte bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("🔧 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))
