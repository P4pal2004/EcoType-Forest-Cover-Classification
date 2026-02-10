import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/final_pipeline.pkl")
    features = joblib.load("models/model_features.pkl")
    class_map = joblib.load("models/class_map.pkl")
    fmin = joblib.load("models/feature_min.pkl")
    fmax = joblib.load("models/feature_max.pkl")
    return pipeline, features, class_map, fmin, fmax

pipeline, features, class_map, fmin, fmax = load_artifacts()

st.title("🌲 Forest Cover Type Prediction")

soil_cols = [c for c in features if c.startswith("Soil_Type")]
wild_cols = [c for c in features if c.startswith("Wilderness_Area")]
num_cols = [c for c in features if c not in soil_cols + wild_cols]

st.sidebar.header("Input Features")

inputs = {}
for col in num_cols:
    lo = float(fmin[col])
    hi = float(fmax[col])
    mid = float((lo + hi) / 2)
    inputs[col] = st.sidebar.slider(col, lo, hi, mid)

soil = st.sidebar.selectbox("Soil Type", soil_cols)
wild = st.sidebar.selectbox("Wilderness Area", wild_cols)

if st.sidebar.button("Predict"):
    row = {c: 0 for c in features}
    for k, v in inputs.items():
        row[k] = v
    row[soil] = 1
    row[wild] = 1

    X = pd.DataFrame([row])[features]

    pred = pipeline.predict(X)[0]
    probs = pipeline.predict_proba(X)[0]

    st.success(f"🌿 Predicted Forest Cover Type: **{class_map[pred]}**")

    prob_df = pd.DataFrame({
        "Cover Type": [class_map[i] for i in range(len(probs))],
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Cover Type"))
