import streamlit as st
import pandas as pd
import joblib
import requests
import os

MODEL_URL = "https://huggingface.co/mp28/ecotype-forest-cover-classifier/resolve/main/final_pipeline.pkl"
FEATURES_URL = "https://huggingface.co/mp28/ecotype-forest-cover-classifier/resolve/main/model_features.pkl"
CLASS_MAP_URL = "https://huggingface.co/mp28/ecotype-forest-cover-classifier/resolve/main/class_map.pkl"

@st.cache_resource
def download_and_load():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("models/final_pipeline.pkl"):
        with open("models/final_pipeline.pkl", "wb") as f:
            f.write(requests.get(MODEL_URL).content)

    if not os.path.exists("models/model_features.pkl"):
        with open("models/model_features.pkl", "wb") as f:
            f.write(requests.get(FEATURES_URL).content)

    if not os.path.exists("models/class_map.pkl"):
        with open("models/class_map.pkl", "wb") as f:
            f.write(requests.get(CLASS_MAP_URL).content)

    pipeline = joblib.load("models/final_pipeline.pkl")
    features = joblib.load("models/model_features.pkl")
    class_map = joblib.load("models/class_map.pkl")

    return pipeline, features, class_map


pipeline, features, class_map = download_and_load()

st.title("🌲 Forest Cover Type Prediction")

soil_cols = [c for c in features if c.startswith("Soil_Type")]
wild_cols = [c for c in features if c.startswith("Wilderness_Area")]
num_cols = [c for c in features if c not in soil_cols + wild_cols]

st.sidebar.header("Input Features")

inputs = {}
for col in num_cols:
    inputs[col] = st.sidebar.slider(col, 0.0, 5000.0, 0.0)

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
    })

    st.bar_chart(prob_df.set_index("Cover Type"))


