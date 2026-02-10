import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

MODEL_URL = "https://huggingface.co/mp28/ecotype-forest-cover-classifier/resolve/main/final_pipeline.pkl"
DATA_URL = "https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset/resolve/main/final_preprocessed_data.csv"

os.makedirs("models", exist_ok=True)

def download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

@st.cache_resource
def load_assets():
    download(MODEL_URL, "models/final_pipeline.pkl")
    download(DATA_URL, "models/final_preprocessed_data.csv")

    pipeline = joblib.load("models/final_pipeline.pkl")
    df = pd.read_csv("models/final_preprocessed_data.csv")

    X = df.drop(columns=["Cover_Type"])
    class_map = {
        0: "Aspen",
        1: "Cottonwood/Willow",
        2: "Douglas-fir",
        3: "Krummholz",
        4: "Lodgepole Pine",
        5: "Ponderosa Pine",
        6: "Spruce/Fir"
    }

    return pipeline, X.columns.tolist(), class_map, X

pipeline, features, class_map, df_sample = load_assets()

st.title("🌲 Forest Cover Type Prediction")

st.sidebar.header("Input Features")

inputs = {}
for col in features:
    lo = float(df_sample[col].min())
    hi = float(df_sample[col].max())
    inputs[col] = st.sidebar.slider(col, lo, hi, float(df_sample[col].mean()))

if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([inputs])[features]
    probs = pipeline.predict_proba(X_input)[0]
    pred = probs.argmax()

    st.success(f"🌿 Predicted Forest Cover Type: **{class_map[pred]}**")
    st.subheader("Prediction Probabilities")
    st.bar_chart(pd.Series(probs, index=[class_map[i] for i in range(len(probs))]))
