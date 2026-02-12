import streamlit as st
import pandas as pd
import joblib
import os
import requests

MODEL_URL = "https://huggingface.co/mp28/ecotype-forest-cover-classifier/resolve/main/final_pipeline_v2.pkl"
DATA_URL = "https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset/resolve/main/final_preprocessed_data.csv"

MODEL_PATH = "models/final_pipeline_v2.pkl"
DATA_PATH = "models/final_preprocessed_data.csv"

os.makedirs("models", exist_ok=True)


def download(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)


@st.cache_resource
def load_assets():
    download(MODEL_URL, MODEL_PATH)
    download(DATA_URL, DATA_PATH)

    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    features = df.drop(columns=["Cover_Type"]).columns.tolist()
    class_map = {
        0: "Aspen",
        1: "Cottonwood/Willow",
        2: "Douglas-fir",
        3: "Krummholz",
        4: "Lodgepole Pine",
        5: "Ponderosa Pine",
        6: "Spruce/Fir"
    }

    return pipeline, features, class_map, df


pipeline, features, class_map, df = load_assets()

st.title("🌲 Forest Cover Type Prediction")

inputs = {}
for col in features:
    inputs[col] = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

if st.sidebar.button("Predict"):
    X = pd.DataFrame([inputs])[features]
    probs = pipeline.predict_proba(X)[0]
    pred = probs.argmax()

    st.success(f"🌿 Predicted Cover Type: **{class_map[pred]}**")
    st.bar_chart(pd.Series(probs, index=[class_map[i] for i in range(len(probs))]))
