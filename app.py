import streamlit as st
import pandas as pd
import joblib
import requests
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_assets():
    model_path = hf_hub_download(
        repo_id="mp28/ecotype-forest-cover-classifier",
        filename="final_pipeline.pkl",
        repo_type="model"
    )

    data_path = hf_hub_download(
        repo_id="mp28/ecotype-forest-cover-dataset",
        filename="final_preprocessed_data.csv",
        repo_type="dataset"
    )

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    features = df.drop(columns=["Cover_Type"]).columns.tolist()
    class_map = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }

    return pipeline, features, class_map, df


pipeline, features, class_map, df = load_assets()

st.title("🌲 Forest Cover Type Prediction")

st.sidebar.header("Input Features")

inputs = {}
for col in features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    inputs[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

X = pd.DataFrame([inputs])[features]

if st.sidebar.button("Predict"):
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]

    st.success(f"🌿 Predicted Cover Type: **{class_map[pred]}**")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": [class_map[i+1] for i in range(len(proba))],
        "Probability": proba
    })
    st.bar_chart(prob_df.set_index("Class"))

st.subheader("Sample Dataset Preview")
st.dataframe(df.head())
