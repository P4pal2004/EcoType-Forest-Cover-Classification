🌲 EcoType: Forest Cover Type Classification








🚀 Live App:
🔗 https://ecotype-forest-cover-classification-mrqengbfnxtcujlook84l2.streamlit.app/

📦 Model on Hugging Face:
https://huggingface.co/mp28/ecotype-forest-cover-classifier

📊 Dataset on Hugging Face:
https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset

📌 Project Overview

EcoType is a machine learning application that predicts forest cover types using cartographic and environmental features such as elevation, slope, soil type, and wilderness area indicators. The project includes full preprocessing, model training, evaluation, and a deployed Streamlit web application for real-time predictions.

This project is inspired by the UCI Forest CoverType dataset and demonstrates real-world classification pipelines and ML deployment skills.

🎯 Forest Cover Classes
Class ID	Cover Type
1	Spruce/Fir
2	Lodgepole Pine
3	Ponderosa Pine
4	Cottonwood/Willow
5	Aspen
6	Douglas-fir
7	Krummholz
🧠 Skills Demonstrated

Exploratory Data Analysis (EDA)

Feature Engineering

Data Preprocessing Pipelines

Random Forest Classification

Model Evaluation & Validation

Model Serialization & Reuse

Streamlit App Development

Cloud Deployment

Hugging Face Model Hosting

🗂️ Project Structure
EcoType-Forest-Cover-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── final_preprocessed_data.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── preprocessing.ipynb
│   ├── EDA.ipynb
│   └── modelling.ipynb
│
├── models/
│   └── final_pipeline_v2.pkl
│
└── data/
    └── raw_data.csv

⚙️ Machine Learning Workflow

Data Cleaning

Removed duplicates

Handled missing values

Feature Engineering

Created hydrology ratios

Encoded wilderness and soil types

Model Training

Random Forest Classifier

Hyperparameter tuning with cross-validation

Evaluation

Achieved ~99% accuracy

Balanced precision/recall across all classes

Deployment

Saved trained pipeline

Integrated into Streamlit UI

🖥️ Streamlit Application Features

Slider-based numeric inputs

Real-time predictions

Class probability visualization

Clean and interactive UI

🚀 How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/P4pal2004/EcoType-Forest-Cover-Classification.git
cd EcoType-Forest-Cover-Classification

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

☁️ Deployment

This project is deployed using Streamlit Cloud and loads model files from Hugging Face Hub for large-file support.

📦 Dataset Source

Hugging Face: https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset

Original: UCI Machine Learning Repository — Forest CoverType Dataset

🧑‍💻 Author

Mahendra Pal
🔗 GitHub: https://github.com/P4pal2004

💼 Aspiring Data Scientist | Machine Learning Engineer
