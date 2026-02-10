# 🌲 EcoType: Forest Cover Type Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-brightgreen.svg)

🚀 **Live App:** *(Add your Streamlit Cloud link here)*  
📦 **Model on Hugging Face:** https://huggingface.co/mp28/ecotype-forest-cover-classifier  
📊 **Dataset on Hugging Face:** https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset  

---

## 📌 Project Overview

EcoType is a machine learning application that predicts **forest cover types** using cartographic and environmental features such as elevation, slope, soil type, and wilderness area indicators. The project includes full preprocessing, model training, evaluation, and a deployed **Streamlit web application** for real-time predictions.

This project is inspired by the **UCI Forest CoverType dataset** and is ideal for demonstrating real-world classification pipelines and ML deployment skills.

---

## 🎯 Forest Cover Classes

| Class ID | Cover Type |
|----------|------------|
| 0 | Aspen |
| 1 | Lodgepole Pine |
| 2 | Ponderosa Pine |
| 3 | Cottonwood/Willow |
| 4 | Douglas-fir |
| 5 | Krummholz |
| 6 | Spruce/Fir |

---

## 🧠 Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing Pipelines
- Random Forest Classification
- Model Evaluation & Validation
- Model Serialization & Reuse
- Streamlit App Development
- Cloud Deployment
- Hugging Face Model Hosting

---

## 🗂️ Project Structure
EcoType-Forest-Cover-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── final_preprocessed_data.csv

│ ├── data_cleaning.ipynb
│ ├── EDA.ipynb
│ ├── modelling.ipynb
│ └── preprocessing.ipynb
├── models/
│ ├── final_pipeline.pkl
│ ├── class_map.pkl
│ ├── model_features.pkl
│ └── scaler.pkl
└── data/
└── raw_data.csv


---

## ⚙️ Machine Learning Workflow

1. **Data Cleaning**
   - Removed duplicates
   - Handled missing values

2. **Feature Engineering**
   - Created hydrology ratio
   - Encoded wilderness and soil types

3. **Model Training**
   - Random Forest Classifier
   - Hyperparameter tuning with cross-validation

4. **Evaluation**
   - Achieved ~99% accuracy
   - Balanced precision/recall across all classes

5. **Deployment**
   - Saved trained pipeline
   - Integrated into Streamlit UI

---

## 🖥️ Streamlit Application Features

- Slider-based numeric inputs
- Dropdown-based soil & wilderness selection
- Real-time predictions
- Class probability visualization
- Clean UI for users

---


## 🚀 How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/P4pal2004/EcoType-Forest-Cover-Classification.git
cd EcoType-Forest-Cover-Classification

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download Model Files

Download from Hugging Face:

🔗 https://huggingface.co/mp28/ecotype-forest-cover-classifier

Place files inside:

models/

5️⃣ Run Streamlit App
streamlit run app.py

☁️ Deployment

This project is deployed using Streamlit Cloud and uses models hosted on Hugging Face Hub for large-file support.

📦 Dataset Source

Dataset hosted on Hugging Face:
🔗 https://huggingface.co/datasets/mp28/ecotype-forest-cover-dataset

Original dataset inspired by:

UCI Machine Learning Repository — Forest CoverType Dataset

🧑‍💻 Author

Mahendra Pal
📧 GitHub: https://github.com/P4pal2004

💼 Aspiring Data Scientist | Machine Learning Engineer