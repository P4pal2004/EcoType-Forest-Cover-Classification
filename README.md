🌲 EcoType — Forest Cover Type Classification

A Machine Learning web application that predicts forest cover types using cartographic and environmental features such as elevation, soil type, and wilderness area.

📌 Project Overview

EcoType is a classification system built using Random Forest to predict forest cover types across landscapes. The model is trained on geospatial data and deployed using Streamlit to provide real-time predictions through an interactive web interface.

🧠 Cover Types (Corrected)
Label	Forest Type
0	Aspen
1	Douglas-fir
2	Krummholz
3	Lodgepole Pine
4	Ponderosa Pine
5	Spruce/Fir
6	Cottonwood/Willow
🚀 Live Demo

👉 Streamlit App: (Add your deployed link here once published)
https://your-app-name.streamlit.app

🖥️ Application Preview
🔹 Input Panel

(Add screenshot here after deployment)


🔹 Prediction Output

(Add screenshot here after deployment)


🛠️ Tech Stack

Python 🐍

Pandas, NumPy

Scikit-learn

Streamlit

Joblib

📂 Project Structure
EcoType-Forest-Cover-Classification/
│── app.py
│── README.md
│── requirements.txt
│── final_preprocessed_data.csv
│── models/
│   ├── final_pipeline.pkl
│   ├── model_features.pkl
│   ├── class_map.pkl
│── notebooks/
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── modelling.ipynb

⚙️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/P4pal2004/EcoType-Forest-Cover-Classification.git
cd EcoType-Forest-Cover-Classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py

📈 Model Performance

Algorithm: Random Forest Classifier

Accuracy: ~99%

Evaluation Metrics: Precision, Recall, F1-score

🌍 Features Used

Elevation

Aspect

Slope

Horizontal/Vertical Distance to Hydrology

Distance to Roadways

Distance to Fire Points

Hillshade (9am, Noon, 3pm)

Wilderness Area (One-hot encoded)

Soil Type (One-hot encoded)

🧪 Example Prediction

Input:

Elevation: 2500
Slope: 15
Wilderness Area: 3
Soil Type: 15


Output:

🌿 Predicted Forest Cover Type: Aspen

📦 Deployment (Streamlit Cloud)

Push your project to GitHub

Go to https://streamlit.io/cloud

Connect your repo

Set app.py as entry file

Deploy 🚀

📜 License

This project is for educational and academic use.

👨‍💻 Author

Mahendra Pal
GitHub: P4pal2004