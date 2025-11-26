# Accident Severity ML App

End-to-end Machine Learning application for predicting accident severity (Slight, Serious, Fatal) using structured data, feature engineering, and ensemble models. Includes a Streamlit UI for interactive predictions, preprocessing encoders, and full local deployment.

---

## 🚦 Features

- Data preprocessing & feature encoding  
- Ensemble model training (Voting/Stacking)  
- Saved ML assets (`best_model.pkl`, `encoders.pkl`)  
- Streamlit app for user-friendly predictions  
- Predicts severity and shows probability distribution for each class  

---

## 📁 Repository Structure

accident-severity-ml-app/
│
├── notebooks/
│ └── notebook.ipynb # EDA + Training (renamed)
│
├── src/
│ ├── preprocessing.py # Data cleaning & encoding helpers
│ ├── training.py # Model training scripts
│ └── utils.py # Utility functions
│
├── app/
│ ├── app.py # Streamlit app
│ ├── best_model.pkl # Trained model
│ └── encoders.pkl # Label encoders
│
├── data/
│ └── sample_input.csv # Optional sample input
│
├── README.md
├── requirements.txt
└── .gitignore


---

## 🚀 Running the Streamlit App Locally

1. Install dependencies:

```bash
pip install -r requirements.txt

Run the app:
streamlit run app.py
The app will open automatically in your browser

| File               | Description                            |
| ------------------ | -------------------------------------- |
| `app.py`           | Streamlit interface for predictions    |
| `best_model.pkl`   | Trained ML model                       |
| `encoders.pkl`     | Label encoders used during training    |
| `notebook.ipynb`   | Original notebook (EDA + training)     |
| `preprocessing.py` | Helper functions for encoding/cleaning |
| `training.py`      | Model training script                  |
| `sample_input.csv` | Example input for testing              |


Prediction Output
Predicted severity: Slight / Serious / Fatal
Probability distribution for each class

Technologies Used
Python
Pandas
NumPy
scikit-learn
Joblib
Streamlit
Ensemble Learning

