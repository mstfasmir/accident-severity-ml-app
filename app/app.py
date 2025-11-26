import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("best_model.pkl")
le_dict = joblib.load("encoders.pkl")

# Features used in training
features = [
    'Sex_of_Casualty', 'Age_of_Casualty', 'Casualty_Severity',
    'Vehicle_Manoeuvre', 'Skidding_and_Overturning', 'Vehicle_Leaving_Carriageway',
    'Hit_Object_off_Carriageway', 'Sex_of_Driver', 'Age_of_Driver',
    'Number_of_Casualties', 'Speed_limit', 'Junction_Control',
    'Light_Conditions', 'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident'
]

st.title("Accident Severity Prediction 🚨")
st.markdown("Please provide the following information:")

# Collect user input
user_input = {}
for feature in features:
    if feature in le_dict:  # categorical
        # Use encoder classes for selectbox
        options = le_dict[feature].classes_
        user_input[feature] = st.selectbox(f"{feature}", options)
    else:  # numeric
        user_input[feature] = st.number_input(feature, min_value=0, max_value=100, value=1)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical columns using the saved LabelEncoders
for col in le_dict:
    if col in input_df.columns:
        input_df[col] = le_dict[col].transform(input_df[col])

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]

    severity_map = {0: "Slight", 1: "Serious", 2: "Fatal"}

    st.subheader("Prediction Results")
    st.write(f"**Predicted Accident Severity:** {severity_map[pred]}")

    st.write("**Prediction Probabilities:**")
    proba_df = pd.DataFrame([pred_proba], columns=[severity_map[i] for i in range(len(pred_proba))])
    st.dataframe(proba_df.T.rename(columns={0: 'Probability'}))
