pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib  # To load your saved model

# Load your trained model (make sure to save it after training)
model = joblib.load('logistic_regression_model.pkl')

# Title of the app
st.title("Employee Retention Prediction")

# Input features for prediction
satisfaction_level = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5)
last_evaluation = st.slider("Last Evaluation Score", min_value=0.0, max_value=1.0, value=0.5)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
work_accident = st.selectbox("Work Accident", [0, 1])  # 0 = No, 1 = Yes
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])  # 0 = No, 1 = Yes
salary = st.selectbox("Salary", ["low", "medium", "high"])

# One-hot encoding for categorical variables
salary_encoded = pd.get_dummies([salary], drop_first=True)[0].to_dict()
data = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    **salary_encoded
}

# Convert input data to DataFrame
input_data = pd.DataFrame([data])

# Predict using the model
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[:, 1]

# Display results
if prediction[0] == 1:
    st.write("The employee is likely to leave.")
else:
    st.write("The employee is likely to stay.")

st.write(f"Probability of leaving: {probability[0]:.2f}")
