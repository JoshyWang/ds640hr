import streamlit as st
import numpy as np

# Coefficients from your fitted logistic regression model
coefficients = {
    'satisfaction_level': -4.147406,
    'last_evaluation': 0.679698,
    'number_project': -0.303713,
    'average_montly_hours': 0.004631,
    'time_spend_company': 0.263568,
    'Work_accident': -1.490668,
    'promotion_last_5years': -1.196919,
    'salary_low': 1.894944,
    'salary_medium': 1.380625,
    'Department_RandD': -0.704288,
    'Department_management': -0.337431,
    'Department_hr': 0.319862,
    'Department_technical': 0.149650,
    'Department_marketing': 0.140880,
    'Department_accounting': 0.105462,
    'Department_support': 0.070365,
    'Department_sales': 0.058106,
    'Department_product_mng': 0.004104,
}

# Streamlit app title
st.title("Employee Left Prediction")

# Input fields for the user to enter data
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
number_project = st.number_input("Number of Projects", min_value=1, value=3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=1, value=200)
time_spend_company = st.number_input("Time Spent in Company (years)", min_value=1, value=3)
Work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])

# Salary input (one-hot encoding)
salary_low = st.selectbox("Salary Level (Low)", [0, 1])
salary_medium = st.selectbox("Salary Level (Medium)", [0, 1])

# Department input (one-hot encoding)
department = st.selectbox("Department", [
    "RandD", "Management", "HR", "Technical", "Marketing",
    "Accounting", "Support", "Sales", "Product Management"
])

# Initialize department features
department_features = {dep: 0 for dep in coefficients if "Department_" in dep}
department_features[f"Department_{department}"] = 1

# Combine all inputs into one dictionary
new_case = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    'salary_low': salary_low,
    'salary_medium': salary_medium,
    **department_features,
}

# Calculate log-odds
log_odds = sum(coefficients[feature] * new_case[feature] for feature in coefficients)

# Convert log-odds to probability
probability = 1 / (1 + np.exp(-log_odds))

# Make a prediction
prediction = 'left' if probability > 0.5 else 'not left'

# Display the results
st.write(f"**Probability of Leaving:** {probability:.4f}")
st.write(f"**Prediction:** {prediction}")

