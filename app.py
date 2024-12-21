import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('loan_status_predictor.pkl')
scaler = joblib.load('vector.pkl')

# Loan prediction form
st.title("Loan Approval Prediction")

# Collecting user input
Gender = st.selectbox("Gender", ['Female', 'Male'])
Married = st.selectbox("Married", ['No', 'Yes'])
Dependents = st.number_input("Dependents", min_value=0, max_value=10)
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['No', 'Yes'])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (Months)", min_value=0)
Credit_History = st.selectbox("Credit History", ['No', 'Yes'])
Property_Area = st.selectbox("Property Area (Urban/Semiurban/Rural)", ['Urban', 'Semiurban', 'Rural'])

# Add "Submit" button
submit_button = st.button("Submit")

# Define the action after the button is clicked
if submit_button:
    # Mapping input data for the model
    gender_map = {'Female': 0, 'Male': 1}
    married_map = {'No': 0, 'Yes': 1}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'No': 0, 'Yes': 1}
    credit_history_map = {'No': 0, 'Yes': 1}
    property_area_map = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}

    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender_map[Gender]],
        'Married': [married_map[Married]],
        'Dependents': [Dependents],
        'Education': [education_map[Education]],
        'Self_Employed': [self_employed_map[Self_Employed]],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [credit_history_map[Credit_History]],
        'Property_Area': [property_area_map[Property_Area]]
    })

    # Scaling the numeric columns (if applicable)
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Prediction
    prediction = model.predict(input_data)

    # Display result with green and red highlights
    if prediction[0] == 1:
        st.success("Loan Status: Approved")  # Green highlight for approved
    else:
        st.error("Loan Status: Not Approved")  # Red highlight for not approved
