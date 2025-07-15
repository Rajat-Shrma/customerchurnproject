# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:39:34 2025

@author: Saurabh
"""

import pickle
import pandas as pd
import streamlit as st

# Load the model
loaded_model = pickle.load(open('C:/Users/Saurabh/Desktop/desktop/churnprediction/customer_churn.pkl', 'rb'))

# Label encoding map
label_encodings = {
    'gender': {'Male': 1, 'Female': 0},
    'SeniorCitizen': {'Yes': 1, 'No': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
}

def main():
    st.title('Customer Churn Prediction')

    # Input fields
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
    partner = st.selectbox("Has Partner", ['Yes', 'No'])
    dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (in months)", min_value=0)
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
    online_backup = st.selectbox("Online Backup", ['No internet service', 'No', 'Yes'])
    device_protection = st.selectbox("Device Protection", ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox("Tech Support", ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox("Streaming TV", ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox("Streaming Movies", ['No internet service', 'No', 'Yes'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predict Churn"):
        input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service,
                                    multiple_lines, internet_service, online_security, online_backup,
                                    device_protection, tech_support, streaming_tv, streaming_movies,
                                    contract, paperless_billing, payment_method, monthly_charges, total_charges]],
                                  columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                           'MonthlyCharges', 'TotalCharges'])

        # Encode categorical data
        for col in label_encodings:
            input_data[col] = input_data[col].map(label_encodings[col])

        # Predict
        prediction = loaded_model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is unlikely to churn.")

if __name__ == '__main__':
    main()