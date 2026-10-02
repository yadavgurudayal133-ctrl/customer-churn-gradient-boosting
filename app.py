import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model = joblib.load(
    Path(__file__).resolve().parent.parent
    / "models"
    / "gradient_boosting_model.pkl"
)

st.title(
    "Customer Churn Prediction"
)

st.write(
    "Predict customer churn using "
    "Gradient Boosting"
)

Tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

partner = st.selectbox(
    "Partner",
    [
        "Yes",
        "No"
    ]
)

dependents = st.selectbox(
    "Dependents",
    [
        "Yes",
        "No"
    ]
)

phone_service = st.selectbox(
    "Phone Service",
    [
        "Yes",
        "No"
    ]
)

online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)
if st.button("Predict Churn"):

    if Tenure <= 12:
        tenure_group = "0-1 Year"
    elif Tenure <= 24:
        tenure_group = "1-2 Years"
    elif Tenure <= 48:
        tenure_group = "2-4 Years"
    else:
        tenure_group = "4+ Years"

    input_data = pd.DataFrame({
        "Gender": [gender],
        "SeniorCitizen": [0],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure": [Tenure],
        "PhoneService": [phone_service],
        "MultipleLines": ["No"],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": [tech_support],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Tenure_group": [tenure_group],
        "average_monthly_spending": [
            total_charges / max(Tenure, 1)
        ],
        "high_monthly_charge": [
            int(monthly_charges > 70)
        ]
    })

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

        st.warning(
            "Recommendation: Offer a retention "
            "plan, discount or long-term contract."
        )

    else:

        st.success(
            "✅ Customer is unlikely to churn"
        )