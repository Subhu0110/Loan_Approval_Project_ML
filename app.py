import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Loan Approval Prediction")

st.sidebar.markdown("""
This application predicts whether a loan application is likely to be **Approved** or **Rejected** using a **Random Forest Classifier**.

### 👨‍💻 Developed By
**Subhansh Yadav**

### 🤖 Model Used
- Random Forest Classifier

### 🚀 Tech Stack
- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction System")

st.markdown(
    """
Predict the approval status of a loan application based on the applicant's financial details.
"""
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------

left, right = st.columns(2)

with left:

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    income = st.number_input(
        "Annual Income",
        min_value=0,
        value=500000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=1000000
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        value=10
    )

    cibil = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=750
    )

with right:

    residential = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=0
    )

    commercial = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=0
    )

    luxury = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=0
    )

    bank = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=0
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

# -----------------------------
# Convert Categories
# -----------------------------

education = 1 if education == "Not Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -----------------------------
# Prediction Button
# -----------------------------

st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    input_data = pd.DataFrame({
        " no_of_dependents": [dependents],
        " income_annum": [income],
        " loan_amount": [loan_amount],
        " loan_term": [loan_term],
        " cibil_score": [cibil],
        " residential_assets_value": [residential],
        " commercial_assets_value": [commercial],
        " luxury_assets_value": [luxury],
        " bank_asset_value": [bank],
        " education_ Not Graduate": [education],
        " self_employed_ Yes": [self_employed]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:

        st.success("✅ Loan Approved")

        st.metric(
            label="Approval Confidence",
            value=f"{probability[1]*100:.2f}%"
        )

        st.balloons()

    else:

        st.error("❌ Loan Rejected")

        st.metric(
            label="Rejection Confidence",
            value=f"{probability[0]*100:.2f}%"
        )

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Built using ❤️ with Streamlit | Machine Learning Project by Subhansh Yadav"
)