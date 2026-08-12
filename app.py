import streamlit as st
import pandas as pd
import pickle


# -------------------------------
# Load Model
# -------------------------------
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# -------------------------------
# Title
# -------------------------------
st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter applicant details below to predict loan approval status."
)


st.divider()


# -------------------------------
# Input Form
# -------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )


    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )


    person_emp_exp = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=60,
        value=5
    )


    person_education = st.selectbox(
        "Education",
        [
            "Associate",
            "Bachelor",
            "Doctorate",
            "High School",
            "Master"
        ]
    )


with col2:

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "MORTGAGE",
            "OTHER",
            "OWN",
            "RENT"
        ]
    )


    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500,
        value=10000
    )


    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )


    previous_loan_defaults = st.selectbox(
        "Previous Loan Default",
        [
            "0",
            "1"
        ]
    )


with col3:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.5
    )


    loan_percent_income = st.slider(
        "Loan Percent Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20
    )


    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=0,
        max_value=80,
        value=5
    )


    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )



st.divider()


# -------------------------------
# Prediction Button
# -------------------------------

if st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
):


    # ---------------------------
    # Create Encoded Input
    # ---------------------------

    input_df = pd.DataFrame({

        "person_age":[person_age],

        "person_income":[person_income],

        "person_emp_exp":[person_emp_exp],

        "loan_amnt":[loan_amnt],

        "loan_int_rate":[loan_int_rate],

        "loan_percent_income":[loan_percent_income],

        "cb_person_cred_hist_length":[
            cb_person_cred_hist_length
        ],

        "credit_score":[credit_score],


        "person_home_ownership_OTHER":[0],

        "person_home_ownership_OWN":[0],

        "person_home_ownership_RENT":[0],


        "loan_intent_EDUCATION":[0],

        "loan_intent_HOMEIMPROVEMENT":[0],

        "loan_intent_MEDICAL":[0],

        "loan_intent_PERSONAL":[0],

        "loan_intent_VENTURE":[0],


        "previous_loan_defaults_on_file_1":[
            int(previous_loan_defaults)
        ],


        "person_education_Bachelor":[0],

        "person_education_Doctorate":[0],

        "person_education_High School":[0],

        "person_education_Master":[0]

    })



    # ---------------------------
    # Apply Encoding
    # ---------------------------

    # Home Ownership

    if person_home_ownership == "OTHER":
        input_df["person_home_ownership_OTHER"] = 1

    elif person_home_ownership == "OWN":
        input_df["person_home_ownership_OWN"] = 1

    elif person_home_ownership == "RENT":
        input_df["person_home_ownership_RENT"] = 1



    # Loan Intent

    if loan_intent == "EDUCATION":
        input_df["loan_intent_EDUCATION"] = 1

    elif loan_intent == "HOMEIMPROVEMENT":
        input_df["loan_intent_HOMEIMPROVEMENT"] = 1

    elif loan_intent == "MEDICAL":
        input_df["loan_intent_MEDICAL"] = 1

    elif loan_intent == "PERSONAL":
        input_df["loan_intent_PERSONAL"] = 1

    elif loan_intent == "VENTURE":
        input_df["loan_intent_VENTURE"] = 1



    # Education

    if person_education == "Bachelor":
        input_df["person_education_Bachelor"] = 1

    elif person_education == "Doctorate":
        input_df["person_education_Doctorate"] = 1

    elif person_education == "High School":
        input_df["person_education_High School"] = 1

    elif person_education == "Master":
        input_df["person_education_Master"] = 1



    # ---------------------------
    # Prediction
    # ---------------------------

    prediction = model.predict(input_df)


    st.divider()


    if prediction[0] == 0:

        st.success(
            "✅ Loan Approved"
        )

    else:

        st.error(
            "❌ Loan Rejected"
        )


    # Probability

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_df)

        approval = probability[0][0] * 100

        rejection = probability[0][1] * 100


        st.subheader("Prediction Confidence")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Approval Probability",
                f"{approval:.2f}%"
            )

        with c2:
            st.metric(
                "Rejection Probability",
                f"{rejection:.2f}%"
            )


    # Show processed data

    with st.expander("View Model Input Data"):
        st.dataframe(input_df)
