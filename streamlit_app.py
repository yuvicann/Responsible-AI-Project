import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load trained model and thresholds
loaded_data = joblib.load("fairness_model_with_thresholds.pkl")
model = loaded_data["model"]
marital_thresholds = loaded_data["marital_thresholds_final"]

# Load the dataset to extract feature names
dataset_path = r"C:\\Users\\nvjad\\Downloads\\responsible ai\\LoanEE2.csv"
# Load CSV with memory optimization
sample_data = pd.read_csv(dataset_path, low_memory=False, dtype=str, nrows=5000)


st.title("Loan Approval Prediction with Fairness Consideration")

st.header("📝 Enter Applicant Details")

# Extract unique values for categorical features
marital_status_options = ["Married", "Single", "Divorced"]
education_options = ["Higher Ed", "Basic Ed", "Vocational"]
home_ownership_options = ["Owner", "Mortgage", "Tenant"]
employment_status_options = ["Employed", "Entrepreneur", "Unemployed"]
work_experience_options = ["Less than 2 Years", "2 to 5 Years", "5 to 10 Years", "10 to 15 Years", "More than 15 Years"]

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", marital_status_options)
education = st.selectbox("Education Level", education_options)
home_ownership = st.selectbox("Home Ownership", home_ownership_options)
employment_status = st.selectbox("Employment Status", employment_status_options)
income = st.number_input("Annual Income (USD)", min_value=1000, max_value=500000, value=50000, step=500)
loan_amount = st.number_input("Loan Amount Requested", min_value=1000, max_value=1000000, value=50000, step=500)
loan_duration = st.slider("Loan Duration (Years)", min_value=1, max_value=30, value=10)
interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
work_experience = st.selectbox("Work Experience", work_experience_options)

# Convert categorical inputs into the correct format (matching training dataset)
input_features = pd.DataFrame([{
    "Age": age,
    "AppliedAmount": loan_amount,
    "LoanDuration": loan_duration,
    "Interest": interest_rate,
    "IncomeTotal": income,

    # Missing Features (set to 0 as default)
    "ExpectedLoss": 0,
    "LiabilitiesTotal": 0,
    "MonthlyPayment": 0,

    # Marital Status Encoding
    "MaritalStatus_1": 1 if marital_status == "Married" else 0,
    "MaritalStatus_3": 1 if marital_status == "Single" else 0,
    "MaritalStatus_4": 1 if marital_status == "Divorced" else 0,

    # Number of Dependents Encoding
    "NrOfDependantslessthan3_1": 0
}])

# Ensure all expected columns are present
model_features = model.feature_names_in_
for feature in model_features:
    if feature not in input_features.columns:
        input_features[feature] = 0  

# Reorder columns to match the model training
input_features = input_features[model_features]

# Apply feature scaling
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_features)

# Convert scaled input to NumPy array
input_data = np.array(scaled_input)

# Predict loan approval
if st.button("Predict Loan Approval"):
    prob = model.predict_proba(input_data)[0][1]

    # Calculate Debt-to-Income Ratio (DTI)
    if income > 0:
        dti = loan_amount / income
    else:
        dti = float("inf")

    # Reduce probability for high DTI applicants
    if dti > 10:
        prob = prob * 0.7  # Reduce probability by 30%
    elif dti > 5:
        prob = prob * 0.85  # Reduce probability by 15%

    # Fine-tune the decision thresholds
    marital_thresholds_final = {
        "Married": marital_thresholds["Married"] * 0.98,
        "Single": marital_thresholds["Single"] * 1.00,
        "Divorced": marital_thresholds["Divorced"] * 0.90
    }

    # Apply dynamic thresholding within the 0.3 - 0.6 range
    adjusted_threshold = np.clip(
        np.percentile(model.predict_proba(input_data)[:, 1], 40), 
        0.30,  
        0.60   
    )

    # Apply a hard rejection for extremely high-risk cases
    if income < 5000 and loan_amount > 100000:
        prediction = 0
    elif prob >= adjusted_threshold:
        prediction = 1
    elif prob >= 0.30 and adjusted_threshold - prob < 0.05:
        prediction = 1
    else:
        prediction = 0

    # Debugging Output
    st.write(f"🔍 Adjusted Raw Probability: {prob:.4f}")
    st.write(f"📊 Final Approval Threshold for {marital_status}: {adjusted_threshold:.4f}")

    # Final Decision Display
    if prediction == 1:
        st.success(f"✅ Loan Approved with {prob:.2%} confidence! (Threshold: {adjusted_threshold:.2f})")
    else:
        st.error(f"❌ Loan Denied with {1 - prob:.2%} confidence. (Threshold: {adjusted_threshold:.2f})")



