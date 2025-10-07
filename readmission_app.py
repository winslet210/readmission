import streamlit as st
import joblib
import pandas as pd

# Define the filename of your saved model
model_filename = 'readmission_risk_model.joblib'

# Load the trained model
try:
    model = joblib.load(model_filename)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Make sure it's in the same directory.")
    st.stop() # Stop the app if the model isn't found

# Define the features used during training (make sure this matches your training features)
features = ['age', 'has_diabetes', 'has_hypertension', 'previous_admissions', 'avg_blood_sugar_last_7_days']

# --- Streamlit App Interface ---
st.title("Readmission Risk Prediction")

st.write("Enter patient details to predict their readmission risk.")

# Input widgets for patient data
age = st.slider("Age", min_value=18, max_value=100, value=50)
has_diabetes = st.checkbox("Has Diabetes")
has_hypertension = st.checkbox("Has Hypertension")
previous_admissions = st.number_input("Number of Previous Admissions", min_value=0, value=0)
avg_blood_sugar_last_7_days = st.number_input("Average Blood Sugar (last 7 days, e.g., mmol/L)", min_value=0.0, value=5.0, format="%.2f")

# Convert checkbox boolean to integer (0 or 1)
has_diabetes_int = 1 if has_diabetes else 0
has_hypertension_int = 1 if has_hypertension else 0

# Button to trigger prediction
if st.button("Predict Risk"):
    # Prepare the input data as a DataFrame
    patient_data = {
        'age': age,
        'has_diabetes': has_diabetes_int,
        'has_hypertension': has_hypertension_int,
        'previous_admissions': previous_admissions,
        'avg_blood_sugar_last_7_days': avg_blood_sugar_last_7_days
    }
    input_df = pd.DataFrame([patient_data])

    # Ensure the column order matches the training data
    input_df = input_df[features]

    # Make the prediction
    risk_probability = model.predict_proba(input_df)[:, 1]
    risk_score = risk_probability[0]

    # Display the result
    st.subheader("Prediction Result:")
    st.write(f"The predicted readmission risk is: **{risk_score:.2f}**")

    # Provide a simple interpretation
    if risk_score > 0.5:
        st.warning("This patient has a higher predicted risk of readmission.")
    else:
        st.info("This patient has a lower predicted risk of readmission.")