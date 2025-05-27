
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = joblib.load("model.pkl")

df_original = pd.read_csv('Medicaldataset.csv')
# Re-perform the preprocessing steps up to splitting and scaling to get the scaler
df_processed = df_original.copy()

numerical_cols_for_outlier = df_processed.select_dtypes(include=np.number).columns

for col in numerical_cols_for_outlier:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
    df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])

max_heart_rate_cap = 220
df_processed['Heart rate'] = np.where(df_processed['Heart rate'] > max_heart_rate_cap, max_heart_rate_cap, df_processed['Heart rate'])
ck_mb_99 = df_processed['CK-MB'].quantile(0.99)
df_processed['CK-MB'] = np.where(df_processed['CK-MB'] > ck_mb_99, ck_mb_99, df_processed['CK-MB'])


X = df_processed.drop(['Result'], axis=1)
numerical_features = X.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
# Fit the scaler on the numerical features of the processed data (before splitting, or on the full processed X)
# Fitting on the full processed X is simpler for deployment if the same preprocessing is applied to input.
X_scaled_for_scaler_fit = X.copy()
X_scaled_for_scaler_fit[numerical_features] = scaler.fit_transform(X[numerical_features])


# Streamlit App
st.title("Heart Attack Prediction")

st.write("""
This application predicts the likelihood of a heart attack based on patient's medical data.
Please enter the patient's information below:
""")

# Input fields for patient data
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
heart_rate = st.number_input("Heart Rate (beats per minute)", min_value=0.0, value=70.0)
systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, value=120.0)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0.0, value=80.0)
blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=0.0, value=100.0)
ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, value=5.0)
troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, value=0.1)

# Create a dictionary from input
input_data = {
    'Age': age,
    'Gender': gender,
    'Heart rate': heart_rate,
    'Systolic blood pressure': systolic_bp,
    'Diastolic blood pressure': diastolic_bp,
    'Blood sugar': blood_sugar,
    'CK-MB': ck_mb,
    'Troponin': troponin
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])
input_df_processed = input_df.copy()
bounds = {
    'Age': (input_df['Age'].quantile(0.25) - 1.5 * (input_df['Age'].quantile(0.75) - input_df['Age'].quantile(0.25)), input_df['Age'].quantile(0.75) + 1.5 * (input_df['Age'].quantile(0.75) - input_df['Age'].quantile(0.25))),
    'Heart rate': (input_df['Heart rate'].quantile(0.25) - 1.5 * (input_df['Heart rate'].quantile(0.75) - input_df['Heart rate'].quantile(0.25)), max_heart_rate_cap), # Use max_heart_rate_cap
    'Systolic blood pressure': (input_df['Systolic blood pressure'].quantile(0.25) - 1.5 * (input_df['Systolic blood pressure'].quantile(0.75) - input_df['Systolic blood pressure'].quantile(0.25)), input_df['Systolic blood pressure'].quantile(0.75) + 1.5 * (input_df['Systolic blood pressure'].quantile(0.75) - input_df['Systolic blood pressure'].quantile(0.25))),
    'Diastolic blood pressure': (input_df['Diastolic blood pressure'].quantile(0.25) - 1.5 * (input_df['Diastolic blood pressure'].quantile(0.75) - input_df['Diastolic blood pressure'].quantile(0.25)), input_df['Diastolic blood pressure'].quantile(0.75) + 1.5 * (input_df['Diastolic blood pressure'].quantile(0.75) - input_df['Diastolic blood pressure'].quantile(0.25))),
    'Blood sugar': (input_df['Blood sugar'].quantile(0.25) - 1.5 * (input_df['Blood sugar'].quantile(0.75) - input_df['Blood sugar'].quantile(0.25)), input_df['Blood sugar'].quantile(0.75) + 1.5 * (input_df['Blood sugar'].quantile(0.75) - input_df['Blood sugar'].quantile(0.25))),
    'CK-MB': (input_df['CK-MB'].quantile(0.25) - 1.5 * (input_df['CK-MB'].quantile(0.75) - input_df['CK-MB'].quantile(0.25)), ck_mb_99),
    'Troponin': (input_df['Troponin'].quantile(0.25) - 1.5 * (input_df['Troponin'].quantile(0.75) - input_df['Troponin'].quantile(0.25)), input_df['Troponin'].quantile(0.75) + 1.5 * (input_df['Troponin'].quantile(0.75) - input_df['Troponin'].quantile(0.25)))
}

for col in bounds:
    lower_bound, upper_bound = bounds[col]
    input_df_processed[col] = np.where(input_df_processed[col] < lower_bound, lower_bound, input_df_processed[col])
    input_df_processed[col] = np.where(input_df_processed[col] > upper_bound, upper_bound, input_df_processed[col])


# Select numerical features from the input DataFrame for scaling
input_numerical_features = input_df_processed.select_dtypes(include=np.number).columns

# Scale the input data using the fitted scaler
input_scaled = input_df_processed.copy()
input_scaled[input_numerical_features] = scaler.transform(input_df_processed[input_numerical_features])


if st.button("Predict Heart Attack Risk"):
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[:, 1]

    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error(f"Based on the provided data, the model predicts a **High Risk** of heart attack.")
    else:
        st.success(f"Based on the provided data, the model predicts a **Low Risk** of heart attack.")

    st.write(f"Probability of positive result: **{prediction_proba[0]:.2f}**")

    st.write("""
    **Disclaimer:** This prediction is based on a machine learning model and should not be considered as medical advice.
    Always consult with a qualified healthcare professional for any health concerns.
    """)

