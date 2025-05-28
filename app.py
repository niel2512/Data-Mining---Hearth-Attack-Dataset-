import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load model and data
model = joblib.load("model.pkl")
df_original = pd.read_csv('Medicaldataset.csv')
df_processed = df_original.copy()

# Outlier handling
numerical_cols = df_processed.select_dtypes(include=np.number).columns
for col in numerical_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower, upper)

max_heart_rate_cap = 220
df_processed['Heart rate'] = np.minimum(df_processed['Heart rate'], max_heart_rate_cap)
ck_mb_cap_99 = df_processed['CK-MB'].quantile(0.99)
df_processed['CK-MB'] = np.minimum(df_processed['CK-MB'], ck_mb_cap_99)

X = df_processed.drop(['Result'], axis=1)
scaler = MinMaxScaler()
X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X.select_dtypes(include=np.number))

# 🌟 App layout and style
st.set_page_config(page_title="Heart Attack Risk Prediction", page_icon="🫀", layout="centered")

st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: red;">Heart Attack Risk Prediction</h1>
        <p style="font-size: 18px;">Masukkan Data Riwayat Penyakit Untuk Di Prediksi</p>
    </div>
""", unsafe_allow_html=True)

# 🎛 Input form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, value=70.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0.0, value=120.0)
    with col2:
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0.0, value=80.0)
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=0.0, value=100.0)
        ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, value=5.0)
        troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, value=0.1)

    submitted = st.form_submit_button("Predict")

# 🧾 Prediction process
if submitted:
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

    input_df = pd.DataFrame([input_data])

    # Outlier bounds clipping
    for col in input_df.columns:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cap = ck_mb_cap_99 if col == 'CK-MB' else max_heart_rate_cap if col == 'Heart rate' else upper
            input_df[col] = np.clip(input_df[col], lower, cap)

    # Scaling
    input_df[input_df.select_dtypes(include=np.number).columns] = scaler.transform(input_df[input_df.select_dtypes(include=np.number).columns])

    # Prediction
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    # 🎯 Display result
    st.subheader("🧾 Hasil Prediksi Model:")

    st.markdown(f"""
    <div style="font-size:18px">
        <strong>Predicted Probability:</strong> <span style="color:green;">{proba:.2f}</span>
    </div>
    <br>
    """, unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error(f"🔴 Pasien Beresiko Tinggi Terkena Serangan Jantung")
    else:
        st.success(f"🟢 Pasien Beresiko Rendah Terkena Serangan Jantung")
