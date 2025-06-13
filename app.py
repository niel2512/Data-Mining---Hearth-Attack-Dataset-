import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load model and data
model = joblib.load("model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl") # Memuat model K-Means
scaler = joblib.load("scaler.pkl") # Memuat scaler yang sudah di-fit

# Deskripsi karakteristik cluster
cluster_info = {
    0: {
        "title": "Cluster 0: Profil Pria dengan Risiko Jantung Lebih Menonjol",
        "description": "Cluster ini <strong>seluruhnya terdiri dari pasien laki-laki</strong>. Karakteristik cluster ini adalah tingkat <strong>Troponin yang lebih tinggi</strong> dibandingkan Cluster 1, yang merupakan indikator kuat adanya stres atau kerusakan pada otot jantung.",
        "icon": "👨‍⚕️"
    },
    1: {
        "title": "Cluster 1: Profil Wanita dengan Usia Relatif Lebih Tua",
        "description": "Cluster ini <strong>seluruhnya terdiri dari pasien perempuan</strong>. Dibandingkan dengan Cluster 0, anggota cluster ini secara rata-rata memiliki <strong>usia yang sedikit lebih tua</strong> dengan <strong>tekanan darah sistolik yang sedikit lebih rendah</strong>.",
        "icon": "👩‍⚕️"
    }
}


# 🌟 App layout and style
st.set_page_config(page_title="Heart Attack Risk Prediction", page_icon="🫀", layout="centered")

st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: red;">Heart Attack Risk Prediction</h1>
        <p style="font-size: 18px;">Masukkan Data Riwayat Pasien Untuk Di Prediksi</p>
    </div>
""", unsafe_allow_html=True)

# 🎛 Input form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: 'Laki-laki' if x == 1 else 'Perempuan')
        heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 70)
        systolic_bp = st.slider("Systolic BP (mmHg)", 80, 220, 120)
    with col2:
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 140, 80)
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50.0, max_value=500.0, value=100.0)
        ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=30.0, value=5.0)
        troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=1.0, value=0.1, format="%.3f")

    submitted = st.form_submit_button("Analisis Risiko", type="primary", use_container_width=True)

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

    numerical_features = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

    input_df_ordered = input_df[numerical_features]

    input_df_scaled = scaler.transform(input_df_ordered)
    input_df_scaled = pd.DataFrame(input_df_scaled, columns=numerical_features)


    # Prediction
    prediction = model.predict(input_df_scaled)[0]
    proba = model.predict_proba(input_df_scaled)[0][1]
    cluster = kmeans_model.predict(input_df_scaled)[0]

    # 🎯 Display result
    st.markdown('<h3 style="text-align:center;">🧾 Hasil Analisis Model: </h3>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Probabilitas Risiko Serangan Jantung", value=f"{proba:.2%}")
        if prediction == 1:
            st.error(f"🔴 Pasien Berisiko Tinggi Terkena Serangan Jantung")
        else:
            st.success(f"🟢 Pasien Berisiko Rendah Terkena Serangan Jantung")

    with res_col2:
        info = cluster_info.get(cluster)
        if info:
            st.markdown(f"<h5>{info['icon']} Analisis Profil Pasien</h5>", unsafe_allow_html=True)
            info_style = "border: 1px solid #043785; border-radius: 0.25rem; background-color: #04337a; padding: 1rem;"
            html_content = f"""
                <div style="{info_style}">
                    <h6 style='margin-top: 0;'>Pasien ini termasuk dalam <strong>{info['title']}</strong></h6>
                    <p style='margin-bottom: 0; font-size:14px'>{info['description']}</p>
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)        
