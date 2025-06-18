import joblib
from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Prediksi Kelulusan Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load('model/model_rf.pkl')
    encoder = joblib.load('model/label_encoder.pkl')  # This is a single LabelEncoder
    features = joblib.load('model/features.pkl')
    return model, encoder, features

model, encoder, features = load_artifacts()

# List of categorical columns that need encoding
CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic', 'subject'
]

# Judul aplikasi
st.title('🎓 Prediksi Kelulusan Siswa')
st.markdown("""
Aplikasi ini memprediksi apakah seorang siswa akan lulus (nilai akhir ≥ 10) berdasarkan karakteristik akademik dan demografik.
""")

# Membuat dua kolom
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Informasi Demografik")
    # Input untuk fitur demografik
    school = st.selectbox("Sekolah", ['GP', 'MS'])
    sex = st.selectbox("Jenis Kelamin", ['F', 'M'])
    age = st.slider("Usia", 15, 22, 17)
    address = st.selectbox("Alamat", ['Urban', 'Rural'])
    famsize = st.selectbox("Ukuran Keluarga", ['LE3', 'GT3'])
    Pstatus = st.selectbox("Status Orang Tua", ['T', 'A'])
    Medu = st.select_slider("Pendidikan Ibu", options=[0, 1, 2, 3, 4], value=2)
    Fedu = st.select_slider("Pendidikan Ayah", options=[0, 1, 2, 3, 4], value=2)
    Mjob = st.selectbox("Pekerjaan Ibu", ['at_home', 'health', 'other', 'services', 'teacher'])
    Fjob = st.selectbox("Pekerjaan Ayah", ['at_home', 'health', 'other', 'services', 'teacher'])
    reason = st.selectbox("Alasan Memilih Sekolah", ['home', 'reputation', 'course', 'other'])
    guardian = st.selectbox("Wali", ['mother', 'father', 'other'])

with col2:
    st.header("📚 Informasi Akademik & Lainnya")
    # Input untuk fitur akademik
    failures = st.slider("Jumlah Nilai Gagal", 0, 4, 0)
    schoolsup = st.selectbox("Dukungan Pendidikan Tambahan", ['yes', 'no'])
    famsup = st.selectbox("Dukungan Keluarga", ['yes', 'no'])
    paid = st.selectbox("Kelas Tambahan Berbayar", ['yes', 'no'])
    activities = st.selectbox("Aktivitas Ekstrakurikuler", ['yes', 'no'])
    nursery = st.selectbox("Pernah PAUD", ['yes', 'no'])
    higher = st.selectbox("Berencana Kuliah", ['yes', 'no'])
    internet = st.selectbox("Akses Internet", ['yes', 'no'])
    romantic = st.selectbox("Dalam Hubungan Romantis", ['yes', 'no'])
    studytime = st.slider("Waktu Belajar (1-4)", 1, 4, 2)
    traveltime = st.slider("Waktu Tempuh ke Sekolah (1-4)", 1, 4, 1)
    freetime = st.slider("Waktu Luang (1-5)", 1, 5, 3)
    goout = st.slider("Frekuensi Keluar dengan Teman (1-5)", 1, 5, 2)
    Dalc = st.slider("Konsumsi Alkohol Hari Kerja (1-5)", 1, 5, 1)
    Walc = st.slider("Konsumsi Alkohol Akhir Pekan (1-5)", 1, 5, 1)
    health = st.slider("Status Kesehatan (1-5)", 1, 5, 3)
    absences = st.slider("Jumlah Ketidakhadiran", 0, 93, 0)
    subject = st.selectbox("Mata Pelajaran", ['Math', 'Portuguese'])

# Tombol prediksi
if st.button('🚀 Prediksi Kelulusan'):
    # Membuat dataframe dari input
    input_data = {
        'school': school,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'Mjob': Mjob,
        'Fjob': Fjob,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': nursery,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'famrel': 4,  # Default value
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences,
        'subject': subject
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col in CATEGORICAL_COLS:
        if col in df_input.columns:
            # Create a temporary encoder for this column
            temp_encoder = LabelEncoder()
            temp_encoder.classes_ = encoder.classes_  # Use the same classes as original encoder
            try:
                df_input[col] = temp_encoder.transform(df_input[col])
            except ValueError:
                # Handle unseen labels by mapping to default (first class)
                df_input[col] = 0
    
    # Pastikan urutan kolom sesuai dengan model
    missing_cols = set(features) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0  # Add missing columns with default value
    
    df_input = df_input[features]
    
    # Prediksi
    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)
    
    # Tampilkan hasil
    st.subheader("🎯 Hasil Prediksi")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction[0] == 1:
            st.success(f"**LULUS** (Nilai ≥ 10)")
            st.metric("Probabilitas", f"{proba[0][1]*100:.2f}%")
        else:
            st.error(f"**TIDAK LULUS** (Nilai < 10)")
            st.metric("Probabilitas", f"{proba[0][0]*100:.2f}%")
    
    with col_result2:
        # Visualisasi probabilitas
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Tidak Lulus', 'Lulus'], proba[0], color=['#ff6b6b', '#51cf66'])
        ax.set_ylabel('Probabilitas')
        ax.set_ylim(0, 1)
        ax.set_title('Probabilitas Prediksi Kelulusan')
        st.pyplot(fig, use_container_width=True)

# Tambahkan visualisasi feature importance
if st.checkbox('📈 Tampilkan Feature Importance'):
    st.subheader("Feature Importance dari Model")
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Translate feature names for better understanding
    feature_translation = {
        'subject': 'Mata Pelajaran',
        'absences': 'Ketidakhadiran',
        'higher': 'Rencana Kuliah',
        'Medu': 'Pendidikan Ibu',
        'reason': 'Alasan Pilih Sekolah',
        'failures': 'Nilai Gagal',
        'Fedu': 'Pendidikan Ayah',
        'studytime': 'Waktu Belajar',
        'goout': 'Frekuensi Keluar',
        'Mjob': 'Pekerjaan Ibu'
    }
    
    feature_importance['Feature'] = feature_importance['Feature'].map(
        lambda x: feature_translation.get(x, x)
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', 
                data=feature_importance.head(10), 
                palette='viridis', ax=ax)
    ax.set_title('10 Fitur Paling Pentik untuk Prediksi')
    ax.set_xlabel('Tingkat Kepentingan')
    ax.set_ylabel('Fitur')
    st.pyplot(fig)

# Informasi tambahan di sidebar
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini menggunakan model Random Forest yang telah dilatih untuk memprediksi kelulusan siswa berdasarkan:
- Karakteristik demografik
- Latar belakang keluarga
- Perilaku siswa
- Performa akademik
""")

st.sidebar.header("Parameter Model")
st.sidebar.text(f"Jumlah Estimator: 300")
st.sidebar.text(f"Kedalaman Maksimal: Tidak Terbatas")
st.sidebar.text(f"Akurasi Model: 86.2%")

st.sidebar.header("Definisi Kelulusan")
st.sidebar.markdown("""
Siswa dinyatakan **LULUS** jika:
- Nilai akhir (G3) ≥ 10

Siswa dinyatakan **TIDAK LULUS** jika:
- Nilai akhir (G3) < 10
""")

st.sidebar.header("Peringatan")
st.sidebar.warning("""
Hasil prediksi merupakan perkiraan berdasarkan model statistik dan tidak menjamin kepastian.
""")