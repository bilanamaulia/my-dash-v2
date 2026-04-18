import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-left: 5px solid;
        margin-bottom: 1rem;
    }
    .dropout-card { border-color: #e74c3c; }
    .graduate-card { border-color: #2ecc71; }
    .enrolled-card { border-color: #f39c12; }
    .prediction-box {
        padding: 2rem; border-radius: 12px; text-align: center; margin: 1.5rem 0;
        font-size: 1.3rem; font-weight: bold;
    }
    .dropout-pred { background: #fdeaea; color: #e74c3c; border: 2px solid #e74c3c; }
    .graduate-pred { background: #eafaf1; color: #27ae60; border: 2px solid #27ae60; }
    .enrolled-pred { background: #fef9e7; color: #f39c12; border: 2px solid #f39c12; }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white; border: none; border-radius: 8px; padding: 0.7rem 2rem;
        font-size: 1rem; font-weight: bold; width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    h1, h2, h3 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'model.pkl')
    le_path = os.path.join(base_dir, 'model', 'label_encoder.pkl')
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data.csv')
    return pd.read_csv(data_path, sep=';')

try:
    model, le = load_model()
    df = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model: {e}")

# ─── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:white; margin:0; font-size:2.2rem;">🎓 Jaya Jaya Institut</h1>
    <p style="color:#a8d8ea; margin:0.5rem 0 0 0; font-size:1.1rem;">
        Student Dropout Prediction System — Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Mahasiswa", "📊 Dashboard Overview", "ℹ️ Informasi Model"])

# ════════════════════════════════════════════════
# TAB 1: PREDICTION
# ════════════════════════════════════════════════
with tab1:
    st.header("Prediksi Status Mahasiswa")
    st.markdown("Masukkan data mahasiswa di bawah ini untuk memprediksi risiko dropout.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Data Pribadi")
            marital_status = st.selectbox("Status Pernikahan", [1, 2, 3, 4, 5, 6],
                format_func=lambda x: {1:"Lajang", 2:"Menikah", 3:"Duda/Janda", 4:"Cerai",
                                        5:"Cerai Legal", 6:"Pisah"}.get(x, str(x)))
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
            age = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
            nationality = st.number_input("Kode Nasionalitas", min_value=1, max_value=109, value=1)
            international = st.selectbox("Mahasiswa Internasional", [0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak")
            displaced = st.selectbox("Pindah Domisili", [0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak")
            special_needs = st.selectbox("Kebutuhan Pendidikan Khusus", [0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak")

        with col2:
            st.subheader("🎓 Data Akademik")
            application_mode = st.number_input("Mode Pendaftaran", min_value=1, max_value=57, value=1)
            application_order = st.number_input("Urutan Pendaftaran", min_value=0, max_value=9, value=1)
            course = st.number_input("Kode Program Studi", min_value=33, max_value=9991, value=9254)
            attendance = st.selectbox("Jadwal Kuliah", [1, 0],
                format_func=lambda x: "Siang" if x == 1 else "Malam")
            prev_qual = st.number_input("Kode Kualifikasi Sebelumnya", min_value=1, max_value=43, value=1)
            prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=130.0)
            admission_grade = st.number_input("Nilai Penerimaan", min_value=0.0, max_value=200.0, value=130.0)
            mothers_qual = st.number_input("Kualifikasi Ibu", min_value=1, max_value=44, value=1)
            fathers_qual = st.number_input("Kualifikasi Ayah", min_value=1, max_value=44, value=1)
            mothers_occ = st.number_input("Pekerjaan Ibu", min_value=0, max_value=194, value=5)
            fathers_occ = st.number_input("Pekerjaan Ayah", min_value=0, max_value=194, value=5)

        with col3:
            st.subheader("📚 Performa Akademik")
            # Sem 1
            st.markdown("**Semester 1**")
            sem1_credited = st.number_input("Unit Dikreditkan Sem 1", 0, 20, 0)
            sem1_enrolled = st.number_input("Unit Diambil Sem 1", 0, 26, 6)
            sem1_evaluations = st.number_input("Evaluasi Sem 1", 0, 45, 6)
            sem1_approved = st.number_input("Unit Lulus Sem 1", 0, 26, 5)
            sem1_grade = st.number_input("Nilai Rata-rata Sem 1", 0.0, 20.0, 13.0)
            sem1_no_eval = st.number_input("Tanpa Evaluasi Sem 1", 0, 12, 0)
            # Sem 2
            st.markdown("**Semester 2**")
            sem2_credited = st.number_input("Unit Dikreditkan Sem 2", 0, 20, 0)
            sem2_enrolled = st.number_input("Unit Diambil Sem 2", 0, 23, 6)
            sem2_evaluations = st.number_input("Evaluasi Sem 2", 0, 33, 6)
            sem2_approved = st.number_input("Unit Lulus Sem 2", 0, 20, 5)
            sem2_grade = st.number_input("Nilai Rata-rata Sem 2", 0.0, 20.0, 13.0)
            sem2_no_eval = st.number_input("Tanpa Evaluasi Sem 2", 0, 12, 0)

            st.subheader("💰 Data Keuangan & Ekonomi")
            debtor = st.selectbox("Status Debitur", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            tuition_uptodate = st.selectbox("UKT Lunas", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            scholarship = st.selectbox("Penerima Beasiswa", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            unemployment = st.number_input("Tingkat Pengangguran (%)", 0.0, 20.0, 10.8)
            inflation = st.number_input("Tingkat Inflasi (%)", -5.0, 5.0, 1.4)
            gdp = st.number_input("GDP", -5.0, 5.0, 1.74)

        submitted = st.form_submit_button("🔮 Prediksi Sekarang")

    if submitted and MODEL_LOADED:
        input_data = pd.DataFrame([{
            'Marital_status': marital_status, 'Application_mode': application_mode,
            'Application_order': application_order, 'Course': course,
            'Daytime_evening_attendance': attendance, 'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade, 'Nacionality': nationality,
            'Mothers_qualification': mothers_qual, 'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ, 'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade, 'Displaced': displaced,
            'Educational_special_needs': special_needs, 'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_uptodate, 'Gender': gender,
            'Scholarship_holder': scholarship, 'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': sem1_credited,
            'Curricular_units_1st_sem_enrolled': sem1_enrolled,
            'Curricular_units_1st_sem_evaluations': sem1_evaluations,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_1st_sem_without_evaluations': sem1_no_eval,
            'Curricular_units_2nd_sem_credited': sem2_credited,
            'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
            'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Curricular_units_2nd_sem_without_evaluations': sem2_no_eval,
            'Unemployment_rate': unemployment, 'Inflation_rate': inflation, 'GDP': gdp
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        pred_label = le.inverse_transform([prediction])[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        css_class = {'Dropout': 'dropout-pred', 'Graduate': 'graduate-pred', 'Enrolled': 'enrolled-pred'}
        emoji = {'Dropout': '⚠️', 'Graduate': '🎉', 'Enrolled': '📖'}

        st.markdown(f"""
        <div class="prediction-box {css_class[pred_label]}">
            {emoji[pred_label]} Prediksi Status: <strong>{pred_label}</strong>
            <br><small>Confidence: {max(probabilities)*100:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Probabilitas per Kelas:**")
            for cls, prob in zip(le.classes_, probabilities):
                color = {'Dropout': '🔴', 'Enrolled': '🟡', 'Graduate': '🟢'}[cls]
                st.progress(float(prob), text=f"{color} {cls}: {prob*100:.1f}%")

        with col_b:
            if pred_label == 'Dropout':
                st.error("""
                **⚠️ RISIKO DROPOUT TERDETEKSI**
                
                Rekomendasi tindakan:
                - Hubungi mahasiswa untuk konseling
                - Cek status pembayaran UKT
                - Evaluasi beban akademik
                - Tawarkan program mentoring
                """)
            elif pred_label == 'Graduate':
                st.success("""
                **🎉 POTENSI LULUS TINGGI**
                
                Mahasiswa ini menunjukkan performa baik.
                Tetap berikan dukungan untuk mempertahankan prestasi.
                """)
            else:
                st.warning("""
                **📖 STATUS MASIH AKTIF**
                
                Perlu monitoring lanjutan. Pantau perkembangan akademik
                di semester berikutnya.
                """)

# ════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ════════════════════════════════════════════════
with tab2:
    st.header("📊 Dashboard Overview - Performa Mahasiswa")

    if MODEL_LOADED:
        # KPI Cards
        total = len(df)
        dropout_n = (df['Status'] == 'Dropout').sum()
        graduate_n = (df['Status'] == 'Graduate').sum()
        enrolled_n = (df['Status'] == 'Enrolled').sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Mahasiswa", f"{total:,}")
        c2.metric("Dropout", f"{dropout_n:,}", f"{dropout_n/total*100:.1f}%", delta_color="inverse")
        c3.metric("Graduate", f"{graduate_n:,}", f"{graduate_n/total*100:.1f}%")
        c4.metric("Enrolled", f"{enrolled_n:,}", f"{enrolled_n/total*100:.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Status Mahasiswa")
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['#e74c3c', '#f39c12', '#2ecc71']
            status_counts = df['Status'].value_counts()
            bars = ax.bar(status_counts.index, status_counts.values, color=colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, status_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f'{val}\n({val/total*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
            ax.set_ylabel('Jumlah')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Pengaruh Pembayaran UKT terhadap Status")
            fig, ax = plt.subplots(figsize=(7, 5))
            tuition_status = pd.crosstab(df['Tuition_fees_up_to_date'], df['Status'], normalize='index') * 100
            tuition_status.index = ['Nunggak', 'Lunas']
            tuition_status.plot(kind='bar', ax=ax, color=colors, edgecolor='white', rot=0)
            ax.set_ylabel('Persentase (%)')
            ax.legend(title='Status')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Distribusi Usia berdasarkan Status")
            fig, ax = plt.subplots(figsize=(7, 5))
            for status, color in zip(['Dropout', 'Enrolled', 'Graduate'], colors):
                data = df[df['Status'] == status]['Age_at_enrollment']
                ax.hist(data, bins=25, alpha=0.6, color=color, label=status, edgecolor='none')
            ax.set_xlabel('Usia')
            ax.set_ylabel('Frekuensi')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()

        with col4:
            st.subheader("Unit Lulus Sem 2 vs Status")
            fig, ax = plt.subplots(figsize=(7, 5))
            df.boxplot(column='Curricular_units_2nd_sem_approved', by='Status', ax=ax,
                       patch_artist=True)
            ax.set_title('')
            ax.set_xlabel('Status')
            ax.set_ylabel('Unit Lulus Semester 2')
            plt.suptitle('')
            st.pyplot(fig)
            plt.close()

# ════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ════════════════════════════════════════════════
with tab3:
    st.header("ℹ️ Informasi Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🤖 Spesifikasi Model
        | Parameter | Nilai |
        |-----------|-------|
        | **Algoritma** | Random Forest Classifier |
        | **n_estimators** | 200 |
        | **Class Weight** | Balanced |
        | **Test Size** | 20% |
        | **Accuracy** | 76.84% |
        | **CV Score (5-fold)** | 77.40% ± 0.79% |

        ### 📊 Metrik per Kelas
        | Kelas | Precision | Recall | F1-Score |
        |-------|-----------|--------|----------|
        | **Dropout** | 0.82 | 0.75 | 0.78 |
        | **Enrolled** | 0.57 | 0.36 | 0.44 |
        | **Graduate** | 0.78 | 0.93 | 0.85 |
        """)

    with col2:
        st.markdown("""
        ### 🏆 Top 5 Fitur Terpenting
        | Rank | Fitur | Importance |
        |------|-------|-----------|
        | 1 | Curricular_units_2nd_sem_approved | 12.60% |
        | 2 | Curricular_units_2nd_sem_grade | 9.33% |
        | 3 | Curricular_units_1st_sem_approved | 6.74% |
        | 4 | Curricular_units_1st_sem_grade | 6.54% |
        | 5 | Admission_grade | 4.74% |

        ### 💡 Insight Kunci
        - Performa akademik (nilai & unit lulus) adalah **faktor dominan** dalam prediksi
        - Status pembayaran UKT dan beasiswa berpengaruh signifikan
        - Usia saat mendaftar turut mempengaruhi risiko dropout
        """)

    st.markdown("---")
    st.markdown("""
    **Dikembangkan oleh:** Bilanawati Maulia
    """)
