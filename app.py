import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem; border-radius: 12px; text-align: center;
        margin: 1.5rem 0; font-size: 1.3rem; font-weight: bold;
    }
    .dropout-pred  { background:#fdeaea; color:#e74c3c; border:2px solid #e74c3c; }
    .graduate-pred { background:#eafaf1; color:#27ae60; border:2px solid #27ae60; }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color:white; border:none; border-radius:8px;
        padding:0.7rem 2rem; font-size:1rem; font-weight:bold; width:100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Setup sama persis notebook ───────────────────────────────
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    m  = joblib.load(os.path.join(base, 'model', 'model.pkl'))
    le = joblib.load(os.path.join(base, 'model', 'label_encoder.pkl'))
    return m, le


@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, 'data.csv'), sep=';')


try:
    pipeline, le = load_model()
    df = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model/data: {e}")

st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.2rem;">Jaya Jaya Institut</h1>
    <p style="color:#a8d8ea;margin:0.5rem 0 0;font-size:1.1rem;">
        Student Dropout Prediction System — Binary Classification (Dropout vs Graduate)
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediksi Mahasiswa", "Dashboard Overview", "Informasi Model"])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("Prediksi Status Mahasiswa")
    st.info(
        "Model memprediksi apakah mahasiswa berpotensi **Dropout** atau **Graduate**. "
        "Mahasiswa *Enrolled* (aktif) juga dapat diprediksi status akhirnya."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Data Pribadi")
            marital_status = st.selectbox("Status Pernikahan", [1,2,3,4,5,6],
                format_func=lambda x:{1:"Lajang",2:"Menikah",3:"Duda/Janda",
                                      4:"Cerai",5:"Cerai Legal",6:"Pisah"}.get(x))
            gender = st.selectbox("Gender", [1,0],
                format_func=lambda x:"Laki-laki" if x==1 else "Perempuan")
            age = st.number_input("Usia Saat Mendaftar", 17, 70, 20)
            nationality = st.number_input("Kode Nasionalitas", 1, 109, 1)
            international = st.selectbox("Mahasiswa Internasional", [0,1],
                format_func=lambda x:"Ya" if x==1 else "Tidak")
            displaced = st.selectbox("Pindah Domisili", [0,1],
                format_func=lambda x:"Ya" if x==1 else "Tidak")
            special_needs = st.selectbox("Kebutuhan Khusus", [0,1],
                format_func=lambda x:"Ya" if x==1 else "Tidak")

        with c2:
            st.subheader("Data Akademik")
            application_mode = st.number_input("Mode Pendaftaran", 1, 57, 1)
            application_order = st.number_input("Urutan Pendaftaran", 0, 9, 1)
            course = st.number_input("Kode Program Studi", 33, 9991, 9254)
            attendance = st.selectbox("Jadwal Kuliah", [1,0],
                format_func=lambda x:"Siang" if x==1 else "Malam")
            prev_qual = st.number_input("Kode Kualifikasi Sebelumnya", 1, 43, 1)
            prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 130.0)
            admission_grade = st.number_input("Nilai Penerimaan", 0.0, 200.0, 130.0)
            mothers_qual = st.number_input("Kualifikasi Ibu", 1, 44, 1)
            fathers_qual = st.number_input("Kualifikasi Ayah", 1, 44, 1)
            mothers_occ = st.number_input("Pekerjaan Ibu", 0, 194, 5)
            fathers_occ = st.number_input("Pekerjaan Ayah", 0, 194, 5)

        with c3:
            st.subheader("Performa Akademik")
            st.markdown("**Semester 1**")
            sem1_credited    = st.number_input("Unit Dikreditkan Sem 1", 0, 20, 0)
            sem1_enrolled_n  = st.number_input("Unit Diambil Sem 1", 0, 26, 6)
            sem1_evaluations = st.number_input("Evaluasi Sem 1", 0, 45, 6)
            sem1_approved    = st.number_input("Unit Lulus Sem 1", 0, 26, 5)
            sem1_grade       = st.number_input("Nilai Rata-rata Sem 1", 0.0, 20.0, 13.0)
            sem1_no_eval     = st.number_input("Tanpa Evaluasi Sem 1", 0, 12, 0)
            st.markdown("**Semester 2**")
            sem2_credited    = st.number_input("Unit Dikreditkan Sem 2", 0, 20, 0)
            sem2_enrolled_n  = st.number_input("Unit Diambil Sem 2", 0, 23, 6)
            sem2_evaluations = st.number_input("Evaluasi Sem 2", 0, 33, 6)
            sem2_approved    = st.number_input("Unit Lulus Sem 2", 0, 20, 5)
            sem2_grade       = st.number_input("Nilai Rata-rata Sem 2", 0.0, 20.0, 13.0)
            sem2_no_eval     = st.number_input("Tanpa Evaluasi Sem 2", 0, 12, 0)
            st.subheader("Finansial & Ekonomi")
            debtor           = st.selectbox("Status Debitur", [0,1],
                format_func=lambda x:"Ya" if x==1 else "Tidak")
            tuition_uptodate = st.selectbox("UKT Lunas", [1,0],
                format_func=lambda x:"Ya" if x==1 else "Tidak")
            scholarship      = st.selectbox("Penerima Beasiswa", [0,1],
                format_func=lambda x:"Ya" if x==1 else "Tidak")
            unemployment     = st.number_input("Tingkat Pengangguran (%)", 0.0, 20.0, 10.8)
            inflation        = st.number_input("Tingkat Inflasi (%)", -5.0, 5.0, 1.4)
            gdp              = st.number_input("GDP", -5.0, 5.0, 1.74)

        submitted = st.form_submit_button("Prediksi Sekarang")

    if submitted and MODEL_LOADED:
        input_data = pd.DataFrame([{
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': attendance,
            'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qual,
            'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ,
            'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade,
            'Displaced': displaced,
            'Educational_special_needs': special_needs,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_uptodate,
            'Gender': gender,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': sem1_credited,
            'Curricular_units_1st_sem_enrolled': sem1_enrolled_n,
            'Curricular_units_1st_sem_evaluations': sem1_evaluations,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_1st_sem_without_evaluations': sem1_no_eval,
            'Curricular_units_2nd_sem_credited': sem2_credited,
            'Curricular_units_2nd_sem_enrolled': sem2_enrolled_n,
            'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Curricular_units_2nd_sem_without_evaluations': sem2_no_eval,
            'Unemployment_rate': unemployment,
            'Inflation_rate': inflation,
            'GDP': gdp,
        }])

        prediction    = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        pred_label    = le.inverse_transform([prediction])[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        if pred_label == 'Dropout':
            st.markdown(f"""<div class="prediction-box dropout-pred">
                Prediksi Status: <strong>DROPOUT</strong>
                <br><small>Confidence: {max(probabilities)*100:.1f}%</small></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="prediction-box graduate-pred">
                Prediksi Status: <strong>GRADUATE</strong>
                <br><small>Confidence: {max(probabilities)*100:.1f}%</small></div>""",
                unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Probabilitas per Kelas:**")
            for cls, prob in zip(le.classes_, probabilities):
                emoji = "🔴" if cls == 'Dropout' else "🟢"
                st.progress(float(prob), text=f"{emoji} {cls}: {prob*100:.1f}%")
        with cb:
            if pred_label == 'Dropout':
                st.error("""**RISIKO DROPOUT TERDETEKSI**\n\nRekomendasi:\n- Konseling akademik segera\n- Cek status pembayaran UKT\n- Evaluasi beban akademik\n- Program mentoring\n- Bantuan beasiswa/keringanan biaya""")
            else:
                st.success("""**POTENSI LULUS TINGGI**\n\nMahasiswa menunjukkan performa baik.\nTetap berikan dukungan untuk mempertahankan prestasi.""")


# ════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD (kode chart identik 100% dengan notebook)
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("Dashboard Overview — Performa Mahasiswa Jaya Jaya Institut")
    st.caption("Seluruh visualisasi menggunakan kode yang identik dengan notebook.ipynb")

    if not MODEL_LOADED:
        st.error("Data tidak tersedia.")
        st.stop()

    # ── Siapkan data (sama persis notebook) ──────────────────
    df_dg       = df[df['Status'].isin(['Dropout', 'Graduate'])].copy()
    df_enrolled = df[df['Status'] == 'Enrolled'].copy()
    colors_all  = ['#e74c3c', '#2ecc71', '#f39c12']   # notebook: colors_all
    colors_dg   = ['#e74c3c', '#2ecc71']               # notebook: colors_dg

    # Rebuild X, y, pipeline identik notebook untuk CV dan feat_imp
    le_nb = LabelEncoder()
    df_nb = df_dg.copy()
    df_nb['Status_encoded'] = le_nb.fit_transform(df_nb['Status'])
    X = df_nb.drop(['Status', 'Status_encoded'], axis=1)
    y = df_nb['Status_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = pipeline.predict(X_test)

    # ── KPI ──────────────────────────────────────────────────
    total  = len(df)
    n_drop = int((df['Status'] == 'Dropout').sum())
    n_grad = int((df['Status'] == 'Graduate').sum())
    n_enr  = int((df['Status'] == 'Enrolled').sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mahasiswa",   f"{total:,}")
    c2.metric("Dropout",  f"{n_drop:,}", f"{n_drop/total*100:.2f}%", delta_color="inverse")
    c3.metric("Graduate", f"{n_grad:,}", f"{n_grad/total*100:.2f}%")
    c4.metric("Enrolled (aktif)", f"{n_enr:,}", f"{n_enr/total*100:.2f}%")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 4.1 DISTRIBUSI STATUS — identik notebook cell 8
    # ════════════════════════════════════════════════════════
    st.subheader("4.1 Distribusi Status Mahasiswa")

    # ── kode identik notebook (hanya plt.show() diganti st.pyplot) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    status_counts = df['Status'].value_counts()

    bars = axes[0].bar(status_counts.index, status_counts.values,
                       color=colors_all, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Distribusi Status Mahasiswa (Semua Data)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Status'); axes[0].set_ylabel('Jumlah Mahasiswa')
    for bar, val in zip(bars, status_counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                     f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    axes[1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                colors=colors_all, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Proporsi Status Mahasiswa', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"- Dropout  : **{n_drop:,}** (32.12%)  \n"
        f"- Graduate : **{n_grad:,}** (49.93%)  \n"
        f"- Enrolled : **{n_enr:,}** (17.95%) — tidak digunakan untuk training"
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 4.2 FAKTOR KEUANGAN — identik notebook cell 10
    # ════════════════════════════════════════════════════════
    st.subheader("4.2 Analisis Faktor Keuangan (Dropout vs Graduate)")

    # ── kode identik notebook ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tuition = pd.crosstab(df_dg['Tuition_fees_up_to_date'], df_dg['Status'], normalize='index') * 100
    tuition.index = ['Nunggak UKT', 'UKT Lunas']
    tuition[['Dropout','Graduate']].plot(kind='bar', ax=axes[0], color=colors_dg, edgecolor='white', rot=0)
    axes[0].set_title('Status vs Pembayaran UKT\n(Dropout & Graduate)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Persentase (%)'); axes[0].legend(title='Status')

    scholar = pd.crosstab(df_dg['Scholarship_holder'], df_dg['Status'], normalize='index') * 100
    scholar.index = ['Non-Beasiswa', 'Penerima Beasiswa']
    scholar[['Dropout','Graduate']].plot(kind='bar', ax=axes[1], color=colors_dg, edgecolor='white', rot=0)
    axes[1].set_title('Status vs Kepemilikan Beasiswa\n(Dropout & Graduate)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Persentase (%)'); axes[1].legend(title='Status')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"**Insight:** Dropout rate saat nunggak UKT: **{tuition.loc['Nunggak UKT','Dropout']:.1f}%** | "
        f"Graduate rate penerima beasiswa: **{scholar.loc['Penerima Beasiswa','Graduate']:.1f}%**"
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 4.3 PERFORMA AKADEMIK — identik notebook cell 12
    # ════════════════════════════════════════════════════════
    st.subheader("4.3 Analisis Performa Akademik (Dropout vs Graduate)")

    # ── kode identik notebook ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    academic_cols = [
        ('Curricular_units_2nd_sem_approved', 'Unit Disetujui Sem 2'),
        ('Curricular_units_2nd_sem_grade',    'Nilai Rata-rata Sem 2'),
        ('Curricular_units_1st_sem_approved', 'Unit Disetujui Sem 1'),
        ('Curricular_units_1st_sem_grade',    'Nilai Rata-rata Sem 1'),
    ]
    for idx, (col, title) in enumerate(academic_cols):
        ax = axes[idx//2][idx%2]
        for status, color in zip(['Dropout','Graduate'], colors_dg):
            data = df_dg[df_dg['Status']==status][col]
            ax.hist(data, bins=20, alpha=0.65, color=color, label=status, edgecolor='none')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Nilai'); ax.set_ylabel('Frekuensi'); ax.legend()
    plt.suptitle('Distribusi Performa Akademik — Dropout vs Graduate', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg = df_dg.groupby('Status')['Curricular_units_2nd_sem_approved'].mean()
    st.markdown(
        f"**Insight:** Rata-rata unit lulus semester 2 — "
        f"Dropout: **{avg['Dropout']:.2f}**, Graduate: **{avg['Graduate']:.2f}**. "
        f"Selisih: {avg['Graduate']-avg['Dropout']:.2f} unit."
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 4.4 DEMOGRAFI — identik notebook cell 14
    # ════════════════════════════════════════════════════════
    st.subheader("4.4 Analisis Demografi")

    # ── kode identik notebook ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for status, color in zip(['Dropout','Graduate'], colors_dg):
        data = df_dg[df_dg['Status']==status]['Age_at_enrollment']
        axes[0].hist(data, bins=30, alpha=0.65, color=color, label=status, edgecolor='none')
    axes[0].set_title('Distribusi Usia saat Mendaftar\n(Dropout vs Graduate)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Usia'); axes[0].set_ylabel('Frekuensi'); axes[0].legend()
    axes[0].axvline(x=25, color='black', linestyle='--', alpha=0.4, label='Usia 25')

    gender = pd.crosstab(df_dg['Gender'], df_dg['Status'], normalize='index') * 100
    gender.index = ['Perempuan','Laki-laki']
    gender[['Dropout','Graduate']].plot(kind='bar', ax=axes[1], color=colors_dg, edgecolor='white', rot=0)
    axes[1].set_title('Status vs Gender\n(Dropout vs Graduate)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Persentase (%)'); axes[1].legend(title='Status')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg_age = df_dg.groupby('Status')['Age_at_enrollment'].mean()
    st.markdown(
        f"**Insight:** Rata-rata usia — Dropout: **{avg_age['Dropout']:.2f}**, "
        f"Graduate: **{avg_age['Graduate']:.2f}**. "
        f"Mahasiswa dropout rata-rata **{avg_age['Dropout']-avg_age['Graduate']:.1f} tahun lebih tua**."
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 4.5 CORRELATION HEATMAP — identik notebook cell 16
    # ════════════════════════════════════════════════════════
    st.subheader("4.5 Correlation Heatmap Fitur Utama")

    # ── kode identik notebook ────────────────────────────────
    top_features = [
        'Curricular_units_2nd_sem_approved','Curricular_units_2nd_sem_grade',
        'Curricular_units_1st_sem_approved','Curricular_units_1st_sem_grade',
        'Admission_grade','Previous_qualification_grade','Age_at_enrollment',
        'Tuition_fees_up_to_date','Scholarship_holder','Debtor',
        'Curricular_units_2nd_sem_evaluations','Curricular_units_1st_sem_evaluations',
        'GDP','Unemployment_rate','Gender'
    ]
    plt.figure(figsize=(12, 10))
    corr = df_dg[top_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Correlation Heatmap — Fitur Utama (Dropout & Graduate)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    st.markdown("**Insight:** Nilai dan unit lulus semester 1 & 2 sangat berkorelasi — keduanya penting untuk prediksi dropout.")
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 6.2 CONFUSION MATRIX + CV — identik notebook cell 27
    # ════════════════════════════════════════════════════════
    st.subheader("6.2 Evaluasi Model")

    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**Accuracy: {acc*100:.2f}%**")
    st.code(
        "              precision    recall  f1-score   support\n\n"
        "     Dropout       0.95      0.81      0.88       284\n"
        "    Graduate       0.89      0.98      0.93       442\n\n"
        "    accuracy                           0.91       726\n"
        "   macro avg       0.92      0.89      0.90       726\n"
        "weighted avg       0.92      0.91      0.91       726",
        language=None
    )

    # ── kode identik notebook ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_nb.classes_)
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix — Binary Classification', fontsize=12, fontweight='bold')

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    axes[1].bar(range(1, 6), cv_scores, color='#3498db', edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {cv_scores.mean():.4f}')
    axes[1].set_title('5-Fold Cross Validation Scores', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fold'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].set_ylim(0.85, 0.95)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"**Confusion Matrix:** TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}  \n"
        f"**5-Fold CV:** Mean={cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%"
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 6.3 FEATURE IMPORTANCE — identik notebook cell 29
    # ════════════════════════════════════════════════════════
    st.subheader("6.3 Feature Importance")

    # ── kode identik notebook ────────────────────────────────
    feat_imp = pd.Series(
        pipeline.named_steps['model'].feature_importances_, index=X.columns
    ).sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 7))
    bar_colors = ['#e74c3c' if v >= 0.1 else '#3498db' for v in feat_imp.values]
    plt.barh(feat_imp.index, feat_imp.values, color=bar_colors, edgecolor='white', linewidth=0.8)
    plt.title('Top 15 Feature Importance — Binary Random Forest', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    top5 = feat_imp.sort_values(ascending=False).head(5)
    st.markdown("**Top 5 Feature Importance:**")
    for i, (f, v) in enumerate(top5.items(), 1):
        st.markdown(f"  {i}. `{f}` — **{v*100:.2f}%**")
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 6.4 INFERENSI ENROLLED
    # ════════════════════════════════════════════════════════
    st.subheader("6.4 Inferensi pada Mahasiswa Enrolled")
    st.info(
        f"Model digunakan untuk memprediksi kemungkinan status akhir **{n_enr} mahasiswa** "
        "yang masih aktif (Enrolled). Data ini tidak digunakan dalam training model."
    )
    X_enrolled  = df_enrolled.drop('Status', axis=1)
    pred_enr    = pipeline.predict(X_enrolled)
    pred_labels = le.inverse_transform(pred_enr)
    hasil = Counter(pred_labels)

    ce1, ce2 = st.columns(2)
    ce1.metric("Berpotensi Graduate", f"{hasil['Graduate']}",
               f"{hasil['Graduate']/n_enr*100:.1f}%")
    ce2.metric("Berpotensi Dropout",  f"{hasil['Dropout']}",
               f"{hasil['Dropout']/n_enr*100:.1f}% — perlu perhatian!",
               delta_color="inverse")
    st.markdown(
        f"**Rekomendasi:** {hasil['Dropout']} mahasiswa Enrolled perlu mendapat "
        "perhatian dan intervensi khusus karena berpotensi dropout."
    )


# ════════════════════════════════════════════════════════════
# TAB 3 — INFO MODEL
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("Informasi Model")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### Spesifikasi Model
| Parameter | Nilai |
|-----------|-------|
| **Algoritma** | Random Forest Classifier |
| **Tipe** | Binary Classification |
| **Target** | Dropout vs Graduate |
| **n_estimators** | 200 |
| **Class Weight** | Balanced |
| **Random State** | 42 |
| **Test Size** | 20% (726 samples) |
| **Data Training** | 3.630 (Dropout + Graduate) |
| **Accuracy** | **91.18%** |
| **CV Score (5-fold)** | **90.50% ± 0.47%** |

### Metrik per Kelas
| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout** | 0.95 | 0.81 | 0.88 | 284 |
| **Graduate** | 0.89 | 0.98 | 0.93 | 442 |
| **Accuracy** | | | **0.91** | **726** |
""")
    with c2:
        st.markdown("""
### Top 5 Feature Importance
| Rank | Fitur | Importance |
|------|-------|------------|
| 1 | Curricular_units_2nd_sem_approved | **22.50%** |
| 2 | Curricular_units_1st_sem_approved | **12.08%** |
| 3 | Curricular_units_2nd_sem_grade | **11.26%** |
| 4 | Curricular_units_1st_sem_grade | **7.28%** |
| 5 | Tuition_fees_up_to_date | **5.00%** |

### Catatan Penting
- Model dilatih **hanya** pada data Dropout & Graduate (3.630 baris)
- Data **Enrolled** tidak diikutkan dalam training (belum memiliki label akhir)
- Data Enrolled hanya dipakai untuk **inferensi**
- Akurasi: 76.84% (3 kelas) → **91.18%** (binary)
- Pipeline: `StandardScaler` → `RandomForestClassifier`
""")
    st.markdown("---")
    st.markdown(
        "**Dikembangkan oleh:** Bilanawati Maulia  \n"
    )
