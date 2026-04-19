# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

**Nama:** Bilanawati Maulia Masruroh
**Username Dicoding:** bilanawati_99  
**Email:** bilanamaulia@gmail.com

---

## Business Understanding

### Latar Belakang
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Meskipun telah mencetak banyak lulusan dengan reputasi baik, institusi ini menghadapi permasalahan serius berupa tingginya angka dropout mahasiswa. Jumlah dropout yang tinggi tentunya menjadi salah satu masalah yang besar bagi sebuah institusi pendidikan.

Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin mahasiswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis
1. Bagaimana mengidentifikasi faktor-faktor utama yang berkontribusi terhadap dropout mahasiswa?
2. Bagaimana membangun model machine learning yang dapat memprediksi apakah seorang mahasiswa akan **Dropout** atau **Graduate**?
3. Bagaimana menyediakan dashboard monitoring yang memudahkan pihak institusi memantau performa mahasiswa?

### Cakupan Proyek
- **Exploratory Data Analysis (EDA)**: Memahami distribusi data dan hubungan antar fitur
- **Data Preprocessing**: Filtering data (Dropout & Graduate saja), encoding, scaling, train-test split
- **Machine Learning Modeling**: Pelatihan model binary classification (Dropout vs Graduate)
- **Business Dashboard**: Visualisasi data untuk monitoring performa mahasiswa via Streamlit
- **Deployment**: Prototype Streamlit yang dapat diakses secara online

### Persiapan

**Sumber data:** [students_performance.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

**Setup environment:**

```bash
# 1. Pastikan Python 3.10+ sudah terinstall
python --version

# 2. Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install seluruh dependencies
pip install -r requirements.txt

# 4. Jalankan prototype machine learning (Streamlit)
streamlit run app.py

# 5. Jalankan notebook untuk melihat seluruh proses analisis
jupyter notebook notebook.ipynb
```

---

## Business Dashboard

Dashboard business dibuat menggunakan **Streamlit** dan dapat diakses secara online melalui Streamlit Community Cloud.

Dashboard menampilkan:
- KPI utama: total mahasiswa, jumlah dropout, graduate, dan enrolled
- Distribusi status mahasiswa (bar chart dan proporsi)
- Analisis dropout rate vs status pembayaran UKT
- Distribusi usia mahasiswa berdasarkan status (Dropout vs Graduate)
- Distribusi unit lulus semester 2 (Dropout vs Graduate)
- Prediksi status akhir mahasiswa Enrolled menggunakan model ML

**Link Dashboard:** (https://my-dash-bila.streamlit.app/)

**Screenshot dashboard:** `bilanawati_99-dashboard.png`

---

## Menjalankan Sistem Machine Learning

### Instalasi dan Menjalankan Lokal

```bash
# 1. Pastikan virtual environment sudah aktif (lihat bagian Persiapan)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

### Akses Online (Streamlit Community Cloud)

**Link Prototype:** (https://my-dash-bila.streamlit.app/)

### Cara Penggunaan Prototype

1. Buka aplikasi melalui link di atas
2. Pilih tab **"Prediksi Mahasiswa"**
3. Isi formulir data mahasiswa:
   - Data Pribadi (usia, gender, status pernikahan)
   - Data Akademik (program studi, nilai masuk, kualifikasi sebelumnya)
   - Performa Akademik Semester 1 & 2 (unit diambil, lulus, nilai rata-rata)
   - Data Finansial (status UKT, beasiswa, debitur)
4. Klik tombol **"Prediksi Sekarang"**
5. Lihat hasil prediksi (Dropout atau Graduate) beserta tingkat kepercayaan model

> **Catatan**: Model memprediksi status **Dropout** atau **Graduate**. Data mahasiswa yang masih berstatus *Enrolled* (aktif) juga dapat diinput untuk mengetahui prediksi status akhir mereka di masa depan — sebagai bagian dari tahap inferensi.

---

## Conclusion

### Temuan Utama

1. **Angka Dropout Signifikan**: Dari 4.424 mahasiswa, terdapat mahasiswa yang dropout dan graduate dalam jumlah yang signifikan. Analisis lengkap tersedia di notebook.

2. **Faktor Akademik Dominan**: Unit kurikulum yang disetujui di semester 2 adalah prediktor terkuat (importance: 22.50%). Mahasiswa dropout rata-rata hanya lulus 1.36 unit di semester 2, jauh di bawah graduate yang rata-rata 5.90 unit.

3. **Faktor Keuangan Kritis**: Mahasiswa yang menunggak UKT memiliki dropout rate sangat tinggi. Penerima beasiswa sebaliknya memiliki graduate rate yang jauh lebih tinggi.

4. **Usia Berpengaruh**: Mahasiswa dropout rata-rata 3.3 tahun lebih tua saat mendaftar, mengindikasikan beban kerja atau keluarga sebagai faktor.

5. **Model Binary lebih Valid**: Dengan hanya menggunakan data Dropout dan Graduate (status yang sudah final), model mencapai akurasi **91.18%** (CV: 90.50% ± 0.47%). Data Enrolled tidak digunakan dalam training karena belum memiliki label akhir — melainkan hanya pada tahap inferensi.

6. **Inferensi Enrolled**: Dari 794 mahasiswa Enrolled, model memprediksi 355 (44.7%) berpotensi dropout dan perlu mendapat perhatian segera.

### Rekomendasi Action Items

1. **Sistem Early Warning Akademik**: Implementasikan monitoring otomatis untuk mahasiswa yang tidak lulus unit apapun di semester pertama sebagai red flag utama. Lakukan intervensi konseling segera.

2. **Program Bantuan Keuangan Proaktif**: Identifikasi mahasiswa yang menunggak UKT sejak awal semester dan tawarkan program cicilan, beasiswa darurat, atau keringanan biaya sebelum mereka memutuskan dropout.

3. **Mentoring Mahasiswa Berisiko Tinggi**: Buat program mentoring dan konseling khusus untuk mahasiswa berusia >25 tahun, mahasiswa dengan nilai rendah di semester pertama, dan mahasiswa yang teridentifikasi berisiko tinggi oleh sistem prediksi ML.

4. **Perluas Program Beasiswa**: Data menunjukkan penerima beasiswa memiliki tingkat kelulusan jauh lebih tinggi. Perluas program beasiswa berbasis kebutuhan dan prestasi.

5. **Prediksi Berkala untuk Mahasiswa Enrolled**: Jalankan model secara rutin pada mahasiswa yang masih aktif (Enrolled) untuk mendeteksi potensi dropout sedini mungkin dan mengambil tindakan preventif.
