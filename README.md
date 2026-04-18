# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

**Nama:** Bilanawati  
**Username Dicoding:** bilanawati_99  
**Email:** bilanamaulia@gmail.com

---

## Business Understanding

### Latar Belakang
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Meskipun telah mencetak banyak lulusan dengan reputasi baik, institusi ini menghadapi permasalahan serius: tingginya angka **dropout mahasiswa yang mencapai 32.12%** dari total 4.424 mahasiswa.

Tingginya angka dropout berdampak negatif terhadap reputasi institusi, pendapatan, dan akreditasi. Oleh karena itu, dibutuhkan sistem prediksi berbasis machine learning untuk mendeteksi sedini mungkin mahasiswa yang berpotensi dropout agar dapat diberikan intervensi dan bimbingan khusus.

### Permasalahan Bisnis
1. Bagaimana mengidentifikasi faktor-faktor utama yang berkontribusi terhadap dropout mahasiswa?
2. Bagaimana membangun model machine learning yang dapat memprediksi status mahasiswa (Dropout / Enrolled / Graduate) secara akurat?
3. Bagaimana menyediakan dashboard monitoring yang memudahkan pihak institusi memantau performa mahasiswa?

### Cakupan Proyek
- **Exploratory Data Analysis (EDA)**: Memahami distribusi data dan hubungan antar fitur
- **Data Preprocessing**: Encoding target variable, scaling, dan train-test split
- **Machine Learning Modeling**: Pelatihan dan evaluasi Random Forest Classifier
- **Business Dashboard**: Visualisasi data menggunakan Metabase (dijalankan via Docker)
- **Deployment**: Prototype Streamlit yang dapat diakses secara online

### Persiapan

**Sumber data:** Dataset performa mahasiswa Jaya Jaya Institut (`data.csv`) — 4.424 baris, 37 kolom, tanpa missing values.

**Setup environment:**
```bash
pip install -r requirements.txt
```

---

## Business Dashboard

Dashboard business dibuat menggunakan **Metabase** untuk memudahkan pihak institusi dalam:
- Memantau distribusi status mahasiswa (Dropout, Enrolled, Graduate)
- Menganalisis pengaruh faktor keuangan (UKT, beasiswa, debitur) terhadap dropout
- Memvisualisasikan performa akademik per semester
- Monitoring tren usia mahasiswa berdasarkan status

**Email Metabase:** root@mail.com  
**Password Metabase:** root123

**Cara menjalankan Metabase via Docker:**
```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```
Akses dashboard di `http://localhost:3000`, login dengan email dan password di atas.

**Screenshot dashboard:** `bilanawati_99-dashboard.png`

---

## Menjalankan Sistem Machine Learning

### Instalasi Lokal

```bash
# 1. Clone atau download proyek
git clone https://github.com/bilanawati99/jaya-jaya-institut.git
cd jaya-jaya-institut

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

### Akses Online (Streamlit Community Cloud)

🔗 **Link Prototype:** [https://jaya-jaya-institut-dropout.streamlit.app](https://jaya-jaya-institut-dropout.streamlit.app)

### Cara Penggunaan Prototype

1. Buka aplikasi melalui link di atas
2. Pilih tab **"🔮 Prediksi Mahasiswa"**
3. Isi formulir data mahasiswa:
   - Data Pribadi (usia, gender, status pernikahan)
   - Data Akademik (program studi, nilai masuk, kualifikasi sebelumnya)
   - Performa Akademik Semester 1 & 2 (unit diambil, lulus, nilai rata-rata)
   - Data Keuangan (status UKT, beasiswa, debitur)
4. Klik tombol **"🔮 Prediksi Sekarang"**
5. Lihat hasil prediksi dan probabilitas per kelas

---

## Conclusion

### Temuan Utama

1. **Angka Dropout Tinggi**: 32.12% mahasiswa (1.421 dari 4.424) melakukan dropout — masalah serius yang membutuhkan penanganan segera.

2. **Faktor Akademik adalah Prediktor Terkuat**: Unit kurikulum yang disetujui dan nilai semester 1 & 2 merupakan prediktor paling signifikan. Mahasiswa yang tidak mengikuti evaluasi (nilai = 0) di semester pertama hampir pasti dropout.

3. **Faktor Keuangan Kritis**: Mahasiswa yang menunggak UKT memiliki dropout rate hingga **86%**. Penerima beasiswa sebaliknya memiliki graduate rate yang jauh lebih tinggi.

4. **Usia Berpengaruh**: Mahasiswa yang lebih tua saat mendaftar (>25 tahun) cenderung lebih berisiko dropout, kemungkinan karena beban kerja atau keluarga.

5. **Model Machine Learning**: Random Forest mencapai akurasi **76.84%** dengan CV score **77.40% ± 0.79%** — stabil dan andal untuk sistem early warning dropout.

### Rekomendasi Action Items

1. **Sistem Early Warning Akademik**: Implementasikan monitoring otomatis untuk mahasiswa yang tidak mengikuti evaluasi di semester pertama. Jika mahasiswa tidak lulus satupun unit mata kuliah di semester 1, segera lakukan intervensi konseling akademik.

2. **Program Bantuan Keuangan Proaktif**: Identifikasi mahasiswa yang menunggak UKT sejak awal semester dan proaktif tawarkan program cicilan, beasiswa darurat, atau keringanan biaya sebelum mereka memutuskan dropout.

3. **Mentoring Mahasiswa Berisiko Tinggi**: Buat program mentoring dan konseling khusus untuk: mahasiswa berusia >25 tahun, mahasiswa dengan IPK rendah di semester pertama, dan mahasiswa yang teridentifikasi berisiko tinggi oleh sistem prediksi ML.

4. **Perluas Program Beasiswa**: Data menunjukkan penerima beasiswa memiliki tingkat kelulusan jauh lebih tinggi. Perluas program beasiswa prestasi dan beasiswa berbasis kebutuhan ekonomi untuk mahasiswa yang berpotensi tapi terkendala finansial.

5. **Dashboard Real-time & Rutinitas Review**: Gunakan business dashboard untuk monitoring performa mahasiswa secara berkala — minimal setiap awal dan akhir semester — sehingga potensi dropout dapat dideteksi dan ditangani sejak dini.
