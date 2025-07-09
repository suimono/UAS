# Implementasi Case-Based Reasoning pada Kasus Tindak Pidana Khusus Korupsi Menggunakan Metode TF-IDF, BERT DAN HYBRID


Penelitian ini bertujuan untuk mengembangkan sistem berbasis Case-Based Reasoning (CBR) guna membantu proses analisis dan pencarian kasus serupa dalam tindak pidana khusus, khususnya korupsi. Untuk meningkatkan kualitas representasi teks dan akurasi temu kembali kasus, digunakan dua pendekatan pemrosesan bahasa alami, yaitu TF-IDF sebagai metode tradisional berbasis statistik, dan BERT sebagai model representasi berbasis pembelajaran mendalam (deep learning).Dan juga metode HYBRID.

# Gambaran Umum Proyek (Project Overview)
Dalam dunia hukum, putusan-putusan pengadilan dan kasus-kasus terdahulu merupakan sumber informasi yang sangat berharga. Proyek ini mengadopsi paradigma Case-Based Reasoning (CBR), di mana solusi untuk masalah baru dicari dengan mengadaptasi solusi dari masalah-masalah serupa yang pernah diselesaikan. Secara spesifik, kami membangun pipeline yang mencakup:

  1. Ekstraksi Data: Mengubah dokumen PDF yang tidak terstruktur menjadi data kasus yang bersih dan terstruktur.
  2. Pembentukan Kueri: Menghasilkan kueri evaluasi dari data kasus yang sudah ada, lengkap dengan ground truth.
  3. Retrieval Kasus: Menemukan kasus-kasus paling relevan dari database menggunakan berbagai algoritma pencarian.
  4. Prediksi Solusi: Memprediksi pasal hukum yang relevan berdasarkan konsensus dari kasus-kasus yang berhasil diretriev.
  5. Evaluasi Kinerja: Mengukur seberapa efektif sistem dalam melakukan retrieval dan prediksi.


        <img width="383" alt="Screenshot 2025-06-28 170704" src="https://github.com/user-attachments/assets/57e2e349-4370-4b6e-aed4-d4df03b30b14" />


## ✅ Alur Proses & Eksekusi

Eksekusinya :


### 1. Ekstrak PDF → Teks Mentah


        python scripts/01_pdf_to_text.py 
        
📁 Output ke: `data/raw/*.txt` 


### 2. Ekstrak Metadata → JSON Terstruktur


        python scripts/02_case_representation.py 
        
📁 Output ke: `data/processed/cases.json`  


### 3. Query & Retrieval

        python scripts/make_queries.py 
        
📁 Output: `exal/queries.json
        
        python scripts/03_retrieval.py      

📁 Output: `results/retrieved_cases.json


### 4. Prediksi Solusi


        python scripts/04_predict.py

📁 Output: `results/predictions_BERT.csv
📁 Output: `results/predictions_Hybrid.csv
📁 Output: `results/predictions_TF_IDF.csv

### 5. Evaluasi

        python scripts/05_evaluation.py

📁 Output: `retrieval_metrics.csv` & `prediction_metrics.csv`


### output prediction_metrics.csv`

        Metrik,BERT,Hybrid,TF-IDF
        Accuracy,87.50%,87.50%,93.75%
        F1-Score,45.71%,46.19%,47.62%
        Precision,60.00%,60.62%,62.50%
        Recall,36.92%,37.31%,38.46%

### output prediction_metrics.csv
        Metrik,BERT,Hybrid,TF-IDF
        F1-Score@5,0.2917,0.2917,0.2917
        MAP,0.7246,0.7246,0.7246
        MRR,0.7246,0.7246,0.7246
        Precision@5,0.1750,0.1750,0.1750
        Recall@5,0.8750,0.8750,0.8750



## 💻 Instalasi

### 🔧 Persiapan

1. Pastikan Python 3.8+ sudah terpasang
2. Install semua dependensi dengan:

```bash
pip install -r requirements.txt
```

### 📦 requirements.txt

## 🧪 Contoh `queries.json`

```json
[
  {
    "query_id": "q1",
    "query_text": "Terdakwa menerima suap dalam proyek pembangunan jalan",
    "ground_truth": "putusan_113_k_pid.sus_2020_..."
  }
]
```

---

## 📓 Menjalankan versi Notebook

> Alternatif interaktif untuk skrip Python

### Jalankan Jupyter:


### jupyter notebook


### File Notebook:

Alur pemasangan seperti File Py di atas dengan urutan 
* 📘 `notebook/01_pdf_to_text.ipynb`
* 📘 `notebook/02_case_representation.ipynb`
* 📘 `notebook/make_queries.ipynb`
* 📘 `notebook/03_retrieval.ipynb`
* 📘 `notebook/04_predict.ipynb`
* 📘 `notebook/05_evaluation.ipynb`




## 📄 Laporan
📂 `reports/laporan_CBR.docx`
Berisi ringkasan tahapan, diagram pipeline, metrik evaluasi, serta diskusi kasus-kasus yang gagal (error analysis).

---

## ✨ Credits

* Dibuat oleh: Ellyas Prambudyas
* Proyek ini ditujukan untuk pembelajaran sistem penalaran berbasis kasus (CBR) dengan domain hukum (putusan pengadilan tindak pidana khusunya KORUPSI).
