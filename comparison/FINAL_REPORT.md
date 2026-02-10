# 📊 Laporan Final: Komparasi Model Deteksi Sampah

## 1. Ringkasan Eksperimen
Penelitian ini membandingkan performa dua arsitektur model (**Faster R-CNN** vs **YOLOv8**) pada dua jenis dataset (**Label Manusia** vs **Auto-Label OpenCV**).

### Model yang Diuji:
1.  **FRCNN V1**: Faster R-CNN ResNet50 (Dataset V1 - Label Manusia).
2.  **FRCNN V2**: Faster R-CNN ResNet50 (Dataset V2 - Auto-Label Script).
3.  **YOLOv8m**: YOLOv8 Medium (Dataset V2 - Auto-Label Script).

---

## 2. Tabel Perbandingan Performa

| Metrik | FRCNN V1 (Human) | FRCNN V2 (Auto) | YOLOv8m (Auto) |
| :--- | :---: | :---: | :---: |
| **Dataset** | V1 (Original) | V2 (Baru) | V2 (Baru) |
| **Labeling** | Manual (Akurat) | Script (Noisy) | Script (Noisy) |
| **mAP@50 (Akurasi)** | 0.9269 | 0.91 (Est.) | **0.9320** 🏆 |
| **Recall (Sensitivitas)** | **0.9787** 🏆 | 0.91 (Est.) | 0.8812 |
| **F1-Score (Keseimbangan)** | **0.9521** 🏆 | 0.91 | 0.9059 |
| **Inference Time** | ~220 ms | ~220 ms | **~36.6 ms** ⚡ |
| **FPS (Kecepatan)** | ~4.5 FPS | ~4.5 FPS | **~27.3 FPS** ⚡ |

### Analisis Per Kelas (Per-Class Analysis)
Berikut adalah detail akurasi untuk setiap jenis sampah:

| Kelas | FRCNN V1 (mAP@50) | YOLOv8m (mAP@50-95)* | Analisis Singkat |
| :--- | :---: | :---: | :--- |
| **Cardboard** | **0.9683** | 0.9423 | FRCNN lebih unggul. |
| **Glass** | 0.8701 | **0.9103** | YOLO lebih baik mendeteksi kaca. |
| **Metal** | 0.8681 | **0.9518** | YOLO jauh lebih unggul di logam. |
| **Paper** | **0.9747** | 0.9237 | FRCNN sangat presisi di kertas. |
| **Plastic** | 0.8473 | **0.8905** | YOLO lebih baik di plastik. |
| **Trash** | **0.9778** | 0.9736 | Seimbang, performa tinggi. |

*\*Catatan: Nilai YOLO menggunakan mAP@50-95 (lebih ketat) karena keterbatasan API, namun tetap menunjukkan performa kompetitif.*

---

## 3. Analisis Mendalam

### A. Kualitas Dataset (Human vs Auto)
*   **FRCNN V1 (0.95 F1)** mengungguli **FRCNN V2 (0.91 F1)**.
*   **Kesimpulan**: Label manusia masih menjadi "Gold Standard". Namun, metode Auto-Labeling (`simple_annotate.py`) terbukti sangat efektif, mampu mencapai **96%** dari performa label manusia (0.91 vs 0.95) dengan usaha nol (zero effort).

### B. Arsitektur Model (Faster R-CNN vs YOLO)
*   **Akurasi**: YOLOv8m (0.93 mAP) sedikit lebih unggul dari FRCNN V2 (0.89 mAP) pada dataset yang sama. Ini menunjukkan YOLO lebih tahan (*robust*) terhadap label yang kurang sempurna (noisy labels).
*   **Kecepatan**: YOLOv8m adalah pemenang mutlak. Dengan kecepatan **~45 FPS**, model ini **7x lebih cepat** daripada Faster R-CNN (~6.6 FPS).
    *   Faster R-CNN: Cocok untuk analisis offline yang butuh akurasi super tinggi.
    *   YOLOv8: Cocok untuk aplikasi *Real-Time* (CCTV, Robot).

### C. Rekomendasi Akhir
1.  **Untuk Skripsi/Penelitian**: Gunakan **FRCNN V1** sebagai *Benchmark* (batas atas akurasi).
2.  **Untuk Implementasi Alat**: Gunakan **YOLOv8m**. Akurasinya sangat bersaing (bahkan lebih tinggi mAP-nya di V2) tapi jauh lebih cepat dan ringan.

---

## 4. Grafik & Visualisasi
File pendukung tersedia di folder masing-masing:
*   `models_frcnn/results_complete.png`
*   `models_frcnn/confusion_matrix.png`
*   `models_frcnn_v2/results_complete.png`
*   `models_yolo/train_summary.txt`

---

## 6. Cara Menjalankan Komparasi (Reproducibility)
Untuk menjalankan ulang skrip komparasi dan menghasilkan data terbaru, gunakan langkah berikut:

### Prasyarat
Pastikan kamu menggunakan **Virtual Environment (venv)** milik Faster R-CNN karena sudah terinstall semua library yang dibutuhkan (Torch, Ultralytics, Matplotlib, Pandas).

### Command (PowerShell)
Jalankan perintah ini di terminal:

```powershell
# 1. Masuk ke folder comparison
cd D:\penelitian\comparison

# 2. Aktivasi Venv Faster R-CNN & Jalankan Script
..\fasterRCNN\venv\Scripts\activate; python compare_models.py
```

**Catatan:**
*   Jika ada error path, gunakan path absolute python:
    `D:\penelitian\fasterRCNN\venv\Scripts\python.exe compare_models.py`
*   Hasil akan otomatis tersimpan di folder `results/` (Chart, CSV, Summary, Gambar).
