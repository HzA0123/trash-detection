# Panduan Menjalankan Komparasi Model (Faster R-CNN vs YOLOv8)

Script ini membandingkan performa model Faster R-CNN (V1) dan YOLOv8m (V2) secara side-by-side.

## 1. Prasyarat
Script ini menggunakan environment dari **Faster R-CNN** karena sudah lengkap (Torch, Ultralytics, Matplotlib, dll).
Jangan buat venv baru di folder ini.

## 2. Cara Menjalankan (Recommended)
Cara paling aman dan anti-ribet adalah menggunakan path absolute ke python di venv `fasterRCNN`.

Buka terminal di folder ini (`D:\penelitian\comparison`), lalu jalankan:

```powershell
D:\penelitian\fasterRCNN\venv\Scripts\python.exe compare_models.py
```

## 3. Output
Setelah script selesai, cek folder `results/` untuk melihat:
1.  **`comparison_summary.txt`**: Rangkuman angka mAP, Recall, F1-Score.
2.  **`comparison_chart.png`**: Grafik perbandingan kecepatan (Inference Time & FPS).
3.  **`comparison_metrics.csv`**: Data detail waktu proses per gambar.
4.  **`compare_*.jpg`**: Gambar hasil deteksi side-by-side.

---
**Troubleshooting:**
Jika muncul error `ModuleNotFoundError`, pastikan kamu menggunakan command di atas (yang pakai `python.exe` dari folder `fasterRCNN\venv`), bukan `python` biasa.
