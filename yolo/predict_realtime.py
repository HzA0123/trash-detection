# predict_realtime.py
from ultralytics import YOLO
import cv2
import time

# ==== KONFIGURASI ====
MODEL_PATH = r"D:\penelitian\yolo\models_yolo\trash_yolov8s\weights\best.pt"  # model hasil training
DEVICE = "cuda"  # "cuda" untuk GPU, "cpu" jika tanpa GPU
CONF_THRESHOLD = 0.3  # confidence minimum untuk menampilkan deteksi
# ======================

# 🧠 Muat model YOLO
print("🧠 Memuat model YOLO...")
model = YOLO(MODEL_PATH)

# Aktifkan kamera
cap = cv2.VideoCapture(0)  # 0 = kamera default
if not cap.isOpened():
    print("❌ Tidak dapat mengakses kamera.")
    exit()

print("✅ Kamera aktif. Tekan 'Q' untuk keluar.")

# Loop deteksi real-time
prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    # Hitung waktu untuk FPS
    start_time = time.time()

    # Jalankan deteksi YOLO
    results = model.predict(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)

    # Render hasil deteksi ke frame
    annotated_frame = results[0].plot()

    # Hitung FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela
cap.release()
cv2.destroyAllWindows()
print("🛑 Deteksi real-time dihentikan.")
