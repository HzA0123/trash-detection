
from ultralytics import YOLO
import os
import time

# --- KONFIGURASI UTAMA ---
MODEL_PATH = r"D:\penelitian\yolo\models_yolo\trash_yolov8s\weights\best.pt"  # Model hasil training
IMAGE_PATH = r"D:\penelitian\yolo\plastikuji1.jpg"  # Gambar target
OUTPUT_PATH = r"D:\penelitian\yolo\results_yolo\predict_single"               # Folder output hasil prediksi
DEVICE = "cuda"  # Gunakan "cuda" untuk GPU, "cpu" kalau tanpa GPU

# --- PERSIAPAN OUTPUT DIRECTORY ---
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- MUAT MODEL ---
print("🧠 Memuat model YOLO...")
model = YOLO(MODEL_PATH)

# --- CEK GAMBAR TARGET ---
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ File gambar tidak ditemukan di lokasi: {IMAGE_PATH}")
else:
    print(f"\n📸 Gambar yang akan diprediksi: {IMAGE_PATH}")

# --- MULAI INFERENSI ---
start_time = time.time()
results = model.predict(
    source=IMAGE_PATH,
    save=True,             # simpan gambar hasil dengan bounding box
    project=OUTPUT_PATH,   # lokasi output
    name="",               # biar hasilnya langsung di folder ini
    exist_ok=True,
    conf=0.25,             # confidence threshold
    device=DEVICE
)
end_time = time.time()

# --- HITUNG WAKTU DAN FPS ---
inference_time = end_time - start_time
fps = 1 / inference_time if inference_time > 0 else 0

# --- CETAK HASIL DETEKSI ---
if results:
    for r in results:
        print("\n🧾 --- Hasil Deteksi ---")
        boxes = r.boxes
        if len(boxes) == 0:
            print("Tidak ada objek terdeteksi dengan confidence >= 0.25.")
        else:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"🟩 ID kelas: {cls_id}, Confidence: {conf:.2f}, Koordinat: {xyxy}")
else:
    print("⚠️ Tidak ada hasil inferensi yang dihasilkan.")

# --- RINGKASAN OUTPUT ---
print("\n✅ Prediksi selesai!")
print(f"📁 Hasil disimpan di: {OUTPUT_PATH}")
print(f"⏱️ Waktu inferensi: {inference_time:.4f} detik")
print(f"⚡ Kecepatan: {fps:.2f} FPS")
