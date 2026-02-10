from ultralytics import YOLO
import time
import os
import torch
import multiprocessing

def main():
    # === 1. Optimasi GPU (opsional tapi direkomendasikan) ===
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # === 2. Mulai timer ===
    start_time = time.time()

    # === 3. Inisialisasi model ===
    model = YOLO('yolov8m.pt') # Upgrade ke Medium

    # === 4. Path output di Drive D ===
    OUTPUT_DIR = "D:/penelitian/yolo/models_yolo"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 5. Training model ===
    results = model.train(
        data='D:/penelitian/yolo/dataset.yaml',  # Path ke dataset YAML
        epochs=50,
        imgsz=512,
        batch=8,                                 # Turunkan ke 8 biar aman di VRAM 4GB
        device=0,                                # Gunakan GPU utama
        project=OUTPUT_DIR,                      # Folder output custom (Drive D)
        name='trash_yolov8m',
        workers=4,                               # Threads DataLoader
        verbose=True,
        patience=10,                             # Early stopping 10 epoch
        save=True,
        exist_ok=True,
        
        # Augmentasi 
        fliplr=0.5,                              # Random Horizontal Flip 50% (Sama seperti FRCNN)
        mosaic=1.0,                              # Mosaic 100% (Kekuatan Penuh YOLO)
        hsv_h=0.015,                             # Color Jitter: Hue
        hsv_s=0.7,                               # Color Jitter: Saturation
        hsv_v=0.4,                               # Color Jitter: Brightness
    )

    # 6. Hitung waktu total
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n==================== TRAINING SELESAI ====================")
    print(f" Model disimpan di: {os.path.abspath(os.path.join(OUTPUT_DIR, 'trash_yolov8m'))}")
    print(f" Total waktu training: {int(hours)} jam {int(minutes)} menit {int(seconds)} detik")

    # === 7. Evaluasi model ===
    print("\n Evaluasi model terhadap data validasi...")
    metrics = model.val()
    print(f" Evaluasi selesai. Hasil mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

    # === 8. Simpan hasil ke log di Drive D ===
    log_path = os.path.join(OUTPUT_DIR, "train_summary.txt")
    with open(log_path, "w") as f:
        f.write("=== YOLOv8 Training Summary ===\n")
        f.write(f"Total waktu training: {int(hours)} jam {int(minutes)} menit {int(seconds)} detik\n")
        f.write(f"mAP@50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@50-95: {metrics.box.map:.4f}\n")
        f.write("Model: yolov8m.pt\n")
        f.write("Patience (early stopping): 10\n")
        f.write("Dataset: TrashNet (6 classes)\n")
    print(f"📝 Ringkasan hasil disimpan di: {log_path}")

# === 9. Tambahkan blok aman untuk Windows ===
if __name__ == "__main__":
    multiprocessing.freeze_support()  # WAJIB di Windows
    main()
