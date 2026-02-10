import os
import shutil
import random
from PIL import Image

# --- KONFIGURASI ---
SOURCE_DIR = r'D:\penelitian\yolo\datasetV2'
DEST_DIR = r'D:\penelitian\yolo\datasetLabeledV2'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
TARGET_SIZE = (512, 512)  # (width, height) resize
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
# -------------------

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_and_save(src_path, dest_path, size):
    """Resize gambar ke ukuran target dan simpan ke lokasi baru."""
    try:
        img = Image.open(src_path)
        img = img.convert("RGB")  # pastikan format seragam
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(dest_path)
    except Exception as e:
        print(f"⚠️ Gagal resize {src_path}: {e}")

def split_data_and_labels():
    print(f"📦 Memulai split dataset + resize...")
    print(f"Sumber: {SOURCE_DIR}")
    print(f"Tujuan: {DEST_DIR}\n")

    if not os.path.isdir(SOURCE_DIR):
        print(f"❌ Error: '{SOURCE_DIR}' tidak ditemukan.")
        return

    random.seed(RANDOM_SEED)

    if os.path.exists(DEST_DIR):
        print(f"🧹 Menghapus folder lama: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)

    # Buat struktur folder output
    for subset in ['train', 'val']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, sub, subset), exist_ok=True)

    class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"Ditemukan {len(class_names)} kelas: {', '.join(class_names)}")

    for class_index, class_name in enumerate(class_names):
        print(f"\n🔹 Memproses kelas: {class_name} ({class_index})")

        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        random.shuffle(files)

        split_point = int(len(files) * TRAIN_RATIO)
        train_files, val_files = files[:split_point], files[split_point:]

        print(f"  Train: {len(train_files)} | Val: {len(val_files)}")

        # Helper untuk memproses file
        def process_files(file_list, subset):
            for i, file_name in enumerate(file_list, 1):
                src_img = os.path.join(source_class_dir, file_name)
                img_dest = os.path.join(DEST_DIR, 'images', subset, file_name)
                lbl_dest = os.path.join(DEST_DIR, 'labels', subset, os.path.splitext(file_name)[0] + ".txt")

                resize_and_save(src_img, img_dest, TARGET_SIZE)

                # bounding box default di tengah
                with open(lbl_dest, "w") as f:
                    f.write(f"{class_index} 0.5 0.5 0.6 0.6\n")

                if i % 100 == 0 or i == len(file_list):
                    print(f"   ✅ {i}/{len(file_list)} {subset} diproses...")

        process_files(train_files, "train")
        process_files(val_files, "val")

    print("\n🎯 Split dan resize dataset selesai!")
    print(f"📁 Dataset tersimpan di: {DEST_DIR}")

if __name__ == "__main__":
    split_data_and_labels()
