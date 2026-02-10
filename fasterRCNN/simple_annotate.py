import cv2
import numpy as np
import os
import shutil
import random
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_DIR = "D:/penelitian/fasterRCNN/datasetV2"
OUTPUT_DIR = "D:/penelitian/fasterRCNN/dataset_v2_final"
TARGET_SIZE = (512, 512)
VAL_SPLIT = 0.2

# Class Mapping (Folder Name -> ID)
# Pastikan nama folder di datasetV2 sesuai (Case Insensitive)
CLASS_MAP = {
    'cardboard': 0,
    'glass': 1,
    'metal': 2,
    'paper': 3,
    'plastic': 4,
    'trash': 5
}

def find_bbox(img):
    """
    Find bounding box of the largest object in the image using OpenCV.
    Returns: (x, y, w, h) in absolute pixels.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding (Otsu's method is usually good for distinct backgrounds)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Filter noise: If box is too small (< 5% of image), ignore
    img_h, img_w = img.shape[:2]
    if w < img_w * 0.05 or h < img_h * 0.05:
        return None
        
    return x, y, w, h

def process_image(src_path, dest_img_path, dest_lbl_path, class_id):
    # Read image
    img = cv2.imread(src_path)
    if img is None:
        print(f"⚠️ Error reading: {src_path}")
        return

    # Resize
    img = cv2.resize(img, TARGET_SIZE)
    
    # Find BBox
    bbox = find_bbox(img)
    
    if bbox is None:
        # Fallback: Use center crop (60% of image)
        h, w = TARGET_SIZE
        box_w, box_h = int(w * 0.6), int(h * 0.6)
        box_x = int((w - box_w) / 2)
        box_y = int((h - box_h) / 2)
        bbox = (box_x, box_y, box_w, box_h)
    
    # Convert to YOLO Format (Normalized Center X, Center Y, W, H)
    x, y, w_box, h_box = bbox
    img_h, img_w = TARGET_SIZE
    
    x_center = (x + w_box / 2) / img_w
    y_center = (y + h_box / 2) / img_h
    width = w_box / img_w
    height = h_box / img_h
    
    # Save Image
    cv2.imwrite(dest_img_path, img)
    
    # Save Label
    with open(dest_lbl_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found at {INPUT_DIR}")
        return

    print(f"🚀 Starting Simple Auto-Annotation...")
    print(f"📂 Input: {INPUT_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    # Collect all images
    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Determine class from folder name
                folder_name = os.path.basename(root).lower()
                if folder_name in CLASS_MAP:
                    all_files.append((os.path.join(root, file), CLASS_MAP[folder_name]))
                else:
                    print(f"⚠️ Skipping file in unknown folder '{folder_name}': {file}")

    print(f"Found {len(all_files)} valid images.")
    random.shuffle(all_files)
    
    # Split Train/Val
    split_idx = int(len(all_files) * (1 - VAL_SPLIT))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Process
    for split, files in [('train', train_files), ('val', val_files)]:
        img_out_dir = os.path.join(OUTPUT_DIR, 'images', split)
        lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels', split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for src_path, class_id in tqdm(files, desc=f"Processing {split}"):
            filename = os.path.basename(src_path)
            parent_name = os.path.basename(os.path.dirname(src_path))
            
            # Cek apakah nama file sudah mengandung nama folder (Case Insensitive)
            if filename.lower().startswith(parent_name.lower()):
                new_filename = filename
            else:
                new_filename = f"{parent_name}_{filename}"
            
            dest_img_path = os.path.join(img_out_dir, new_filename)
            dest_lbl_path = os.path.join(lbl_out_dir, os.path.splitext(new_filename)[0] + ".txt")
            
            process_image(src_path, dest_img_path, dest_lbl_path, class_id)
            
    print(f"\n✅ Selesai! Dataset tersimpan di: {OUTPUT_DIR}")
    print(f"👉 Silakan cek beberapa gambar di folder output untuk memastikan kotaknya pas.")

if __name__ == "__main__":
    main()
