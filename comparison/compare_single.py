import torch
import torchvision
import cv2
import numpy as np
import os
import sys
import argparse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from ultralytics import YOLO

# === CONFIGURATION ===
FRCNN_MODEL_PATH = r"D:\penelitian\fasterRCNN\models_frcnn\best_model.pth"
YOLO_MODEL_PATH = r"D:\penelitian\yolo\models_yolo\trash_yolov8m\weights\best.pt"
OUTPUT_DIR = r"D:\penelitian\comparison\results-single"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === IMAGE TO TEST (Ganti path ini sesuai keinginan) ===
IMAGE_PATH = r"D:\penelitian\yolo\datasetLabeledV2\images\val\Metal201.jpg" 

# === FRCNN SETUP ===
def get_frcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def frcnn_inference(model, image_path, device, threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = T.ToTensor()(image_rgb).to(device)
    
    with torch.no_grad():
        prediction = model([image_tensor])[0]
        
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]

# === VISUALIZATION ===
def draw_boxes(image, boxes, scores, labels, title, color=(0, 255, 0)):
    img_copy = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        # Handle label mapping (FRCNN is 1-based, YOLO is 0-based usually, but here we map to strings)
        # For FRCNN: label is index in CLASSES (if we assume 1-based mapping from training)
        # For YOLO: label is index in CLASSES
        
        # Note: In compare_models.py we used: class_name = CLASSES[label - 1] if isinstance(label, (int, np.integer)) else label
        # Let's stick to that logic for safety, assuming FRCNN returns 1-based indices.
        
        if isinstance(label, (int, np.integer)):
             # Heuristic: if label is 0-based (YOLO) it might be < len(CLASSES). 
             # If it's 1-based (FRCNN), it might be <= len(CLASSES) but > 0.
             # In compare_models.py we treated them differently implicitly or the labels were already correct.
             # Let's be explicit.
             pass

        # Simple mapping based on title to be safe
        if title == "FRCNN":
             class_name = CLASSES[label - 1] # FRCNN is 1-based
        else:
             class_name = CLASSES[label] if label < len(CLASSES) else str(label) # YOLO is 0-based

        # Draw Box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        text = f"{class_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    # Add Title
    cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img_copy

def main():
    # Gunakan path yang sudah diset di atas
    img_path = IMAGE_PATH
    
    # Hapus tanda kutip jika user tidak sengaja menambahkannya di variabel
    img_path = img_path.strip('"').strip("'")

    if not os.path.exists(img_path):
        print(f"❌ Error: File tidak ditemukan: {img_path}")
        return

    print(f"🚀 Processing: {img_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load FRCNN
    print("Loading Faster R-CNN...")
    frcnn = get_frcnn_model(num_classes=7)
    frcnn.load_state_dict(torch.load(FRCNN_MODEL_PATH, map_location=device))
    frcnn.to(device)
    frcnn.eval()

    # 2. Load YOLO
    print("Loading YOLOv8m...")
    yolo = YOLO(YOLO_MODEL_PATH)

    # 3. Inference
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("❌ Error: Could not read image with OpenCV.")
        return

    # FRCNN
    print("Running Faster R-CNN...")
    f_boxes, f_scores, f_labels = frcnn_inference(frcnn, img_path, device)
    frcnn_img = draw_boxes(original_img, f_boxes, f_scores, f_labels, "FRCNN", (0, 0, 255)) # Red

    # YOLO
    print("Running YOLOv8m...")
    results = yolo(img_path, verbose=False)[0]
    y_boxes = results.boxes.xyxy.cpu().numpy()
    y_scores = results.boxes.conf.cpu().numpy()
    y_labels = results.boxes.cls.cpu().numpy().astype(int) # YOLO is 0-based
    yolo_img = draw_boxes(original_img, y_boxes, y_scores, y_labels, "YOLO", (0, 255, 0)) # Green

    # 4. Combine and Save
    # Resize if necessary to match heights (usually same if same source image)
    combined = np.hstack((frcnn_img, yolo_img))
    
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, f"single_compare_{filename}")
    cv2.imwrite(save_path, combined)
    
    print(f"✅ Comparison saved to: {save_path}")
    
    # Optional: Show image if local (but this is remote agent, so just print path)
    # cv2.imshow("Comparison", combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
