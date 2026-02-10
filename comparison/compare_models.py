import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import os
import random
import glob
from ultralytics import YOLO
import sys
import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Add fasterRCNN to path to import dataset utils
sys.path.append(r"D:\penelitian\fasterRCNN")
from dataset import TrashDataset
from utils import get_transform, collate_fn



# === CONFIGURATION ===
FRCNN_MODEL_PATH = r"D:\penelitian\fasterRCNN\models_frcnn\best_model.pth" 
YOLO_MODEL_PATH = r"D:\penelitian\yolo\models_yolo\trash_yolov8m\weights\best.pt"
TEST_IMAGES_DIR = r"D:\penelitian\yolo\datasetLabeledV2\images\val"
OUTPUT_DIR = r"D:\penelitian\comparison\results"
NUM_SAMPLES = 5
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === FRCNN SETUP ===
def get_frcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def frcnn_inference(model, image_path, device, threshold=0.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = T.ToTensor()(image_rgb).to(device)
    
    with torch.no_grad():
        prediction = model([image_tensor])[0]
        
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    # Filter by threshold
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]

def evaluate_frcnn(model, data_loader, device):
    print("📊 Evaluating Faster R-CNN (Calculating mAP, F1, Recall)...")
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    
    for images, targets in tqdm.tqdm(data_loader, desc="FRCNN Eval"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)
            
        metric.update(outputs, targets)
        
    results = metric.compute()

    map50 = results['map_50'].item()
    recall = results['mar_100'].item()
    
    # Per-class metrics
    # Note: map_per_class is mAP50-95. map_50_per_class is not standard output in this version.
    map50_per_class = results['map_per_class'].cpu().numpy()
    recall_per_class = results['mar_100_per_class'].cpu().numpy()
    
    # Estimate F1 (Harmonic Mean of Precision@50 and Recall)
    f1 = 2 * (map50 * recall) / (map50 + recall + 1e-6)
    
    return map50, recall, f1, map50_per_class, recall_per_class

# === VISUALIZATION ===
def draw_boxes(image, boxes, scores, labels, title, color=(0, 255, 0)):
    img_copy = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        class_name = CLASSES[label - 1] if isinstance(label, (int, np.integer)) else label 
        
        # Draw Box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        text = f"{class_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return img_copy

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Load FRCNN
    print("Loading Faster R-CNN...")
    frcnn = get_frcnn_model(num_classes=7) # 6 + background
    frcnn.load_state_dict(torch.load(FRCNN_MODEL_PATH, map_location=device))
    frcnn.to(device)
    frcnn.eval()
    
    # 2. Load YOLO
    print("Loading YOLOv8m...")
    yolo = YOLO(YOLO_MODEL_PATH)
    
    # 3. Quantitative Evaluation (Full Metrics)
    print("\n🚀 Running Full Evaluation (mAP, Recall, F1)...")
    
    # --- FRCNN Eval ---
    # Need dataset with labels
    dataset_val = TrashDataset(root_dir=r"D:\penelitian\yolo\datasetLabeledV2", split='val', transforms=get_transform(train=False))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    
    frcnn_map, frcnn_recall, frcnn_f1, frcnn_map_per_class, frcnn_recall_per_class = evaluate_frcnn(frcnn, data_loader_val, device)
    
    # --- YOLO Eval ---
    print("\n📊 Evaluating YOLOv8m...")
    yolo_metrics = yolo.val(data=r"D:\penelitian\yolo\dataset.yaml", split='val', verbose=False)
    yolo_map = yolo_metrics.box.map50
    yolo_recall = yolo_metrics.box.r.mean() # Average recall
    yolo_f1 = 2 * (yolo_map * yolo_recall) / (yolo_map + yolo_recall + 1e-6)
    

    
    # YOLO per-class
    try:
        yolo_maps_per_class = yolo_metrics.box.maps 
    except Exception:
        yolo_maps_per_class = []
        

    
    # 5. Inference Speed Comparison (Per Image)
    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    if not image_files:
        print("No images found!")
        return
    
    print(f"\n🚀 Running Speed Comparison on {len(image_files)} images...")
    
    frcnn_times = []
    yolo_times = []
    
    # CSV Header
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "comparison_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "FRCNN_Time_ms", "YOLO_Time_ms", "Difference_ms"])
        
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            
            # Warmup for first few images
            is_warmup = i < 5
            
            # --- FRCNN ---
            start = cv2.getTickCount()
            frcnn_inference(frcnn, img_path, device)
            end = cv2.getTickCount()
            t_frcnn = (end - start) / cv2.getTickFrequency() * 1000
            
            # --- YOLO ---
            start = cv2.getTickCount()
            yolo(img_path, verbose=False)
            end = cv2.getTickCount()
            t_yolo = (end - start) / cv2.getTickFrequency() * 1000
            
            if not is_warmup:
                frcnn_times.append(t_frcnn)
                yolo_times.append(t_yolo)
                writer.writerow([filename, f"{t_frcnn:.2f}", f"{t_yolo:.2f}", f"{t_frcnn - t_yolo:.2f}"])
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(image_files)} images...")

    # Statistics
    avg_frcnn = np.mean(frcnn_times)
    avg_yolo = np.mean(yolo_times)
    fps_frcnn = 1000 / avg_frcnn
    fps_yolo = 1000 / avg_yolo
    
    print(f"\n📊 Speed Results:")
    print(f"Faster R-CNN: {avg_frcnn:.2f} ms ({fps_frcnn:.2f} FPS)")
    print(f"YOLOv8m     : {avg_yolo:.2f} ms ({fps_yolo:.2f} FPS)")

    # Plotting
    labels = ['Faster R-CNN', 'YOLOv8m']
    times = [avg_frcnn, avg_yolo]
    fps = [fps_frcnn, fps_yolo]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Inference Time (ms) - Lower is Better', color=color)
    bars1 = ax1.bar(labels, times, color=color, alpha=0.6, label='Time (ms)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms', ha='center', va='bottom')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('FPS - Higher is Better', color=color)
    ax2.plot(labels, fps, color=color, marker='o', linewidth=3, label='FPS', markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    
    for i, v in enumerate(fps):
        ax2.text(i, v + 0.5, f'{v:.1f} FPS', color='red', ha='center', fontweight='bold')

    plt.title('Performance Comparison: Faster R-CNN vs YOLOv8m')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_chart.png"))
    print(f"✅ Chart saved to {os.path.join(OUTPUT_DIR, 'comparison_chart.png')}")

    # --- Visual Samples (Top 5) ---
    print("\nGenerating visual samples...")
    samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    
    for img_path in samples:
        filename = os.path.basename(img_path)
        original_img = cv2.imread(img_path)
        
        # FRCNN
        f_boxes, f_scores, f_labels = frcnn_inference(frcnn, img_path, device)
        frcnn_img = draw_boxes(original_img, f_boxes, f_scores, f_labels, "FRCNN", (0, 0, 255))
        
        # YOLO
        results = yolo(img_path, verbose=False)[0]
        y_boxes = results.boxes.xyxy.cpu().numpy()
        y_scores = results.boxes.conf.cpu().numpy()
        y_labels = results.boxes.cls.cpu().numpy().astype(int) + 1
        yolo_img = draw_boxes(original_img, y_boxes, y_scores, y_labels, "YOLO", (0, 255, 0))
        
        combined = np.hstack((frcnn_img, yolo_img))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"compare_{filename}"), combined)

    # === SAVE FINAL SUMMARY (Moved to end to include speed stats) ===
    summary_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== FINAL MODEL COMPARISON ===\n")
        f.write(f"Dataset: {TEST_IMAGES_DIR}\n\n")
        
        f.write("1. OVERALL PERFORMANCE (Accuracy)\n")
        f.write(f"   Metric      | Faster R-CNN (V1) | YOLOv8m (V2)\n")
        f.write(f"   ----------- | ----------------- | ------------\n")
        f.write(f"   mAP@50      | {frcnn_map:.4f}            | {yolo_map:.4f}\n")
        f.write(f"   Recall      | {frcnn_recall:.4f}            | {yolo_recall:.4f}\n")
        f.write(f"   F1-Score    | {frcnn_f1:.4f}            | {yolo_f1:.4f}\n\n")
        
        f.write("2. PER-CLASS ANALYSIS (mAP@50 for FRCNN, mAP@50-95 for YOLO*)\n")
        f.write("   *Note: YOLO API provides mAP50-95 per class easily. Comparison is indicative.\n")
        f.write(f"   {'Class':<12} | {'FRCNN (mAP@50)':<15} | {'YOLO (mAP@50-95)':<15}\n")
        f.write(f"   {'-'*12} | {'-'*15} | {'-'*15}\n")
        
        for i, class_name in enumerate(CLASSES):
            f_score = frcnn_map_per_class[i] if i < len(frcnn_map_per_class) else 0.0
            y_score = yolo_maps_per_class[i] if i < len(yolo_maps_per_class) else 0.0
            f.write(f"   {class_name:<12} | {f_score:.4f}          | {y_score:.4f}\n")
            
        f.write("\n3. SPEED PERFORMANCE (Inference Time)\n")
        f.write(f"   Metric           | Faster R-CNN      | YOLOv8m\n")
        f.write(f"   ---------------- | ----------------- | ------------\n")
        f.write(f"   Avg Time (Image) | {avg_frcnn:.2f} ms        | {avg_yolo:.2f} ms\n")
        f.write(f"   FPS              | {fps_frcnn:.2f} FPS         | {fps_yolo:.2f} FPS\n")
        f.write(f"   Speedup Factor   | 1x (Baseline)     | {avg_frcnn/avg_yolo:.1f}x Faster\n\n")

        f.write("4. Conclusion\n")
        if frcnn_f1 > yolo_f1:
            f.write("   - Accuracy: Faster R-CNN is better (Higher F1).\n")
        else:
            f.write("   - Accuracy: YOLOv8m is better (Higher F1).\n")
            
        if avg_yolo < avg_frcnn:
            f.write("   - Speed: YOLOv8m is significantly faster.\n")
            
    print(f"✅ Summary saved to {summary_path}")
    print("Comparison Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
