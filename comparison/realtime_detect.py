import torch
import torchvision
import cv2
import numpy as np
import os
import sys
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from ultralytics import YOLO

# === CONFIGURATION ===
FRCNN_MODEL_PATH = r"D:\penelitian\fasterRCNN\models_frcnn\best_model.pth"
YOLO_MODEL_PATH = r"D:\penelitian\yolo\models_yolo\trash_yolov8m\weights\best.pt"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === FRCNN SETUP ===
def get_frcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def frcnn_inference(model, frame, device, threshold=0.5):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        
        # Label Mapping
        if title == "FRCNN":
             class_name = CLASSES[label - 1] if label - 1 < len(CLASSES) else str(label)
        else:
             class_name = CLASSES[label] if label < len(CLASSES) else str(label)

        # Draw Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        text = f"{class_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return image

def apply_nms(boxes, scores, labels, iou_threshold=0.3):
    # Apply Non-Maximum Suppression to filter overlapping boxes
    if len(boxes) == 0:
        return boxes, scores, labels
        
    # Convert to tensor for torchvision NMS
    boxes_t = torch.tensor(boxes)
    scores_t = torch.tensor(scores)
    
    # NMS
    keep_indices = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    keep_indices = keep_indices.numpy()
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def nothing(x):
    pass

def main():
    print("🚀 Initializing Real-Time Detection...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading Faster R-CNN...")
    frcnn = get_frcnn_model(num_classes=7)
    frcnn.load_state_dict(torch.load(FRCNN_MODEL_PATH, map_location=device))
    frcnn.to(device)
    frcnn.eval()

    print("Loading YOLOv8m...")
    yolo = YOLO(YOLO_MODEL_PATH)

    # === WARMUP ===
    print("🔥 Warming up models...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warmup FRCNN
    frcnn_inference(frcnn, dummy_frame, device)
    # Warmup YOLO
    yolo(dummy_frame, verbose=False)
    print("✅ Warmup done!")

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Create Window and Trackbar
    cv2.namedWindow('Real-Time Detection')
    cv2.createTrackbar('Threshold', 'Real-Time Detection', 15, 100, nothing) # Default 15%

    print("\n=== CONTROLS ===")
    print("'1' : Faster R-CNN Mode")
    print("'2' : YOLOv8m Mode")
    print("'3' : Split Screen (Compare)")
    print("'q' : Quit")
    
    mode = '2' # Default to YOLO (Faster)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get Threshold from Trackbar
        conf_thresh = cv2.getTrackbarPos('Threshold', 'Real-Time Detection') / 100.0
        
        start_time = time.time()
        
        # Common RGB conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if mode == '1': # FRCNN
            boxes, scores, labels = frcnn_inference(frcnn, frame, device, threshold=conf_thresh)
            # Apply Extra NMS
            boxes, scores, labels = apply_nms(boxes, scores, labels, iou_threshold=0.3)
            
            draw_boxes(frame, boxes, scores, labels, "FRCNN", (0, 0, 255))
            cv2.putText(frame, "Mode: Faster R-CNN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if len(boxes) == 0:
                 cv2.putText(frame, "No Detections", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        elif mode == '2': # YOLO
            # YOLO conf argument sets the threshold
            # Pass RGB frame to be safe/consistent
            results = yolo(frame_rgb, verbose=False, conf=conf_thresh)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)
            draw_boxes(frame, boxes, scores, labels, "YOLO", (0, 255, 0))
            cv2.putText(frame, "Mode: YOLOv8m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if len(boxes) == 0:
                 cv2.putText(frame, "No Detections", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif mode == '3': # Split Screen
            frame_frcnn = frame.copy()
            frame_yolo = frame.copy()
            
            # FRCNN
            f_boxes, f_scores, f_labels = frcnn_inference(frcnn, frame_frcnn, device, threshold=conf_thresh)
            f_boxes, f_scores, f_labels = apply_nms(f_boxes, f_scores, f_labels, iou_threshold=0.3)
            draw_boxes(frame_frcnn, f_boxes, f_scores, f_labels, "FRCNN", (0, 0, 255))
            
            # YOLO
            # Pass RGB frame
            results = yolo(frame_rgb, verbose=False, conf=conf_thresh)[0]
            y_boxes = results.boxes.xyxy.cpu().numpy()
            y_scores = results.boxes.conf.cpu().numpy()
            y_labels = results.boxes.cls.cpu().numpy().astype(int)
            draw_boxes(frame_yolo, y_boxes, y_scores, y_labels, "YOLO", (0, 255, 0))
            
            # Combine
            frame = np.hstack((frame_frcnn, frame_yolo))
            if frame.shape[1] > 1920:
                frame = cv2.resize(frame, (1920, int(frame.shape[0] * 1920 / frame.shape[1])))

        # FPS Calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf_thresh:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Real-Time Detection', frame)
        
        key = cv2.waitKey(1)
        if key != -1:
            # Handle Quit
            if (key & 0xFF) == ord('q'):
                break
            
            # Handle Modes
            elif (key & 0xFF) == ord('1'):
                mode = '1'
            elif (key & 0xFF) == ord('2'):
                mode = '2'
            elif (key & 0xFF) == ord('3'):
                mode = '3'
                
            # Handle Threshold (Arrows or -/+)
            # Note: Arrow keys can be platform specific. 
            # Windows: Left=2424832, Right=2555904 (raw) or via 0xFF check if mapped
            # We add - and = as robust fallbacks
            current_thresh = cv2.getTrackbarPos('Threshold', 'Real-Time Detection')
            
            # Left Arrow (common codes) or '-'
            if key == 81 or key == 2424832 or (key & 0xFF) == ord('-') or (key & 0xFF) == ord(','):
                cv2.setTrackbarPos('Threshold', 'Real-Time Detection', max(0, current_thresh - 5))
                
            # Right Arrow (common codes) or '+'/'='
            elif key == 83 or key == 2555904 or (key & 0xFF) == ord('=') or (key & 0xFF) == ord('.'):
                cv2.setTrackbarPos('Threshold', 'Real-Time Detection', min(100, current_thresh + 5))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
