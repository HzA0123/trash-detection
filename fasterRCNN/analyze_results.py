import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from dataset import TrashDataset
from utils import collate_fn, get_transform
from tqdm import tqdm
import numpy as np
import os
import argparse
from torchvision.ops import box_iou

# === CONFIGURATION ===
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# HARDCODED PATHS (Edit di sini)
DATASET_ROOT = "D:/penelitian/fasterRCNN/datasetLabeledV2"
MODEL_PATH = "D:/penelitian/models_frcnn_v2/model_epoch_15.pth" # Epoch 15 is the best (F1 0.9190)
CSV_PATH = "D:/penelitian/models_frcnn_v2/training_log.csv"
OUTPUT_DIR = "D:/penelitian/models_frcnn_v2"

def plot_metrics(csv_path, output_dir):
    print(f"📊 Plotting metrics from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    # Create 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Results: {os.path.basename(output_dir)}', fontsize=16)

    # 1. Loss
    sns.lineplot(ax=axes[0, 0], x=df['Epoch'], y=df['Loss'], marker='o', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    # 2. F1-Score
    sns.lineplot(ax=axes[0, 1], x=df['Epoch'], y=df['F1-Score'], marker='o', color='green')
    axes[0, 1].set_title('F1-Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')

    # 3. Precision (mAP@50)
    sns.lineplot(ax=axes[0, 2], x=df['Epoch'], y=df['Precision'], marker='o', color='blue')
    axes[0, 2].set_title('Precision (mAP@50)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')

    # 4. Recall
    sns.lineplot(ax=axes[1, 0], x=df['Epoch'], y=df['Recall'], marker='o', color='orange')
    axes[1, 0].set_title('Recall (mAR@100)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')

    # 5. Inference Time
    sns.lineplot(ax=axes[1, 1], x=df['Epoch'], y=df['Inference Time (ms)'], marker='o', color='purple')
    axes[1, 1].set_title('Inference Time (ms)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ms')

    # 6. FPS
    sns.lineplot(ax=axes[1, 2], x=df['Epoch'], y=df['FPS'], marker='o', color='brown')
    axes[1, 2].set_title('Frames Per Second (FPS)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('FPS')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    save_path = os.path.join(output_dir, 'results_complete.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Complete metrics plot saved to {save_path}")
    plt.close()

def get_model(num_classes, model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def compute_confusion_matrix(model, data_loader, device, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    print("🔍 Computing Confusion Matrix (this may take a while)...")
    
    # Matrix: Rows = True Class, Cols = Predicted Class
    # Last index is for "Background" (False Positive / False Negative)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    for images, targets in tqdm(data_loader, desc="Inference"):
        images = list(img.to(device) for img in images)
        
        with torch.no_grad():
            outputs = model(images)
            
        for i, output in enumerate(outputs):
            target = targets[i]
            
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            
            pred_boxes = output['boxes'].to(device)
            pred_scores = output['scores'].to(device)
            pred_labels = output['labels'].to(device)
            
            # Filter by confidence
            keep = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
                
            # Match predictions to ground truth
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = box_iou(gt_boxes, pred_boxes)
                
                # For each GT, find best matching Pred
                matched_gt = set()
                matched_pred = set()
                
                for g_idx in range(len(gt_boxes)):
                    # Find best match for this GT
                    best_iou, p_idx = torch.max(ious[g_idx], dim=0)
                    
                    if best_iou >= iou_threshold:
                        # Match found!
                        true_cls = gt_labels[g_idx].item() - 1 # 0-indexed
                        pred_cls = pred_labels[p_idx].item() - 1 # 0-indexed
                        
                        cm[true_cls, pred_cls] += 1
                        matched_gt.add(g_idx)
                        matched_pred.add(p_idx.item())
                    else:
                        # No match for this GT -> False Negative (Background)
                        true_cls = gt_labels[g_idx].item() - 1
                        cm[true_cls, num_classes] += 1 # Last col is Background
                        
                # Handle False Positives (Preds with no GT match)
                for p_idx in range(len(pred_boxes)):
                    if p_idx not in matched_pred:
                        pred_cls = pred_labels[p_idx].item() - 1
                        cm[num_classes, pred_cls] += 1 # Last row is Background
            
            elif len(gt_boxes) > 0:
                # All GTs are missed
                for label in gt_labels:
                    cm[label.item()-1, num_classes] += 1
            
            elif len(pred_boxes) > 0:
                # All Preds are wrong (Ghost detections)
                for label in pred_labels:
                    cm[num_classes, label.item()-1] += 1

    return cm

def plot_confusion_matrix(cm, classes, output_dir):
    # Add 'Background' to classes
    labels = classes + ['Background']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix saved to {save_path}")
    plt.close()

def main():
    print(f"🚀 Starting Analysis...")
    print(f"📂 Dataset: {DATASET_ROOT}")
    print(f"🤖 Model: {MODEL_PATH}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Plot Metrics
    plot_metrics(CSV_PATH, OUTPUT_DIR)
    
    # 2. Confusion Matrix
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load Dataset (Val)
    dataset_test = TrashDataset(DATASET_ROOT, split='val', transforms=get_transform(train=False))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Load Model
    model = get_model(num_classes=7, model_path=MODEL_PATH, device=device)
    if model is None: return
    
    # Compute & Plot
    cm = compute_confusion_matrix(model, data_loader_test, device, num_classes=6)
    plot_confusion_matrix(cm, CLASSES, OUTPUT_DIR)

if __name__ == "__main__":
    main()
