import os
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from dataset import TrashDataset
from utils import collate_fn, get_transform
from tqdm import tqdm
import numpy as np
import csv
import datetime
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# === CONFIGURATION (EDIT THIS ON LAPTOP 2) ===
DATASET_ROOT = "D:/penelitian/fasterRCNN/datasetLabeledV2" # <--- GANTI INI SESUAI LOKASI DATASET DI LAPTOP 2
OUTPUT_DIR = "D:/penelitian/models_frcnn_v2"
NUM_EPOCHS = 30
BATCH_SIZE = 4
NUM_WORKERS = 6 # Sesuaikan dengan CPU Laptop 2

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    metric_logger = []
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch")
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed Precision Training
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_value = losses.item()
        metric_logger.append(loss_value)
        pbar.set_postfix({"Loss": f"{loss_value:.4f}"})
        
    return np.mean(metric_logger)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    inference_times = []
    
    print("Running evaluation...")
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Measure inference time
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        
        # Time per batch -> Time per image
        batch_time = end_time - start_time
        inference_times.append(batch_time / len(images))
        
        metric.update(outputs, targets)

    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    # Compute metrics
    results = metric.compute()
    map50 = results['map_50'].item()
    mar_100 = results['mar_100'].item() # Recall
    
    # Precision Proxy
    precision = map50 
    
    # Calculate F1 Score
    f1_score = 2 * (precision * mar_100) / (precision + mar_100 + 1e-7)
    
    print(f"\n⚡ Avg Inference Time: {avg_inference_time*1000:.2f} ms | FPS: {fps:.2f}")
    print(f"📊 Precision (mAP50): {precision:.4f} | Recall: {mar_100:.4f} | F1: {f1_score:.4f}")
    
    return avg_inference_time, precision, mar_100, f1_score, fps

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main():
    # === Configuration ===
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    num_classes = 7 
    
    # Check if dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Error: Dataset not found at {DATASET_ROOT}")
        print("Please edit DATASET_ROOT in train_v2.py to match your laptop's path.")
        return

    print(f"📂 Loading dataset from: {DATASET_ROOT}")
    dataset = TrashDataset(DATASET_ROOT, split='train', transforms=get_transform(train=True))
    dataset_test = TrashDataset(DATASET_ROOT, split='val', transforms=get_transform(train=False))

    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=10)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Inference Time (ms)", "FPS", "Precision", "Recall", "F1-Score", "Timestamp"])
    
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    total_start_time = time.time()
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, scaler)
        lr_scheduler.step()
        
        avg_inf_time, precision, recall, f1, fps = evaluate(model, data_loader_test, device)
        
        # Save Best Model
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New Best Model! (F1: {best_f1:.4f}) Saved to {save_path}")

        # Save Last Model (Every Epoch - Overwrite)
        last_save_path = os.path.join(OUTPUT_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_save_path)
        
        early_stopping(f1)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break
        
        # Save Checkpoint Every 10 Epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Checkpoint saved to {save_path}")
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_loss:.4f}", 
                f"{avg_inf_time*1000:.2f}", 
                f"{fps:.2f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
            
        epoch_duration = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1} done in {epoch_duration:.2f}s | Loss: {avg_loss:.4f} | F1: {f1:.4f}")

    total_time = time.time() - total_start_time
    print(f"\n==================== TRAINING SELESAI ====================")
    print(f"🕒 Total waktu: {total_time/3600:.2f} jam")
    print(f"📂 Log: {csv_path}")
    print(f"🏆 Best F1: {best_f1:.4f}")
    
    # === AUTOMATIC ANALYSIS ===
    print("\n📊 Starting Automatic Analysis...")
    import analyze_results
    
    # 1. Plot Metrics
    analyze_results.plot_metrics(csv_path, OUTPUT_DIR)
    
    # 2. Confusion Matrix
    print("🔍 Generating Confusion Matrix for Best Model...")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    
    if os.path.exists(best_model_path):
        # Load Best Model
        best_model = get_model_instance_segmentation(num_classes)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.to(device)
        best_model.eval()
        
        # Compute & Plot
        cm = analyze_results.compute_confusion_matrix(best_model, data_loader_test, device, num_classes=6)
        analyze_results.plot_confusion_matrix(cm, analyze_results.CLASSES, OUTPUT_DIR)
        print("✅ Analysis Complete! Check results_complete.png and confusion_matrix.png")
    else:
        print("⚠️ Best model not found, skipping Confusion Matrix.")

if __name__ == "__main__":
    main()
