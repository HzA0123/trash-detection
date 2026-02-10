"""
Batch prediction helper for YOLOv8 (Ultralytics).

Features:
- Run predictions on all images in a folder (default: validation folder)
- Lower default confidence to 0.25 to reduce false negatives
- Warm-up loop to get stable FPS measurement
- Save per-image detections to JSON and a summary CSV
- Run `model.val` at the end to produce mAP/precision/recall/F1

Usage (from repository root or yolo folder):
    python predict_batch.py
    python predict_batch.py --weights models_yolo\trash_yolov8s\weights\best.pt --source dataset-labeled\images\val --conf 0.25 --imgsz 640
"""

import argparse
import json
import os
import time
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Batch predict + metrics for YOLOv8")
    p.add_argument("--weights", type=str, default=r"models_yolo\trash_yolov8s\weights\best.pt", help="Path to weights")
    p.add_argument("--source", type=str, default=r"dataset-labeled\images\val", help="Folder with images to predict")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="cuda", help='"cuda" or "cpu"')
    p.add_argument("--out", type=str, default=r"results_yolo\predict_batch", help="Output folder for detections")
    p.add_argument("--warmup", type=int, default=10, help="Number of warm-up runs to stabilize timing")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    print(f"Loading model {args.weights} on device={args.device} ...")
    model = YOLO(str(args.weights))

    # collect image paths
    imgs = sorted([p for p in src.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    if len(imgs) == 0:
        print("No images found in source folder.")
        return

    # Warm-up on first image
    print(f"Warm-up: running {args.warmup} inference passes on first image to stabilize timings...")
    first = str(imgs[0])
    for i in range(args.warmup):
        _ = model.predict(source=first, conf=0.25, device=args.device, imgsz=args.imgsz)

    detections = []
    times = []

    print(f"Running predictions on {len(imgs)} images with conf={args.conf} imgsz={args.imgsz} ...")
    for p in imgs:
        t0 = time.time()
        results = model.predict(source=str(p), conf=args.conf, device=args.device, imgsz=args.imgsz, save=False)
        t1 = time.time()
        elapsed = t1 - t0
        times.append(elapsed)

        # parse results (Ultralytics Results object)
        per_img = {"image": str(p), "time": elapsed, "boxes": []}
        if results:
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = [float(x) for x in box.xyxy[0].tolist()]
                        per_img['boxes'].append({"class": cls_id, "conf": conf, "xyxy": xyxy})
                    except Exception:
                        # fallback parsing
                        pass

        detections.append(per_img)

    # save detections JSON
    json_path = out / "detections.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, indent=2)

    # save summary CSV
    csv_path = out / "summary.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("image,time_s,num_boxes\n")
        for d in detections:
            f.write(f"{d['image']},{d['time']:.4f},{len(d['boxes'])}\n")

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print("\n--- Summary ---")
    print(f"Images: {len(imgs)}")
    print(f"Avg inference time per image: {avg_time:.4f} s")
    print(f"Estimated FPS: {fps:.2f}")
    print(f"Detections saved to: {json_path}")
    print(f"Summary CSV: {csv_path}")

    # Run dataset validation (mAP/F1) using Ultralytics val if a data yaml is present
    data_yaml = Path('dataset.yaml')
    if data_yaml.exists():
        print("\nRunning model.val to compute mAP/precision/recall/F1 on dataset.yaml ...")
        try:
            val_res = model.val(data=str(data_yaml))
            print(val_res)
        except Exception as e:
            print("Error running model.val:", e)
    else:
        print("dataset.yaml not found in current folder; skipping model.val")


if __name__ == '__main__':
    main()
