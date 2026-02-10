from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO(r'D:\penelitian\yolo\models_yolo\trash_yolov8s\weights\best.pt')

# Set paths
val_dir = r'dataset-labeled/images/val'
out_dir = r'results_yolo/predict_finetuneA'
os.makedirs(out_dir, exist_ok=True)

# Run predictions on all validation images
results = model.predict(source=val_dir, conf=0.25)

# Save annotated images
for r in results:
    img = r.plot()
    basename = os.path.basename(r.path)
    outpath = os.path.join(out_dir, basename.replace('.jpg','_annotated.jpg'))
    try:
        cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception:
        cv2.imwrite(outpath, img)
    print('SAVED:', outpath)