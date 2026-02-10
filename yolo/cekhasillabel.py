import cv2
import os

image_path = r"D:\penelitian\yolo\dataset-labeled\images\train\cardboard79.jpg"
label_path = r"D:\penelitian\yolo\dataset-labeled\labels\train\cardboard79.txt"

# Baca gambar
img = cv2.imread(image_path)
h, w, _ = img.shape

# Baca file label YOLO
with open(label_path, "r") as f:
    for line in f:
        cls, x_c, y_c, bw, bh = map(float, line.split())
        x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h

        # Konversi ke koordinat pojok kiri atas dan kanan bawah
        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Gambar kotak merah dan label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"class {int(cls)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Preview Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
