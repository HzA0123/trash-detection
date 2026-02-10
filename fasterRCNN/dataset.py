import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TrashDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            split (string): 'train' or 'val'
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'labels', split)
        
        # Filter valid images
        self.imgs = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Class mapping (must match dataset.yaml)
        # 0: cardboard, 1: glass, 2: metal, 3: paper, 4: plastic, 5: trash
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load Image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Load Label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            w, h = img.size
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO (x_c, y_c, w, h) normalized -> COCO (x_min, y_min, x_max, y_max) absolute
                    x_min = (x_center - width / 2) * w
                    y_min = (y_center - height / 2) * h
                    x_max = (x_center + width / 2) * w
                    y_max = (y_center + height / 2) * h
                    
                    # Clip to image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    # Ensure valid box
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(cls_id + 1) # Add 1 because 0 is background in FRCNN
        
        # Convert to tensor
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Negative example (background)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
