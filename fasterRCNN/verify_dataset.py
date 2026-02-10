from dataset import TrashDataset
from utils import get_transform
import torch
import os

def verify():
    data_path = "D:/penelitian/yolo/dataset-labeled"
    if not os.path.exists(data_path):
        print(f"❌ Dataset path not found: {data_path}")
        return

    print(f"Checking dataset at: {data_path}")
    
    try:
        dataset = TrashDataset(data_path, split='train', transforms=get_transform(train=True))
        print(f"✅ Dataset initialized. Size: {len(dataset)}")
        
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"✅ Sample loaded successfully.")
            print(f"   Image shape: {img.shape}")
            print(f"   Target keys: {target.keys()}")
            print(f"   Boxes shape: {target['boxes'].shape}")
            print(f"   Labels: {target['labels']}")
        else:
            print("⚠️ Dataset is empty!")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
