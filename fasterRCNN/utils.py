import torch
import torchvision.transforms as T
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return T.functional.to_tensor(image), target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Image is Tensor (C, H, W)
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # bbox: [x_min, y_min, x_max, y_max]
            # flipped x_min = width - old_x_max
            # flipped x_max = width - old_x_min
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.transform = T.ColorJitter(brightness=brightness, contrast=contrast)
        
    def __call__(self, image, target):
        # Apply to PIL Image
        image = self.transform(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    if train:
        # 1. Color Jitter (on PIL image)
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2))
    
    # 2. Convert to Tensor
    transforms.append(ToTensor())
    
    if train:
        # 3. Random Horizontal Flip (on Tensor, handles boxes)
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)
