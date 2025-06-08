from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import numpy as np


def decode_segmap(label):
    color_map = torch.tensor([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
        [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
        [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
        [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
        [  0, 80,100], [  0,  0,230], [119, 11, 32]  # 19 classes
    ], dtype=torch.uint8)
    
    segmap = torch.zeros((label.shape[0], label.shape[1], 3), dtype=torch.uint8)
    for i in range(len(color_map)):
        segmap[label == i] = color_map[i]
    return segmap


class CityscapeDataset(Dataset):
    def __init__(self, root, processor, split='train', size=(1024, 1024)):
        self.root = Path(root)
        self.processor = processor
        self.split = split
        self.size = size
        
        self.image = self.root / 'leftImg8bit' / self.split
        self.label = self.root / 'gtFine' / self.split
        
        self.data = list(self.image.glob('**/*.png'))
        self.data.sort()
        self.labels = list(self.label.glob('**/*_trainIds.png'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_path = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        
        if self.size == (1024, 1024):
            processed = self.processor(images=image, segmentation_maps=label, return_tensors="pt")
        else:
            processed = self.processor(images=image, segmentation_maps=label, return_tensors="pt", 
                                       size=self.size, do_resize=True)
            
        pixel_values = processed["pixel_values"].squeeze()     # [3, H, W]
        labels = processed["labels"].squeeze()                 # [H, W]
        return {"pixel_values": pixel_values, "labels": labels}
    
    
class UGMDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        
        self.data = list((self.root / 'camera' / 'rgb').glob('*.png'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)
        
        return {
            'image': image,
            'path': image_path
        }