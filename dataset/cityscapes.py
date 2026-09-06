import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from transformers import SegformerImageProcessor
from torchvision.datasets import Cityscapes

class CityscapeDataset(Dataset):
    def __init__(self, root, processor=None, split='train', size=(1024, 1024)):
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
        
        if self.processor is None:
            return image.resize(self.size), label
        
        if self.size == (1024, 1024):
            processed = self.processor(images=image, segmentation_maps=label, return_tensors="pt")
        else:
            processed = self.processor(images=image, segmentation_maps=label, return_tensors="pt", 
                                       size=self.size, do_resize=True)
            
        pixel_values = processed["pixel_values"].squeeze()     # [3, H, W]
        labels = processed["labels"].squeeze()                 # [H, W]
        return pixel_values, labels
    

class CityscapeDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=8, num_workers=4, size=(1024, 1024)):
        super().__init__()
        self.root = root
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage=None):
        self.train_dataset = CityscapeDataset(self.root, self.processor, "train", self.size)
        self.val_dataset = CityscapeDataset(self.root, self.processor, "val", self.size)
        self.test_dataset = CityscapeDataset(self.root, self.processor, "test", self.size)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        
if __name__ == "__main__":
    root = '/media/esr/ssd0/cityscapes'
    dm = CityscapeDataModule(root, batch_size=4, num_workers=2, size=(256, 256))
    dm.setup()
    
    for batch in dm.train_dataloader():
        pixel_values, labels = batch
        print(pixel_values.shape, labels.shape)
        break