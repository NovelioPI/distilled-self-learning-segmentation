import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from itertools import chain
import numpy as np
from transformers import SegformerImageProcessor


class UGMDataset(Dataset):
    def __init__(self, root, return_teacher_logits=False, size=(1024, 1024), transform=None):
        self.root = Path(root)
        self.return_teacher_logits = return_teacher_logits
        self.size = size
        self.transform = transforms.ToTensor() if transform is None else transform
        
        self.data = list((self.root / 'camera' / 'rgb').glob('*.png'))
        self.data.sort()
        
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        
        if self.return_teacher_logits:
            teacher_logits_dir = self.root / 'camera' / 'seg' / 'pseudo_labels_segformer_b5' / 'logits'
            self.teacher_logits = list(teacher_logits_dir.glob('*.npy'))
            self.teacher_logits.sort()
            
            assert len(self.data) == len(self.teacher_logits), "Mismatch between image and logits count."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.processor(images=image, return_tensors="pt", size=self.size)['pixel_values'][0]
        
        if self.return_teacher_logits:
            logits_path = self.teacher_logits[idx]
            logits = torch.from_numpy(np.load(logits_path)).float()
            return image, logits

        return image
    
    def get_image_path(self, idx):
        if 0 <= idx < len(self.data):
            return self.data[idx]
        else:
            raise IndexError("Index out of range for dataset.")
        
        
class UGMDataModule(pl.LightningDataModule):
    def __init__(self, root, return_teacher_logits=False, batch_size=4, num_workers=4, size=(1024, 1024)):
        super(UGMDataModule, self).__init__()
        self.root = root
        self.return_teacher_logits = return_teacher_logits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        
    def setup(self, stage=None):
        def flatten_ranges(*ranges):
            return list(chain.from_iterable(ranges))
        
        dataset = UGMDataset(self.root, self.return_teacher_logits, size=self.size)
        
        around_mipa = Subset(dataset, flatten_ranges(range(0, 690), range(2350, 3760), range(9985, 10139))) # 2254 data
        around_vokasi = Subset(dataset, flatten_ranges(range(690, 2030), range(2030, 2350))) # 1660 data
        agro = Subset(dataset, flatten_ranges(range(3760, 4990), range(9330, 9985))) # 1885 data
        lembah = Subset(dataset, flatten_ranges(range(4990, 6670))) # 1680 data
        inside_ugm = Subset(dataset, flatten_ranges(range(6670, 9330))) # 2660 data
        
        self.train_dataset = ConcatDataset([around_mipa, agro, inside_ugm])  # 6800/10139 data (0.67)
        self.val_dataset = around_vokasi # 1660/10139 data (0.16)
        self.test_dataset = lembah # 1680/10139 data (0.17)
        
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
    root = '/media/esr/ssd0/dataset/2025-01-10/'
    dm = UGMDataModule(root, return_teacher_logits=True, batch_size=4, num_workers=4, size=(1024, 1024))
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    for images, logits in train_loader:
        print(f"Train batch - Images shape: {images.shape}, Logits shape: {logits.shape if logits is not None else 'N/A'}")
        break