import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from itertools import chain
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(size=(1024, 1024)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=size[0], width=size[1], p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform(size=(1024, 1024)):
    return A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class UGMDataset(Dataset):
    def __init__(self, root, return_teacher_logits=False, size=(1024, 1024), transform=None):
        self.root = Path(root)
        self.return_teacher_logits = return_teacher_logits
        self.size = size
        self.num_classes = 19
        self.ignore_index = 255
        
        self.data = list((self.root / 'camera' / 'rgb').glob('*.png'))
        self.data.sort()
        
        self.transform = transform

        if self.return_teacher_logits:
            teacher_logits_dir = self.root / 'camera' / 'seg' / 'pseudo_labels_segformer_b5' / 'logits'
            self.teacher_logits = list(teacher_logits_dir.glob('*.npy'))
            self.teacher_logits.sort()
            assert len(self.data) == len(self.teacher_logits), "Mismatch between image and logits count."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB').resize(self.size, Image.BILINEAR)
        image = np.array(image)

        if self.return_teacher_logits:
            logits_path = self.teacher_logits[idx]
            logits = np.load(logits_path)
            logits = np.transpose(logits, (1, 2, 0))
            
        if self.transform:
            augmented = self.transform(image=image, mask=logits if self.return_teacher_logits else None)
            image = augmented['image']
            if self.return_teacher_logits:
                logits = augmented['mask'].permute(2, 0, 1)
                return image, logits

        if self.return_teacher_logits:
            return image, torch.tensor(logits, dtype=torch.float32).permute(2, 0, 1)
        else:
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
        def flatten_index_ranges(*ranges):
            return list(chain.from_iterable(ranges))

        # Define datasets with mode-specific transforms
        train_data = UGMDataset(self.root, self.return_teacher_logits, size=self.size, transform=get_train_transform(self.size))
        val_data = UGMDataset(self.root, self.return_teacher_logits, size=self.size, transform=get_val_transform(self.size))

        around_mipa     = flatten_index_ranges(range(0, 690), range(2350, 3760), range(9985, 10139))  # 2254
        agro            = flatten_index_ranges(range(3760, 4990), range(9330, 9985))  # 1885
        inside_ugm      = flatten_index_ranges(range(6670, 9330))  # 2660
        around_vokasi   = flatten_index_ranges(range(690, 2030), range(2030, 2350))  # 1660
        lembah          = flatten_index_ranges(range(4990, 6670))  # 1680
        
        self.train_dataset = Subset(train_data, list(chain(around_mipa, agro, inside_ugm)))  # 6800/10139 data (0.67)
        self.val_dataset = Subset(val_data, around_vokasi) # 1660/10139 data (0.16)
        self.test_dataset = Subset(val_data, lembah) # 1680/10139 data (0.17)

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
    dm = UGMDataModule(root, return_teacher_logits=True, batch_size=4, num_workers=4, size=(256, 256))
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"Number of training samples: {len(dm.train_dataset)}")

    for images, logits in train_loader:
        print(f"Train batch - Images shape: {images.shape}, Logits shape: {logits.shape if logits is not None else 'N/A'}")
        break
