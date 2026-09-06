import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from transformers import SegformerImageProcessor

class UnlabeledDataset(Dataset):
    def __init__(self, 
                 root="dataset", 
                 size=(1024, 1024),
                 split=None,
                 ):
        self.root = Path(root)
        self.size = size
        self.split = split if split != 'predict' else None
        self.num_classes = 19
        
        if self.split is None:
            self.images = list((self.root / "camera/rgb").glob("*.png"))
            self.images.sort()
        else:
            with open(self.root / f"{self.split}.json", 'r') as file:
                self.data = json.load(file)
            self.data = [v for k, v in self.data.items()]
            self.images = [self.root / f"camera/rgb/{img}.png" for img in self.data]
            self.images.sort()

        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=size[0], width=size[1], p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.transform = A.Compose([
            A.Resize(height=size[0], width=size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        teacher_path = self.root / "camera/seg/pseudo_labels_segformer_b5"
        
        if self.split is None:
            self.teacher_logits = [p for p in (teacher_path / "logits").glob("*.npy")]
        else:
            self.teacher_logits = [teacher_path / f"logits/{name}.npy" for name in self.data]
        self.teacher_logits.sort()
        self.augmentation.add_targets({'t_logits': 'mask'})
        self.transform.add_targets({'t_logits': 'mask'})

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(self.size, Image.BILINEAR)
        image = np.array(image)
        
        t_logits = np.load(self.teacher_logits[idx])
        t_logits = np.transpose(t_logits, (1, 2, 0))
            
        if self.split == "train":
            output = self.augmentation(image=image, t_logits=t_logits)
        else:
            output = self.transform(image=image, t_logits=t_logits)
        image = output["image"]
        t_logits = output["t_logits"]
        
        t_prob = torch.softmax(torch.tensor(t_logits), dim=0)
        label = torch.argmax(t_prob, dim=0)
        conf = torch.max(t_prob, dim=0).values

        return image, label, conf
    
class LabeledDataset(Dataset):
    def __init__(self, root, split='train', size=(1024, 1024)):
        self.root = Path(root)
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
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


class UGMDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root,
                 batch_size=4, 
                 num_workers=4, 
                 size=(1024, 1024)):
        super(UGMDataModule, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage=None):
        self.train_ul = UnlabeledDataset(root=self.root, size=self.size, split="train")
        self.val_ul = UnlabeledDataset(root=self.root, size=self.size, split="val")
        self.test_ul = UnlabeledDataset(root=self.root, size=self.size, split="test")

        self.train_l = LabeledDataset(root=self.root, split="train", size=self.size)
        self.val_l = LabeledDataset(root=self.root, split="val", size=self.size)
        self.test_l = LabeledDataset(root=self.root, split="test", size=self.size)

    def train_dataloader(self):
        return DataLoader(
            self.train_ul,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ul,
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
