import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from itertools import chain
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class UGMDataset(Dataset):
    def __init__(self, 
                 root="dataset", 
                 size=(1024, 1024),
                 split=None,
                 return_teacher_logits=False,
                 use_refinement=False,
                 **kwargs
                 ):
        self.root = Path(root)
        self.size = size
        self.split = split if split != 'predict' else None
        self.return_teacher_logits = return_teacher_logits
        self.use_refinement = use_refinement
        self.num_classes = 19
        self.use_transform = kwargs.get('use_transform', True)
        
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

        if self.use_transform:
            self.transform = A.Compose([
                A.Resize(height=size[0], width=size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=size[0], width=size[1]),
                ToTensorV2()
            ])
        
        teacher_path = self.root / "camera/seg/pseudo_labels_segformer_b5"
        if return_teacher_logits:
            if self.split is None:
                self.teacher_logits = [p for p in (teacher_path / "logits").glob("*.npy")]
            else:
                self.teacher_logits = [teacher_path / f"logits/{name}.npy" for name in self.data]
            self.teacher_logits.sort()
            self.augmentation.add_targets({'t_logits': 'mask'})
            self.transform.add_targets({'t_logits': 'mask'})
        if use_refinement:
            if self.split is None:
                self.refined_labels = [p for p in (teacher_path / "refined-1.0_pseudo_labels").glob("*.png") if '_colorized' not in p.stem]
            else:
                self.refined_labels = [teacher_path / f"refined-1.0_pseudo_labels/{name}.png" for name in self.data]
            self.refined_labels.sort()
            self.augmentation.add_targets({'refined_label': 'mask'})
            self.transform.add_targets({'refined_label': 'mask'})

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(self.size, Image.BILINEAR)
        image = np.array(image)
        
        if self.return_teacher_logits:
            t_logits = np.load(self.teacher_logits[idx])
            t_logits = np.transpose(t_logits, (1, 2, 0))
        
        if self.use_refinement:
            refined_label = Image.open(self.refined_labels[idx]).convert("L")
            refined_label = np.array(refined_label)
            
        if self.split == "train":
            output = self.augmentation(image=image, t_logits=t_logits if self.return_teacher_logits else None, 
                                    refined_label=refined_label if self.use_refinement else None)
        else:
            output = self.transform(image=image, t_logits=t_logits if self.return_teacher_logits else None, 
                                    refined_label=refined_label if self.use_refinement else None)
            
        image = output["image"]
        t_logits = output["t_logits"].permute(2, 0, 1) if self.return_teacher_logits else None
        refined_label = output["refined_label"].long() if self.use_refinement else None
        
        if self.return_teacher_logits and self.use_refinement:
            return image, t_logits, refined_label
        elif self.return_teacher_logits:
            return image, t_logits
        elif self.use_refinement:
            return image, refined_label
        else:
            return image
            
    def get_image_path(self, idx):
        if 0 <= idx < len(self.images):
            return self.images[idx]
        else:
            raise IndexError("Index out of range for dataset.")


class UGMDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root, 
                 return_teacher_logits=False, 
                 use_refinement=False,
                 entropy_threshold=1.0,
                 batch_size=4, 
                 num_workers=4, 
                 size=(1024, 1024)):
        super(UGMDataModule, self).__init__()
        self.root = root
        self.return_teacher_logits = return_teacher_logits
        self.use_refinement = use_refinement
        self.entropy_threshold = entropy_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage=None):
        self.train_dataset = UGMDataset(
            root=self.root,
            size=self.size,
            split="train",
            return_teacher_logits=self.return_teacher_logits,
            use_refinement=self.use_refinement,
            entropy_threshold=self.entropy_threshold
        )
        self.val_dataset = UGMDataset(
            root=self.root,
            size=self.size,
            split="val",
            return_teacher_logits=self.return_teacher_logits,
            use_refinement=self.use_refinement,
            entropy_threshold=self.entropy_threshold
        )
        self.test_dataset = UGMDataset(
            root=self.root,
            size=self.size,
            split="test",
            return_teacher_logits=self.return_teacher_logits,
            use_refinement=self.use_refinement,
            entropy_threshold=self.entropy_threshold
        )
        self.predict_dataset = UGMDataset(
            root=self.root,
            size=self.size,
            split="predict",
            return_teacher_logits=self.return_teacher_logits,
            use_refinement=self.use_refinement,
            entropy_threshold=self.entropy_threshold
        )

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
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    root = '/media/esr/ssd0/dataset/2025-01-10/'
    
    # Check the dataset directly
    ugm_dataset = UGMDataset(root, return_teacher_logits=True, size=(256, 256), use_refinement=True, entropy_threshold=0.25)
    for i in range(len(ugm_dataset)):
        image, t_logits, refined_label = ugm_dataset[i]
        print(f"Image shape: {image.shape}, Teacher logits shape: {t_logits.shape if t_logits is not None else 'N/A'}, Refined label shape: {refined_label.shape if refined_label is not None else 'N/A'}")
        break
        
    
    
    # Check the data module
    dm = UGMDataModule(root, return_teacher_logits=True, batch_size=4, num_workers=4, size=(256, 256), use_refinement=True)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"Number of training samples: {len(dm.train_dataset)}")

    for batch in train_loader:
        images, t_logits, refined_labels = batch
        print(f"Batch size: {images.shape}, Teacher logits shape: {t_logits.shape if t_logits is not None else 'N/A'}, Refined labels shape: {refined_labels.shape if refined_labels is not None else 'N/A'}")
        break
