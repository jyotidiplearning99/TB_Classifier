# src/dataset_no_clahe.py

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def autocrop_nonzero(img):
    """Remove black borders/frames"""
    m = img > 0
    if not m.any():
        return img
    ys, xs = np.where(m)
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

class TBDatasetNoCLAHE(Dataset):
    """
    TB Dataset WITHOUT CLAHE (for DA+DB testing)
    """
    
    def __init__(self, csv_file, transform=None, use_clahe=False):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_clahe = use_clahe
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Cannot load image: {row['image_path']}")
        
        # Convert to grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 8-bit
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Auto-crop black borders
        image = autocrop_nonzero(image)
        
        # Resize if cropping changed size dramatically
        if image.shape[0] < 256 or image.shape[1] < 256:
            image = cv2.resize(image, (512, 512))
        
        # Apply CLAHE ONLY if flag is True
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        # Stack to 3 channels
        image = np.stack([image, image, image], axis=-1)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'patient_id': row['patient_id'],
            'dataset': row['dataset']
        }

def get_valid_transforms(img_size=512):
    """Validation transforms"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
