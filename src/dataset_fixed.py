# src/dataset_fixed.py - ROBUST LOADING

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TBDatasetRobust(Dataset):
    """
    TB Dataset with ROBUST 8/16-bit image loading
    Fixes domain shift from bit depth differences
    """
    
    def __init__(self, csv_file, transform=None, grayscale=True):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.grayscale = grayscale
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ROBUST LOADING: Handle 8-bit, 12-bit, 16-bit images
        image = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Cannot load image: {row['image_path']}")
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            if self.grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # CRITICAL FIX: Normalize to 8-bit range FIRST
        if image.dtype != np.uint8:
            # Handle 12-bit, 16-bit images
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE on normalized 8-bit image
        if self.grayscale:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            
            # Stack to 3 channels for pretrained models
            image = np.stack([image, image, image], axis=-1)
        else:
            # Apply CLAHE on L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
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

# Keep same transforms
def get_train_transforms(img_size=512):
    """Conservative augmentations for medical imaging"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-7, 7),
            shear=(-3, 3),
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # Increased from 0.1
            contrast_limit=0.2,    # Increased from 0.1
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(
            max_holes=4,
            min_height=int(img_size * 0.02),
            min_width=int(img_size * 0.02),
            max_height=int(img_size * 0.05),
            max_width=int(img_size * 0.05),
            fill_value=0,
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

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

def get_weighted_sampler(csv_path, seed=1337):
    """Weighted sampler"""
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().to_dict()
    weights = df['label'].map(lambda y: 1.0 / counts[y]).values.astype('float32')
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g
    )
