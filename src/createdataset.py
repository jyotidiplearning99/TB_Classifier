# src/dataset.py

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TBDataset(Dataset):
    """TB Chest X-ray Dataset with advanced preprocessing"""
    
    def __init__(self, csv_file, transform=None, grayscale=True):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.grayscale = grayscale
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image - grayscale mode for CXR
        if self.grayscale:
            image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot load image: {row['image_path']}")
            
            # Apply CLAHE on grayscale (better for medical imaging)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            
            # Stack to 3 channels for pretrained models
            image = np.stack([image, image, image], axis=-1)
        else:
            image = cv2.imread(row['image_path'])
            if image is None:
                raise ValueError(f"Cannot load image: {row['image_path']}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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

def get_train_transforms(img_size=512):
    """
    Medical imaging appropriate augmentations
    Conservative for chest X-rays - no heavy distortions
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Geometric - conservative for medical imaging
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-7, 7),
            shear=(-3, 3),
            p=0.5
        ),
        
        # Noise augmentations - mild
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        
        # Dropout - IMPROVED: with min sizes to avoid 0-sized holes
        A.CoarseDropout(
            max_holes=4,
            min_height=int(img_size * 0.02),  # Min size to avoid 0
            min_width=int(img_size * 0.02),
            max_height=int(img_size * 0.05),
            max_width=int(img_size * 0.05),
            fill_value=0,
            p=0.2
        ),
        
        # Normalization - ImageNet stats
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# Safe import for older PyTorch versions
try:
    from torch.utils.data import WeightedRandomSampler
except ImportError:
    from torch.utils.data.sampler import WeightedRandomSampler
    

def get_valid_transforms(img_size=512):
    """Validation transforms - minimal preprocessing"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_weighted_sampler(csv_path, seed=1337):
    """
    Create weighted sampler for imbalanced data with reproducible sampling
    Inverse frequency weighting
    """
    df = pd.read_csv(csv_path)
    
    # Count classes
    counts = df['label'].value_counts().to_dict()
    
    # Calculate weights (inverse frequency)
    weights = df['label'].map(lambda y: 1.0 / counts[y]).values.astype('float32')
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g
    )
    
    print(f"✓ Weighted sampler created (seed={seed})")
    print(f"  Class 0: {counts.get(0, 0)} samples, weight: {1.0/counts.get(0, 1):.4f}")
    print(f"  Class 1: {counts.get(1, 0)} samples, weight: {1.0/counts.get(1, 1):.4f}")
    
    return sampler

# Test dataset
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING DATASET")
    print("="*60 + "\n")
    
    dataset = TBDataset(
        csv_file='metadata/train.csv',
        transform=get_train_transforms(),
        grayscale=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Label: {sample['label']}")
    print(f"Patient ID: {sample['patient_id']}")
    
    # Test sampler
    sampler = get_weighted_sampler('metadata/train.csv', seed=42)
    print(f"\n✓ Sampler created with reproducible seed")
    
    print("\n" + "="*60)
