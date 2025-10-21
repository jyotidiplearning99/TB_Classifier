# src/data_quality_check.py

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataQualityChecker:
    """Check data quality and image properties"""
    
    def __init__(self, metadata_csv='metadata/tb_complete_metadata.csv'):
        self.df = pd.read_csv(metadata_csv)
        self.issues = []
    
    def check_images(self):
        """Check if all images can be loaded"""
        print("\n🔍 Checking image integrity...")
        
        corrupt_images = []
        image_properties = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                img_path = row['image_path']
                img = cv2.imread(img_path)
                
                if img is None:
                    corrupt_images.append(img_path)
                else:
                    h, w = img.shape[:2]
                    channels = img.shape[2] if len(img.shape) == 3 else 1
                    
                    image_properties.append({
                        'height': h,
                        'width': w,
                        'channels': channels,
                        'aspect_ratio': w/h
                    })
            except Exception as e:
                corrupt_images.append(img_path)
                self.issues.append(f"Error loading {img_path}: {e}")
        
        print(f"\n✓ Checked {len(self.df)} images")
        print(f"  - Valid: {len(self.df) - len(corrupt_images)}")
        print(f"  - Corrupt: {len(corrupt_images)}")
        
        if corrupt_images:
            print("\n⚠️  Corrupt images found:")
            for img in corrupt_images[:10]:  # Show first 10
                print(f"  - {img}")
        
        # Analyze properties
        if image_properties:
            props_df = pd.DataFrame(image_properties)
            
            print(f"\n📏 Image Properties:")
            print(f"  Height: {props_df['height'].min()}-{props_df['height'].max()} (mean: {props_df['height'].mean():.0f})")
            print(f"  Width: {props_df['width'].min()}-{props_df['width'].max()} (mean: {props_df['width'].mean():.0f})")
            print(f"  Aspect ratio: {props_df['aspect_ratio'].min():.2f}-{props_df['aspect_ratio'].max():.2f}")
            
            # Visualize
            self.plot_distributions(props_df)
        
        return corrupt_images
    
    def plot_distributions(self, props_df):
        """Plot image property distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(props_df['height'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Height Distribution')
        axes[0, 0].set_xlabel('Height (pixels)')
        
        axes[0, 1].hist(props_df['width'], bins=50, edgecolor='black')
        axes[0, 1].set_title('Width Distribution')
        axes[0, 1].set_xlabel('Width (pixels)')
        
        axes[1, 0].hist(props_df['aspect_ratio'], bins=50, edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
        
        axes[1, 1].scatter(props_df['width'], props_df['height'], alpha=0.3)
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        plt.savefig('data_quality_report.png', dpi=300)
        print(f"\n✓ Quality report saved to: data_quality_report.png")
        plt.close()

if __name__ == "__main__":
    checker = DataQualityChecker()
    corrupt = checker.check_images()
