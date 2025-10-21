# src/create_metadata.py

import os
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

class MetadataCreator:
    """Create unified metadata CSV for all TB datasets"""
    
    def __init__(self, base_dir='tb_datasets', output_dir='metadata'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.all_data = []
    
    def process_tbx11k(self):
        """Process TBX11K dataset"""
        print("\n📝 Processing TBX11K...")
        
        tbx11k_path = self.base_dir / 'tbx11k'
        
        # Find all images
        image_files = list(tbx11k_path.rglob('*.png')) + list(tbx11k_path.rglob('*.jpg'))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc="Processing TBX11K"):
            # Extract label from directory structure
            # TBX11K typically has folders like: Tuberculosis, Normal, etc.
            parent_dir = img_path.parent.name.lower()
            
            if 'tuberculosis' in parent_dir or 'tb' in parent_dir or 'sick' in parent_dir:
                label = 1  # TB positive
            elif 'normal' in parent_dir or 'healthy' in parent_dir:
                label = 0  # TB negative
            else:
                # Try to infer from filename
                filename = img_path.stem.lower()
                if 'tb' in filename or 'tuberculosis' in filename:
                    label = 1
                else:
                    label = 0  # Default to negative
            
            # Extract patient ID from filename
            patient_id = f"tbx11k_{img_path.stem}"
            
            self.all_data.append({
                'image_path': str(img_path.absolute()),
                'relative_path': str(img_path.relative_to(self.base_dir)),
                'label': label,
                'patient_id': patient_id,
                'dataset': 'tbx11k',
                'filename': img_path.name,
                'image_size': img_path.stat().st_size
            })
        
        print(f"✓ Processed {len(image_files)} TBX11K images")
    
    def process_tb_chest_xray(self):
        """Process TB Chest X-ray Database"""
        print("\n📝 Processing TB Chest X-ray Database...")
        
        tb_path = self.base_dir / 'tb_chest_xray'
        
        image_files = list(tb_path.rglob('*.png')) + list(tb_path.rglob('*.jpg'))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc="Processing TB Chest X-ray"):
            # This dataset usually has clear directory structure
            path_parts = img_path.parts
            
            label = 0  # Default
            for part in path_parts:
                part_lower = part.lower()
                if 'tuberculosis' in part_lower or 'tb' in part_lower:
                    label = 1
                    break
                elif 'normal' in part_lower:
                    label = 0
                    break
            
            patient_id = f"tb_chest_{img_path.stem}"
            
            self.all_data.append({
                'image_path': str(img_path.absolute()),
                'relative_path': str(img_path.relative_to(self.base_dir)),
                'label': label,
                'patient_id': patient_id,
                'dataset': 'tb_chest_xray',
                'filename': img_path.name,
                'image_size': img_path.stat().st_size
            })
        
        print(f"✓ Processed {len(image_files)} TB Chest X-ray images")
    
    def process_shenzhen(self):
        """Process Shenzhen dataset"""
        print("\n📝 Processing Shenzhen dataset...")
        
        shenzhen_path = self.base_dir / 'shenzhen'
        
        image_files = list(shenzhen_path.rglob('*.png')) + list(shenzhen_path.rglob('*.jpg'))
        
        print(f"Found {len(image_files)} images")
        
        # Look for metadata file
        metadata_file = None
        for meta_file in ['ClinicalReadings.txt', 'metadata.txt', 'labels.txt']:
            potential_path = shenzhen_path / meta_file
            if potential_path.exists():
                metadata_file = potential_path
                break
        
        # Parse metadata if exists
        metadata_dict = {}
        if metadata_file:
            print(f"Found metadata file: {metadata_file.name}")
            with open(metadata_file, 'r') as f:
                for line in f:
                    # Parse format: filename,label or similar
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        label_str = parts[1].strip().lower()
                        label = 1 if 'tuberculosis' in label_str or 'abnormal' in label_str else 0
                        metadata_dict[filename] = label
        
        for img_path in tqdm(image_files, desc="Processing Shenzhen"):
            # Check metadata first
            if img_path.name in metadata_dict:
                label = metadata_dict[img_path.name]
            else:
                # Infer from directory structure
                parent_dir = img_path.parent.name.lower()
                if 'tuberculosis' in parent_dir or 'tb' in parent_dir:
                    label = 1
                else:
                    label = 0
            
            patient_id = f"shenzhen_{img_path.stem}"
            
            self.all_data.append({
                'image_path': str(img_path.absolute()),
                'relative_path': str(img_path.relative_to(self.base_dir)),
                'label': label,
                'patient_id': patient_id,
                'dataset': 'shenzhen',
                'filename': img_path.name,
                'image_size': img_path.stat().st_size
            })
        
        print(f"✓ Processed {len(image_files)} Shenzhen images")
    
    def process_montgomery(self):
        """Process Montgomery dataset"""
        print("\n📝 Processing Montgomery dataset...")
        
        montgomery_path = self.base_dir / 'montgomery'
        
        image_files = list(montgomery_path.rglob('*.png')) + list(montgomery_path.rglob('*.jpg'))
        
        print(f"Found {len(image_files)} images")
        
        # Look for metadata
        metadata_file = None
        for meta_file in ['ClinicalReadings.txt', 'metadata.txt', 'labels.txt']:
            potential_path = montgomery_path / meta_file
            if potential_path.exists():
                metadata_file = potential_path
                break
        
        metadata_dict = {}
        if metadata_file:
            print(f"Found metadata file: {metadata_file.name}")
            with open(metadata_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        label_str = parts[1].strip().lower()
                        label = 1 if 'tuberculosis' in label_str or 'abnormal' in label_str else 0
                        metadata_dict[filename] = label
        
        for img_path in tqdm(image_files, desc="Processing Montgomery"):
            if img_path.name in metadata_dict:
                label = metadata_dict[img_path.name]
            else:
                parent_dir = img_path.parent.name.lower()
                label = 1 if 'tuberculosis' in parent_dir or 'tb' in parent_dir else 0
            
            patient_id = f"montgomery_{img_path.stem}"
            
            self.all_data.append({
                'image_path': str(img_path.absolute()),
                'relative_path': str(img_path.relative_to(self.base_dir)),
                'label': label,
                'patient_id': patient_id,
                'dataset': 'montgomery',
                'filename': img_path.name,
                'image_size': img_path.stat().st_size
            })
        
        print(f"✓ Processed {len(image_files)} Montgomery images")
    
    def create_metadata(self):
        """Create complete metadata CSV"""
        print("\n" + "="*60)
        print("🏗️  CREATING UNIFIED METADATA")
        print("="*60)
        
        # Process all datasets
        self.process_tbx11k()
        self.process_tb_chest_xray()
        self.process_shenzhen()
        self.process_montgomery()
        
        # Create DataFrame
        df = pd.DataFrame(self.all_data)
        
        # Statistics
        print("\n" + "="*60)
        print("📊 METADATA STATISTICS")
        print("="*60)
        
        print(f"\nTotal images: {len(df):,}")
        print(f"\nBy dataset:")
        print(df['dataset'].value_counts())
        
        print(f"\nBy label:")
        print(f"  TB Negative (0): {(df['label']==0).sum():,}")
        print(f"  TB Positive (1): {(df['label']==1).sum():,}")
        
        print(f"\nClass distribution:")
        print(f"  Negative: {(df['label']==0).sum()/len(df)*100:.1f}%")
        print(f"  Positive: {(df['label']==1).sum()/len(df)*100:.1f}%")
        
        # Save complete metadata
        output_path = self.output_dir / 'tb_complete_metadata.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Complete metadata saved to: {output_path}")
        
        # Save per-dataset metadata
        for dataset_name in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset_name]
            dataset_path = self.output_dir / f'tb_{dataset_name}_metadata.csv'
            dataset_df.to_csv(dataset_path, index=False)
            print(f"✓ {dataset_name} metadata saved to: {dataset_path}")
        
        # Create train/val/test splits
        self.create_splits(df)
        
        return df
    
    def create_splits(self, df):
        """Create train/validation/test splits"""
        from sklearn.model_selection import train_test_split
        
        print("\n" + "="*60)
        print("✂️  CREATING DATA SPLITS")
        print("="*60)
        
        # Stratified split by label
        train_val_df, test_df = train_test_split(
            df, 
            test_size=0.15, 
            stratify=df['label'],
            random_state=42
        )
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.15,  # 0.15 of 0.85 = ~0.128 of total
            stratify=train_val_df['label'],
            random_state=42
        )
        
        # Save splits
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        print(f"\nTrain set: {len(train_df):,} images")
        print(f"  - Negative: {(train_df['label']==0).sum():,} ({(train_df['label']==0).sum()/len(train_df)*100:.1f}%)")
        print(f"  - Positive: {(train_df['label']==1).sum():,} ({(train_df['label']==1).sum()/len(train_df)*100:.1f}%)")
        
        print(f"\nValidation set: {len(val_df):,} images")
        print(f"  - Negative: {(val_df['label']==0).sum():,} ({(val_df['label']==0).sum()/len(val_df)*100:.1f}%)")
        print(f"  - Positive: {(val_df['label']==1).sum():,} ({(val_df['label']==1).sum()/len(val_df)*100:.1f}%)")
        
        print(f"\nTest set: {len(test_df):,} images")
        print(f"  - Negative: {(test_df['label']==0).sum():,} ({(test_df['label']==0).sum()/len(test_df)*100:.1f}%)")
        print(f"  - Positive: {(test_df['label']==1).sum():,} ({(test_df['label']==1).sum()/len(test_df)*100:.1f}%)")
        
        print(f"\n✓ Splits saved to metadata/")

if __name__ == "__main__":
    creator = MetadataCreator(base_dir='tb_datasets', output_dir='metadata')
    df = creator.create_metadata()
    
    print("\n" + "="*60)
    print("✅ METADATA CREATION COMPLETE!")
    print("="*60)
