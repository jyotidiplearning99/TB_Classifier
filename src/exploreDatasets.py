# src/explore_datasets.py

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

class DatasetExplorer:
    """Explore and analyze downloaded TB datasets"""
    
    def __init__(self, base_dir='tb_datasets'):
        self.base_dir = Path(base_dir)
        self.stats = {}
    
    def explore_directory(self, path):
        """Recursively explore directory structure"""
        structure = defaultdict(list)
        
        for root, dirs, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg', '.dcm']:
                    rel_path = Path(root).relative_to(path)
                    structure[str(rel_path)].append(file)
        
        return structure
    
    def count_images(self, dataset_path):
        """Count images by extension"""
        counts = defaultdict(int)
        
        for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
            count = len(list(dataset_path.rglob(f'*{ext}')))
            if count > 0:
                counts[ext] = count
        
        return counts
    
    def analyze_tbx11k(self):
        """Analyze TBX11K dataset structure"""
        print("\n" + "="*60)
        print("📊 ANALYZING TBX11K DATASET")
        print("="*60)
        
        tbx11k_path = self.base_dir / 'tbx11k'
        
        if not tbx11k_path.exists():
            print("❌ TBX11K not found")
            return
        
        # Explore structure
        structure = self.explore_directory(tbx11k_path)
        
        print(f"\nDirectory structure:")
        for dir_name, files in sorted(structure.items()):
            print(f"  {dir_name}: {len(files)} files")
        
        # Count images
        counts = self.count_images(tbx11k_path)
        total = sum(counts.values())
        
        print(f"\nImage counts:")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")
        print(f"  TOTAL: {total}")
        
        self.stats['tbx11k'] = {
            'path': str(tbx11k_path),
            'total_images': total,
            'structure': dict(structure)
        }
    
    def analyze_tb_chest_xray(self):
        """Analyze TB Chest X-ray Database"""
        print("\n" + "="*60)
        print("📊 ANALYZING TB CHEST X-RAY DATABASE")
        print("="*60)
        
        tb_path = self.base_dir / 'tb_chest_xray'
        
        if not tb_path.exists():
            print("❌ TB Chest X-ray not found")
            return
        
        structure = self.explore_directory(tb_path)
        
        print(f"\nDirectory structure:")
        for dir_name, files in sorted(structure.items()):
            print(f"  {dir_name}: {len(files)} files")
        
        counts = self.count_images(tb_path)
        total = sum(counts.values())
        
        print(f"\nImage counts:")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")
        print(f"  TOTAL: {total}")
        
        self.stats['tb_chest_xray'] = {
            'path': str(tb_path),
            'total_images': total,
            'structure': dict(structure)
        }
    
    def analyze_shenzhen(self):
        """Analyze Shenzhen dataset"""
        print("\n" + "="*60)
        print("📊 ANALYZING SHENZHEN DATASET")
        print("="*60)
        
        shenzhen_path = self.base_dir / 'shenzhen'
        
        if not shenzhen_path.exists():
            print("❌ Shenzhen not found")
            return
        
        structure = self.explore_directory(shenzhen_path)
        
        print(f"\nDirectory structure:")
        for dir_name, files in sorted(structure.items()):
            print(f"  {dir_name}: {len(files)} files")
        
        counts = self.count_images(shenzhen_path)
        total = sum(counts.values())
        
        print(f"\nImage counts:")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")
        print(f"  TOTAL: {total}")
        
        self.stats['shenzhen'] = {
            'path': str(shenzhen_path),
            'total_images': total,
            'structure': dict(structure)
        }
    
    def analyze_montgomery(self):
        """Analyze Montgomery dataset"""
        print("\n" + "="*60)
        print("📊 ANALYZING MONTGOMERY DATASET")
        print("="*60)
        
        montgomery_path = self.base_dir / 'montgomery'
        
        if not montgomery_path.exists():
            print("❌ Montgomery not found")
            return
        
        structure = self.explore_directory(montgomery_path)
        
        print(f"\nDirectory structure:")
        for dir_name, files in sorted(structure.items()):
            print(f"  {dir_name}: {len(files)} files")
        
        counts = self.count_images(montgomery_path)
        total = sum(counts.values())
        
        print(f"\nImage counts:")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")
        print(f"  TOTAL: {total}")
        
        self.stats['montgomery'] = {
            'path': str(montgomery_path),
            'total_images': total,
            'structure': dict(structure)
        }
    
    def analyze_all(self):
        """Analyze all datasets"""
        print("\n" + "🔍"*30)
        print("DATASET EXPLORATION")
        print("🔍"*30)
        
        self.analyze_tbx11k()
        self.analyze_tb_chest_xray()
        self.analyze_shenzhen()
        self.analyze_montgomery()
        
        self.print_summary()
        self.save_stats()
    
    def print_summary(self):
        """Print overall summary"""
        print("\n" + "="*60)
        print("📈 OVERALL SUMMARY")
        print("="*60)
        
        total_images = sum(
            dataset.get('total_images', 0) 
            for dataset in self.stats.values()
        )
        
        print(f"\nDatasets analyzed: {len(self.stats)}")
        print(f"Total images: {total_images:,}")
        
        print("\nBreakdown by dataset:")
        for name, info in self.stats.items():
            count = info.get('total_images', 0)
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"  {name:20} {count:6,} images ({percentage:5.1f}%)")
        
        print("\n" + "="*60)
    
    def save_stats(self):
        """Save statistics to JSON"""
        with open('dataset_statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n✓ Statistics saved to: dataset_statistics.json")

if __name__ == "__main__":
    explorer = DatasetExplorer(base_dir='tb_datasets')
    explorer.analyze_all()
