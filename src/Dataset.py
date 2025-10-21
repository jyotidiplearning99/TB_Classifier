# complete_download.py

import os
import sys
import json
import kaggle
import zipfile
import urllib.request
from pathlib import Path
import shutil

class TBDatasetDownloader:
    """Complete TB dataset downloader"""
    
    def __init__(self, base_dir='tb_datasets'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.datasets_info = {
            'tbx11k': {
                'name': 'TBX11K',
                'size': '11,200 images',
                'resolution': '512x512',
                'downloaded': False
            },
            'tb_chest_xray': {
                'name': 'TB Chest X-ray Database',
                'size': 'Multiple collections',
                'resolution': 'Various',
                'downloaded': False
            },
            'shenzhen': {
                'name': 'Shenzhen Hospital',
                'size': '~662 images',
                'resolution': 'Various',
                'downloaded': False
            },
            'montgomery': {
                'name': 'Montgomery County',
                'size': '~138 images',
                'resolution': 'Various',
                'downloaded': False
            }
        }
    
    def check_kaggle_credentials(self):
        """Check if Kaggle API is configured"""
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_config.exists():
            print("❌ Kaggle API credentials not found!")
            print("\nSetup Instructions:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section and click 'Create New API Token'")
            print("3. This downloads kaggle.json")
            print("4. Place it in:")
            print(f"   - Linux/Mac: {Path.home() / '.kaggle' / 'kaggle.json'}")
            print(f"   - Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)")
            return False
        
        print("✓ Kaggle credentials found!")
        return True
    
    def download_tbx11k(self):
        """Download TBX11K dataset"""
        dataset_dir = self.base_dir / 'tbx11k'
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            print("\n" + "="*60)
            print("Downloading TBX11K (11,200 images)...")
            print("="*60)
            
            # Try multiple Kaggle dataset identifiers
            kaggle_ids = [
                'usmanshams/tbx-11',
                'vbookshelf/tbx11k-simplified'
            ]
            
            for kaggle_id in kaggle_ids:
                try:
                    kaggle.api.dataset_download_files(
                        kaggle_id,
                        path=str(dataset_dir),
                        unzip=True
                    )
                    print(f"✓ Downloaded from {kaggle_id}")
                    self.datasets_info['tbx11k']['downloaded'] = True
                    return str(dataset_dir)
                except Exception as e:
                    print(f"Failed with {kaggle_id}: {e}")
                    continue
            
            print("❌ Could not download TBX11K from Kaggle")
            print("Manual download: https://www.kaggle.com/datasets/usmanshams/tbx-11")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None
    
    def download_tb_chest_xray(self):
        """Download TB Chest X-ray Database"""
        dataset_dir = self.base_dir / 'tb_chest_xray'
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            print("\n" + "="*60)
            print("Downloading TB Chest X-ray Database...")
            print("="*60)
            
            kaggle.api.dataset_download_files(
                'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                path=str(dataset_dir),
                unzip=True
            )
            
            print("✓ TB Chest X-ray Database downloaded!")
            self.datasets_info['tb_chest_xray']['downloaded'] = True
            return str(dataset_dir)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Manual download: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        
        return None
    
    def download_shenzhen(self):
        """Download Shenzhen Hospital dataset"""
        dataset_dir = self.base_dir / 'shenzhen'
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            print("\n" + "="*60)
            print("Downloading Shenzhen Hospital dataset...")
            print("="*60)
            
            url = "http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"
            zip_path = dataset_dir / "ChinaSet_AllFiles.zip"
            
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Clean up
            zip_path.unlink()
            
            print("✓ Shenzhen dataset downloaded!")
            self.datasets_info['shenzhen']['downloaded'] = True
            return str(dataset_dir)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Manual download: http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip")
        
        return None
    
    def download_montgomery(self):
        """Download Montgomery County dataset"""
        dataset_dir = self.base_dir / 'montgomery'
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            print("\n" + "="*60)
            print("Downloading Montgomery County dataset...")
            print("="*60)
            
            url = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
            zip_path = dataset_dir / "NLM-MontgomeryCXRSet.zip"
            
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Clean up
            zip_path.unlink()
            
            print("✓ Montgomery dataset downloaded!")
            self.datasets_info['montgomery']['downloaded'] = True
            return str(dataset_dir)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Manual download: http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip")
        
        return None
    
    def download_all(self):
        """Download all TB datasets"""
        print("\n" + "🔬"*30)
        print("TB DATASET DOWNLOADER")
        print("🔬"*30 + "\n")
        
        # Check Kaggle credentials
        has_kaggle = self.check_kaggle_credentials()
        
        # Download datasets
        if has_kaggle:
            self.download_tbx11k()
            self.download_tb_chest_xray()
        
        self.download_shenzhen()
        self.download_montgomery()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print download summary"""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60 + "\n")
        
        for key, info in self.datasets_info.items():
            status = "✓ Downloaded" if info['downloaded'] else "❌ Failed"
            print(f"{info['name']:30} {status}")
            print(f"  Size: {info['size']}, Resolution: {info['resolution']}")
        
        print("\n" + "="*60)
        print(f"All datasets saved to: {self.base_dir.absolute()}")
        print("="*60 + "\n")

# Run the downloader
if __name__ == "__main__":
    downloader = TBDatasetDownloader(base_dir='tb_datasets')
    downloader.download_all()
