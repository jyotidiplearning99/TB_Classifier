# src/prepare_independent_dataset.py

import os
import pandas as pd
from pathlib import Path

def prepare_da_db_dataset(base_dir='tb_datasets/tbx11k/TBX11K', output_csv='metadata/independent_test.csv'):
    """
    Prepare DA and DB datasets as independent validation set
    
    Naming convention:
    - Files starting with 'n' or 'nx' = Normal (label 0)
    - Files starting with 'p' or 'px' = Positive TB (label 1)
    """
    
    base_path = Path(base_dir)
    
    print(f"\n{'='*60}")
    print("📂 PREPARING INDEPENDENT DATASET (DA+DB)")
    print(f"{'='*60}")
    print(f"Base path: {base_path}\n")
    
    # DA and DB path
    da_db_path = base_path / 'imgs' / 'extra' / 'da+db'
    
    if not da_db_path.exists():
        print(f"❌ Path not found: {da_db_path}")
        return None
    
    print(f"✓ Found DA+DB path: {da_db_path}\n")
    
    # Collect all images
    all_data = []
    
    # Process train and val folders
    for subset in ['train', 'val']:
        subset_path = da_db_path / subset
        
        if not subset_path.exists():
            print(f"⚠️  Path not found: {subset_path}")
            continue
        
        print(f"📁 Processing {subset}/")
        
        # Find all images
        image_files = list(subset_path.glob('*.png')) + list(subset_path.glob('*.jpg'))
        
        print(f"   Found {len(image_files)} images")
        
        n_normal = 0
        n_positive = 0
        
        for img_path in image_files:
            filename = img_path.stem.lower()
            
            # Determine label based on filename prefix
            if filename.startswith('n') or filename.startswith('nx'):
                # Normal/Negative case
                label = 0
                n_normal += 1
            elif filename.startswith('p') or filename.startswith('px'):
                # Positive/Patient case (TB)
                label = 1
                n_positive += 1
            else:
                # Unknown naming - skip
                print(f"   ⚠️  Unknown naming pattern: {img_path.name}")
                continue
            
            all_data.append({
                'image_path': str(img_path.absolute()),
                'label': label,
                'patient_id': f"dadb_{img_path.stem}",
                'dataset': 'da_db_independent',
                'subset': subset
            })
        
        print(f"   ✓ Labeled: {n_normal} normal, {n_positive} TB positive")
    
    if len(all_data) == 0:
        print("\n❌ No images could be processed!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("✅ INDEPENDENT DATASET PREPARED")
    print(f"{'='*60}")
    print(f"Total images:     {len(df)}")
    print(f"  From train:     {(df['subset']=='train').sum()}")
    print(f"  From val:       {(df['subset']=='val').sum()}")
    print(f"\nClass distribution:")
    print(f"  TB Negative (0): {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    print(f"  TB Positive (1): {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    print(f"\nDataset origin:")
    print(f"  DA (Belarus):    {sum(1 for p in df['patient_id'] if not 'x' in p.split('_')[1][1:])}")
    print(f"  DB (Belarus):    {sum(1 for p in df['patient_id'] if 'x' in p.split('_')[1][1:])}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Show sample
    print("Sample of prepared data:")
    print(df[['patient_id', 'label', 'subset']].head(10).to_string(index=False))
    print("...")
    print(df[['patient_id', 'label', 'subset']].tail(5).to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = prepare_da_db_dataset()
    
    if df is not None:
        print("\n✅ SUCCESS! Ready for independent evaluation!")
        print("\nNext step:")
        print("  python src/evaluate_independent.py")
    else:
        print("\n❌ Failed to prepare dataset")
