# src/train_production.py
"""
Production-Ready Medical Image Classifier
- Multi-site training with domain adaptation
- Auto-learning per-domain thresholds
- Dataset boosting for small domains
- Temperature scaling support
- Full deployment artifacts
"""

import os
from pathlib import Path
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve,
    average_precision_score, brier_score_loss, 
    roc_curve, confusion_matrix
)
import random
import cv2
import hashlib

from model import MedicalTBClassifier  # ✅ FIXED
from dataset_fixed import TBDatasetRobust, get_train_transforms, get_valid_transforms

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def optimize_runtime():
    cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        smoothed = targets * (1 - 2*self.smoothing) + self.smoothing
        return self.bce(inputs, smoothed)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}  # ✅ FIXED
        self.steps = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        self.steps += 1
        if self.steps < 100:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                    (1 - self.decay) * param.data)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()  # ✅ Memory cleanup

def get_weighted_sampler_with_boost(csv_path, seed=1337, dataset_boost=None):
    """
    ✅ NEW: Weighted sampler with dataset boosting
    Makes small/shifted domains count more during training
    """
    df = pd.read_csv(csv_path)
    
    # Class weights
    counts = df['label'].value_counts().to_dict()
    weights = df['label'].map(lambda y: 1.0 / counts[y]).values.astype('float32')
    
    # ✅ Dataset boost
    if dataset_boost:
        print("\n📊 Dataset boost applied:")
        for dataset_name, boost_factor in dataset_boost.items():
            mask = df['dataset'] == dataset_name
            n_boosted = mask.sum()
            weights[mask] *= boost_factor
            print(f"  {dataset_name}: {n_boosted} samples × {boost_factor}")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g
    )

class ProductionTrainer:
    """Production-ready medical image classifier"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == "cuda")
        
        self.setup_cache()
        
        print(f"\n{'='*70}")
        print(f"🚀 {config.get('task_name', 'Medical Image Classifier')}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = MedicalTBClassifier(
            model_name=config['model_name'],
            pretrained=True,
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        if self.use_cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.ema = EMA(self.model, decay=config.get('ema_decay', 0.999))
        
        # Data
        self.setup_data()
        
        # Loss
        self.criterion = LabelSmoothingBCELoss(
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        total_steps = config['num_epochs'] * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_cuda else None
        
        # ✅ Domain thresholds tracking
        self.domain_thresholds = {}
        
        # Tracking
        self.best_auc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.init_logging()
    
    def setup_cache(self):
        project_dir = self.config.get('project_dir', os.getcwd())
        cache_root = Path(project_dir) / '.cache'
        
        os.environ['HF_HOME'] = str(cache_root / 'huggingface')
        os.environ['TORCH_HOME'] = str(cache_root / 'torch')
        
        for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
            os.makedirs(cache_dir, exist_ok=True)
    
    def setup_data(self):
        """✅ FIXED: Proper prefetch_factor handling"""
        train_dataset = TBDatasetRobust(
            csv_file=self.config['train_csv'],
            transform=get_train_transforms(self.config.get('img_size', 512)),
            grayscale=self.config.get('grayscale', True)
        )
        
        val_dataset = TBDatasetRobust(
            csv_file=self.config['val_csv'],
            transform=get_valid_transforms(self.config.get('img_size', 512)),
            grayscale=self.config.get('grayscale', True)
        )
        
        # Weighted sampling with boost
        use_weighted = self.config.get('use_weighted_sampler', True)
        dataset_boost = self.config.get('dataset_boost', None)
        
        if use_weighted:
            train_sampler = get_weighted_sampler_with_boost(
                self.config['train_csv'],
                seed=self.config.get('seed', 1337),
                dataset_boost=dataset_boost
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        
        # ✅ FIXED: Safe prefetch_factor
        nw = self.config.get('num_workers', 2)
        
        loader_kwargs = {
            'num_workers': nw,
            'pin_memory': self.use_cuda,
            'persistent_workers': (nw > 0),
        }
        
        # ✅ Only add prefetch_factor if workers > 0
        if nw > 0:
            loader_kwargs['prefetch_factor'] = 2
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            sampler=train_sampler,
            shuffle=shuffle if train_sampler is None else False,
            drop_last=True,
            **loader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8) * 2,
            shuffle=False,
            **loader_kwargs
        )
        
        print(f"\n✓ Train: {len(train_dataset)} images, {len(self.train_loader)} batches")
        print(f"✓ Val: {len(val_dataset)} images, {len(self.val_loader)} batches")
    
    def init_logging(self):
        log_file = self.output_dir / 'training_log.csv'
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,train_auc,val_auc,val_prauc,val_f1,"
                   "sensitivity,specificity,sens_at_95spec,brier,threshold,lr\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=self.use_cuda)
            labels = batch['label'].to(self.device, non_blocking=self.use_cuda).unsqueeze(1)
            
            if self.use_cuda:
                images = images.to(memory_format=torch.channels_last)
            
            self.optimizer.zero_grad()
            
            if self.use_cuda:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.ema.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            labels_np = labels.detach().cpu().numpy().ravel()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = epoch_loss / len(self.train_loader)
        train_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0
        
        return avg_loss, train_auc
    
    def validate(self, epoch, use_ema=True):
        if use_ema and self.ema.steps >= 100:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_datasets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device, non_blocking=self.use_cuda)
                labels = batch['label'].numpy().ravel()
                
                if self.use_cuda:
                    images = images.to(memory_format=torch.channels_last)
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                preds = torch.sigmoid(outputs).cpu().numpy().ravel()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # ✅ FIXED: Safe dataset extraction
                ds = batch.get('dataset', None)
                if ds is None:
                    all_datasets.extend(['unknown'] * len(labels))
                else:
                    all_datasets.extend(list(ds))
        
        if use_ema and self.ema.steps >= 100:
            self.ema.restore()
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        auc = roc_auc_score(all_labels, all_preds)
        prauc = average_precision_score(all_labels, all_preds)
        brier = brier_score_loss(all_labels, all_preds)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        
        binary_preds = (all_preds >= optimal_threshold).astype(int)
        f1 = f1_score(all_labels, binary_preds)
        
        # Sensitivity at 95% specificity
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        idx = np.searchsorted(fpr, 0.05, side='left')
        idx = min(max(idx, 0), len(tpr) - 1)
        sens_at_95spec = tpr[idx]
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ✅ Per-domain analysis with threshold learning
        self.analyze_domains(all_preds, all_labels, all_datasets, optimal_threshold)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1} Validation Results")
        print(f"{'='*70}")
        print(f"ROC-AUC:          {auc:.4f}")
        print(f"PR-AUC:           {prauc:.4f}")
        print(f"Brier Score:      {brier:.4f}")
        print(f"F1 Score:         {f1:.4f}")
        print(f"Sensitivity:      {sensitivity:.4f}")
        print(f"Specificity:      {specificity:.4f}")
        print(f"Sens@95%Spec:     {sens_at_95spec:.4f}")
        print(f"Optimal Thresh:   {optimal_threshold:.4f}")
        print(f"{'='*70}\n")
        
        return auc, prauc, f1, sensitivity, specificity, sens_at_95spec, brier, optimal_threshold
    
    def analyze_domains(self, preds, labels, datasets, threshold):
        """
        ✅ IMPROVED: Learn per-domain thresholds with stability check
        """
        unique_domains = set(datasets)
        
        if len(unique_domains) > 1:
            print("\n📈 Per-Domain Analysis:")
            
            min_samples = self.config.get('min_domain_samples', 30)  # ✅ Raised from 10
            
            for domain in sorted(unique_domains):
                mask = np.array(datasets) == domain
                n = int(mask.sum())
                
                # ✅ Higher threshold for stability
                if n >= min_samples:
                    domain_preds = preds[mask]
                    domain_labels = labels[mask]
                    
                    if len(np.unique(domain_labels)) > 1:
                        domain_auc = roc_auc_score(domain_labels, domain_preds)
                        
                        # Use global threshold
                        binary = (domain_preds >= threshold).astype(int)
                        tn, fp, fn, tp = confusion_matrix(domain_labels, binary, labels=[0,1]).ravel()
                        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        # Find domain-specific optimal threshold
                        prec, rec, thresh_vals = precision_recall_curve(domain_labels, domain_preds)
                        f1_vals = 2 * (prec * rec) / (prec + rec + 1e-8)
                        opt_idx = np.argmax(f1_vals[:-1])
                        domain_threshold = thresh_vals[opt_idx] if len(thresh_vals) > 0 else threshold
                        
                        # ✅ Store domain threshold with validation flag
                        self.domain_thresholds[domain] = {
                            'threshold': float(domain_threshold),
                            'auc': float(domain_auc),
                            'sensitivity': float(sens),
                            'specificity': float(spec),
                            'n_samples': n,
                            'validated': True  # ✅ Has enough samples
                        }
                        
                        print(f"  {domain:25} n={n:4d} AUC={domain_auc:.3f} "
                              f"Sens={sens:.3f} Spec={spec:.3f} OptThr={domain_threshold:.3f}")
                else:
                    # ✅ Mark as not validated
                    print(f"  {domain:25} n={n:4d} (too few for domain-specific threshold)")
                    if n > 0:
                        self.domain_thresholds[domain] = {
                            'threshold': float(threshold),  # Use global
                            'n_samples': n,
                            'validated': False  # ✅ Not enough samples
                        }
    
    def save_checkpoint(self, epoch, metrics):
        """✅ IMPROVED: Save all deployment artifacts"""
        auc = metrics['auc']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        improved = (auc > self.best_auc + 1e-4)
        
        if improved:
            self.best_auc = auc
            self.best_epoch = epoch + 1
            
            # Save model
            model_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, model_path)
            
            # ✅ Compute model checksum
            with open(model_path, 'rb') as f:
                model_checksum = hashlib.sha256(f.read()).hexdigest()[:16]
            
            # ✅ Save model metadata
            model_meta = {
                'version': '2.0',
                'task': self.config.get('task_name', 'Medical Classification'),
                'epoch': epoch + 1,
                'metrics': {
                    'val_auc': float(auc),
                    'val_prauc': float(metrics['prauc']),
                    'val_sensitivity': float(metrics['sensitivity']),
                    'val_specificity': float(metrics['specificity']),
                    'brier_score': float(metrics['brier'])
                },
                'preprocessing': {
                    'img_size': self.config.get('img_size', 512),
                    'grayscale': self.config.get('grayscale', True),
                    'normalization': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    },
                    'clahe': {
                        'enabled': True,
                        'clip_limit': 2.0,
                        'tile_grid_size': [8, 8]
                    }
                },
                'model': {
                    'backbone': self.config['model_name'],
                    'dropout': self.config.get('dropout', 0.3),
                    'checksum': model_checksum
                }
            }
            
            with open(self.output_dir / 'model_meta.json', 'w') as f:
                json.dump(model_meta, f, indent=2)
            
            # ✅ Save thresholds (both formats for compatibility)
            thresholds_data = {
                'default': float(metrics['threshold']),
                'domain_specific': self.domain_thresholds,  # Service format
                'domains': self.domain_thresholds  # Trainer format
            }
            
            with open(self.output_dir / 'thresholds.json', 'w') as f:
                json.dump(thresholds_data, f, indent=2)
            
            print(f"✓ Saved best model (AUC: {auc:.4f})")
            print(f"  Model metadata: model_meta.json")
            print(f"  Thresholds: thresholds.json")
        
        torch.save(checkpoint, self.output_dir / 'latest_model.pth')
    
    def train(self):
        print(f"\n{'='*70}")
        print("🚀 STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.config['num_epochs']):
            train_loss, train_auc = self.train_epoch(epoch)
            
            val_metrics = self.validate(epoch)
            auc, prauc, f1, sens, spec, sens95, brier, threshold = val_metrics
            
            metrics = {
                'auc': auc,
                'prauc': prauc,
                'f1': f1,
                'sensitivity': sens,
                'specificity': spec,
                'sens_at_95spec': sens95,
                'brier': brier,
                'threshold': threshold
            }
            
            self.save_checkpoint(epoch, metrics)
            
            # ✅ Log with LR
            lr_now = self.scheduler.get_last_lr()[0]
            with open(self.output_dir / 'training_log.csv', 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_auc:.6f},{auc:.6f},"
                       f"{prauc:.6f},{f1:.6f},{sens:.6f},{spec:.6f},"
                       f"{sens95:.6f},{brier:.6f},{threshold:.6f},{lr_now:.8f}\n")
            
            # Early stopping
            improved = (auc > self.best_auc + 1e-4)
            self.patience_counter = 0 if improved else self.patience_counter + 1
            
            if self.patience_counter >= self.config.get('patience', 15):
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best ROC-AUC:     {self.best_auc:.4f} (epoch {self.best_epoch})")
        print(f"Model saved to:   {self.output_dir}")
        print(f"{'='*70}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'task_name': 'TB Classification',
            'project_dir': '/scratch/project_2010751/TB_Classifier',
            'train_csv': 'metadata/train_v2.csv',
            'val_csv': 'metadata/val_v2.csv',
            'img_size': 512,
            'grayscale': True,
            'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
            'dropout': 0.3,
            'batch_size': 8,
            'num_epochs': 100,
            'lr': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'patience': 20,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'num_workers': 2,
            'seed': 1337,
            'output_dir': 'outputs/tb_production',
            'use_weighted_sampler': True,
            'dataset_boost': {'da_db_independent': 10},
            'min_domain_samples': 30  # ✅ Minimum for domain-specific threshold
        }
    
    set_seed(config.get('seed', 1337))
    optimize_runtime()
    
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print("="*70)
    
    trainer = ProductionTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


