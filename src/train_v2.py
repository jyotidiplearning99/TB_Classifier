# src/train_v2.py

import os
PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"
os.environ['TRANSFORMERS_CACHE'] = f"{PROJECT_DIR}/.cache/huggingface"

for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
    os.makedirs(cache_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve,
    average_precision_score, brier_score_loss, 
    roc_curve, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from pathlib import Path
import contextlib
import pandas as pd
import random
import cv2

from model import MedicalTBClassifier
from dataset_fixed import TBDatasetRobust, get_train_transforms, get_valid_transforms, get_weighted_sampler

def set_seed(seed=1337):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"✓ Random seed set to {seed}")

def optimize_runtime():
    """Runtime optimizations"""
    cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        print("✓ Runtime optimizations applied")

class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with label smoothing"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        smoothed = targets * (1 - 2*self.smoothing) + self.smoothing
        return self.bce(inputs, smoothed)

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}  # ← ADD THIS LINE
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

def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_pred)

def safe_prauc(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        try:
            return average_precision_score(y_true, y_pred)
        except:
            return float('nan')

class RobustTBTrainer:
    """
    Robust TB Classifier Training with Diverse Data
    Version 2: Includes DA+DB for better generalization
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == "cuda")
        
        print(f"\n{'='*70}")
        print(f"🚀 ROBUST TB CLASSIFIER TRAINING V2")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Training with DIVERSE multi-site data including DA+DB")
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = MedicalTBClassifier(
            model_name=config['model_name'],
            pretrained=True,
            dropout=config['dropout']
        ).to(self.device)
        
        if self.use_cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
            print("✓ Model using channels_last format")
        
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # EMA
        self.ema = EMA(self.model, decay=0.999)
        
        # Data
        self.setup_data()
        
        # Loss
        self.criterion = LabelSmoothingBCELoss(smoothing=0.1)  # Increased smoothing
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        total_steps = config['num_epochs'] * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config['min_lr']
        )
        
        # Mixed precision
        self.autocast = torch.cuda.amp.autocast if self.use_cuda else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler() if self.use_cuda else None
        
        # Tracking
        self.best_auc = 0
        self.best_prauc = 0
        self.best_threshold = 0.5
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Log file
        with open(self.output_dir / 'training_log.csv', 'w') as f:
            f.write("epoch,train_loss,train_auc,val_auc,val_prauc,val_f1,"
                   "sensitivity,specificity,sens_at_95spec,brier,threshold\n")
    
    def setup_data(self):
        print("\n📊 Setting up datasets...")
        
        train_dataset = TBDatasetRobust(
            csv_file=self.config['train_csv'],
            transform=get_train_transforms(self.config['img_size']),
            grayscale=True
        )
        
        val_dataset = TBDatasetRobust(
            csv_file=self.config['val_csv'],
            transform=get_valid_transforms(self.config['img_size']),
            grayscale=True
        )
        
        # Weighted sampler
        train_sampler = get_weighted_sampler(
            self.config['train_csv'],
            seed=self.config.get('seed', 1337)
        )
        
        use_persistent = self.config['num_workers'] > 0
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.use_cuda,
            drop_last=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.use_cuda,
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None
        )
        
        print(f"✓ Train: {len(train_dataset)} images, {len(self.train_loader)} batches")
        print(f"✓ Val: {len(val_dataset)} images, {len(self.val_loader)} batches")
        if use_persistent:
            print(f"✓ Using persistent workers with prefetch_factor=2")
    
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
            
            with self.autocast():
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
        all_preds = np.asarray(all_preds).ravel()
        all_labels = np.asarray(all_labels).ravel()
        train_auc = safe_auc(all_labels, all_preds)
        
        return avg_loss, train_auc
    
    def validate(self, epoch, use_ema=True):
        if use_ema and self.ema.steps >= 100:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device, non_blocking=self.use_cuda)
                labels = batch['label'].numpy().ravel()
                
                if self.use_cuda:
                    images = images.to(memory_format=torch.channels_last)
                
                with self.autocast():
                    outputs = self.model(images)
                
                preds = torch.sigmoid(outputs).cpu().numpy().ravel()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_patient_ids.extend(batch['patient_id'])
        
        if use_ema and self.ema.steps >= 100:
            self.ema.restore()
        
        all_preds = np.asarray(all_preds).ravel()
        all_labels = np.asarray(all_labels).ravel()
        
        auc = safe_auc(all_labels, all_preds)
        prauc = safe_prauc(all_labels, all_preds)
        
        try:
            brier = brier_score_loss(all_labels, all_preds)
        except:
            brier = float('nan')
        
        try:
            precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        except:
            optimal_threshold = 0.5
        
        binary_preds = (all_preds >= optimal_threshold).astype(int)
        f1 = f1_score(all_labels, binary_preds)
        
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            idx = np.searchsorted(fpr, 0.05, side='left')
            idx = min(max(idx, 0), len(tpr) - 1)
            sens_at_95spec = tpr[idx]
        except:
            sens_at_95spec = float('nan')
        
        try:
            tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        except:
            sensitivity = float('nan')
            specificity = float('nan')
        
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
        
        pred_df = pd.DataFrame({
            'patient_id': all_patient_ids,
            'true_label': all_labels,
            'predicted_prob': all_preds,
            'predicted_label': binary_preds
        })
        pred_df.to_csv(self.output_dir / f'predictions_epoch_{epoch+1}.csv', index=False)
        
        return auc, prauc, f1, sensitivity, specificity, sens_at_95spec, brier, optimal_threshold
    
    def save_checkpoint(self, epoch, metrics):
        auc, prauc, threshold = metrics['auc'], metrics['prauc'], metrics['threshold']
        
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
            self.best_prauc = prauc
            self.best_threshold = threshold
            self.best_epoch = epoch + 1
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            
            import shutil
            src = self.output_dir / f'predictions_epoch_{epoch+1}.csv'
            dst = self.output_dir / 'predictions_best_epoch.csv'
            if src.exists():
                shutil.copy(src, dst)
            
            print(f"✓ Saved best model (AUC: {auc:.4f}, PR-AUC: {prauc:.4f})")
        
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
            
            with open(self.output_dir / 'training_log.csv', 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_auc:.6f},{auc:.6f},"
                       f"{prauc:.6f},{f1:.6f},{sens:.6f},{spec:.6f},"
                       f"{sens95:.6f},{brier:.6f},{threshold:.6f}\n")
            
            improved = (auc > self.best_auc + 1e-4)
            self.patience_counter = 0 if improved else self.patience_counter + 1
            
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best ROC-AUC:     {self.best_auc:.4f} (epoch {self.best_epoch})")
        print(f"Best PR-AUC:      {self.best_prauc:.4f}")
        print(f"Best Threshold:   {self.best_threshold:.4f}")
        print(f"Model saved to:   {self.output_dir}")
        print(f"{'='*70}\n")

def main():
    set_seed(1337)
    optimize_runtime()
    
    config = {
        # Data - V2 with DA+DB
        'train_csv': 'metadata/train_v2.csv',
        'val_csv': 'metadata/val_v2.csv',
        'img_size': 512,
        'grayscale': True,
        
        # Model
        'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'dropout': 0.3,
        
        # Training
        'batch_size': 8,
        'num_epochs': 100,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'patience': 20,  # Increased for diverse data
        
        # System
        'num_workers': 2,
        'seed': 1337,
        'output_dir': 'outputs/tb_v2_robust'
    }
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION V2")
    print("="*70)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("="*70)
    
    trainer = RobustTBTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
