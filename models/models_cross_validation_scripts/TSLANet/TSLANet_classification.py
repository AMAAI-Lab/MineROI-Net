import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import os
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

import dataloader
from dataloader_tslanet import get_datasets
from utils_tslanet import get_clf_report, save_copy_of_files, str2bool, random_masking_3D
torch.set_float32_matmul_precision('medium')
from lightning.pytorch.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# Run the preprocessing




class MetricsCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer, 'logged_metrics'):
            train_loss = trainer.logged_metrics.get('train_loss', None)
            train_acc = trainer.logged_metrics.get('train_acc', None)
            train_f1 = trainer.logged_metrics.get('train_f1', None)
            
            if train_loss is not None:
                self.train_losses.append(float(train_loss))
            if train_acc is not None:
                self.train_accs.append(float(train_acc))
            if train_f1 is not None:
                self.train_f1s.append(float(train_f1))
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(trainer, 'logged_metrics'):
            val_loss = trainer.logged_metrics.get('val_loss', None)
            val_acc = trainer.logged_metrics.get('val_acc', None)
            val_f1 = trainer.logged_metrics.get('val_f1', None)
            
            if val_loss is not None:
                self.val_losses.append(float(val_loss))
            if val_acc is not None:
                self.val_accs.append(float(val_acc))
            if val_f1 is not None:
                self.val_f1s.append(float(val_f1))


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        # Normalize energy
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if args.adaptive_filter:
            # Adaptive High Frequency Mask
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        self.input_layer = nn.Linear(args.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # Classifier head
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]

        preds, target = self.model.pretrain(data)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)

        # Use weighted CrossEntropyLoss with mild label smoothing
        if hasattr(args, 'class_weights'):
            self.register_buffer('class_weights', args.class_weights)
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=0.0
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

        
        # Store predictions for confusion matrix
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

   
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        # Warmup + Cosine Annealing
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Store predictions for confusion matrix (only during testing)
        if mode == "test":
            self.test_predictions.extend(preds.argmax(dim=-1).cpu().numpy())
            self.test_targets.extend(labels.cpu().numpy())
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        # Generate confusion matrix and classification report
        if self.test_predictions and self.test_targets:
            self.generate_confusion_matrix()
            self.generate_classification_report()

    def generate_confusion_matrix(self):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(self.test_targets, self.test_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=args.class_names, yticklabels=args.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        cm_path = os.path.join(CHECKPOINT_PATH, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nConfusion matrix saved to: {cm_path}")

    

    def generate_classification_report(self):
        """Generate and save classification report"""
        report = classification_report(
            self.test_targets, self.test_predictions, 
            target_names=args.class_names, output_dict=True
        )
        
        # Save classification report as JSON
        report_path = os.path.join(CHECKPOINT_PATH, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print classification report
        print("\nClassification Report:")
        print("=" * 60)
        report_str = classification_report(
            self.test_targets, self.test_predictions, 
            target_names=args.class_names
        )
        print(report_str)
        print(f"Classification report saved to: {report_path}")




def save_training_plots(metrics_callback, save_path):
    """Generate and save training progress plots as separate images"""
    
    epochs = range(1, len(metrics_callback.train_losses) + 1)
    
    # 1. Loss plot
    if metrics_callback.train_losses and metrics_callback.val_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, metrics_callback.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, metrics_callback.val_losses, 'r-', label='Test Loss', linewidth=2)
        # ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(save_path, 'training_loss.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {loss_path}")
    
    # 2. Accuracy plot
    if metrics_callback.train_accs and metrics_callback.val_accs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, metrics_callback.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, metrics_callback.val_accs, 'r-', label='Test Accuracy', linewidth=2)
        # ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = os.path.join(save_path, 'training_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to: {acc_path}")
    
    # 3. F1 Score plot
    if metrics_callback.train_f1s and metrics_callback.val_f1s:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, metrics_callback.train_f1s, 'b-', label='Train F1', linewidth=2)
        ax.plot(epochs, metrics_callback.val_f1s, 'r-', label='Test F1', linewidth=2)
        # ax.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        f1_path = os.path.join(save_path, 'training_f1.png')
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"F1 plot saved to: {f1_path}")
    
    print(f"\nAll training plots saved to: {save_path}")



def save_experiment_results(args, acc_results, f1_results, save_path):
    """Save comprehensive experiment results"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    results = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset': os.path.basename(args.data_path),
            'model_id': args.model_id,
            'run_description': run_description
        },
        'model_parameters': {
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'patch_size': args.patch_size,
            'dropout_rate': args.dropout_rate,
            'masking_ratio': args.masking_ratio,
            'ICB': args.ICB,
            'ASB': args.ASB,
            'adaptive_filter': args.adaptive_filter,
            'load_from_pretrained': args.load_from_pretrained
        },
        'training_parameters': {
            'num_epochs': args.num_epochs,
            'pretrain_epochs': args.pretrain_epochs,
            'batch_size': args.batch_size,
            'train_lr': args.train_lr,
            'pretrain_lr': args.pretrain_lr
        },
        'dataset_info': {
            'num_classes': args.num_classes,
            'seq_len': args.seq_len,
            'num_channels': args.num_channels,
            'class_names': args.class_names
        },
        'results': {
            'accuracy': acc_results,
            'f1_score': f1_results
        }
    }
    
    # Save as JSON
    results_path = os.path.join(save_path, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: {os.path.basename(args.data_path)}")
    print(f"Model Configuration: dim={args.emb_dim}, depth={args.depth}")
    print(f"Components: ASB={args.ASB}, ICB={args.ICB}, AF={args.adaptive_filter}")
    print(f"Pretraining: {args.load_from_pretrained}")
    print("-" * 80)
    print("FINAL RESULTS:")
    print(f"Test Accuracy:  {acc_results['test']:.4f}")
    print(f"Val Accuracy:   {acc_results['val']:.4f}")
    print(f"Test F1 Score:  {f1_results['test']:.4f}")
    print(f"Val F1 Score:   {f1_results['val']:.4f}")
    print("=" * 80)
    print(f"All results saved to: {save_path}")
    print("=" * 80)


def pretrain_model():
    print("\nStarting Pretraining Phase...")
    print("=" * 50)
    
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    
    # Setup metrics tracking for pretraining
    pretrain_metrics = MetricsCallback()
    
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            pretrain_metrics,
            TQDMProgressBar(refresh_rate=1)  # Single progress bar for all pretraining epochs
        ],
        logger=WandbLogger(
            project=f"tslanet_{SEQ_LEN}",
            name=f"pretrain_{run_description}",
            save_dir=CHECKPOINT_PATH
        ) 
        # logger=TensorBoardLogger(CHECKPOINT_PATH, name="pretrain_logs")
    )
    
    L.seed_everything(args.seed)

    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nPretraining completed! Best model saved at: {pretrain_checkpoint_callback.best_model_path}")
    
    return pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    print("\nStarting Main Training Phase...")
    print("=" * 50)
    
    # Setup metrics tracking for main training
    train_metrics = MetricsCallback()
    
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1.0,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            train_metrics,
            TQDMProgressBar(refresh_rate=1)
        ],
        logger=WandbLogger(
            project=f"tslanet_{SEQ_LEN}",
            name=f"train_{run_description}",
            save_dir=CHECKPOINT_PATH
        )
    )
    
    L.seed_everything(args.seed)

    # Initialize training model
    model = model_training()
    
    # Load pretrained weights if available
    if args.load_from_pretrained and pretrained_model_path:
        print(f"Loading pretrained weights from: {pretrained_model_path}")
        pretrained_checkpoint = torch.load(pretrained_model_path)
        pretrained_state_dict = pretrained_checkpoint['state_dict']
        model_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                model_state_dict[new_key] = value
        model.model.load_state_dict(model_state_dict, strict=False)
        print("✓ Pretrained weights loaded successfully!")
    else:
        print("Training from scratch (no pretraining)")

    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # last_ckpt_path = os.path.join(CHECKPOINT_PATH, 'last.ckpt')
    # model = model_training.load_from_checkpoint(last_ckpt_path)

    # Test best model on validation and test set
    print("\nEvaluating model on validation set...")
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    
    # Clear accumulated predictions from validation run
    model.test_predictions = []
    model.test_targets = []

    print("\nEvaluating model on test set...")
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)



    # ========================================
    # READ CLASSIFICATION REPORT FOR ACCURATE METRICS ✅
    # ========================================
    report_path = os.path.join(CHECKPOINT_PATH, 'classification_report.json')
    with open(report_path, 'r') as f:
        classification_report_data = json.load(f)
    
    # Extract accuracy and macro F1 from the classification report (TEST DATA)
    test_accuracy_from_report = classification_report_data['accuracy']
    test_macro_f1_from_report = classification_report_data['macro avg']['f1-score']

    # Use classification report values for results
    acc_result = {
        "test": test_accuracy_from_report,      # ✅ From classification report
        "val": test_accuracy_from_report        # From PyTorch Lightning
    }
    f1_result = {
        "test": test_macro_f1_from_report,      # ✅ From classification report (macro F1)
        "val": test_macro_f1_from_report         # From PyTorch Lightning
    }


    # # Confusion matrix will be generated here with only test data
    # acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    # f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    # ✅ LOG BEST MODEL METRICS TO WANDB
    if trainer.logger:
        log_dict = {
            "best_model/val_acc": test_accuracy_from_report,
            "best_model/val_f1": test_macro_f1_from_report,
            "best_model/val_loss": val_result[0].get("test_loss", None),
            "best_model/test_acc": test_accuracy_from_report, #test_result[0]["test_acc"],
            "best_model/test_f1": test_macro_f1_from_report, #test_result[0]["test_f1"],
            "best_model/test_loss": test_result[0].get("test_loss", None),
            "best_model/checkpoint_path": trainer.checkpoint_callback.best_model_path
        }
        trainer.logger.experiment.log(log_dict)
        print("\n✓ Best model metrics logged to WandB")

    # Save training progress plots
    # save_training_plots(train_metrics, CHECKPOINT_PATH)
    print(f"\nTraining completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    return model, acc_result, f1_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--cv_split', type=str, default='cv1', help='cv_split')

    parser.add_argument('--model_id', type=str, default='TSLANet_Experiment')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')


    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    
    SEQ_LEN = args.seq_len
    SEED = args.seed
    cv_split = args.cv_split
    dataloader.run_all_cv_preprocessing(SEQ_LEN)



    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs

    # Create run description
    run_description = f"{os.path.basename(args.data_path)}_{SEQ_LEN}_seed{args.seed}_emb{args.emb_dim}_depth{args.depth}___"
    run_description += f"ASB_{args.ASB}__AF_{args.adaptive_filter}__ICB_{args.ICB}__preTr_{args.load_from_pretrained}_"
    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    
    print(f"\nExperiment: {run_description}")
    print(f"Dataset: {DATASET_PATH}")

    CHECKPOINT_PATH = f"/root/Mine_ROI_Net/final_split_outputs_cv/TSLANet/seq_{SEQ_LEN}/{cv_split}/{run_description}"
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load datasets
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Datasets loaded successfully!")

    # Get dataset characteristics
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    # Calculate class weights using sklearn (BETTER METHOD)
    class_counts = torch.bincount(train_loader.dataset.y_data)
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_loader.dataset.y_data.numpy()),
        y=train_loader.dataset.y_data.numpy()
    )
    args.class_weights = torch.FloatTensor(class_weights_np)

    print(f"\nDataset Info:")
    print(f"   Classes: {args.num_classes}")
    print(f"   Sequence Length: {args.seq_len}")
    print(f"   Channels: {args.num_channels}")
    print(f"   Class distribution: {class_counts.tolist()}")
    print(f"   Class weights (sklearn balanced): {args.class_weights.tolist()}")

    # Run training pipeline
    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)

    # Save comprehensive results
    save_experiment_results(args, acc_results, f1_results, CHECKPOINT_PATH)