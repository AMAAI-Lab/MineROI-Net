# transformer_final_split.py
import dataloader
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import random
import pandas as pd
import itertools
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# =======================
# Run the preprocessing dataloader to prepare data
# =======================
CV_SPLIT = "cv3"  # Change as needed: cv1, cv2, cv3
SEQ_LEN = 60
dataloader.run_all_cv_preprocessing(SEQ_LEN)

# =======================
# Set Seed for Reproducibility
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Config
# =====================================================================
SAVE_DIR = f"/root/Mine_ROI_Net/country_wise_data/seq_{SEQ_LEN}_cv/{CV_SPLIT}"
OUT_DIR = f"/root/Mine_ROI_Net/final_split_outputs_cv/Mine_ROI_Net/seq_{SEQ_LEN}/{CV_SPLIT}"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Load preprocessed .pt files
# =====================================================================
print("Loading preprocessed data...")
train_dict = torch.load(os.path.join(SAVE_DIR, "train_trans.pt"))

train_seqs = train_dict["samples"].numpy()
train_labels = train_dict["labels"].numpy()


print("\nTrain class distribution:")
print(pd.Series(train_labels).value_counts())



# =====================================================================
# Load test .pt (adjust TEST_FILE if yours is named test.pt)
# =====================================================================
val_FILE = "val_trans.pt"  # change to "val.pt" if that's your filename
val_path = os.path.join(SAVE_DIR, val_FILE)
val_dict = torch.load(val_path)

val_seqs = val_dict["samples"].numpy()
val_labels = val_dict["labels"].numpy()
print("val class distribution:")
print(pd.Series(val_labels).value_counts())



# =====================================================================
# Dataset Class
# =====================================================================
class MiningDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create datasets
train_dataset = MiningDataset(train_seqs, train_labels)
val_dataset = MiningDataset(val_seqs, val_labels)

# Compute class weights (once)
input_dim = train_seqs.shape[2]
classes = np.unique(train_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print("Class weights:", class_weights)

# =====================================================================
# Model Classes
# =====================================================================
class SpectralFeatureExtractor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(num_features, 2, dtype=torch.float32) * 0.02)
        
    def forward(self, x):
        B, L, C = x.shape
        x = x.transpose(1, 2)
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight.unsqueeze(0).unsqueeze(-1)
        x_out = torch.fft.irfft(x_weighted, n=L, dim=2, norm='ortho')
        return x_out.transpose(1, 2)


class ChannelMixing(nn.Module):
    def __init__(self, num_features, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features // reduction)
        self.fc2 = nn.Linear(num_features // reduction, num_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x_pooled = x.mean(dim=1)
        x_weighted = self.fc2(self.act(self.fc1(x_pooled)))
        out = identity * x_weighted.unsqueeze(1)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.3, num_classes=3):
        super(HybridTransformerModel, self).__init__()
        
        # Keep your spectral and channel mixing features
        self.spectral = SpectralFeatureExtractor(input_dim)
        self.channel_mix = ChannelMixing(input_dim)
        
        # Project input to d_model dimensions if needed
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN,  dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Important: matches your data format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, seq):
        # Feature extraction
        seq = self.spectral(seq)
        seq = self.channel_mix(seq)
        
        # Project to model dimension
        seq = self.input_projection(seq)
        
        # Add positional encoding
        seq = self.pos_encoder(seq)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(seq)
        
        # Global average pooling or use [CLS] token approach
        # Option 1: Mean pooling
        pooled = transformer_out.mean(dim=1)
        
        # Option 2: Use last token (uncomment to use instead)
        # pooled = transformer_out[:, -1, :]
        
        # Classification
        output = self.classifier(pooled)
        return output
    

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    all_loss = 0.0
    with torch.no_grad():
        for seq, label in loader:
            seq = torch.as_tensor(seq).float().to(device)
            label = torch.as_tensor(label).long().to(device)
            logits = model(seq)
            loss = criterion(logits, label)
            all_loss += loss.item()
            probs = torch.softmax(logits, dim=1)  # Get probabilities
            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = all_loss / len(loader)
    return np.array(all_true), np.array(all_preds), np.array(all_probs), avg_loss

def plot_curves(train_losses, test_losses, train_accs, test_accs, title_prefix, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='test Loss')
    ax1.set_title(f"{title_prefix} - Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(test_accs, label='test Acc')
    ax2.set_title(f"{title_prefix} - Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)




def plot_roc_curve(y_true, y_probs, class_names, out_png):
    """Plot ROC curve for multi-class classification"""
    n_classes = len(class_names)
    
    # Binarize the labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fig, ax = plt.subplots(figsize=(4, 4))  # Same size as confusion matrix
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=15, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=15, fontweight='bold', labelpad=8)
    
    # Match tick label style with confusion matrix
    ax.tick_params(axis='both', labelsize=12)
    
    # Compact legend inside plot
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    
    # Match tight layout with confusion matrix
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    
# =====================================================================
# Hyperparameter Grid
# =====================================================================
param_grids = {
    30: {
        'batch_size': [64],
        'lr': [0.0001],
        'd_model': [64],
        'nhead': [2],
        'num_layers': [2],
        'dim_feedforward': [256],
        'dropout': [0.2],
        'wd': [0.00001],
        'epochs': [20]
    },
    60: {
        'batch_size': [64],
        'lr': [0.0001],
        'd_model': [64],           # Replaces hidden_dim
        'nhead': [4],                 # Number of attention heads
        'num_layers': [2],
        'dim_feedforward': [256], # FFN hidden dimension
        'dropout': [0.2],
        'wd': [0.00001],
        'epochs': [20]
    }
}

param_grid = param_grids[SEQ_LEN]

# Generate all combinations
combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
print(f"\nTotal experiments for SEQ_LEN={SEQ_LEN}: {len(combinations)}\n")

# =====================================================================
# Run Experiments
# =====================================================================
# =====================================================================
# Run Experiments: save best-by-val-loss, then test it, log everything
# =====================================================================

SEEDS = [42] 


results = []

for SEED in SEEDS:
    print(f"\n{'#'*80}")
    print(f"### RUNNING WITH SEED: {SEED} ###")
    print(f"{'#'*80}\n")
    
    set_seed(SEED)  # Set the seed
    
    for idx, params in enumerate(combinations):
        print(f"\n{'='*70}")
        print(f"Seed {SEED} | Experiment {idx+1}/{len(combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*70}\n")
    
    # Hyperparams
        BATCH_SIZE = params['batch_size']
        LR         = params['lr']
        EPOCHS     = params['epochs']
        WEIGHT_DECAY = params['wd']

        # W&B run - ADD SEED TO NAME
        run_name = f"seed{SEED}_exp{idx+1}_d{params['d_model']}_h{params['nhead']}_l{params['num_layers']}_lr{params['lr']}_bs{params['batch_size']}_drop{params['dropout']}_feed{params['dim_feedforward']}"

  
        wandb.init(
            project=f"transformer_{SEQ_LEN}_reproduce_seed",
            config={**params, "weight_decay": WEIGHT_DECAY, "num_classes": NUM_CLASSES},
            name=run_name,
            dir= f"./{SEQ_LEN}_reproduce_seed",  # specify wandb directory if needed
            reinit=True
        )

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = HybridTransformerModel(
            input_dim=input_dim,
            d_model=params['d_model'],          # Changed from hidden_dim
            nhead=params['nhead'],              # New parameter
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],  # New parameter
            dropout=params['dropout'],
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        # Loss/Opt/Sched
    global criterion  # so evaluate_model can use it
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Tracking
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUT_DIR, f"{run_name}_best_val.pth")

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # Train
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        preds_tr, true_tr = [], []

        for seq, label in train_loader:
            seq = torch.as_tensor(seq).float().to(DEVICE)
            label = torch.as_tensor(label).long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss.item())
            preds_tr.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            true_tr.extend(label.detach().cpu().numpy())

        train_loss = float(np.mean(batch_losses))
        train_acc  = accuracy_score(true_tr, preds_tr)

        # val
        yv, pv, _, val_loss = evaluate_model(model, val_loader, DEVICE)

        val_acc = accuracy_score(yv, pv)

        # Record
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc)

        # Save best-by-train-loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss={val_loss:.4f}")

        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
        })

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    # torch.save(model.state_dict(), best_model_path)
    # print(f"Model saved at final epoch {EPOCHS} with train_loss={train_loss:.4f}")

    # =========================
    # # After training: reload best and EVALUATE (val & val)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)

    yt, pt, probs_t, val_loss = evaluate_model(model, val_loader, DEVICE)

   
    val_acc_final = accuracy_score(yt, pt)
    val_f1_final  = f1_score(yt, pt, average='macro')

    # Calculate AUC (One-vs-Rest for multi-class)
    try:
        val_auc = roc_auc_score(yt, probs_t, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        val_auc = None

    print(f"val AUC: {val_auc:.4f}")

    # Reports
    val_report = classification_report(yt, pt, output_dict=True, digits=4)

    # Confusion matrices
    cm_val = confusion_matrix(yt, pt)
    classes_sorted = list(np.unique(np.concatenate([yv, yt])))

    # Save curves figure
    curves_png = os.path.join(OUT_DIR, f"{run_name}_curves.png")
    plot_curves(train_losses, val_losses, train_accs, val_accs, title_prefix=run_name, out_png=curves_png)

    # Save & log confusion matrices as images
    def save_cm(cm, title, out_png, class_names):
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, interpolation='nearest')
        plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks); ax.set_xticklabels(class_names)
        ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close(fig)

    cm_val_png = os.path.join(OUT_DIR, f"{run_name}_cm_val.png")
    save_cm(cm_val, f"{run_name} - val CM", cm_val_png, class_names=classes_sorted)


    class_names_roc = ['Unprofitable', 'Marginal', 'Profitable']

    roc_val_png = os.path.join(OUT_DIR, f"{run_name}_roc_val.png")

    plot_roc_curve(yt, probs_t, class_names_roc, roc_val_png)

    # W&B logging of artifacts
    wandb.log({
        "final/val/loss": val_loss,
        "final/val/acc":  val_acc_final,
        "final/val/f1":   val_f1_final,
        "final/val/auc":  val_auc,
        "plots/curves":    wandb.Image(curves_png),
        "plots/cm_val":   wandb.Image(cm_val_png),
        "plots/roc_val": wandb.Image(roc_val_png),
    })


    # Save reports locally
    pd.DataFrame(val_report).transpose().to_csv(os.path.join(OUT_DIR, f"{run_name}_val_report.csv"))

    # Record summary row
    results.append({
        **params,
        "final_train_loss": train_loss,  
        "val_loss":     val_loss,
        "val_acc":      val_acc_final,
        "val_f1":       val_f1_final,
        "val_auc":      val_auc,
        "ckpt_path": best_model_path

    })

    wandb.finish()
    print(f"Done: {run_name} | final_val_loss={val_loss:.4f} | val_acc={val_acc_final:.4f} | val_f1={val_f1_final:.4f}")

results_df = pd.DataFrame(results).sort_values('val_f1', ascending=False)

print("\n" + "="*80)
print("HYPERPARAMETER TUNING RESULTS (Top 10 by val_f1)")

print("="*80)
print(results_df.head(10).to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/hyperparameter_results.csv", index=False)
print(f"\nFull results saved to {OUT_DIR}/hyperparameter_results.csv")




























