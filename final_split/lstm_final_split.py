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

SEQ_LEN = 60
dataloader.run_all_preprocessing(SEQ_LEN)

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
SAVE_DIR = f"/root/MineROI-Net/country_wise_data/seq_{SEQ_LEN}"
OUT_DIR = f"/root/MineROI-Net/final_split_outputs/LSTM/seq_{SEQ_LEN}"
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
TEST_FILE = "test_trans.pt"  # change to "test.pt" if that's your filename
test_path = os.path.join(SAVE_DIR, TEST_FILE)
test_dict = torch.load(test_path)

test_seqs = test_dict["samples"].numpy()
test_labels = test_dict["labels"].numpy()
print("Test class distribution:")
print(pd.Series(test_labels).value_counts())



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
test_dataset = MiningDataset(test_seqs, test_labels)

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


class HybridLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, num_classes=3):
        super(HybridLSTMModel, self).__init__()
        
        self.spectral = SpectralFeatureExtractor(input_dim)
        self.channel_mix = ChannelMixing(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, seq):
        seq = self.spectral(seq)
        seq = self.channel_mix(seq)
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]
        output = self.classifier(last_hidden)
        return output



# =====================================================================
# Plots
# =====================================================================

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
            probs = torch.softmax(logits, dim=1)  
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
    
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fig, ax = plt.subplots(figsize=(4, 4)) 
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  
    
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
    ax.tick_params(axis='both', labelsize=12)    
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
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
        'hidden_dim': [16],
        'num_layers': [2],
        'dropout': [0.3],
        'wd' : [0.00001],
        'epochs' : [20]
    },
    60: {
        'batch_size': [64],
        'lr': [0.0001],
        'hidden_dim': [16],
        'num_layers': [2],
        'dropout': [0.3],
        'wd' : [0.00001],
        'epochs' : [20]
    }
}

param_grid = param_grids[SEQ_LEN]

# Generate all combinations
combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
print(f"\nTotal experiments for SEQ_LEN={SEQ_LEN}: {len(combinations)}\n")

# =====================================================================
# Run Experiments
# =====================================================================
# SEEDS = [42,1234,60,321,10,500] 
SEEDS = [42] 

results = []

for SEED in SEEDS:
    print(f"\n{'#'*80}")
    print(f"### RUNNING WITH SEED: {SEED} ###")
    print(f"{'#'*80}\n")
    
    set_seed(SEED)  
    
    for idx, params in enumerate(combinations):
        print(f"\n{'='*70}")
        print(f"Seed {SEED} | Experiment {idx+1}/{len(combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*70}\n")
    
        BATCH_SIZE = params['batch_size']
        LR         = params['lr']
        HIDDEN_DIM = params['hidden_dim']
        NUM_LAYERS = params['num_layers']
        DROPOUT    = params['dropout']
        EPOCHS     = params['epochs']
        WEIGHT_DECAY = params['wd']

        run_name = f"seed{SEED}_exp{idx+1}_h{HIDDEN_DIM}_l{NUM_LAYERS}_lr{LR}_bs{BATCH_SIZE}_d{DROPOUT}"
        wandb.init(
            project= f"lstm_{SEQ_LEN}_reproduce",
            config={**params, "weight_decay": WEIGHT_DECAY, "num_classes": NUM_CLASSES},
            name=run_name,
            dir= f"./{SEQ_LEN}_reproduce",
            reinit=True
        )

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = HybridLSTMModel(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        global criterion  
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        best_model_path = os.path.join(OUT_DIR, f"{run_name}_final.pth")

        train_losses, test_losses = [], []
        train_accs,   test_accs   = [], []

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

            # test
            yv, pv, _, test_loss = evaluate_model(model, test_loader, DEVICE)

            test_acc = accuracy_score(yv, pv)

            # Record
            train_losses.append(train_loss); test_losses.append(test_loss)
            train_accs.append(train_acc);    test_accs.append(test_acc)

            # Log epoch metrics
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc":  train_acc,
                "test/loss":   test_loss,
                "test/acc":    test_acc,
                "lr":         scheduler.get_last_lr()[0],
            })

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | test_acc {test_acc:.4f}")

        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved at final epoch {EPOCHS} with train_loss={train_loss:.4f}")

        # =========================
        # Final Evaluation
        # =========================
        yt, pt, probs_t, test_loss = evaluate_model(model, test_loader, DEVICE)
    
        test_acc_final = accuracy_score(yt, pt)
        test_f1_final  = f1_score(yt, pt, average='macro')

        # Calculate AUC (One-vs-Rest for multi-class)
        try:
            test_auc = roc_auc_score(yt, probs_t, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"AUC calculation error: {e}")
            test_auc = None

        print(f"Test AUC: {test_auc:.4f}")

        # Reports
        test_report = classification_report(yt, pt, output_dict=True, digits=4)

        # Confusion matrices
        cm_test = confusion_matrix(yt, pt)
        classes_sorted = list(np.unique(np.concatenate([yv, yt])))

        # Save curves figure
        curves_png = os.path.join(OUT_DIR, f"{run_name}_curves.png")
        plot_curves(train_losses, test_losses, train_accs, test_accs, title_prefix=run_name, out_png=curves_png)

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

        cm_test_png = os.path.join(OUT_DIR, f"{run_name}_cm_test.png")
        save_cm(cm_test, f"{run_name} - Test CM", cm_test_png, class_names=classes_sorted)


        class_names_roc = ['Unprofitable', 'Marginal', 'Profitable']

        roc_test_png = os.path.join(OUT_DIR, f"{run_name}_roc_test.png")

        plot_roc_curve(yt, probs_t, class_names_roc, roc_test_png)

        # W&B logging of artifacts
        wandb.log({
            "final/test/loss": test_loss,
            "final/test/acc":  test_acc_final,
            "final/test/f1":   test_f1_final,
            "final/test/auc":  test_auc,
            "plots/curves":    wandb.Image(curves_png),
            "plots/cm_test":   wandb.Image(cm_test_png),
            "plots/roc_test": wandb.Image(roc_test_png),
        })


        # Save reports locally
        pd.DataFrame(test_report).transpose().to_csv(os.path.join(OUT_DIR, f"{run_name}_test_report.csv"))

        # Record summary row
        results.append({
            "seed": SEED,
            **params,
            "final_train_loss": train_loss,  
            "test_loss":     test_loss,
            "test_acc":      test_acc_final,
            "test_f1":       test_f1_final,
            "test_auc":      test_auc,
            "ckpt_path": best_model_path

        })

        wandb.finish()
        print(f"Done: {run_name} | final_test_loss={test_loss:.4f} | test_acc={test_acc_final:.4f} | test_f1={test_f1_final:.4f}")

results_df = pd.DataFrame(results).sort_values('test_f1', ascending=False)

print("\n" + "="*80)
print("HYPERPARAMETER TUNING RESULTS (Top 10 by test_f1)")

print("="*80)
print(results_df.head(10).to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/hyperparameter_results.csv", index=False)
print(f"\nFull results saved to {OUT_DIR}/hyperparameter_results.csv")




























