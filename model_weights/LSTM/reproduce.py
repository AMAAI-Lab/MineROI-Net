import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================================================
# Config: change these paths for your machine
# =========================================================
SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessed data for a given split (e.g., cv1)
DATA_DIR = "/root/Mine_ROI_Net/country_wise_data/seq_60_cv/cv1/"

# Path to the trained weights you want to evaluate
CKPT_PATH = "/root/Mine_ROI_Net/model_weights/LSTM/seq_60_cv1.pth"

NUM_CLASSES = 3

# Which file to evaluate on – you can switch to 'train_lstm.pt' or 'val_lstm.pt'
EVAL_FILE = "val_trans.pt"   # or "test_lstm.pt" if you have it


# =========================================================
# Dataset wrapper
# =========================================================
class MiningDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =========================================================
# Model definition (same as in training)
# =========================================================
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


# =========================================================
# Evaluation helper
# =========================================================
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for seq, label in loader:
            seq = torch.as_tensor(seq).float().to(device)
            label = torch.as_tensor(label).long().to(device)

            logits = model(seq)
            pred = torch.argmax(logits, dim=1)

            all_true.extend(label.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return all_true, all_pred, acc, f1


# =========================================================
# Main: load data, load weights, run inference
# =========================================================
def main():
    # ---------- Load preprocessed data ----------
    eval_dict = torch.load(os.path.join(DATA_DIR, EVAL_FILE))
    eval_seqs = eval_dict["samples"].numpy()   # [N, L, F]
    eval_labels = eval_dict["labels"].numpy()  # [N]

    print(f"Loaded {EVAL_FILE}: {eval_seqs.shape[0]} samples")

    dataset = MiningDataset(eval_seqs, eval_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    input_dim = eval_seqs.shape[2]
    

    # # Model - seq 30
    # model = HybridLSTMModel(
    #     input_dim=input_dim,
    #     hidden_dim=16,
    #     num_layers=2,
    #     dropout=0.3,
    #     num_classes=NUM_CLASSES
    # ).to(DEVICE)

    # Model - seq 60
    model = HybridLSTMModel(
        input_dim=input_dim,
        hidden_dim=16,
        num_layers=2,
        dropout=0.3,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # ---------- Load trained weights ----------
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from: {CKPT_PATH}")

    # ---------- Evaluate ----------
    y_true, y_pred, acc, f1 = evaluate(model, loader, DEVICE)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
