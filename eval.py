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
DATA_DIR = "/root/TsLanet/reproduce_transformer/country_wise_data/seq_30"

# Path to the trained weights you want to evaluate
CKPT_PATH = "/root/TsLanet/reproduce_transformer/results_seq_30/seed42_exp1_d64_h2_l2_lr0.0001_bs64_drop0.2_feed256_final.pth"

NUM_CLASSES = 3

# Which file to evaluate on – you can switch to 'train_lstm.pt' or 'val_lstm.pt'
EVAL_FILE = "test_lstm.pt"   # or "test_lstm.pt" if you have it


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
        self.complex_weight = nn.Parameter(
            torch.randn(num_features, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.transpose(1, 2)                           # [B, C, L]
        x_fft = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT over time
        weight = torch.view_as_complex(self.complex_weight)  # [C]
        x_weighted = x_fft * weight.unsqueeze(0).unsqueeze(-1)
        x_out = torch.fft.irfft(x_weighted, n=L, dim=2, norm="ortho")
        return x_out.transpose(1, 2)                    # [B, L, C]


class ChannelMixing(nn.Module):
    def __init__(self, num_features, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features // reduction)
        self.fc2 = nn.Linear(num_features // reduction, num_features)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, L, F]
        identity = x
        x_pooled = x.mean(dim=1)          # [B, F]
        x_weighted = self.fc2(self.act(self.fc1(x_pooled)))  # [B, F]
        out = identity * x_weighted.unsqueeze(1)             # broadcast over L
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HybridTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        num_classes=3,
    ):
        super().__init__()
        self.spectral = SpectralFeatureExtractor(input_dim)
        self.channel_mix = ChannelMixing(input_dim)

        self.input_projection = (
            nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, seq):
        # seq: [B, L, F]
        seq = self.spectral(seq)
        seq = self.channel_mix(seq)
        seq = self.input_projection(seq)
        seq = self.pos_encoder(seq)
        z = self.transformer_encoder(seq)      # [B, L, D]
        pooled = z.mean(dim=1)                 # [B, D]
        out = self.classifier(pooled)          # [B, num_classes]
        return out


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

    # ---------- Build model with SAME hparams as training ----------
    model = HybridTransformerModel(
        input_dim=input_dim,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,         # must match training config
        num_classes=NUM_CLASSES,
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