import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib



################################################ prepare_datasets ################################################

def prepare_datasets(SEQ_LEN=30):
    """Prepare datasets for each country or for any list of CSV files.
    
    Steps:
        1. Load CSV
        2. Encode categorical columns
        3. Build sliding windows per machine
        4. Time-based split (train=first 80%, val=test=last 20%)
        5. Fit scaler on training only
        6. Save train/val/test .pt files + scaler
    
    Args:
        SEQ_LEN: Window length for time-series.
        datasets: Optional list of dicts specifying dataset name/path/output_dir.
                  If None, the default 3-country set is used."""
    
    # Default datasets (3 countries)
    datasets = [
        { 
            "name": "texas",
            "data_path": "/root/MineROI-Net/country_wise_data/final_texas.csv",
            "save_dir": "/root/MineROI-Net/country_wise_data/texas_seq"
        },
        {
            "name": "china",
            "data_path": "/root/MineROI-Net/country_wise_data/final_china.csv",
            "save_dir": "/root/MineROI-Net/country_wise_data/china_seq"
        },
        {
            "name": "ethiopia",
            "data_path": "/root/MineROI-Net/country_wise_data/final_ethiopia.csv",
            "save_dir": "/root/MineROI-Net/country_wise_data/ethiopia_seq"
        }
    ]
    
    # Loop through each dataset
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset['name'].upper()}")
        print(f"{'='*70}")
        
        DATA_PATH = dataset["data_path"]
        SAVE_DIR = dataset["save_dir"]
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Load & sort
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["machine_name", "date"]).reset_index(drop=True)
        print("All_columns =", df.columns.tolist())
        
        target_col = "roi_category_id"
        
        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in ["date", "machine_name"]]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Feature columns
        feature_cols = [c for c in df.columns if c not in ["date", target_col, "machine_name"]]
        print("Used_features_for_model =", feature_cols)
        
        # Build sliding windows per machine
        X_raw, y, last_dates, machines = [], [], [], []
        
        for m, sub in df.groupby("machine_name"):
            sub = sub.reset_index(drop=True)
            for i in range(len(sub) - SEQ_LEN + 1):
                win = sub.iloc[i:i+SEQ_LEN]
                X_raw.append(win[feature_cols].values)
                y.append(int(win.iloc[-1][target_col]))
                last_dates.append(pd.to_datetime(win.iloc[-1]["date"]))
                machines.append(m)
        
        X_raw = np.asarray(X_raw, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        last_dates = pd.to_datetime(pd.Series(last_dates))
        print("Total windows:", X_raw.shape[0])
        
        # Time-based split
        train_cut = last_dates.quantile(0.80)
        train_mask = last_dates <= train_cut
        val_mask = last_dates > train_cut
        test_mask = last_dates > train_cut
        
        # Split data
        X_train_raw = X_raw[train_mask]
        X_val_raw = X_raw[val_mask]
        X_test_raw = X_raw[test_mask]
        
        # Reshape to 2D for scaling
        N_train, L, C = X_train_raw.shape
        N_val = X_val_raw.shape[0]
        N_test = X_test_raw.shape[0]
        
        X_train_2d = X_train_raw.reshape(-1, C)
        X_val_2d = X_val_raw.reshape(-1, C)
        X_test_2d = X_test_raw.reshape(-1, C)
        
        # Fit scaler on training data only
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_val_scaled = scaler.transform(X_val_2d)
        X_test_scaled = scaler.transform(X_test_2d)
        
        # Reshape back to 3D
        X_train_scaled = X_train_scaled.reshape(N_train, L, C)
        X_val_scaled = X_val_scaled.reshape(N_val, L, C)
        X_test_scaled = X_test_scaled.reshape(N_test, L, C)
        
        # Transpose to [N, C, L]
        X_train_final = X_train_scaled.transpose(0, 2, 1)
        X_val_final = X_val_scaled.transpose(0, 2, 1)
        X_test_final = X_test_scaled.transpose(0, 2, 1)
        
        # Pack and save
        def pack(X_data, mask):
            return {
                "samples": torch.from_numpy(X_data),
                "labels": torch.from_numpy(y[mask]),
                "last_dates": last_dates[mask].astype("int64").values,
                "machines": np.array(machines, dtype=object)[mask]
            }
        
        train_dict = pack(X_train_final, train_mask)
        val_dict = pack(X_val_final, val_mask)
        test_dict = pack(X_test_final, test_mask)
        
        torch.save(train_dict, os.path.join(SAVE_DIR, "train.pt"))
        torch.save(val_dict, os.path.join(SAVE_DIR, "val.pt"))
        torch.save(test_dict, os.path.join(SAVE_DIR, "test.pt"))
        joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))
        
        print(f"✓ Saved to {SAVE_DIR}")
        print(f"  Train: {train_mask.sum()} samples")
        print(f"  Val:  {val_mask.sum()} samples")
        print(f"  Test:  {test_mask.sum()} samples")
        print(f"  Train cutoff: {train_cut.date()}")
    
    print(f"\n{'='*70}")
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print(f"{'='*70}")


################################################ combine_datasets ################################################

def combine_datasets(SEQ_LEN=30):
    """Combine multiple country-level preprocessed datasets (train/val/test) into a 
    single unified dataset.

    This function assumes that each country has already been processed by 
    prepare_datasets(), producing:

        country_name_seq/
            ├── train.pt
            ├── val.pt
            ├── test.pt
            └── scaler.joblib

    Steps performed:
        1. Load train/val/test splits from each country.
        2. Concatenate:
            - samples      → torch.float32 tensors [N, C, L]
            - labels       → class labels
            - last_dates   → timestamps for each window
            - machines     → machine identifiers (hashed)
        3. Save the merged train/val/test datasets into:
            country_wise_data/seq_<SEQ_LEN>/

    Args:
        SEQ_LEN (int): Length of the time-series window; determines output directory.

    Outputs:
        Saves three files in /root/MineROI-Net/country_wise_data/seq_<SEQ_LEN>/ :
            - train.pt
            - val.pt
            - test.pt"""
    
    countries = ["china", "ethiopia", "texas"]
    SAVE_DIR = f"/root/MineROI-Net/country_wise_data/seq_{SEQ_LEN}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    def load_split(country, split):
        path = f"/root/MineROI-Net/country_wise_data/{country}_seq/{split}.pt"
        if os.path.exists(path):
            print(f"Loading {path} ...")
            return torch.load(path, weights_only=False)
        else:
            print(f"Skipping {path} (not found)")
            return None
    
    def combine_splits(split):
        all_samples, all_labels, all_dates, all_machines = [], [], [], []
        
        for country in countries:
            data = load_split(country, split)
            if data is None:
                continue
            
            all_samples.append(data["samples"])
            all_labels.append(data["labels"])
            all_dates.append(torch.tensor(data["last_dates"]))
            all_machines.append(torch.tensor([hash(m) for m in data["machines"]]))
        
        combined = {
            "samples": torch.cat(all_samples, dim=0),
            "labels": torch.cat(all_labels, dim=0),
            "last_dates": torch.cat(all_dates, dim=0),
            "machines": torch.cat(all_machines, dim=0)
        }
        print(f"{split} combined → {combined['samples'].shape[0]} samples")
        return combined
    
    train_combined = combine_splits("train")
    val_combined = combine_splits("val")
    test_combined = combine_splits("test")
    
    torch.save(train_combined, os.path.join(SAVE_DIR, "train.pt"))
    torch.save(val_combined, os.path.join(SAVE_DIR, "val.pt"))
    torch.save(test_combined, os.path.join(SAVE_DIR, "test.pt"))
    
    print(f"✓ Combined datasets saved in {SAVE_DIR}")



################################################ convert_to_trans_format ################################################

def convert_to_trans_format(SEQ_LEN=30):
    """Convert preprocessed datasets from channel-first format [B, C, L] 
    to sequence-first format [B, L, C], which is required by Transformer and LSTM-based architectures.

    Input format  (saved by prepare_datasets or combine_datasets):
        samples: [B, C, L]
            B = number of windows
            C = number of features
            L = sequence length (e.g., 30 or 60)

    Output format (saved as <split>_trans.pt):
        samples: [B, L, C]

    Steps performed:
        1. Load 'train.pt', 'val.pt', and 'test.pt' from:
            country_wise_data/seq_<SEQ_LEN>/
        2. Transpose samples from (C, L) → (L, C) for each batch.
        3. Preserve labels, timestamps (last_dates), and machine identifiers.
        4. Save new files:
            train_trans.pt
            val_trans.pt
            test_trans.pt
    
    Args:
        SEQ_LEN (int): Window length; determines which folder to load from."""
    
    BASE_DIR = f"/root/MineROI-Net/country_wise_data/seq_{SEQ_LEN}"
    files = ["train", "val", "test"]
    
    for fname in files:
        src = os.path.join(BASE_DIR, f"{fname}.pt")
        dst = os.path.join(BASE_DIR, f"{fname}_trans.pt")
        
        print(f"\nProcessing: {fname}.pt -> {fname}_trans.pt")
        
        data = torch.load(src, map_location="cpu", weights_only=False)
        X = data["samples"]
        X_trans = X.transpose(1, 2).contiguous()
        
        new_data = {
            "samples": X_trans.float(),
            "labels": data["labels"].type_as(data["labels"]),
            "last_dates": data["last_dates"],
            "machines": data["machines"],
            "meta": {"format": "BLC", "created_from": f"{fname}.pt"}
        }
        
        torch.save(new_data, dst)
        
        chk = torch.load(dst, map_location="cpu", weights_only=False)
        print(f"  ✓ samples: {tuple(chk['samples'].shape)}")
        print(f"  ✓ labels: {tuple(chk['labels'].shape)}")
    
    print("\n✓ All files converted successfully!")


def run_all_preprocessing(SEQ_LEN=30):
    """Run all preprocessing steps"""
    print("Step 1: Preparing individual datasets...")
    prepare_datasets(SEQ_LEN)
    
    print("\nStep 2: Combining datasets...")
    combine_datasets(SEQ_LEN)
    
    print("\nStep 3: Converting to LSTM and Transformer format...")
    convert_to_trans_format(SEQ_LEN)
    
    print("\n" + "="*70)
    print("ALL PREPROCESSING COMPLETE!")
    print("="*70)


# If running this file directly
if __name__ == "__main__":
    run_all_preprocessing(SEQ_LEN=60)