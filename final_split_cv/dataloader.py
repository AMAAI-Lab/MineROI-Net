# dataloader_cv.py
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

def prepare_cv_datasets(SEQ_LEN=30):
    """Prepare cross-validation datasets for each country"""
    
    # Define datasets
    datasets = [
        {"name": "texas", "path": "/root/MineROI-Net/country_wise_data/final_texas.csv"},
        {"name": "china", "path": "/root/MineROI-Net/country_wise_data/final_china.csv"},
        {"name": "ethiopia", "path": "/root/MineROI-Net/country_wise_data/final_ethiopia.csv"}
    ]
    
    # Define CV splits
    cv_splits = {
        "cv1": {"train_q": 0.50, "val_q": 0.60},
        "cv2": {"train_q": 0.60, "val_q": 0.70},
        "cv3": {"train_q": 0.70, "val_q": 0.80}
    }
    
    # Process each country
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset['name'].upper()}")
        print(f"{'='*70}")
        
        DATA_PATH = dataset["path"]
        
        # Load & sort
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["machine_name", "date"]).reset_index(drop=True)
        
        target_col = "roi_category_id"
        
        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in ["date", "machine_name"]]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Feature columns
        feature_cols = [c for c in df.columns if c not in ["date", target_col, "machine_name"]]
        
        # Build windows per machine (WITHOUT scaling)
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
        print(f"Total windows: {X_raw.shape[0]}")
        
        # Process each CV split
        for cv_name, splits in cv_splits.items():
            print(f"\n  Processing {cv_name}...")
            SAVE_DIR = f"/root/MineROI-Net/country_wise_data/{dataset['name']}/{cv_name}"
            os.makedirs(SAVE_DIR, exist_ok=True)
            
            # Time-based split
            train_cut = last_dates.quantile(splits["train_q"])
            val_cut = last_dates.quantile(splits["val_q"])
            
            train_mask = last_dates <= train_cut
            val_mask = (last_dates > train_cut) & (last_dates <= val_cut)
            test_mask = val_mask 
            
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
            
            print(f"    ✓ {cv_name}: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    print(f"\n{'='*70}")
    print("ALL COUNTRY CV DATASETS PROCESSED!")
    print(f"{'='*70}")


def combine_cv_datasets(SEQ_LEN=30):
    """Combine all countries for each CV fold"""
    
    countries = ["china", "ethiopia", "texas"]
    cv_folds = ["cv1", "cv2", "cv3"]
    
    BASE_DIR = "/root/MineROI-Net/country_wise_data/"
    SAVE_DIR = f"/root/MineROI-Net/country_wise_data/seq_{SEQ_LEN}_cv"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    def load_split(country, cv, split):
        path = os.path.join(BASE_DIR, country, cv, f"{split}.pt")
        if os.path.exists(path):
            print(f"  Loading {country}/{cv}/{split}.pt ...")
            return torch.load(path, weights_only=False)
        else:
            print(f"  Skipping {path} (not found)")
            return None
    
    def combine_splits(split, cv):
        all_samples, all_labels, all_dates, all_machines = [], [], [], []
        
        for country in countries:
            data = load_split(country, cv, split)
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
        print(f"  {split} combined → {combined['samples'].shape[0]} samples")
        return combined
    
    # Process each CV fold
    for cv in cv_folds:
        print(f"\n{'='*70}")
        print(f"Processing {cv.upper()}")
        print(f"{'='*70}")
        
        cv_save_dir = os.path.join(SAVE_DIR, cv)
        os.makedirs(cv_save_dir, exist_ok=True)
        
        train_combined = combine_splits("train", cv)
        val_combined = combine_splits("val", cv)
        test_combined = combine_splits("test", cv)
        
        torch.save(train_combined, os.path.join(cv_save_dir, "train.pt"))
        torch.save(val_combined, os.path.join(cv_save_dir, "val.pt"))
        torch.save(test_combined, os.path.join(cv_save_dir, "test.pt"))
        
        print(f"  ✓ Saved to {cv_save_dir}")
    
    print(f"\n{'='*70}")
    print("ALL CV FOLDS COMBINED!")
    print(f"{'='*70}")


def convert_to_trans_format(SEQ_LEN=30):
    """Convert [B,C,L] to [B,L,C] for all CV folds"""
    
    cv_folds = ["cv1", "cv2", "cv3"]
    splits = ["train", "val", "test"]
    BASE_DIR = f"/root/MineROI-Net/country_wise_data/seq_{SEQ_LEN}_cv"
    
    for cv in cv_folds:
        print(f"\n{'='*70}")
        print(f"Converting {cv.upper()} to trans format")
        print(f"{'='*70}")
        
        for split in splits:
            src = os.path.join(BASE_DIR, cv, f"{split}.pt")
            dst = os.path.join(BASE_DIR, cv, f"{split}_trans.pt")
            
            print(f"  {split}.pt → {split}_trans.pt")
            
            data = torch.load(src, map_location="cpu", weights_only=False)
            X = data["samples"]
            X_trans = X.transpose(1, 2).contiguous()
            
            new_data = {
                "samples": X_trans.float(),
                "labels": data["labels"].type_as(data["labels"]),
                "last_dates": data["last_dates"],
                "machines": data["machines"],
                "meta": {"format": "BLC", "created_from": f"{split}.pt"}
            }
            
            torch.save(new_data, dst)
            
            chk = torch.load(dst, map_location="cpu", weights_only=False)
            print(f"    ✓ samples: {tuple(chk['samples'].shape)}")
    
    print(f"\n{'='*70}")
    print("ALL CV FOLDS CONVERTED TO trans FORMAT!")
    print(f"{'='*70}")


def run_all_cv_preprocessing(SEQ_LEN=30):
    """Run all CV preprocessing steps"""
    print("Step 1: Preparing individual country CV datasets...")
    prepare_cv_datasets(SEQ_LEN)
    
    print("\nStep 2: Combining countries for each CV fold...")
    combine_cv_datasets(SEQ_LEN)
    
    print("\nStep 3: Converting to trans format...")
    convert_to_trans_format(SEQ_LEN)
    
    print("\n" + "="*70)
    print("ALL CV PREPROCESSING COMPLETE!")
    print("="*70)


# If running this file directly
if __name__ == "__main__":
    run_all_cv_preprocessing(SEQ_LEN=60)