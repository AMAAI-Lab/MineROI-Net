import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from typing import Dict, List, Any


################################################ helpers ################################################
def _print_split_stats(
    y: np.ndarray,
    machines: np.ndarray,
    mask: np.ndarray,
    split_name: str,
    scenario: str,
    save_dir: str,
    all_classes: np.ndarray = None,
):
    """
    Prints:
      1) class counts for this split
      2) machine counts for this split
      3) machine x class crosstab
    Also saves CSV files in save_dir/stats/
    """
    y_split = y[mask]
    m_split = machines[mask]

    if all_classes is None:
        all_classes = np.sort(np.unique(y))

    stats_dir = os.path.join(save_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # 1) class counts
    class_counts = (
        pd.Series(y_split, name="roi_category_id")
        .value_counts()
        .reindex(all_classes, fill_value=0)
        .sort_index()
    )
    class_pct = (class_counts / max(len(y_split), 1) * 100).round(2)

    print(f"\n[{scenario}] {split_name.upper()} - Class counts")
    print(class_counts.to_string())
    print(f"[{scenario}] {split_name.upper()} - Class %")
    print(class_pct.to_string())

    pd.DataFrame({
        "roi_category_id": class_counts.index,
        "count": class_counts.values,
        "percent": class_pct.values
    }).to_csv(
        os.path.join(stats_dir, f"{split_name}_class_counts.csv"),
        index=False
    )

    # 2) machine counts (total samples per machine)
    machine_counts = (
        pd.Series(m_split, name="machine_name")
        .value_counts()
        .sort_index()
    )
    print(f"\n[{scenario}] {split_name.upper()} - Counts by machine")
    print(machine_counts.to_string())

    machine_counts.rename("count").to_csv(
        os.path.join(stats_dir, f"{split_name}_machine_counts.csv")
    )

    # 3) machine x class table
    machine_class = pd.crosstab(
        pd.Series(m_split, name="machine_name"),
        pd.Series(y_split, name="roi_category_id")
    ).reindex(columns=all_classes, fill_value=0).sort_index()

    print(f"\n[{scenario}] {split_name.upper()} - Machine x Class counts")
    print(machine_class.to_string())

    machine_class.to_csv(
        os.path.join(stats_dir, f"{split_name}_machine_class_counts.csv")
    )


def _print_combined_stats(combined: dict, split_name: str, save_dir: str):
    y = combined["labels"].cpu().numpy() if torch.is_tensor(combined["labels"]) else np.asarray(combined["labels"])
    m = np.asarray(combined["machines"])

    all_classes = np.sort(np.unique(y))
    stats_dir = os.path.join(save_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    class_counts = pd.Series(y, name="roi_category_id").value_counts().reindex(all_classes, fill_value=0).sort_index()
    machine_counts = pd.Series(m, name="machine_name").value_counts().sort_index()
    machine_class = pd.crosstab(pd.Series(m, name="machine_name"), pd.Series(y, name="roi_category_id")).reindex(columns=all_classes, fill_value=0).sort_index()

    print(f"\n[COMBINED] {split_name.upper()} class counts:\n{class_counts.to_string()}")
    print(f"\n[COMBINED] {split_name.upper()} machine counts:\n{machine_counts.to_string()}")
    print(f"\n[COMBINED] {split_name.upper()} machine x class:\n{machine_class.to_string()}")

    class_counts.rename("count").to_csv(os.path.join(stats_dir, f"{split_name}_class_counts.csv"))
    machine_counts.rename("count").to_csv(os.path.join(stats_dir, f"{split_name}_machine_counts.csv"))
    machine_class.to_csv(os.path.join(stats_dir, f"{split_name}_machine_class_counts.csv"))


def _discover_datasets(base_final_dir: str, base_seq_dir: str) -> List[Dict[str, str]]:
    datasets = []
    for fname in sorted(os.listdir(base_final_dir)):
        if fname.startswith("final_elec_") and fname.endswith(".csv"):
            scenario = fname.replace("final_", "").replace(".csv", "")  # elec_0.1
            datasets.append({
                "name": scenario,
                "data_path": os.path.join(base_final_dir, fname),
                "save_dir": os.path.join(base_seq_dir, scenario),
            })
    return datasets


def _build_windows(df: pd.DataFrame, seq_len: int, target_col: str, feature_cols: List[str]):
    """
    Build sliding windows per machine, preserving temporal order.
    Output:
      X_raw: [N, L, C]
      y: [N]
      last_dates: pd.Series[N]
      machines: np.array[N]
    """
    X_raw, y, last_dates, machines = [], [], [], []

    for m, sub in df.groupby("machine_name"):
        sub = sub.sort_values("date").reset_index(drop=True)
        if len(sub) < seq_len:
            continue

        for i in range(len(sub) - seq_len + 1):
            win = sub.iloc[i:i + seq_len]
            X_raw.append(win[feature_cols].to_numpy(dtype=np.float32))
            y.append(int(win.iloc[-1][target_col]))
            last_dates.append(pd.to_datetime(win.iloc[-1]["date"]))
            machines.append(m)

    if len(X_raw) == 0:
        return None

    X_raw = np.asarray(X_raw, dtype=np.float32)  # [N, L, C]
    y = np.asarray(y, dtype=np.int64)
    last_dates = pd.to_datetime(pd.Series(last_dates))
    machines = np.asarray(machines, dtype=object)

    return X_raw, y, last_dates, machines


################################################ prepare_datasets ################################################

def prepare_datasets(
    SEQ_LEN: int = 60,
    scaler_mode: str = "global",      # "global" or "per_scenario"
    val_equals_test: bool = True      # keep your convenience behavior by default
):
    """
    Prepare datasets per electricity scenario, with configurable scaling strategy.

    scaler_mode:
      - "global": one scaler fitted on combined TRAIN rows from all scenarios (recommended for one combined model)
      - "per_scenario": one scaler per scenario (your previous behavior)

    val_equals_test:
      - True: val and test are identical (your current convenience setup)
      - False: uses 80/10/10 by time quantiles
    """

    BASE_FINAL_DIR = "/root/Mine_ROI_Net/version_2/data_collection/final"
    BASE_SEQ_DIR   = "/root/Mine_ROI_Net/version_2/data_collection/seq_60"
    os.makedirs(BASE_SEQ_DIR, exist_ok=True)

    assert scaler_mode in {"global", "per_scenario"}, "scaler_mode must be 'global' or 'per_scenario'"

    datasets = _discover_datasets(BASE_FINAL_DIR, BASE_SEQ_DIR)
    if len(datasets) == 0:
        print(f"No files found in {BASE_FINAL_DIR} matching final_elec_*.csv")
        return

    # 1) Load all scenario data first
    dfs: Dict[str, pd.DataFrame] = {}
    for ds in datasets:
        df = pd.read_csv(ds["data_path"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["machine_name", "date"]).reset_index(drop=True)
        dfs[ds["name"]] = df

    # 2) Global categorical encoding (consistent codes across scenarios)
    all_cat_cols = set()
    for df in dfs.values():
        cols = df.select_dtypes(include=["object"]).columns.tolist()
        cols = [c for c in cols if c not in ["date", "machine_name"]]
        all_cat_cols.update(cols)

    for col in sorted(all_cat_cols):
        pool = []
        for _, df in dfs.items():
            if col in df.columns:
                pool.append(df[col].astype(str))
        if len(pool) == 0:
            continue

        enc = LabelEncoder().fit(pd.concat(pool, axis=0).values)

        for name, df in dfs.items():
            if col in df.columns:
                dfs[name][col] = enc.transform(df[col].astype(str))

    # 3) Build windows + split masks per scenario
    scenario_cache: Dict[str, Dict[str, Any]] = {}
    all_train_rows_2d = []  # for global scaler

    for ds in datasets:
        scenario = ds["name"]
        save_dir = ds["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        df = dfs[scenario]
        target_col = "roi_category_id"
        feature_cols = [c for c in df.columns if c not in ["date", target_col, "machine_name"]]

        built = _build_windows(df, SEQ_LEN, target_col, feature_cols)
        if built is None:
            print(f"[WARN] {scenario}: no windows created (data shorter than SEQ_LEN={SEQ_LEN}). Skipping.")
            continue

        X_raw, y, last_dates, machines = built
        n_total = len(y)

        # Time-based split
        train_cut = last_dates.quantile(0.80)
        train_mask = (last_dates <= train_cut).to_numpy()

        if val_equals_test:
            val_mask = (last_dates > train_cut).to_numpy()
            test_mask = (last_dates > train_cut).to_numpy()
        else:
            val_cut = last_dates.quantile(0.90)
            val_mask = ((last_dates > train_cut) & (last_dates <= val_cut)).to_numpy()
            test_mask = (last_dates > val_cut).to_numpy()

        X_train_raw = X_raw[train_mask]
        X_val_raw = X_raw[val_mask]
        X_test_raw = X_raw[test_mask]

        if X_train_raw.shape[0] == 0:
            print(f"[WARN] {scenario}: empty training split after cutoff. Skipping.")
            continue

        if scaler_mode == "global":
            _, _, C = X_train_raw.shape
            all_train_rows_2d.append(X_train_raw.reshape(-1, C))

        scenario_cache[scenario] = {
            "save_dir": save_dir,
            "feature_cols": feature_cols,
            "X_raw": X_raw,
            "y": y,
            "last_dates": last_dates,
            "machines": machines,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "train_cut": train_cut,
            "n_total": n_total,
        }


        all_classes = np.sort(np.unique(y))

        _print_split_stats(
            y=y,
            machines=machines,
            mask=train_mask,
            split_name="train",
            scenario=scenario,
            save_dir=save_dir,
            all_classes=all_classes
        )

        _print_split_stats(
            y=y,
            machines=machines,
            mask=test_mask,
            split_name="test",
            scenario=scenario,
            save_dir=save_dir,
            all_classes=all_classes
        )


        print(f"\n{'='*70}")
        print(f"Processing: {scenario.upper()}")
        print(f"All_columns = {df.columns.tolist()}")
        print(f"Used_features_for_model = {feature_cols}")
        print(f"Total windows: {n_total}")
        print(f"Train cutoff: {train_cut.date()}")
        print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    if len(scenario_cache) == 0:
        print("No valid scenarios to process.")
        return

    # 4) Fit scaler(s)
    global_scaler = None
    if scaler_mode == "global":
        stacked = np.vstack(all_train_rows_2d)  # [sum(N_train*L), C]
        global_scaler = MinMaxScaler()
        global_scaler.fit(stacked)
        joblib.dump(global_scaler, os.path.join(BASE_SEQ_DIR, "global_scaler.joblib"))
        print(f"\n[Scaler] Global scaler fitted on combined training rows: {stacked.shape}")

    # 5) Scale + save each scenario
    def _pack(X_data_ncl, y_arr, last_dates_ser, machines_arr, mask_arr, scenario_name: str):
        return {
            "samples": torch.from_numpy(X_data_ncl).float(),  # [N, C, L]
            "labels": torch.from_numpy(y_arr[mask_arr]).long(),
            "last_dates": last_dates_ser[mask_arr].astype("int64").to_numpy(),
            "machines": machines_arr[mask_arr],
            "scenario": np.array([scenario_name] * int(mask_arr.sum()), dtype=object),
        }

    for scenario, d in scenario_cache.items():
        X_raw = d["X_raw"]
        y = d["y"]
        last_dates = d["last_dates"]
        machines = d["machines"]

        train_mask = d["train_mask"]
        val_mask = d["val_mask"]
        test_mask = d["test_mask"]

        X_train_raw = X_raw[train_mask]
        X_val_raw = X_raw[val_mask]
        X_test_raw = X_raw[test_mask]

        N_train, L, C = X_train_raw.shape
        N_val = X_val_raw.shape[0]
        N_test = X_test_raw.shape[0]

        X_train_2d = X_train_raw.reshape(-1, C)
        X_val_2d = X_val_raw.reshape(-1, C)
        X_test_2d = X_test_raw.reshape(-1, C)

        if scaler_mode == "per_scenario":
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_2d)
            X_val_scaled = scaler.transform(X_val_2d)
            X_test_scaled = scaler.transform(X_test_2d)
        else:
            scaler = global_scaler
            X_train_scaled = scaler.transform(X_train_2d)
            X_val_scaled = scaler.transform(X_val_2d)
            X_test_scaled = scaler.transform(X_test_2d)

        # reshape back
        X_train_scaled = X_train_scaled.reshape(N_train, L, C)
        X_val_scaled = X_val_scaled.reshape(N_val, L, C)
        X_test_scaled = X_test_scaled.reshape(N_test, L, C)

        # [N, L, C] -> [N, C, L] for your existing pipeline
        X_train_final = X_train_scaled.transpose(0, 2, 1)
        X_val_final = X_val_scaled.transpose(0, 2, 1)
        X_test_final = X_test_scaled.transpose(0, 2, 1)

        train_dict = _pack(X_train_final, y, last_dates, machines, train_mask, scenario)
        val_dict = _pack(X_val_final, y, last_dates, machines, val_mask, scenario)
        test_dict = _pack(X_test_final, y, last_dates, machines, test_mask, scenario)

        save_dir = d["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        torch.save(train_dict, os.path.join(save_dir, "train.pt"))
        torch.save(val_dict, os.path.join(save_dir, "val.pt"))
        torch.save(test_dict, os.path.join(save_dir, "test.pt"))
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

        print(f"✓ Saved scenario tensors to {save_dir}")

    print(f"\n{'='*70}")
    print(f"ALL DATASETS PROCESSED SUCCESSFULLY! (scaler_mode={scaler_mode})")
    print(f"{'='*70}")


################################################ combine_datasets ################################################

def combine_datasets(SEQ_LEN=60):
    """
    Combine multiple scenario preprocessed datasets into one unified dataset.
    """

    BASE_SEQ_DIR = "/root/Mine_ROI_Net/version_2/data_collection/seq_60"
    SAVE_DIR = f"/root/Mine_ROI_Net/version_2/data_collection/seq_combined_{SEQ_LEN}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    scenarios = sorted([
        d for d in os.listdir(BASE_SEQ_DIR)
        if d.startswith("elec_") and os.path.isdir(os.path.join(BASE_SEQ_DIR, d))
    ])

    print(f"Found {len(scenarios)} scenarios in {BASE_SEQ_DIR}")
    if len(scenarios) == 0:
        print("No scenario folders found (expected folders like elec_0.1, elec_0.2, ...).")
        return

    def load_split(scenario, split):
        path = os.path.join(BASE_SEQ_DIR, scenario, f"{split}.pt")
        if os.path.exists(path):
            print(f"Loading {path} ...")
            return torch.load(path, weights_only=False)
        else:
            print(f"Skipping {path} (not found)")
            return None

    def combine_splits(split):
        all_samples, all_labels, all_dates, all_machines, all_scenarios = [], [], [], [], []

        for scenario in scenarios:
            data = load_split(scenario, split)
            if data is None:
                continue

            all_samples.append(data["samples"])
            all_labels.append(data["labels"])
            all_dates.append(torch.as_tensor(data["last_dates"]))
            all_machines.append(data["machines"])

            if "scenario" in data:
                all_scenarios.append(data["scenario"])
            else:
                n = data["samples"].shape[0]
                all_scenarios.append(np.array([scenario] * n, dtype=object))

        if len(all_samples) == 0:
            return None

        combined = {
            "samples": torch.cat(all_samples, dim=0),   # [N, C, L]
            "labels": torch.cat(all_labels, dim=0),
            "last_dates": torch.cat(all_dates, dim=0),
            "machines": np.concatenate(all_machines),
            "scenarios": np.concatenate(all_scenarios),
        }
        print(f"{split} combined → {combined['samples'].shape[0]} samples")
        return combined

    train_combined = combine_splits("train")
    val_combined = combine_splits("val")
    test_combined = combine_splits("test")

    if train_combined is None or val_combined is None or test_combined is None:
        print("Failed to combine one or more splits.")
        return

    # ✅ Print combined stats AFTER split objects exist
    _print_combined_stats(train_combined, "train", SAVE_DIR)
    _print_combined_stats(test_combined, "test", SAVE_DIR)
    # optional
    # _print_combined_stats(val_combined, "val", SAVE_DIR)

    torch.save(train_combined, os.path.join(SAVE_DIR, "train.pt"))
    torch.save(val_combined, os.path.join(SAVE_DIR, "val.pt"))
    torch.save(test_combined, os.path.join(SAVE_DIR, "test.pt"))

    # If global scaler exists, copy it into combined folder
    global_scaler_path = os.path.join(BASE_SEQ_DIR, "global_scaler.joblib")
    if os.path.exists(global_scaler_path):
        scaler = joblib.load(global_scaler_path)
        joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))
        print(f"✓ Global scaler copied to {os.path.join(SAVE_DIR, 'scaler.joblib')}")

    print(f"✓ Combined datasets saved in {SAVE_DIR}")



################################################ convert_to_trans_format ################################################

def convert_to_trans_format(SEQ_LEN=60):
    """
    Convert [B, C, L] -> [B, L, C] for Transformer/LSTM style input.
    """
    BASE_DIR = f"/root/Mine_ROI_Net/version_2/data_collection/seq_combined_{SEQ_LEN}"
    files = ["train", "val", "test"]

    for fname in files:
        src = os.path.join(BASE_DIR, f"{fname}.pt")
        dst = os.path.join(BASE_DIR, f"{fname}_trans.pt")

        print(f"\nProcessing: {fname}.pt -> {fname}_trans.pt")
        data = torch.load(src, map_location="cpu", weights_only=False)

        X = data["samples"]                         # [B, C, L]
        X_trans = X.transpose(1, 2).contiguous()   # [B, L, C]

        new_data = {
            "samples": X_trans.float(),
            "labels": data["labels"].type_as(data["labels"]),
            "last_dates": data["last_dates"],
            "machines": data["machines"],
            "scenarios": data.get("scenarios", None),
            "meta": {"format": "BLC", "created_from": f"{fname}.pt"}
        }

        torch.save(new_data, dst)

        chk = torch.load(dst, map_location="cpu", weights_only=False)
        print(f"  ✓ samples: {tuple(chk['samples'].shape)}")
        print(f"  ✓ labels: {tuple(chk['labels'].shape)}")

    print("\n✓ All files converted successfully!")


################################################ run all ################################################

def run_all_preprocessing(
    SEQ_LEN=60,
    scaler_mode="global",   # "global" recommended for one combined model
    val_equals_test=True    # keep your current behavior
):
    print("Step 1: Preparing individual datasets...")
    prepare_datasets(
        SEQ_LEN=SEQ_LEN,
        scaler_mode=scaler_mode,
        val_equals_test=val_equals_test
    )

    print("\nStep 2: Combining datasets...")
    combine_datasets(SEQ_LEN=SEQ_LEN)

    print("\nStep 3: Converting to LSTM/Transformer format...")
    convert_to_trans_format(SEQ_LEN=SEQ_LEN)

    print("\n" + "=" * 70)
    print("ALL PREPROCESSING COMPLETE!")
    print("=" * 70)






if __name__ == "__main__":
    run_all_preprocessing(
        SEQ_LEN=60,
        scaler_mode="global",   # change to "per_scenario" if needed
        val_equals_test=True
    )
