"""
Feature Engineering and Final Dataset Preparation

This script performs feature engineering on ROI datasets for three regions
(Texas, China, Ethiopia) and prepares the final datasets for modeling.

Steps:
1. Load ROI data for each region
2. Calculate network hashrate and derived features
3. Engineer global time series features (returns, EMAs, volatility, etc.)
4. Create ROI categories for classification
5. Clean and save final datasets

Usage:
    python prepare_final_datasets.py

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np


# Configuration
DATASETS = {
    'texas': {
        'input': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_texas.csv",
        'output': "/root/Mine_ROI_Net/country_wise_data/final_texas.csv"
    },
    'china': {
        'input': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_china.csv",
        'output': "/root/Mine_ROI_Net/country_wise_data/final_china.csv"
    },
    'ethiopia': {
        'input': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_ethiopia.csv",
        'output': "/root/Mine_ROI_Net/country_wise_data/final_ethiopia.csv"
    }
}

DATE_COL = "date"
ID_COL = "machine_name"
TARGET_COL = 'ROI'


def create_roi_categories(
    df: pd.DataFrame,
    target_col: str,
    # bins = (-np.inf, 0.8, 1.0, 1.2, 1.5, np.inf),
    # bins = (-np.inf, 0, 0.8, 1.1, 1.5, np.inf),
    # bins = (-np.inf, 0, 0.8, 1.5, np.inf),
    bins = (-np.inf, 0, 1, np.inf),
    threshold: float = 1.5
):
    """
    Assumes ROI is a multiple (e.g., 1.0 = break-even, 1.5 = 1.5x).
    Creates:
      - roi_category: string labels named by ROI range (e.g., 'roi_[1.2,1.5)x', 'roi_≥1.5x')
      - roi_category_id: ordered codes 0..K-1 (low -> high ROI)
      - take_machine: 1 if ROI >= threshold, else 0
    """
    # Build human-readable labels from bins
    edges = list(bins)
    labels = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        if np.isneginf(a) and np.isposinf(b):
            labels.append("roi_all")
        elif np.isneginf(a):
            labels.append(f"roi_<{b:.2f}x")
        elif np.isposinf(b):
            labels.append(f"roi_≥{a:.2f}x")
        else:
            labels.append(f"roi_[{a:.2f},{b:.2f})x")

    # Bin using left-closed, right-open intervals: [a, b)
    cats = pd.cut(df[target_col], bins=edges, labels=labels, right=False, include_lowest=True, ordered=True)

    # Attach to df
    out = df.copy()
    out["roi_category"] = cats
    out["roi_category_id"] = out["roi_category"].cat.codes  # 0..K-1
    # out["take_machine"] = (out[target_col] >= threshold).astype(int)

    return out


def process_dataset(csv_path, output_path, region_name):
    """Process a single region dataset."""
    print(f"\n{'='*70}")
    print(f"Processing {region_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    print(f"✓ Loaded {len(df)} records")
    
    # Sort by machine and date
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
    print(f"✓ Unique machines: {df['machine_name'].nunique()}")
    
    # Calculate network hashrate and derived features
    print("Calculating network features...")
    df["network_hashrate_hs"] = (df["difficulty"] * (2**32)) / 600
    df["network_hashrate"] = df["network_hashrate_hs"] / 1e12
    df["capex_per_th"] = df["machine_price"] / df["machine_hashrate"]
    df["hashrate_share"] = df["machine_hashrate"] / df["network_hashrate"]
    df["revenue_per_watt"] = df["Revenue_Potential"] / df["power"]
    
    # Global time series features
    print("Engineering global features...")
    GLOBAL_BASE = ["bitcoin_price", "difficulty", "fees", "block_reward"]
    have = [c for c in GLOBAL_BASE if c in df.columns]
    
    print(f"  Initial shape: {df.shape}")
    print(f"  Missing values before engineering: {df.isnull().sum().sum()}")
    
    global_df = (df[[DATE_COL] + have]
                     .drop_duplicates(subset=DATE_COL, keep="first")
                     .sort_values(DATE_COL)
                     .reset_index(drop=True))
    print(f"  Global features shape: {global_df.shape}")
    
    px = global_df["bitcoin_price"].astype(float)
    diff = global_df["difficulty"].astype(float)
    fees = global_df["fees"].astype(float)
    
    # Engineering global features
    global_df["ret_1"] = px.pct_change(1)
    global_df["ret_7"] = px.pct_change(7)
    global_df["ret_30"] = px.pct_change(30)
    global_df["ema_10"] = px.ewm(span=10, adjust=False).mean()
    global_df["ema_30"] = px.ewm(span=30, adjust=False).mean()
    global_df["ema_cross_10_30"] = global_df["ema_10"] - global_df["ema_30"]
    global_df["vol_30"] = global_df["ret_1"].rolling(30).std()
    
    roll_max = px.cummax()
    global_df["drawdown"] = (px / roll_max - 1.0)
    
    global_df["log_diff"] = np.log(diff)
    global_df["d_logdiff_1"] = global_df["log_diff"].diff(1)
    global_df["d_logdiff_7"] = global_df["log_diff"].diff(7)
    global_df["d_logdiff_30"] = global_df["log_diff"].diff(30)
    
    subsidy = global_df["block_reward"] * 6 * 24
    global_df["fee_share"] = fees / (fees + subsidy)
    global_df["fee_share_7"] = global_df["fee_share"].rolling(7).mean()
    global_df["fee_share_30"] = global_df["fee_share"].rolling(30).mean()
    global_df["hashprice_proxy"] = px / diff
    
    global_df["month"] = pd.to_datetime(global_df[DATE_COL]).dt.month
    global_df["month_sin"] = np.sin(2 * np.pi * global_df["month"] / 12)
    global_df["month_cos"] = np.cos(2 * np.pi * global_df["month"] / 12)
    
    engineered_cols = [
        "ret_1","ret_7","ret_30","ema_10","ema_30","ema_cross_10_30",
        "vol_30","drawdown","log_diff","d_logdiff_1","d_logdiff_7","d_logdiff_30",
        "fee_share","fee_share_7","fee_share_30","hashprice_proxy","month_sin","month_cos"
    ]
    
    df = df.merge(global_df[[DATE_COL] + engineered_cols], on=DATE_COL, how="left")
    
    # Clean NaNs
    print("Cleaning NaN values...")
    df = (df.groupby(ID_COL, group_keys=False)
                .apply(lambda g: g.dropna())
                .reset_index(drop=True))
    
    print(f"  Shape after cleaning: {df.shape}")
    print(f"  Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    
    # Create ROI categories
    print("Creating ROI categories...")
    df = create_roi_categories(df, target_col=TARGET_COL)
    
    # Drop columns
    print("Dropping unnecessary columns...")
    df = df.drop(columns=[
           'relesed_date', 
           'machine_available',  'ROI', 'breakeven_days', 'network_hashrate_hs',
           'network_hashrate', 'capex_per_th', 'hashrate_share',
           'revenue_per_watt', 'ret_1', 'ret_7', 'ret_30', 'ema_10', 'ema_30',
           'ema_cross_10_30', 'vol_30', 'drawdown', 'log_diff', 'd_logdiff_1',
           'd_logdiff_7', 'd_logdiff_30', 'fee_share', 'fee_share_7',
           'fee_share_30', 'hashprice_proxy', 'month_sin', 'month_cos',
           'roi_category'], axis=1)
    
    print(f"  Final shape: {df.shape}")
    print(f"  Final columns: {len(df.columns)}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return df


def main():
    """Main function to process all datasets."""
    print("="*70)
    print("FEATURE ENGINEERING AND FINAL DATASET PREPARATION")
    print("="*70)
    
    results = {}
    
    # Process each region
    for region_name, config in DATASETS.items():
        try:
            df = process_dataset(
                csv_path=config['input'],
                output_path=config['output'],
                region_name=region_name
            )
            results[region_name] = df
        except Exception as e:
            print(f"\n✗ Error processing {region_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Regions processed: {len(results)}/{len(DATASETS)}")
    for region_name, df in results.items():
        print(f"\n{region_name.upper()}:")
        print(f"  Total records: {len(df)}")
        print(f"  Unique machines: {df['machine_name'].nunique()}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  ROI category distribution:")
        print(f"    {df['roi_category_id'].value_counts().sort_index().to_dict()}")
        print(f"  Output: {DATASETS[region_name]['output']}")
    print("="*70)


if __name__ == "__main__":
    main()