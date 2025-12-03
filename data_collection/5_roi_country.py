"""
Calculate ROI and Breakeven Days for Multiple Regions

This script calculates Return on Investment (ROI) and breakeven days for Bitcoin mining
across three geographical regions: Texas, Ethiopia, and China.

Data Sources (Section 4.1 of the paper):
-----------------------------------------
1. China: Monthly electricity prices
2. Ethiopia: Monthly electricity prices
3. Texas: Monthly electricity prices from Kaggle dataset

Preprocessing:
--------------
Monthly data is repeated for each day of the month to create daily time series.
This method is applied uniformly across all three regions to ensure consistency
with other daily features in the dataset.

Usage:
    python calculate_roi_all_regions.py

Requirements:
    pip install pandas numpy tqdm
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration
FEATURES_PATH = "/root/Mine_ROI_Net/data_collection/full_feature_data.csv"

ELECTRICITY_DATA = {
    'texas': {
        'path': "/root/Mine_ROI_Net/data_collection/electricity_data/texas_residential_daily_df.csv",
        'fillna_value': 0.1549,
        'output': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_texas.csv"
    },
    'ethiopia': {
        'path': "/root/Mine_ROI_Net/data_collection/electricity_data/ethiopia_electricity_prices_daily.csv",
        'fillna_value': 0.01,
        'output': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_ethiopia.csv"
    },
    'china': {
        'path': "/root/Mine_ROI_Net/data_collection/electricity_data/china_electricity_prices_daily.csv",
        'fillna_value': 0.08,
        'output': "/root/Mine_ROI_Net/data_collection/electricity_data/roi_china.csv"
    }
}


def calculate_roi_and_breakeven(row, full_df):
    start_date = row['date']
    machine_name = row['machine_name']
    machine_cost = row['machine_price']
    hashrate = row['machine_hashrate'] * 1e12  # TH/s to H/s
    power = row['power']

    end_date = start_date + pd.DateOffset(months=12)
    future_df = full_df[(full_df['date'] >= start_date) &
                        (full_df['date'] <= end_date) &
                        (full_df['machine_name'] == machine_name)].copy()

    if len(future_df) < 360:
        return pd.Series({'ROI': np.nan, 'breakeven_days': np.nan})

    total_profit = 0
    cumulative_profit = 0
    breakeven_days = None

    for i, (_, day_data) in enumerate(future_df.iterrows()):
        block_reward = day_data['block_reward']
        btc_price = day_data['bitcoin_price']
        difficulty = day_data['difficulty']
        electricity_rate = day_data['electricity_rate']
        fees = day_data['fees']

        btc_mined = ((hashrate * 86400) / (difficulty * (2**32))) * (block_reward + (fees/144))
        revenue = btc_mined * btc_price
        electricity_cost = (power * 24 / 1000) * electricity_rate
        daily_profit = revenue - electricity_cost

        total_profit += daily_profit
        cumulative_profit += daily_profit

        if breakeven_days is None and cumulative_profit >= machine_cost:
            breakeven_days = i + 1  # 1-based day count

    if breakeven_days is None:
        breakeven_days = 365  # Max penalty if not reached

    roi = total_profit / machine_cost
    return pd.Series({'ROI': roi, 'breakeven_days': breakeven_days})


def load_features():
    """Load and prepare features dataset."""
    print("Loading features dataset...")
    features = pd.read_csv(FEATURES_PATH)
    features = features.set_index('date')
    features.index = pd.to_datetime(features.index)
    
    # Drop Unnamed: 0 column if it exists
    if 'Unnamed: 0' in features.columns:
        features = features.drop(columns=['Unnamed: 0'], axis=1)
    
    print(f"✓ Loaded {len(features)} feature records")
    return features


def load_electricity_data(elec_path):
    """Load and prepare electricity data."""
    elec = pd.read_csv(elec_path)
    
    # Drop Unnamed: 0 column if it exists
    if 'Unnamed: 0' in elec.columns:
        elec = elec.drop(columns=['Unnamed: 0'], axis=1)
    
    elec = elec.set_index('date')
    elec.index = pd.to_datetime(elec.index)
    
    # Rename price column to electricity_rate
    elec = elec.rename(columns={'price': 'electricity_rate'})
    
    return elec


def process_region(region_name, region_config, features):
    """Process a single region: merge data, fill NaN, calculate ROI."""
    print("\n" + "="*70)
    print(f"Processing {region_name.upper()}")
    print("="*70)
    
    # Load electricity data
    print(f"Loading electricity data from: {region_config['path']}")
    elec = load_electricity_data(region_config['path'])
    print(f"✓ Loaded {len(elec)} electricity records")
    
    # Merge with features
    print("Merging with features...")
    final_df = features.merge(elec, left_index=True, right_index=True, how='left')
    print(f"✓ Merged dataset has {len(final_df)} records")
    
    # Check for missing values
    null_count = final_df['electricity_rate'].isnull().sum()
    if null_count > 0:
        print(f"  Filling {null_count} missing electricity_rate values with {region_config['fillna_value']}")
        final_df['electricity_rate'] = final_df['electricity_rate'].fillna(region_config['fillna_value'])
    
    # Reset index
    final_df = final_df.reset_index()
    df = final_df
    
    # Calculate ROI and breakeven days
    print(f"\nCalculating ROI and breakeven days for {region_name}...")
    tqdm.pandas()
    
    # Make sure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply function to each row
    df[['ROI', 'breakeven_days']] = df.progress_apply(
        lambda row: calculate_roi_and_breakeven(row, df), axis=1
    )
    
    # Save to CSV
    df.to_csv(region_config['output'], index=False)
    print(f"\n✓ Saved results to: {region_config['output']}")
    print(f"  Total records: {len(df)}")
    print(f"  Valid ROI calculations: {df['ROI'].notna().sum()}")
    
    return df


def main():
    """Main function to process all regions."""
    print("="*70)
    print("CALCULATING ROI AND BREAKEVEN DAYS FOR ALL REGIONS")
    print("="*70)
    
    # Load features once
    features = load_features()
    
    # Process each region
    results = {}
    for region_name, region_config in ELECTRICITY_DATA.items():
        try:
            df = process_region(region_name, region_config, features)
            results[region_name] = df
        except Exception as e:
            print(f"\n✗ Error processing {region_name}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Regions processed: {len(results)}/{len(ELECTRICITY_DATA)}")
    for region_name, df in results.items():
        print(f"\n{region_name.upper()}:")
        print(f"  Total records: {len(df)}")
        print(f"  Valid ROI: {df['ROI'].notna().sum()}")
        print(f"  Output: {ELECTRICITY_DATA[region_name]['output']}")
    print("="*70)


if __name__ == "__main__":
    main()