"""
prepare_all_miners_datasets.py

Script to prepare complete datasets for ALL ASIC miners by combining
blockchain data with machine specifications and calculating revenue potential.

This script processes all 20 miners automatically without requiring arguments.

This script:
1. Loads machine price data for each miner
2. Loads blockchain data (BTC price, difficulty, hashrate, etc.)
3. Merges the datasets
4. Adds machine specifications (hashrate, power, efficiency, release date)
5. Calculates block rewards based on halving schedule
6. Calculates machine age since release
7. Calculates days since last Bitcoin halving
8. Filters data to only include dates after machine availability
9. Calculates daily revenue potential
10. Saves individual dataset for each miner
11. Combines all datasets and saves as full_feature_data.csv

Usage:
    python prepare_all_miners_datasets.py

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# Machine specifications database
MACHINE_SPECS = {
    's9': {
        'hashrate': 'XX.X',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's19pro': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's15': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's17pro': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm32': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's7': {
        'hashrate': 'X.XX',
        'power': 'XXXX',
        'efficiency': 'XXX',
        'release_date': 'XXXX-XX-XX'
    },
    't17': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's19jpro': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm21s': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm10s': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's19kpro': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's21': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm30s': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'ka3': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'r4': {
        'hashrate': 'X.X',
        'power': 'XXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    't19': {
        'hashrate': 'XX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's19xp': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    's19apro': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm50s': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    },
    'm53': {
        'hashrate': 'XXX',
        'power': 'XXXX',
        'efficiency': 'XX',
        'release_date': 'XXXX-XX-XX'
    }
}

# Configuration - Update these paths as needed
BLOCKCHAIN_DATA_PATH = '/root/Mine_ROI_Net/data_collection/blockchain_data.csv'
MINER_DATA_DIR = 'miner_data'
OUTPUT_DIR = 'complete_datasets'
COMBINED_OUTPUT_PATH = 'full_feature_data.csv'


def load_machine_data(miner_name, miner_data_dir):
    """
    Load machine price data for a specific miner.
    
    Parameters:
    -----------
    miner_name : str
        Name of the miner (e.g., 'm53', 's19pro')
    miner_data_dir : str
        Directory containing miner price CSV files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and machine_price column
    """
    
    file_path = os.path.join(miner_data_dir, f'{miner_name}_data.csv')
    
    if not os.path.exists(file_path):
        print(f"  ✗ File not found: {file_path}")
        return None
    
    machine_data = pd.read_csv(file_path)
    machine_data = machine_data.set_index('date')
    
    # Rename the miner column to 'machine_price'
    machine_data.rename(columns={miner_name: 'machine_price'}, inplace=True)
    
    return machine_data


def load_blockchain_data(blockchain_csv):
    """
    Load blockchain data (BTC price, difficulty, hashrate, etc.).
    
    Parameters:
    -----------
    blockchain_csv : str
        Path to blockchain data CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and blockchain metrics
    """
    
    if not os.path.exists(blockchain_csv):
        raise FileNotFoundError(f"Blockchain data file not found: {blockchain_csv}")
    
    merged_df = pd.read_csv(blockchain_csv)
    merged_df = merged_df.set_index('date')
    
    return merged_df


def add_machine_specifications(df, miner_name):
    """
    Add machine specifications (hashrate, power, efficiency, release date).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined DataFrame
    miner_name : str
        Name of the miner
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with machine specifications added
    """
    
    if miner_name not in MACHINE_SPECS:
        raise ValueError(f"Machine specifications not found for {miner_name}")
    
    specs = MACHINE_SPECS[miner_name]
    
    df['machine_hashrate'] = specs['hashrate']
    df['power'] = specs['power']
    df['efficiency'] = specs['efficiency']
    df['relesed_date'] = specs['release_date']
    
    return df


def add_block_rewards(df):
    """
    Add block reward column based on Bitcoin halving schedule.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with block_reward column added
    """
    
    df.index = pd.to_datetime(df.index)
    
    # Bitcoin halving schedule
    date_ranges = [
        (pd.Timestamp('2012-11-28'), 50),
        (pd.Timestamp('2016-07-09'), 25),
        (pd.Timestamp('2020-05-11'), 12.5),
        (pd.Timestamp('2024-04-19'), 6.25),
        (pd.Timestamp.max, 3.125)
    ]
    
    def get_value(date):
        for end_date, value in date_ranges:
            if date <= end_date:
                return value
        return 3.125
    
    df['block_reward'] = df.index.map(get_value)
    
    return df


def add_machine_age(df):
    """
    Calculate machine age in days since release date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with relesed_date column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with age_days column added
    """
    
    # Convert 'relesed_date' column to datetime
    df['relesed_date'] = pd.to_datetime(df['relesed_date'])
    
    # Calculate age in days
    df['age_days'] = (df.index - df['relesed_date']).dt.days
    
    return df


def add_days_since_halving(df):
    """
    Calculate days since last Bitcoin halving event.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with days_since_halving column added
    """
    
    # Known Bitcoin halving dates
    halving_dates = [
        pd.to_datetime("2012-11-28"),
        pd.to_datetime("2016-07-09"),
        pd.to_datetime("2020-05-11"),
        pd.to_datetime("2024-04-20"),
        pd.to_datetime("2028-03-26")
    ]
    
    def compute_days_since_halving(date):
        past_halvings = [d for d in halving_dates if d < date]
        if not past_halvings:
            return None
        last_halving = max(past_halvings)
        return (date - last_halving).days
    
    df["days_since_halving"] = df.index.map(compute_days_since_halving)
    
    return df


def filter_by_availability(df, release_date):
    """
    Filter data to only include dates after machine availability.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    release_date : str
        Machine release date
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    
    df["machine_available"] = df.index >= pd.Timestamp(release_date)
    df = df[df['machine_available'] == True]
    df = df.dropna()
    
    return df


def calculate_revenue_potential(df):
    """
    Calculate daily revenue potential for the mining machine.
    
    Revenue formula:
    Revenue = (machine_hashrate * block_reward * seconds_per_day * bitcoin_price) / (difficulty * 2^32)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with required columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Revenue_Potential column added
    """
    
    # Convert to appropriate data types
    df['machine_hashrate'] = df['machine_hashrate'].astype(np.float64)
    df['power'] = df['power'].astype(np.float64)
    df['efficiency'] = df['efficiency'].astype(np.float64)
    
    # Calculate revenue potential
    seconds_per_day = 86400
    df["Revenue_Potential"] = (
        (df['machine_hashrate'] * (10**12)) * 
        df['block_reward'] * 
        seconds_per_day * 
        df['bitcoin_price']
    ) / (df['difficulty'] * (2**32))
    
    return df


def prepare_miner_dataset(miner_name, blockchain_data, miner_data_dir, output_dir):
    """
    Complete pipeline to prepare miner dataset.
    
    Parameters:
    -----------
    miner_name : str
        Name of the miner (e.g., 'm53', 's19pro')
    blockchain_data : pd.DataFrame
        Blockchain data DataFrame
    miner_data_dir : str
        Directory containing miner price data
    output_dir : str
        Directory to save output CSV
    
    Returns:
    --------
    pd.DataFrame or None
        Processed DataFrame if successful, None otherwise
    """
    
    try:
        print(f"\nProcessing {miner_name}...")
        
        # Load machine data
        machine_data = load_machine_data(miner_name, miner_data_dir)
        if machine_data is None:
            return None
        
        # Merge datasets
        combined_df = blockchain_data.merge(machine_data, on='date', how='left')
        
        # Add features
        combined_df = add_machine_specifications(combined_df, miner_name)
        combined_df = add_block_rewards(combined_df)
        combined_df = add_machine_age(combined_df)
        combined_df = add_days_since_halving(combined_df)
        
        # Filter and clean
        release_date = MACHINE_SPECS[miner_name]['release_date']
        df = filter_by_availability(combined_df, release_date)
        
        # Calculate revenue
        df = calculate_revenue_potential(df)
        
        # Add machine_name column
        # Use lowercase for consistency (handle M32 -> m32)
        machine_name_normalized = miner_name.lower()
        df['machine_name'] = machine_name_normalized
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'{miner_name}_complete_dataset.csv')
        df.to_csv(output_path)
        
        print(f"  ✓ {miner_name}: {len(df)} records saved to {output_path}")
        print(f"    Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ {miner_name}: Error - {str(e)}")
        return None


def combine_all_datasets(all_dataframes, output_path):
    """
    Combine all miner datasets into a single CSV file.
    
    Parameters:
    -----------
    all_dataframes : dict
        Dictionary mapping miner names to their DataFrames
    output_path : str
        Path to save the combined CSV file
    """
    
    print("\n" + "="*70)
    print("COMBINING ALL DATASETS")
    print("="*70)
    
    # Filter out None values (failed miners)
    valid_dataframes = [df for df in all_dataframes.values() if df is not None]
    
    if not valid_dataframes:
        print("✗ No valid dataframes to combine!")
        return
    
    print(f"Combining {len(valid_dataframes)} datasets...")
    
    # Reset index before concatenating to include date as a column
    dataframes_with_date = []
    for df in valid_dataframes:
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)  # Convert date index to column
        dataframes_with_date.append(df_copy)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes_with_date, ignore_index=True)
    
    print(f"✓ Combined dataset created")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Total columns: {len(combined_df.columns)}")
    
    # Show distribution by machine
    print(f"\nRecords per machine:")
    machine_counts = combined_df['machine_name'].value_counts().sort_index()
    for machine, count in machine_counts.items():
        print(f"  {machine}: {count}")
    
    # Save combined dataset with date column
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Combined dataset saved to: {output_path}")
    print(f"  Columns: {list(combined_df.columns)}")
    
    return combined_df


def prepare_all_miners():
    """
    Process all miners and create complete datasets.
    """
    
    print("="*70)
    print("PREPARING COMPLETE DATASETS FOR ALL MINERS")
    print("="*70)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Created output directory: {OUTPUT_DIR}")
    
    # Load blockchain data once (shared across all miners)
    print(f"\nLoading blockchain data from: {BLOCKCHAIN_DATA_PATH}")
    try:
        blockchain_data = load_blockchain_data(BLOCKCHAIN_DATA_PATH)
        print(f"✓ Loaded {len(blockchain_data)} blockchain records")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure blockchain_data.csv exists in the current directory.")
        return
    
    # Process each miner and store results
    print(f"\nProcessing {len(MACHINE_SPECS)} miners...")
    print("-"*70)
    
    successful = 0
    failed = 0
    all_dataframes = {}
    
    for miner_name in MACHINE_SPECS.keys():
        df = prepare_miner_dataset(
            miner_name=miner_name,
            blockchain_data=blockchain_data.copy(),  # Pass a copy to avoid modification
            miner_data_dir=MINER_DATA_DIR,
            output_dir=OUTPUT_DIR
        )
        
        if df is not None:
            successful += 1
            all_dataframes[miner_name] = df
        else:
            failed += 1
    
    # Combine all datasets into one file
    if all_dataframes:
        combine_all_datasets(all_dataframes, COMBINED_OUTPUT_PATH)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total miners processed: {len(MACHINE_SPECS)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"\nIndividual datasets: {OUTPUT_DIR}/")
    print(f"Combined dataset: {COMBINED_OUTPUT_PATH}")
    print("="*70)


if __name__ == "__main__":
    prepare_all_miners()
