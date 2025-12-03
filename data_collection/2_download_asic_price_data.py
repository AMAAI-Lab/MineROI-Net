"""
download_asic_price_data.py

Script to download historical ASIC miner price data from Hashrate Index API.
Downloads price indices for various efficiency categories and calculates individual
miner prices based on their hashrate and efficiency.
Data is filtered from January 22, 2018 to September 21, 2025.
Creates separate DataFrames for each miner model.

Usage:
    python download_asic_price_data.py --api-key YOUR_API_KEY --output asic_prices.csv

Requirements:
    pip install requests pandas
"""

import requests
import pandas as pd
import argparse
import sys
import os
from datetime import datetime


def download_asic_price_index(api_key, currency='USD', span='ALL'):
    """
    Download ASIC price index data from Hashrate Index API.
    
    Parameters:
    -----------
    api_key : str
        Your Hashrate Index API key
    currency : str, optional
        Currency for prices (default: USD)
    span : str, optional
        Time span for data. Options: 3M, 6M, 1Y, ALL (default: ALL)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with price index data by efficiency category
    """
    
    url = "https://api.hashrateindex.com/v1/hashrateindex/asic/price-index"
    
    headers = {
        "Accept": "application/json",
        "X-Hi-Api-Key": api_key
    }
    
    params = {
        "currency": currency,
        "span": span
    }
    
    print("Downloading ASIC price index data...")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json().get("data", [])
        
        if not data:
            print("✗ No data returned from API")
            return None
        
        price_df = pd.DataFrame(data)
        print(f"✓ Successfully downloaded {len(price_df)} records")
        
        return price_df
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to fetch data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def process_asic_prices(price_df):
    """
    Process ASIC price data: format dates and set index.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Raw price data from API
    
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with date index
    """
    
    print("\nProcessing price data...")
    
    # Convert timestamp to datetime
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    
    # Format to date only (remove time)
    price_df['timestamp'] = price_df['timestamp'].dt.strftime('%Y-%m-%d')
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    
    # Set index
    price_df = price_df.set_index('timestamp')
    price_df.index.name = 'date'
    
    # Sort by date ascending
    price_df = price_df.sort_index(ascending=True)
    
    print("✓ Data processed and sorted")
    
    return price_df


def calculate_miner_prices(price_df):
    """
    Calculate individual miner prices based on hashrate and efficiency.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        DataFrame with efficiency category prices
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns for each miner model
    """
    
    # Define miner specifications: (hashrate in TH/s, efficiency category column)
    machine_info = {
        's9':       (13.5,  'above68'),    # efficiency = 98 W/TH
        's19pro':   (110,   '25to38'),     # efficiency = 30 W/TH
        's15':      (28,    '38to68'),     # efficiency = 57 W/TH
        's17pro':   (50,    '38to68'),     # efficiency = 40 W/TH
        'M32':      (62,    '38to68'),     # efficiency = 54 W/TH
        's7':       (4.73,  'above68'),    # efficiency = 273 W/TH
        't17':      (40,    '38to68'),     # efficiency = 55 W/TH
        's19jpro':  (100,   '25to38'),     # efficiency = 30 W/TH
        'm21s':     (56,    '38to68'),     # efficiency = 60 W/TH
        'm10s':     (55,    '38to68'),     # efficiency = 64 W/TH
        's19kpro':  (120,   '19to25'),     # efficiency = 23 W/TH
        's21':      (200,   'under19'),    # efficiency = 18 W/TH
        'm30s':     (112,   '25to38'),     # efficiency = 31 W/TH
        'ka3':      (166,   '19to25'),     # efficiency = 19 W/TH
        'r4':       (8.7,   'above68'),    # efficiency = 97 W/TH
        't19':      (88,    '25to38'),     # efficiency = 38 W/TH
        's19xp':    (141,   '19to25'),     # efficiency = 21 W/TH
        's19apro':  (104,   '25to38'),     # efficiency = 31 W/TH
        'm50s':     (136,   '19to25'),     # efficiency = 24 W/TH
        'm53':      (226,   '25to38')      # efficiency = 29 W/TH
    }
    
    print("\nCalculating individual miner prices...")
    
    # Create new columns for each miner
    for miner, (hashrate, column) in machine_info.items():
        if column in price_df.columns:
            price_df[miner] = price_df[column] * hashrate
            print(f"  ✓ Calculated prices for {miner} (hashrate: {hashrate} TH/s)")
        else:
            print(f"  ✗ Efficiency column '{column}' not found for miner {miner}")
    
    return price_df


def extract_individual_miner_dataframes(price_df):
    """
    Extract separate DataFrames for each miner (drop NaN values).
    Each DataFrame has 'date' as index and miner name as column.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        DataFrame with all miner prices
    
    Returns:
    --------
    dict
        Dictionary of miner name -> DataFrame with date index and single column
    """
    
    miners = ['s9', 's19pro', 's15', 's17pro', 'M32', 's7', 't17', 's19jpro',
              'm21s', 'm10s', 's19kpro', 's21', 'm30s', 'ka3', 'r4', 't19',
              's19xp', 's19apro', 'm50s', 'm53']
    
    miner_dataframes = {}
    
    print("\nExtracting individual miner DataFrames...")
    
    for miner in miners:
        if miner in price_df.columns:
            # Create a DataFrame with just this miner's data
            miner_df = price_df[[miner]].dropna()
            miner_dataframes[miner] = miner_df
            
            # Also store as separate variable for backward compatibility
            globals()[f'{miner}_data'] = miner_df
            
            print(f"  ✓ {miner}_data: {len(miner_df)} rows × {len(miner_df.columns)} columns")
        else:
            print(f"  ✗ {miner}: not found in DataFrame")
    
    return miner_dataframes


def save_individual_miner_csvs(miner_dataframes, output_dir='miner_data'):
    """
    Save each miner's DataFrame to a separate CSV file.
    
    Parameters:
    -----------
    miner_dataframes : dict
        Dictionary of miner name -> DataFrame
    output_dir : str
        Directory to save individual CSV files
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created directory: {output_dir}")
    
    print(f"\nSaving individual miner CSV files to {output_dir}/...")
    
    for miner_name, miner_df in miner_dataframes.items():
        output_path = os.path.join(output_dir, f'{miner_name}_data.csv')
        miner_df.to_csv(output_path)
        print(f"  ✓ Saved {miner_name}_data.csv ({len(miner_df)} rows)")


def download_and_process_asic_data(api_key, output_path='asic_prices.csv', 
                                   currency='USD', span='ALL',
                                   save_individual=True, individual_dir='miner_data'):
    """
    Complete pipeline to download and process ASIC price data.
    Data is filtered from January 22, 2018 to September 21, 2025.
    
    Parameters:
    -----------
    api_key : str
        Your Hashrate Index API key
    output_path : str
        Path to save the complete output CSV file
    currency : str, optional
        Currency for prices (default: USD)
    span : str, optional
        Time span for data (default: ALL)
    save_individual : bool, optional
        Whether to save individual miner CSVs (default: True)
    individual_dir : str, optional
        Directory for individual miner CSV files (default: miner_data)
    """
    
    print("="*60)
    print("ASIC Price Data Download Script")
    print("="*60)
    
    # Download data
    price_df = download_asic_price_index(api_key, currency, span)
    
    if price_df is None:
        print("\n✗ Failed to download data. Exiting.")
        return None, None
    
    # Process data
    price_df = process_asic_prices(price_df)
    
    # Calculate miner prices
    price_df = calculate_miner_prices(price_df)
    
    # Filter data from January 22, 2018 to September 21, 2025
    print("\nFiltering data...")
    start_date = pd.to_datetime('2018-01-22')
    end_date = pd.to_datetime('2025-09-21')
    
    price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
    
    print(f"✓ Data filtered from {start_date.date()} to {end_date.date()}")
    
    # Extract individual miner DataFrames
    miner_dataframes = extract_individual_miner_dataframes(price_df)
    
    # Save complete dataset
    price_df.to_csv(output_path)
    print(f"\n✓ Complete data saved to: {output_path}")
    print(f"  Total records: {len(price_df)}")
    print(f"  Date range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"  Total columns: {len(price_df.columns)}")
    
    # Save individual miner CSVs
    if save_individual:
        save_individual_miner_csvs(miner_dataframes, individual_dir)
    
    # Display sample data from one miner
    if 'm53' in miner_dataframes:
        print(f"\nSample data from m53_data:")
        print(miner_dataframes['m53'].head())
        print("...")
        print(miner_dataframes['m53'].tail())
        print(f"\n{len(miner_dataframes['m53'])} rows × {len(miner_dataframes['m53'].columns)} columns")
        print(f"dtype: {miner_dataframes['m53']['m53'].dtype}")
    
    return price_df, miner_dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download ASIC price data from Hashrate Index API (filtered from 2018-01-22 to 2025-09-21)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='Your Hashrate Index API key (required)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='asic_prices.csv',
        help='Output CSV file path for complete data (default: asic_prices.csv)'
    )
    
    parser.add_argument(
        '--currency',
        type=str,
        default='USD',
        help='Currency for prices (default: USD)'
    )
    
    parser.add_argument(
        '--span',
        type=str,
        default='ALL',
        choices=['3M', '6M', '1Y', 'ALL'],
        help='Time span for data: 3M, 6M, 1Y, ALL (default: ALL)'
    )
    
    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='Do not save individual miner CSV files'
    )
    
    parser.add_argument(
        '--individual-dir',
        type=str,
        default='miner_data',
        help='Directory for individual miner CSV files (default: miner_data)'
    )
    
    args = parser.parse_args()
    
    # Download and process data
    price_df, miner_dataframes = download_and_process_asic_data(
        api_key=args.api_key,
        output_path=args.output,
        currency=args.currency,
        span=args.span,
        save_individual=not args.no_individual,
        individual_dir=args.individual_dir
    )
    
    if miner_dataframes:
        print("\n" + "="*60)
        print("Summary of Individual Miner DataFrames:")
        print("="*60)
        for miner_name, miner_df in miner_dataframes.items():
            print(f"{miner_name}_data: {len(miner_df)} rows × {len(miner_df.columns)} columns")