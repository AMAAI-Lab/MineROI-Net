"""
download_blockchain_data.py

Script to download historical blockchain data from blockchain.info API.
Downloads Bitcoin price, difficulty, transaction fees, hashrate, and miners revenue.

Usage:
    python download_blockchain_data.py --output blockchain_data.csv

Requirements:
    pip install requests pandas
"""

import requests
import pandas as pd
import argparse
from datetime import datetime


def get_blockchain_data(data_type, start_year=2009, end_year=None):
    """
    Download blockchain data from blockchain.info API.
    
    Parameters:
    -----------
    data_type : str
        Type of data to download. Options: 'fees', 'hashrate', 'revenue', 
        'difficulty', 'bitcoin_price'
    start_year : int, optional
        Starting year for data collection (default: 2009)
    end_year : int, optional
        Ending year for data collection (default: current year)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and requested data column
    """
    
    options = {
        'fees': "transaction-fees",      # Transaction fees in BTC
        'hashrate': "hash-rate",         # Th/s - Estimated terahashes per second
        'revenue': "miners-revenue",     # USD - Total miner revenue (block rewards + fees)
        'difficulty': "difficulty",      # Relative mining difficulty
        'bitcoin_price': 'market-price', # USD - Bitcoin price
    }
    
    if data_type not in options:
        raise ValueError(f"Invalid data_type. Choose from: {list(options.keys())}")
    
    url = 'https://api.blockchain.info/charts/' + options[data_type]
    
    # Timespan of 6 years gives daily granularity
    timespan = 6
    
    if end_year is None:
        end_year = datetime.now().year
    
    # Base parameters for API request
    base_params = {
        'start': '2009-01-01',
        'timespan': f'{timespan}years',
        'format': 'json'
    }
    
    # Collect all data points
    values = []
    
    print(f"Downloading {data_type} data...")
    
    for year in range(start_year, end_year + 1, timespan):
        request_params = base_params.copy()
        request_params['start'] = f'{year}-01-01'
        
        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            values.extend(data['values'])
            
            print(f"  ✓ Data from {year} onwards added")
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading data for year {year}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(values)
    
    # Set display format for floats
    pd.set_option('display.float_format', '{:.24f}'.format)
    
    # Convert Unix timestamp to datetime
    df['x'] = pd.to_datetime(df['x'], unit='s')
    df.set_index('x', inplace=True)
    
    # Rename value column to data type
    df.rename(columns={'y': data_type}, inplace=True)
    
    return df


def download_all_blockchain_data(output_path='blockchain_data.csv', start_year=2009, end_year=None):
    """
    Download all blockchain data types and merge into single CSV.
    
    Parameters:
    -----------
    output_path : str
        Path to save the output CSV file
    start_year : int, optional
        Starting year for data collection (default: 2009)
    end_year : int, optional
        Ending year for data collection (default: current year)
    """
    
    print("="*60)
    print("Blockchain Data Download Script")
    print("="*60)
    
    data_types = ['bitcoin_price', 'difficulty', 'fees', 'hashrate', 'revenue']
    
    dataframes = {}
    
    # Download each data type
    for data_type in data_types:
        try:
            df = get_blockchain_data(data_type, start_year, end_year)
            dataframes[data_type] = df
            print(f"✓ Successfully downloaded {data_type}")
        except Exception as e:
            print(f"✗ Failed to download {data_type}: {e}")
            return
    
    print("\nMerging data...")
    
    # Merge all dataframes on date index
    merged_df = dataframes['bitcoin_price']
    
    for data_type in ['difficulty', 'fees', 'hashrate', 'revenue']:
        merged_df = pd.merge(
            merged_df, 
            dataframes[data_type],
            left_index=True, 
            right_index=True,
            how='outer'  # Use outer join to keep all dates
        )
    
    # Reset index to make date a column
    merged_df.index.name = 'date'
    merged_df.reset_index(inplace=True)

    # Filter data from January 17, 2009 to September 23, 2025 (this is the range we used to get results in our paper)
    print("\nFiltering data...")
    start_date = pd.to_datetime('2009-01-17')
    end_date = pd.to_datetime('2025-09-23')
    
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df[(merged_df['date'] >= start_date) & (merged_df['date'] <= end_date)]
    
    print(f"✓ Data filtered from {start_date.date()} to {end_date.date()}")

    
    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Data successfully saved to: {output_path}")
    print(f"  Total records: {len(merged_df)}")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"\nColumns: {list(merged_df.columns)}")
    print(f"\nFirst few rows:")
    print(merged_df.head())
    print(f"\Last few rows:")
    print(merged_df.tail())

    
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download historical blockchain data from blockchain.info API'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='blockchain_data.csv',
        help='Output CSV file path (default: blockchain_data.csv)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2009,
        help='Starting year for data collection (default: 2009)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help='Ending year for data collection (default: current year)'
    )
    
    args = parser.parse_args()
    
    # Download data
    download_all_blockchain_data(
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year
    )