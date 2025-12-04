# Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI prediction

This repository contains the code and dataset accompanying the paper "Smart Timing for Mining: A Deep Learning
Framework for Bitcoin Hardware ROI prediction" by Sithumi Wickramasinghe (PhD candidate), Prof. Bikramjith Das, and Prof. Dorien Herremans.

<div align="center">
  <img src="MineROI-Net.png" alt="MineROI-Net Architecture" width="800"/>
</div>

We propose MineROI-Net a Transformer-based architecture for **timing Bitcoin mining hardware purchases**.  
Given a 30- or 60-day window of machine, market, and network features, the model predicts whether
buying a specific ASIC miner on a given day will be:

- **Unprofitable** (ROI ≤ 0)  
- **Marginal** (0 < ROI < 1)  
- **Profitable** (ROI ≥ 1 within 365 days)

This repository contains the PyTorch implementation, data preprocessing pipeline, and experiments
used in our paper.

---

# Overview

Bitcoin mining is capital-intensive and highly sensitive to market cycles, halving events, hardware
efficiency, and electricity prices. MineROI-Net formulates the **hardware acquisition timing**
problem as a **multi-class time series classification task**.

The model combines:

1. **Spectral Feature Extractor**  
   - FFT-based layer with learnable complex weights to highlight important frequency components
     (halving cycles, difficulty adjustment cycles, etc.).

2. **Channel Mixing Module**  
   - Squeeze-and-Excitation style feature re-weighting that captures cross-feature interactions
     (e.g., price vs. electricity cost dominance) in a lightweight way.

3. **Transformer Encoder**  
   - Standard multi-head self-attention over the processed sequence, followed by global pooling and
     a classification head.

MineROI-Net outperforms LSTM-based and TSLANet baselines on data from **20 ASIC miners (2015–2024)**,
achieving strong accuracy and macro-F1 while being economically well-behaved (very high precision
for profitable and unprofitable periods).

---



# Repository structure

A minimal structure (yours may contain more folders):

```text
MineROI-Net/
│
├── country_wise_data/
│   ├── final_china.csv
│   ├── final_ethiopia.csv
│   ├── final_texas.csv
│   ├── china_seq/           # (generated sequences for china)
│   ├── ethiopia_seq/        # (generated sequences for ethiopia)
│   ├── texas_seq/           # (generated sequences for texas)
│   └── seq_30/              # (final data for model when loock back window = 30)
│   └── seq_60/              # (final data for model when loock back window = 60)
│   └── ...
│
├── final_split/
│   ├── dataloader.py                # preprocessing + CV pipeline
│   └── transformer_final_split.py   # MineROI-Net model + training loop
│   └── ...
│
├── results_seq_30/      # final results when loock back window = 30 
├── results_seq_60/      # final results when loock back window = 60 
└── README.md
```





# Data creation pipeline steps
The data_collection folder consists of 6 sequential scripts that download, process, and prepare datasets for 20 different ASIC mining machines across three regions (Texas, China, and Ethiopia). 
> **Note**  
> Due to data-source restrictions, we do not redistribute raw ASIC pricing data.  
> Follow the data sources listed in the paper and following scripts to reconstruct the dataset, or plug in your own mining data.

### Step 1: Download Blockchain Data
**Script:** `1_download_blockchain_data.py`

Downloads historical Bitcoin blockchain data from blockchain.info API.
**Data collected:**
- Bitcoin price (USD)
- Network difficulty
- Transaction fees (BTC)
- Network hashrate (TH/s)
- Miners revenue (USD)

**Date range:** January 17, 2009 to September 23, 2025

<!-- **Usage:**
```bash
python 1_download_blockchain_data.py --output blockchain_data.csv
``` -->

**Output:** `blockchain_data.csv`

---

### Step 2: Download ASIC Price Data
**Script:** `2_download_asic_price_data.py`

Downloads historical ASIC miner prices from Hashrate Index API.

**Miners covered:** 20 ASIC models (S9, S19 Pro, S15, S17 Pro, M32, S7, T17, S19j Pro, M21S, M10S, S19k Pro, S21, M30S, KA3, R4, T19, S19 XP, S19a Pro, M50S, M53)

**Date range:** January 22, 2018 to September 21, 2025

<!-- **Usage:**
```bash
python 2_download_asic_price_data.py --api-key YOUR_API_KEY
``` -->


**Output:** 
- `asic_prices.csv` (complete dataset)
- `miner_data/` (individual CSVs for each miner)

---

### Step 3: Prepare Electricity Data
**Script:** `3_electricity_data.py`

**Note:** This is a placeholder script. You need to prepare electricity price data for three regions:

**Required files:**
- `texas_residential_daily_df.csv`
- `china_electricity_prices_daily.csv`
- `ethiopia_electricity_prices_daily.csv`

**Format:** Each file should contain:
- `date` column (YYYY-MM-DD)
- `price` column (USD per kWh)

**Data source:** See Section 4.1 in the paper. Monthly electricity prices are repeated for each day of the month to create daily time series.

---

### Step 4: Prepare Miner Datasets
**Script:** `4_prepare_miner_dataset.py`

Combines blockchain data with ASIC specifications and calculates features for all 20 miners.

**What it does:**
- Loads blockchain data and miner price data
- Adds machine specifications (hashrate, power, efficiency, release date)
- Calculates block rewards based on Bitcoin halving schedule
- Calculates machine age since release date
- Calculates days since last Bitcoin halving
- Filters data to dates after each machine's release
- Calculates daily revenue potential

<!-- **Usage:**
```bash
python 4_prepare_miner_dataset.py
``` -->

**Output:** `full_feature_data.csv` (combined dataset with all miners and features)

---

### Step 5: Calculate ROI by Country
**Script:** `5_roi_country.py`

Calculates Return on Investment (ROI) for each miner in each region.

**What it does:**
- Merges feature data with electricity prices for each region
- Calculates 12-month forward ROI for each machine on each date

<!-- **Usage:**
```bash
python 5_roi_country.py
``` -->

**Output:**
- `roi_texas.csv`
- `roi_china.csv`
- `roi_ethiopia.csv`

---

### Step 6: Create Target Variable
**Script:** `6_create_target.py`

Performs feature engineering and creates classification target for modeling.

  - Category 0: ROI < 0 (Loss)
  - Category 1: 0 ≤ ROI < 1 (Partial recovery)
  - Category 2: ROI ≥ 1 (Profitable)
- Cleans data and removes unnecessary columns

<!-- **Usage:**
```bash
python 6_create_target.py
``` -->

**Output:**
- `final_texas.csv`
- `final_china.csv`
- `final_ethiopia.csv`

---

## Complete Pipeline Execution

Run all steps in sequence:
```bash
# Step 1: Download blockchain data
python 1_download_blockchain_data.py

# Step 2: Download ASIC prices (requires API key)
python 2_download_asic_price_data.py --api-key YOUR_API_KEY

# Step 3: Prepare electricity data (manual - see script comments)
python 3_electricity_data.py

# Step 4: Combine data and prepare miner datasets
python 4_prepare_miner_dataset.py

# Step 5: Calculate ROI for each region
python 5_roi_country.py

# Step 6: Create final datasets with targets
python 6_create_target.py
```

---



## Final Dataset Features

Each final dataset (`final_*.csv`) contains:

**Features:**
- `date`: Date
- `bitcoin_price`: Bitcoin price (USD)
- `difficulty`: Network difficulty
- `fees`: Transaction fees (BTC)
- `hashrate`: Network hashrate (TH/s)
- `revenue`: Miners revenue (USD)
- `machine_price`: ASIC miner price (USD)
- `machine_hashrate`: Miner hashrate (TH/s)
- `power`: Power consumption (W)
- `efficiency`: Energy efficiency (W/TH)
- `block_reward`: Block subsidy (BTC)
- `age_days`: Days since miner release
- `days_since_halving`: Days since last Bitcoin halving
- `Revenue_Potential`: Daily revenue potential (USD)
- `electricity_rate`: Electricity price (USD/kWh)
- `machine_name`: Miner model identifier

**Target:**
- `roi_category_id`: ROI category (0=Loss, 1=Partial, 2=Profitable)

---







# MineROI-Net Model Training

The **models** folder consists of two main components:
1. **dataloader.py** - Prepares time-series data for model training
2. **transformer_final_split.py** - Trains and evaluates the transformer model

---


## Script: `dataloader.py`

Prepares the final datasets for training by creating time-series windows and splitting data.

```python
from dataloader import run_all_preprocessing

# Create time-series windows and prepare datasets
run_all_preprocessing(SEQ_LEN=30)
```

**What it does:**
- Creates 30-day sliding windows from time-series data
- Splits data: 80% train, 20% val/test (time-based)
- Scales features with MinMaxScaler
- Combines data from all three regions
- Converts to transformer format `[Batch, Length, Channels]`

**Output:** `seq_30/train_trans.pt`, `val_trans.pt`, `test_trans.pt`

---


## Script: `transformer_final_split.py`
```bash
python transformer_final_split.py
```

**What it does:**
- Trains transformer classifier on preprocessed data
- Predicts ROI categories: 0 (Loss), 1 (Partial), 2 (Profitable)
- Uses early stopping and saves best checkpoint
- Reports accuracy, precision, recall, F1-score

**Output:** `checkpoints/best_model.pth`

---







