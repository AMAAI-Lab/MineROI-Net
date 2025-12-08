# Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI prediction

<!-- **[Paper](https://arxiv.org/abs/2512.05402)** -->

This repository contains the code and dataset accompanying the paper **[Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI Prediction](https://arxiv.org/abs/2512.05402)** by Sithumi Wickramasinghe (PhD candidate), Prof. Bikramjith Das, and Prof. Dorien Herremans.

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



<!-- # Repository structure

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
``` -->



# Data Collection Pipeline

The `data_collection/` folder contains 6 sequential scripts that download, process, and prepare datasets for 20 ASIC mining machines across three regions (Texas, China, Ethiopia).

> **Note:** Due to data-source restrictions, we do not redistribute raw ASIC pricing data. Follow the data sources in Section 4.1 of the paper and these scripts to reconstruct the dataset.

---

## Pipeline Steps

### Step 1: Download Blockchain Data
**Script:** `1_download_blockchain_data.py`

Downloads Bitcoin price, network difficulty, transaction fees, hashrate, and miners revenue from blockchain.info API.

**Date range:** Jan 17, 2009 - Sep 23, 2025  
**Output:** `blockchain_data.csv`

---

### Step 2: Download ASIC Price Data
**Script:** `2_download_asic_price_data.py`

Downloads historical prices for 20 ASIC miners from Hashrate Index API.

**Requires:** API key from [hashrateindex.com/api](https://hashrateindex.com/api)  
**Date range:** Jan 22, 2018 - Sep 21, 2025  
**Output:** `asic_prices.csv`, `miner_data/` (individual CSVs)

---

### Step 3: Prepare Electricity Data
**Script:** `3_electricity_data.py`

Prepare electricity price data for three regions. See Section 4.1 in paper for data sources.

**Required files:**
- `texas_residential_daily_df.csv`
- `china_electricity_prices_daily.csv`
- `ethiopia_electricity_prices_daily.csv`

**Format:** `date` (YYYY-MM-DD) and `price` (USD/kWh) columns

---

### Step 4: Prepare Miner Datasets
**Script:** `4_prepare_miner_dataset.py`

Combines blockchain data with ASIC specifications for all 20 miners.

**Calculations:**
- Machine specifications (hashrate, power, efficiency, release date)
- Block rewards (Bitcoin halving schedule)
- Machine age and days since halving
- Daily revenue potential

**Output:** `full_feature_data.csv`

---

### Step 5: Calculate ROI by Country
**Script:** `5_roi_country.py`

Calculates 12-month forward ROI for each miner in each region.

**Output:** `roi_texas.csv`, `roi_china.csv`, `roi_ethiopia.csv`

---

### Step 6: Create Target Variable
**Script:** `6_create_target.py`

Engineers time-series features and creates ROI classification target.

**ROI Categories:**
- 0: Unprofitable (ROI < 0)
- 1: Marginal (0 ≤ ROI < 1)
- 2: Profitable (ROI ≥ 1)

**Output:** `final_texas.csv`, `final_china.csv`, `final_ethiopia.csv`

---

## Quick Start

Run all steps in sequence:
```bash
python 1_download_blockchain_data.py
python 2_download_asic_price_data.py --api-key YOUR_API_KEY
python 3_electricity_data.py
python 4_prepare_miner_dataset.py
python 5_roi_country.py
python 6_create_target.py
```

---

## Final Dataset

Each `final_*.csv` contains:

**Key Features:**
- `date`, `bitcoin_price`, `difficulty`, `fees`, `hashrate`, `revenue`
- `machine_price`, `machine_hashrate`, `power`, `efficiency`
- `block_reward`, `age_days`, `days_since_halving`
- `Revenue_Potential`, `electricity_rate`, `machine_name`

**Target:**
- `roi_category_id`: 0 (Unprofitable), 1 (Marginal), 2 (Profitable)

---



## Dataset Sample

| date       | bitcoin_price | difficulty        | fees   | hashrate      | revenue      | machine_price | machine_hashrate | power | efficiency | block_reward | age_days | days_since_halving | Revenue_Potential | machine_name | electricity_rate | roi_category_id |
|------------|----------------|--------------------|--------|----------------|--------------|----------------|-------------------|--------|------------|--------------|----------|---------------------|--------------------|----------------|-------------------|------------------|
| 2024-09-18 | 60304.22       | 9.27E13            | 7.02   | 6.36E8         | 2.64E7       | 833.94         | 226               | 6554   | 29         | 3.125        | 1265     | 151                 | 9.245              | m53            | 0.0767            | 0                |
| 2024-09-19 | 61683.91       | 9.27E13            | 9.94   | 5.62E8         | 2.50E7       | 833.94         | 226               | 6554   | 29         | 3.125        | 1266     | 152                 | 9.457              | m53            | 0.0767            | 0                |
| 2024-09-20 | 62938.20       | 9.27E13            | 9.51   | 6.68E8         | 2.93E7       | 833.94         | 226               | 6554   | 29         | 3.125        | 1267     | 153                 | 9.649              | m53            | 0.0767            | 0                |
| 2024-09-21 | 63213.19       | 9.27E13            | 6.08   | 6.22E8         | 2.70E7       | 833.94         | 226               | 6554   | 29         | 3.125        | 1268     | 154                 | 9.691              | m53            | 0.0767            | 0                |

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







