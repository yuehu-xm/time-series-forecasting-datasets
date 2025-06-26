# Time Series Forecasting Datasets

[中文版本](README_CN.md) | [English Version](README_EN.md)

This repository contains multiple classic datasets for time series forecasting research. These datasets are widely used for evaluating time series forecasting models in academic research and industrial applications.

## Dataset Overview

### 📊 Included Datasets

| Dataset | Description | Time Range | Features |
|---------|-------------|------------|----------|
| **electricity.csv** | Electricity load data | Power consumption time series | Multivariate |
| **traffic.csv** | Traffic flow data | Road traffic flow time series | Multivariate |
| **ETTh1.csv** | Electricity Transforming Temperature (hourly) | Hourly sampled transformer data | 7 features |
| **ETTh2.csv** | Electricity Transforming Temperature (hourly) | Hourly sampled transformer data | 7 features |
| **ETTm1.csv** | Electricity Transforming Temperature (minutely) | 15-minute sampled transformer data | 7 features |
| **ETTm2.csv** | Electricity Transforming Temperature (minutely) | 15-minute sampled transformer data | 7 features |
| **Exchange.csv** | Exchange rate data | Multi-country currency exchange rate time series | 8 currency pairs |
| **Weather.csv** | Weather data | Meteorological observation time series | 21 weather features |

## 🚀 Quick Start

### Data Loading Example

```python
import pandas as pd
import numpy as np

# Load dataset
def load_dataset(file_path):
    """Load time series dataset"""
    data = pd.read_csv(file_path)
    return data

# Example: Load ETT dataset
ett_data = load_dataset('ETTh1.csv')
print(f"Data shape: {ett_data.shape}")
print(f"Columns: {ett_data.columns.tolist()}")
```

### Basic Data Exploration

```python
import matplotlib.pyplot as plt

# Basic data information
def explore_dataset(data, dataset_name):
    print(f"\n=== {dataset_name} Dataset Information ===")
    print(f"Data shape: {data.shape}")
    print(f"Time range: {data.iloc[0, 0]} to {data.iloc[-1, 0]}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(data.iloc[:, 1])  # Plot first feature
    plt.title(f'{dataset_name} - Time Series Visualization')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
```

## 📈 Dataset Details

### ETT (Electricity Transforming Temperature) Dataset
- **ETTh1/ETTh2**: Hourly electricity transformer temperature data
- **ETTm1/ETTm2**: 15-minute electricity transformer temperature data
- **Features**: Contains target variable OT (Oil Temperature) and 6 other load features

### Electricity Dataset
- Electricity load data from UCI
- Contains electricity consumption time series from multiple customers
- Suitable for multivariate time series forecasting tasks

### Traffic Dataset
- Traffic flow data from California highways
- Contains hourly traffic flow records from multiple sensors
- Suitable for traffic prediction and urban planning research

### Exchange Dataset
- Contains exchange rate data from 8 countries
- Daily exchange rates based on Singapore Dollar
- Suitable for financial time series forecasting

### Weather Dataset
- Weather observation data from German weather stations
- Contains 21 meteorological features (temperature, humidity, wind speed, etc.)
- High-frequency observation data at 10-minute intervals

## 🔧 Usage Recommendations

### Data Preprocessing
```python
def preprocess_data(data, target_col, seq_len=96, pred_len=24):
    """
    Time series data preprocessing
    
    Args:
        data: Raw data
        target_col: Target column name
        seq_len: Input sequence length
        pred_len: Prediction length
    """
    # Data normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Separate features and target variable
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    # Standardization
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))
    
    return features_scaled, target_scaled, scaler
```

### Recommended Machine Learning Models
- **Traditional Methods**: ARIMA, SARIMA, Prophet
- **Deep Learning**: LSTM, GRU, Transformer
- **Latest Methods**: Informer, Autoformer, PatchTST

## 📚 Related Papers

If you use these datasets in your research, please consider citing the relevant papers:

- ETT Dataset: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- For other datasets, please refer to their official documentation for original sources

## 🤝 Contributing

Welcome to submit issues and improvement suggestions! If you have new time series datasets you'd like to add to this repository, please submit a Pull Request.

## 📄 License

The datasets in this repository follow their original licenses. Please check the specific license requirements for each dataset before use.

## 📞 Contact

For questions or suggestions, please contact us through GitHub Issues.

---

**Note**: Some large dataset files are stored using Git LFS. Please ensure Git LFS is installed when cloning the repository for the first time.

```bash
# Install Git LFS
git lfs install

# Clone repository
git clone <repository-url>
```

