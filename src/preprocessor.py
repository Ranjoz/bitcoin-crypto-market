"""
Data Preprocessing Module
========================

Functions for cleaning and preprocessing cryptocurrency data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean cryptocurrency data by handling dates, missing values, and duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw cryptocurrency data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned cryptocurrency data
    """
    df_clean = df.copy()
    
    # Convert Date column to datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Set Date as index
    df_clean.set_index('Date', inplace=True)
    
    # Sort by date
    df_clean.sort_index(inplace=True)
    
    # Remove duplicates (keep first occurrence)
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Handle missing values
    # Forward fill for small gaps
    df_clean = df_clean.fillna(method='ffill', limit=3)
    
    # Interpolate for remaining gaps
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    df_clean[numeric_columns] = df_clean[numeric_columns].interpolate(method='linear')
    
    return df_clean


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-related features to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned cryptocurrency data
        
    Returns:
    --------
    pd.DataFrame
        Data with additional price features
    """
    df_features = df.copy()
    
    # Daily returns
    df_features['Daily_Return'] = df_features['Close'].pct_change()
    
    # Log returns
    df_features['Log_Return'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
    
    # Price change
    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
    
    # Price range
    df_features['Price_Range'] = df_features['High'] - df_features['Low']
    df_features['Range_Pct'] = (df_features['High'] - df_features['Low']) / df_features['Open'] * 100
    
    # Moving averages
    df_features['MA7'] = df_features['Close'].rolling(window=7).mean()
    df_features['MA30'] = df_features['Close'].rolling(window=30).mean()
    df_features['MA90'] = df_features['Close'].rolling(window=90).mean()
    
    return df_features


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility and volume features to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with price features
        
    Returns:
    --------
    pd.DataFrame
        Data with additional volatility features
    """
    df_vol = df.copy()
    
    # Rolling volatility
    df_vol['Volatility_30'] = df_vol['Daily_Return'].rolling(window=30).std()
    df_vol['Volatility_90'] = df_vol['Daily_Return'].rolling(window=90).std()
    
    # Cumulative returns
    df_vol['Cumulative_Return'] = (1 + df_vol['Daily_Return']).cumprod()
    
    # Volume features
    df_vol['Volume_Change'] = df_vol['Volume'].pct_change()
    df_vol['Volume_MA30'] = df_vol['Volume'].rolling(window=30).mean()
    
    return df_vol


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-based features to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with existing features
        
    Returns:
    --------
    pd.DataFrame
        Data with additional date features
    """
    df_date = df.copy()
    
    # Extract date components
    df_date['Year'] = df_date.index.year
    df_date['Month'] = df_date.index.month
    df_date['DayOfWeek'] = df_date.index.dayofweek
    df_date['Quarter'] = df_date.index.quarter
    df_date['DayOfYear'] = df_date.index.dayofyear
    
    return df_date


def preprocess_crypto_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for cryptocurrency data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw cryptocurrency data
        
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed data ready for analysis
    """
    print("Starting data preprocessing...")
    
    # Clean data
    df_clean = clean_data(df)
    print(f"✅ Data cleaned: {len(df_clean)} records after cleaning")
    
    # Add features
    df_features = add_price_features(df_clean)
    df_features = add_volatility_features(df_features)
    df_features = add_date_features(df_features)
    
    print(f"✅ Features added: {len(df_features.columns)} total columns")
    
    return df_features


def preprocess_multiple_cryptos(crypto_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess multiple cryptocurrency datasets.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
        
    Returns:
    --------
    dict
        Dictionary of preprocessed cryptocurrency DataFrames
    """
    processed_data = {}
    
    for crypto_name, df in crypto_data.items():
        print(f"\nProcessing {crypto_name}...")
        processed_data[crypto_name] = preprocess_crypto_data(df)
    
    return processed_data


def get_common_date_range(crypto_data: Dict[str, pd.DataFrame]) -> tuple:
    """
    Get the common date range across all cryptocurrency datasets.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
        
    Returns:
    --------
    tuple
        (start_date, end_date) of common range
    """
    start_dates = []
    end_dates = []
    
    for df in crypto_data.values():
        if not df.empty:
            start_dates.append(df.index.min())
            end_dates.append(df.index.max())
    
    common_start = max(start_dates) if start_dates else None
    common_end = min(end_dates) if end_dates else None
    
    return common_start, common_end