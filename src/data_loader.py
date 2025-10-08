"""
Data Loading Module
==================

Functions for loading cryptocurrency data from CSV files.
"""

import pandas as pd
import os
from typing import Dict, List, Optional
import warnings


def load_crypto_data(filename: str, crypto_name: str) -> pd.DataFrame:
    """
    Load cryptocurrency data from a CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    crypto_name : str
        Name of the cryptocurrency for identification
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cryptocurrency data and added 'Crypto' column
    """
    try:
        df = pd.read_csv(filename)
        df['Crypto'] = crypto_name
        return df
    except Exception as e:
        print(f"Error loading {crypto_name} data from {filename}: {e}")
        return pd.DataFrame()


def load_multiple_cryptos(data_dir: str, crypto_list: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple cryptocurrency datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
    crypto_list : list, optional
        List of cryptocurrency names to load. If None, loads all available.
        
    Returns:
    --------
    dict
        Dictionary with crypto names as keys and DataFrames as values
    """
    crypto_data = {}
    
    # Define mapping of file names to crypto names
    crypto_files = {
        'Bitcoin': 'coin_Bitcoin.csv',
        'Ethereum': 'coin_Ethereum.csv',
        'XRP': 'coin_XRP.csv',
        'Litecoin': 'coin_Litecoin.csv',
        'Cardano': 'coin_Cardano.csv',
        'Polkadot': 'coin_Polkadot.csv',
        'Solana': 'coin_Solana.csv',
        'Dogecoin': 'coin_Dogecoin.csv',
        'Aave': 'coin_Aave.csv',
        'BinanceCoin': 'coin_BinanceCoin.csv',
        'ChainLink': 'coin_ChainLink.csv',
        'Cosmos': 'coin_Cosmos.csv',
        'CryptocomCoin': 'coin_CryptocomCoin.csv',
        'EOS': 'coin_EOS.csv',
        'Iota': 'coin_Iota.csv',
        'Monero': 'coin_Monero.csv',
        'NEM': 'coin_NEM.csv',
        'Stellar': 'coin_Stellar.csv',
        'Tether': 'coin_Tether.csv',
        'Tron': 'coin_Tron.csv',
        'Uniswap': 'coin_Uniswap.csv',
        'USDCoin': 'coin_USDCoin.csv',
        'WrappedBitcoin': 'coin_WrappedBitcoin.csv'
    }
    
    # Filter to requested cryptos if specified
    if crypto_list:
        crypto_files = {name: file for name, file in crypto_files.items() if name in crypto_list}
    
    # Load each cryptocurrency
    for crypto_name, filename in crypto_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = load_crypto_data(filepath, crypto_name)
            if not df.empty:
                crypto_data[crypto_name] = df
                print(f"✅ Loaded {crypto_name}: {len(df)} records")
            else:
                print(f"❌ Failed to load {crypto_name}")
        else:
            print(f"⚠️  File not found: {filepath}")
    
    return crypto_data


def get_available_cryptos(data_dir: str) -> List[str]:
    """
    Get list of available cryptocurrency datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
        
    Returns:
    --------
    list
        List of available cryptocurrency names
    """
    available = []
    for file in os.listdir(data_dir):
        if file.startswith('coin_') and file.endswith('.csv'):
            crypto_name = file.replace('coin_', '').replace('.csv', '')
            available.append(crypto_name)
    return sorted(available)


def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame has the expected cryptocurrency data structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool
        True if structure is valid, False otherwise
    """
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True