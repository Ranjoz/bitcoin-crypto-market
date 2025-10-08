"""
Analysis Module
==============

Core analysis functions for cryptocurrency market analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive summary statistics for cryptocurrency data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap', 'Daily_Return']
    stats = df[numeric_columns].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame(index=['skewness', 'kurtosis', 'volatility'])
    for col in numeric_columns:
        if col in df.columns:
            additional_stats.loc['skewness', col] = df[col].skew()
            additional_stats.loc['kurtosis', col] = df[col].kurtosis()
            if col == 'Daily_Return':
                additional_stats.loc['volatility', col] = df[col].std() * np.sqrt(252)  # Annualized
    
    return pd.concat([stats, additional_stats])


def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various volatility metrics for a cryptocurrency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data with Daily_Return column
        
    Returns:
    --------
    dict
        Dictionary of volatility metrics
    """
    returns = df['Daily_Return'].dropna()
    
    metrics = {
        'daily_volatility': returns.std(),
        'annualized_volatility': returns.std() * np.sqrt(252),
        'volatility_30d': df['Daily_Return'].rolling(30).std().mean(),
        'volatility_90d': df['Daily_Return'].rolling(90).std().mean(),
        'coefficient_of_variation': returns.std() / abs(returns.mean()) if returns.mean() != 0 else np.inf
    }
    
    # Parkinson volatility (using High-Low)
    if all(col in df.columns for col in ['High', 'Low']):
        hl_ratio = np.log(df['High'] / df['Low'])
        parkinson_vol = np.sqrt((1/(4*np.log(2))) * np.mean(hl_ratio**2))
        metrics['parkinson_volatility'] = parkinson_vol
    
    return metrics


def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate risk metrics for a cryptocurrency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data
        
    Returns:
    --------
    dict
        Dictionary of risk metrics
    """
    returns = df['Daily_Return'].dropna()
    cumulative_returns = (1 + returns).cumprod()
    
    # Maximum drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Value at Risk (VaR) at 95% confidence
    var_95 = np.percentile(returns, 5)
    
    # Expected Shortfall (Conditional VaR)
    es_95 = returns[returns <= var_95].mean()
    
    metrics = {
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'expected_shortfall_95': es_95,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    }
    
    return metrics


def calculate_correlation_matrix(crypto_data: Dict[str, pd.DataFrame], 
                                price_col: str = 'Close', 
                                method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix between cryptocurrencies.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
    price_col : str
        Column to use for correlation calculation
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    # Align all datasets to common date range
    price_data = {}
    
    for name, df in crypto_data.items():
        if price_col in df.columns:
            price_data[name] = df[price_col]
    
    # Create DataFrame with aligned dates
    prices_df = pd.DataFrame(price_data)
    
    # Calculate correlation
    correlation_matrix = prices_df.corr(method=method)
    
    return correlation_matrix


def calculate_bitcoin_dominance(crypto_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate Bitcoin dominance over time.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames including Bitcoin
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Bitcoin dominance metrics
    """
    if 'Bitcoin' not in crypto_data:
        raise ValueError("Bitcoin data required for dominance calculation")
    
    # Combine market cap data
    market_caps = {}
    for name, df in crypto_data.items():
        if 'Market Cap' in df.columns:
            market_caps[name] = df['Market Cap']
    
    market_cap_df = pd.DataFrame(market_caps)
    
    # Calculate total market cap and Bitcoin dominance
    dominance_df = pd.DataFrame(index=market_cap_df.index)
    dominance_df['Total_Market_Cap'] = market_cap_df.sum(axis=1)
    dominance_df['Bitcoin_Market_Cap'] = market_cap_df['Bitcoin']
    dominance_df['Altcoin_Market_Cap'] = dominance_df['Total_Market_Cap'] - dominance_df['Bitcoin_Market_Cap']
    dominance_df['Bitcoin_Dominance'] = (dominance_df['Bitcoin_Market_Cap'] / dominance_df['Total_Market_Cap']) * 100
    
    return dominance_df


def analyze_seasonal_patterns(df: pd.DataFrame, value_col: str = 'Daily_Return') -> Dict[str, pd.DataFrame]:
    """
    Analyze seasonal patterns in cryptocurrency data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data with date features
    value_col : str
        Column to analyze for seasonal patterns
        
    Returns:
    --------
    dict
        Dictionary containing seasonal analysis results
    """
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col} not found in data")
    
    results = {}
    
    # Monthly patterns
    monthly_stats = df.groupby('Month')[value_col].agg(['mean', 'std', 'count'])
    results['monthly'] = monthly_stats
    
    # Day of week patterns
    dow_stats = df.groupby('DayOfWeek')[value_col].agg(['mean', 'std', 'count'])
    results['day_of_week'] = dow_stats
    
    # Quarterly patterns
    quarterly_stats = df.groupby('Quarter')[value_col].agg(['mean', 'std', 'count'])
    results['quarterly'] = quarterly_stats
    
    # Yearly patterns
    yearly_stats = df.groupby('Year')[value_col].agg(['mean', 'std', 'count'])
    results['yearly'] = yearly_stats
    
    return results


def compare_cryptocurrencies(crypto_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare key metrics across cryptocurrencies.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
        
    Returns:
    --------
    pd.DataFrame
        Comparison table of key metrics
    """
    comparison_data = []
    
    for name, df in crypto_data.items():
        if df.empty:
            continue
            
        # Basic metrics
        metrics = {
            'Cryptocurrency': name,
            'Start_Date': df.index.min(),
            'End_Date': df.index.max(),
            'Days_Available': len(df),
            'Avg_Price': df['Close'].mean() if 'Close' in df.columns else np.nan,
            'Min_Price': df['Close'].min() if 'Close' in df.columns else np.nan,
            'Max_Price': df['Close'].max() if 'Close' in df.columns else np.nan,
            'Current_Price': df['Close'].iloc[-1] if 'Close' in df.columns else np.nan,
            'Total_Return': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100 if 'Close' in df.columns else np.nan,
            'Avg_Daily_Return': df['Daily_Return'].mean() * 100 if 'Daily_Return' in df.columns else np.nan,
            'Volatility': df['Daily_Return'].std() * np.sqrt(252) * 100 if 'Daily_Return' in df.columns else np.nan,
            'Current_Market_Cap': df['Market Cap'].iloc[-1] if 'Market Cap' in df.columns else np.nan,
            'Avg_Volume': df['Volume'].mean() if 'Volume' in df.columns else np.nan
        }
        
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df.set_index('Cryptocurrency')