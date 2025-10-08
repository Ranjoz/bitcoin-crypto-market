"""
Visualization Module
===================

Functions for creating visualizations of cryptocurrency data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Set style defaults
plt.style.use('default')
sns.set_palette("husl")


def plot_price_history(df: pd.DataFrame, crypto_name: str, 
                      save_path: Optional[str] = None) -> None:
    """
    Plot price history for a cryptocurrency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data
    crypto_name : str
        Name of the cryptocurrency
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Price plot
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5)
    if 'MA30' in df.columns:
        ax1.plot(df.index, df['MA30'], label='30-day MA', alpha=0.7)
    if 'MA90' in df.columns:
        ax1.plot(df.index, df['MA90'], label='90-day MA', alpha=0.7)
    
    ax1.set_title(f'{crypto_name} Price History', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume plot
    ax2.bar(df.index, df['Volume'], alpha=0.6, color='orange')
    ax2.set_title('Trading Volume', fontsize=14)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame,
                           save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Cryptocurrency Price Correlation Matrix', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_volatility_comparison(crypto_data: Dict[str, pd.DataFrame],
                             save_path: Optional[str] = None) -> None:
    """
    Plot volatility comparison across cryptocurrencies.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
    save_path : str, optional
        Path to save the plot
    """
    volatility_data = []
    
    for name, df in crypto_data.items():
        if 'Daily_Return' in df.columns:
            vol = df['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized %
            volatility_data.append({'Cryptocurrency': name, 'Volatility': vol})
    
    vol_df = pd.DataFrame(volatility_data)
    vol_df = vol_df.sort_values('Volatility', ascending=True)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(vol_df['Cryptocurrency'], vol_df['Volatility'])
    
    # Color gradient
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Cryptocurrency Volatility Comparison\n(Annualized)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Volatility (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(vol_df['Volatility']):
        plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bitcoin_dominance(dominance_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
    """
    Plot Bitcoin dominance over time.
    
    Parameters:
    -----------
    dominance_df : pd.DataFrame
        DataFrame with Bitcoin dominance data
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Market cap plot
    ax1.fill_between(dominance_df.index, 0, dominance_df['Bitcoin_Market_Cap'], 
                     alpha=0.7, label='Bitcoin Market Cap', color='orange')
    ax1.fill_between(dominance_df.index, dominance_df['Bitcoin_Market_Cap'], 
                     dominance_df['Total_Market_Cap'], 
                     alpha=0.7, label='Altcoin Market Cap', color='blue')
    
    ax1.set_title('Cryptocurrency Market Capitalization', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Market Cap (USD)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dominance plot
    ax2.plot(dominance_df.index, dominance_df['Bitcoin_Dominance'], 
             color='orange', linewidth=2)
    ax2.fill_between(dominance_df.index, dominance_df['Bitcoin_Dominance'], 
                     alpha=0.3, color='orange')
    
    ax2.set_title('Bitcoin Dominance (%)', fontsize=14)
    ax2.set_ylabel('Dominance (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_seasonal_patterns(seasonal_data: Dict[str, pd.DataFrame], 
                          pattern_type: str = 'monthly',
                          save_path: Optional[str] = None) -> None:
    """
    Plot seasonal patterns.
    
    Parameters:
    -----------
    seasonal_data : dict
        Dictionary with seasonal analysis results
    pattern_type : str
        Type of pattern to plot ('monthly', 'day_of_week', 'quarterly')
    save_path : str, optional
        Path to save the plot
    """
    if pattern_type not in seasonal_data:
        raise ValueError(f"Pattern type '{pattern_type}' not found in data")
    
    data = seasonal_data[pattern_type]
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(range(len(data)), data['mean'] * 100, 
                   yerr=data['std'] * 100, capsize=5, alpha=0.7)
    
    # Color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Customize labels based on pattern type
    if pattern_type == 'monthly':
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        title = 'Average Monthly Returns'
    elif pattern_type == 'day_of_week':
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        title = 'Average Returns by Day of Week'
    elif pattern_type == 'quarterly':
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        title = 'Average Quarterly Returns'
    else:
        labels = [str(i) for i in data.index]
        title = f'Seasonal Pattern: {pattern_type}'
    
    plt.xticks(range(len(data)), labels)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Average Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(data['mean'] * 100):
        plt.text(i, v + (0.1 if v >= 0 else -0.3), f'{v:.2f}%', 
                ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_interactive_price_chart(df: pd.DataFrame, crypto_name: str) -> go.Figure:
    """
    Create interactive price chart using Plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cryptocurrency data
    crypto_name : str
        Name of the cryptocurrency
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxis=True,
                       vertical_spacing=0.1,
                       subplot_titles=(f'{crypto_name} Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                  row=1, col=1)
    
    # Moving averages
    if 'MA30' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'],
                               name='30-day MA',
                               line=dict(color='orange', width=1)),
                      row=1, col=1)
    
    if 'MA90' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA90'],
                               name='90-day MA',
                               line=dict(color='red', width=1)),
                      row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        name='Volume',
                        marker_color='rgba(0,100,80,0.6)'),
                  row=2, col=1)
    
    fig.update_layout(
        title=f'{crypto_name} Interactive Chart',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig


def plot_multiple_crypto_comparison(crypto_data: Dict[str, pd.DataFrame],
                                   normalize: bool = True,
                                   save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple cryptocurrencies.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
    normalize : bool
        Whether to normalize prices to base 100
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    for name, df in crypto_data.items():
        if 'Close' in df.columns and not df.empty:
            prices = df['Close'].dropna()
            
            if normalize:
                # Normalize to base 100
                prices = (prices / prices.iloc[0]) * 100
                
            plt.plot(prices.index, prices, label=name, linewidth=2, alpha=0.8)
    
    plt.title('Cryptocurrency Price Comparison' + 
              (' (Normalized to Base 100)' if normalize else ''), 
              fontsize=16, fontweight='bold')
    plt.ylabel('Price' + (' (Base 100)' if normalize else ' (USD)'), fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log' if not normalize else 'linear')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()