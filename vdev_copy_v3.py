import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm
from plotly.subplots import make_subplots
import datetime
from sklearn.linear_model import LinearRegression
import time

# Set page configuration
st.set_page_config(
    page_title="Stock Pattern Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stProgress"] div[role="progressbar"] {
            border-radius: 10px; /* Adjust for more rounding */
            height: 15px !important; /* Adjust height */
    }
    h1, h2, h3 {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        height: 2.5rem;
        font-weight: 500;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .stDataFrame {
        border-radius: 5px;
        border: 1px solid #f0f2f6;
    }
    .css-18e3th9 {
        padding-top: 0.5rem;
    }
    .css-1kyxreq {
        justify-content: center;
        align-items: center;
    }
    .stAlert {
        border-radius: 4px;
    }
    .stSelectbox label, .stDateInput label {
        font-weight: 500;
    }
    .css-1v0mbdj {
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        font-weight: 600;
    }
    /* Remove extra padding */
    .css-12oz5g7 {
        padding-top: 1rem;
    }
    /* Tighten spacing in sidebar */
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Reduce spacing between widgets */
    .css-ocqkz7 {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_pattern' not in st.session_state:
    st.session_state.selected_pattern = None
if 'scan_cancelled' not in st.session_state:
    st.session_state.scan_cancelled = False
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'min_confidence': 0.7,
        'max_patterns': 5
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_cached(symbol, start_date, end_date):
    return fetch_stock_data(symbol, start_date, end_date)

@st.cache_data
def load_stock_symbols():
    try:
        with open("stock_symbols.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("stock_symbols.txt not found")
        return []
    
    

from scipy.signal import savgol_filter

from scipy.signal import savgol_filter
import yfinance as yf

def fetch_stock_data(symbol, start_date, end_date):
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_str, end=end_str)
        
        if df.empty or len(df) < 50:
            print(f"{symbol}: Insufficient data - {len(df)} rows")
            return None
            
        df = df.reset_index()
        df['Close_Smoothed'] = savgol_filter(df['Close'], window_length=11, polyorder=2)
        df = calculate_moving_average(df, window=50)
        df = calculate_rsi(df, window=50)
        
        # Debug: Check for NaN
        if pd.isna(df['MA'].iloc[-1]) or pd.isna(df['RSI'].iloc[-1]):
            print(f"{symbol}: MA or RSI is NaN - MA: {df['MA'].iloc[-1]}, RSI: {df['RSI'].iloc[-1]}")
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
        return None
    
def calculate_moving_average(df, window=50):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=50):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def find_extrema(df, order=5):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

def find_peaks(data, min_change=0.005):
    """
    Loosened peak detection: only checks immediate neighbors with a smaller min_change.
    """
    peaks = []
    for i in range(1, len(data) - 1):  # Reduced window from 2 to 1
        if (data['Close'].iloc[i] > data['Close'].iloc[i-1] * (1 + min_change) and 
            data['Close'].iloc[i] > data['Close'].iloc[i+1] * (1 + min_change)):
            peaks.append(i)
    return peaks

def find_valleys(data, min_change=0.005):
    """
    Loosened valley detection: only checks immediate neighbors with a smaller min_change.
    """
    valleys = []
    for i in range(1, len(data) - 1):  # Reduced window from 2 to 1
        if (data['Close'].iloc[i] < data['Close'].iloc[i-1] * (1 - min_change) and 
            data['Close'].iloc[i] < data['Close'].iloc[i+1] * (1 - min_change)):
            valleys.append(i)
    return valleys

import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from scipy.signal import argrelextrema

import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def detect_head_and_shoulders(data, min_peaks=3, min_troughs=2, sma_window=50, peak_order=5):
    """
    Detect Head and Shoulders patterns with improved logic.
    
    Args:
        data (pd.DataFrame): Stock data with 'Date' and 'Close'.
        min_peaks (int): Minimum number of peaks required.
        min_troughs (int): Minimum number of troughs required.
        sma_window (int): Window for SMA trend analysis.
        peak_order (int): Order for peak/trough detection.
        
    Returns:
        list: Detected H&S patterns with details.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
        raise ValueError("Input data must be a DataFrame with 'Close' column")
    
    if len(data) < sma_window:
        print("Not enough data for trend analysis")
        return []
    
    # Step 1: Trend analysis using SMA
    data = data.copy()
    sma = data['Close'].rolling(window=sma_window).mean()
    sma_slope = (sma.iloc[-1] - sma.iloc[0]) / len(sma)
    
    # Require significant upward trend
    if sma_slope <= 0:
        print("No prior uptrend detected")
        return []
    
    # Step 2: Detect peaks and troughs
    peaks = argrelextrema(data['Close'].values, np.greater, order=peak_order)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=peak_order)[0]
    
    if len(peaks) < min_peaks or len(troughs) < min_troughs:
        print(f"Not enough peaks ({len(peaks)}) or troughs ({len(troughs)}) for H&S")
        return []
    
    patterns = []
    
    # Step 3: Iterate through potential patterns
    for i in range(len(peaks) - 2):
        ls_idx = peaks[i]  # Left shoulder
        h_idx = peaks[i+1]  # Head
        rs_idx = peaks[i+2]  # Right shoulder
        
        # Validate peak progression
        if not (data['Close'].iloc[ls_idx] < data['Close'].iloc[h_idx] > data['Close'].iloc[rs_idx]):
            continue
            
        # Find troughs between peaks
        ls_troughs = [t for t in troughs if ls_idx < t < h_idx]
        rs_troughs = [t for t in troughs if h_idx < t < rs_idx]
        
        if not ls_troughs or not rs_troughs:
            continue
            
        # Take the highest trough between LS and H, and between H and RS
        t1_idx = ls_troughs[np.argmax(data['Close'].iloc[ls_troughs])]
        t2_idx = rs_troughs[np.argmax(data['Close'].iloc[rs_troughs])]
        
        # Neckline validation
        neckline_slope = (data['Close'].iloc[t2_idx] - data['Close'].iloc[t1_idx]) / (t2_idx - t1_idx)
        
        # Right shoulder should be similar height to left shoulder
        rs_price = data['Close'].iloc[rs_idx]
        ls_price = data['Close'].iloc[ls_idx]
        if abs(rs_price - ls_price) > 0.1 * ls_price:  # Within 10% of LS price
            continue
            
        # Find breakout point
        breakout_idx = None
        for j in range(rs_idx, min(rs_idx + 20, len(data))):
            neckline_price = data['Close'].iloc[t1_idx] + neckline_slope * (j - t1_idx)
            if data['Close'].iloc[j] < neckline_price * 0.98:  # 2% below neckline
                breakout_idx = j
                break
                
        # Calculate pattern metrics
        pattern_height = data['Close'].iloc[h_idx] - max(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
        target_price = (data['Close'].iloc[t1_idx] + neckline_slope * (rs_idx - t1_idx)) - pattern_height
        
        patterns.append({
            'left_shoulder': ls_idx,
            'head': h_idx,
            'right_shoulder': rs_idx,
            'neckline_trough1': t1_idx,
            'neckline_trough2': t2_idx,
            'breakout': breakout_idx,
            'neckline_slope': neckline_slope,
            'target_price': target_price,
            'pattern_height': pattern_height,
            'confidence': min(1.0, pattern_height / data['Close'].iloc[h_idx])  # Simple confidence metric
        })
    
    print(f"Total H&S patterns detected: {len(patterns)}")
    return patterns

def plot_head_and_shoulders(df, patterns, stock_name=""):
    """
    Enhanced plotting function for Head and Shoulders patterns.
    
    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        patterns (list): List of detected H&S patterns.
        stock_name (str): Name of the stock for title.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    if not patterns:
        print("No patterns to plot")
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'], 
        mode='lines', 
        name='Price', 
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    # Color cycle for multiple patterns
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, pattern in enumerate(patterns):
        color = colors[i % len(colors)]
        
        # Get key points
        ls_idx = pattern['left_shoulder']
        h_idx = pattern['head']
        rs_idx = pattern['right_shoulder']
        t1_idx = pattern['neckline_trough1']
        t2_idx = pattern['neckline_trough2']
        breakout_idx = pattern['breakout']
        
        # Neckline calculation
        neckline_x = [df['Date'].iloc[t1_idx], df['Date'].iloc[t2_idx]]
        neckline_y = [df['Close'].iloc[t1_idx], df['Close'].iloc[t2_idx]]
        
        # Extend neckline for plotting
        extended_neckline_x = [
            df['Date'].iloc[max(0, t1_idx - 5)],
            df['Date'].iloc[min(len(df)-1, t2_idx + 5)]
        ]
        extended_neckline_y = [
            df['Close'].iloc[t1_idx] - 5 * pattern['neckline_slope'],
            df['Close'].iloc[t2_idx] + 5 * pattern['neckline_slope']
        ]
        
        # Plot neckline
        fig.add_trace(go.Scatter(
            x=extended_neckline_x,
            y=extended_neckline_y,
            mode='lines',
            line=dict(color=color, dash='dash', width=1.5),
            name=f'Neckline {i+1}'
        ), row=1, col=1)
        
        # Plot key points
        for point, label in zip([ls_idx, h_idx, rs_idx], ['LS', 'H', 'RS']):
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[point]],
                y=[df['Close'].iloc[point]],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=label,
                textposition='top center',
                showlegend=False
            ), row=1, col=1)
        
        # Plot breakout if exists
        if breakout_idx:
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[breakout_idx]],
                y=[df['Close'].iloc[breakout_idx]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='x'),
                name=f'Breakout {i+1}',
            ), row=1, col=1)
            
            # Plot target line
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[breakout_idx], df['Date'].iloc[-1]],
                y=[pattern['target_price'], pattern['target_price']],
                mode='lines',
                line=dict(color=color, dash='dot', width=1),
                name=f'Target {i+1}'
            ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color='#7f7f7f', opacity=0.6)
    ), row=2, col=1)
    
    # Layout
    # Layout
    fig.update_layout(
        title=f'Head & Shoulders Patterns: {stock_name}',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1
        )
    )
    
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    return fig
def detect_double_bottom(df, order=5, tolerance=0.05, min_pattern_length=10, max_patterns=3):
    """
    Detect Double Bottom patterns with a prior downtrend, similar bottoms, and breakout confirmation.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', and 'Volume'.
        order (int): Window size for peak/trough detection.
        tolerance (float): Max % difference between bottom prices (increased to 5%).
        min_pattern_length (int): Min days between bottoms.
        max_patterns (int): Max number of patterns to return.

    Returns:
        list: Detected Double Bottom patterns.
    """
    # Detect troughs and peaks
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    patterns = []

    if len(troughs) < 2 or len(peaks) < 1:
        print(f"Not enough troughs ({len(troughs)}) or peaks ({len(peaks)}) for Double Bottom")
        return patterns

    print(f"Troughs: {len(troughs)}, Peaks: {len(peaks)}, Data length: {len(df)}")

    for i in range(len(troughs) - 1):
        # Step 1: Identify First Bottom
        trough1_idx = troughs[i]
        price1 = df['Close'].iloc[trough1_idx]

        # Validate prior downtrend (simple slope check over 20 days)
        downtrend_lookback = 20
        if trough1_idx < downtrend_lookback:
            continue
        prior_data = df['Close'].iloc[trough1_idx - downtrend_lookback:trough1_idx]
        if prior_data.iloc[-1] >= prior_data.iloc[0]:  # Not a downtrend
            print(f"Trough {trough1_idx}: No prior downtrend")
            continue

        # Step 2: Find Temporary High (Neckline) after First Bottom
        between_peaks = [p for p in peaks if trough1_idx < p < troughs[i + 1]]
        if not between_peaks:
            continue
        neckline_idx = max(between_peaks, key=lambda p: df['Close'].iloc[p])
        neckline_price = df['Close'].iloc[neckline_idx]

        # Step 3: Identify Second Bottom
        trough2_idx = troughs[i + 1]
        if trough2_idx - trough1_idx < min_pattern_length:
            print(f"Troughs {trough1_idx}-{trough2_idx}: Too close ({trough2_idx - trough1_idx} days)")
            continue

        price2 = df['Close'].iloc[trough2_idx]

        # Check if bottoms are at similar levels (within 5% tolerance)
        if abs(price1 - price2) / min(price1, price2) > tolerance:
            print(f"Troughs {trough1_idx}-{trough2_idx}: Bottoms not similar ({price1:.2f} vs {price2:.2f})")
            continue

        # Step 4: Confirm Breakout
        breakout_idx = None
        for idx in range(trough2_idx, min(len(df), trough2_idx + 30)):  # Look forward 30 days
            if df['Close'].iloc[idx] > neckline_price:
                breakout_idx = idx
                break

        # Calculate pattern metrics
        pattern_height = neckline_price - min(price1, price2)
        target_price = neckline_price + pattern_height

        # Confidence based on breakout and similarity
        confidence = 0.5  # Base confidence
        if breakout_idx:
            confidence += 0.3  # Boost for breakout
        confidence += (1 - abs(price1 - price2) / min(price1, price2) / tolerance) * 0.2  # Similarity bonus

        # Step 5: Store pattern (no strict confidence filter here)
        patterns.append({
            'trough1': trough1_idx,
            'trough2': trough2_idx,
            'neckline': neckline_idx,
            'neckline_price': neckline_price,
            'breakout': breakout_idx,
            'target': target_price,
            'pattern_height': pattern_height,
            'trough_prices': (price1, price2),
            'confidence': min(0.99, confidence),
            'status': 'confirmed' if breakout_idx else 'forming'
        })
        print(f"Pattern detected: T1 {df['Date'].iloc[trough1_idx]}, T2 {df['Date'].iloc[trough2_idx]}, Neckline {df['Date'].iloc[neckline_idx]}, Confidence: {confidence:.2f}")

    # Sort by confidence and limit to max_patterns
    patterns = sorted(patterns, key=lambda x: -x['confidence'])[:max_patterns]
    print(f"Total Double Bottom patterns detected: {len(patterns)}")
    return patterns

def plot_double_bottom(df, pattern_points, stock_name=""):
    """
    Plot Double Bottom patterns with clear W formation, neckline, breakout, and target.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        pattern_points (list): List of detected Double Bottom patterns.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Price", line=dict(color='#1E88E5')),
                  row=1, col=1)

    for idx, pattern in enumerate(pattern_points):
        color = '#E91E63'  # Consistent color for simplicity

        # Key points
        trough1_idx = pattern['trough1']
        trough2_idx = pattern['trough2']
        neckline_idx = pattern['neckline']
        trough1_date = df['Date'].iloc[trough1_idx]
        trough2_date = df['Date'].iloc[trough2_idx]
        neckline_date = df['Date'].iloc[neckline_idx]
        neckline_price = pattern['neckline_price']
        breakout_date = df['Date'].iloc[pattern['breakout']] if pattern['breakout'] else None

        # Find the peak between the two troughs
        start_idx = trough1_idx
        end_idx = trough2_idx
        segment = df.iloc[start_idx:end_idx+1]
        peak_idx = segment['Close'].idxmax()
        peak_date = df['Date'].iloc[peak_idx]
        peak_price = df['Close'].iloc[peak_idx]

        # Plot the W formation (connecting lines)
        fig.add_trace(go.Scatter(
            x=[trough1_date, peak_date, trough2_date],
            y=[pattern['trough_prices'][0], peak_price, pattern['trough_prices'][1]],
            mode='lines',
            line=dict(color=color, width=2),
            name=f'W Formation ({idx+1})',
            showlegend=False
        ), row=1, col=1)

        # Plot troughs
        fig.add_trace(go.Scatter(x=[trough1_date], y=[pattern['trough_prices'][0]], mode="markers+text",
                                text=["B1"], textposition="bottom center", marker=dict(size=12, color=color),
                                name=f"Bottom 1 ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[trough2_date], y=[pattern['trough_prices'][1]], mode="markers+text",
                                text=["B2"], textposition="bottom center", marker=dict(size=12, color=color),
                                name=f"Bottom 2 ({idx+1})"), row=1, col=1)

        # Plot neckline (extended)
        neckline_start_date = df['Date'].iloc[max(0, trough1_idx - 5)]  # Extend left
        neckline_end_date = df['Date'].iloc[min(len(df)-1, trough2_idx + 5)]  # Extend right
        fig.add_trace(go.Scatter(
            x=[neckline_start_date, neckline_end_date], 
            y=[neckline_price, neckline_price], 
            mode="lines",
            line=dict(color=color, dash='dash'), 
            name=f"Neckline ({idx+1})"
        ), row=1, col=1)

        # Plot breakout and target
        if breakout_date:
            fig.add_trace(go.Scatter(x=[breakout_date], y=[df['Close'].iloc[pattern['breakout']]],
                                    mode="markers+text", text=["Breakout"], textposition="top center",
                                    marker=dict(size=12, color=color), name=f"Breakout ({idx+1})"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[breakout_date, df['Date'].iloc[-1]], y=[pattern['target']] * 2,
                                    mode="lines", line=dict(color=color, dash='dot'), name=f"Target ({idx+1})"),
                          row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker=dict(color='#26A69A', opacity=0.7)),
                  row=2, col=1)

    fig.update_layout(
        title=f"Double Bottom Patterns for {stock_name}", 
        height=800, 
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1
        )
    )
    return fig
    
def detect_cup_and_handle(df, order=10, cup_min_bars=20, handle_max_retrace=0.5):
    """
    Detect Cup and Handle patterns with relaxed constraints for better detection.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        order (int): Window size for peak/trough detection.
        cup_min_bars (int): Minimum bars for cup duration (reduced to 20).
        handle_max_retrace (float): Max handle retracement (increased to 0.5).

    Returns:
        list: Detected Cup and Handle patterns.
    """
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    patterns = []

    if len(peaks) < 2 or len(troughs) < 1:
        print(f"Not enough peaks ({len(peaks)}) or troughs ({len(troughs)}) for Cup and Handle")
        return patterns

    print(f"Peaks: {len(peaks)}, Troughs: {len(troughs)}, Data length: {len(df)}")

    for i in range(len(peaks) - 1):
        # Step 1: Relaxed Uptrend Precondition (30 days prior to left rim)
        left_peak_idx = peaks[i]
        uptrend_lookback = 30
        if left_peak_idx < uptrend_lookback:
            continue
        prior_data = df['Close'].iloc[left_peak_idx - uptrend_lookback:left_peak_idx]
        if prior_data.iloc[-1] <= prior_data.iloc[0] * 1.05:  # Allow 5% flatness
            print(f"Peak {left_peak_idx}: No prior uptrend")
            continue

        # Step 2: Detect Cup Formation
        cup_troughs = [t for t in troughs if t > left_peak_idx]
        if not cup_troughs:
            continue
        cup_bottom_idx = cup_troughs[0]
        right_peaks = [p for p in peaks if p > cup_bottom_idx]
        if not right_peaks:
            continue
        right_peak_idx = right_peaks[0]

        if right_peak_idx - left_peak_idx < cup_min_bars:
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Cup too short ({right_peak_idx - left_peak_idx} bars)")
            continue

        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]

        # Validate cup: Rims within 10% and depth 20%-60% of uptrend move
        if abs(right_peak_price - left_peak_price) / left_peak_price > 0.10:
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Rims not similar ({left_peak_price:.2f} vs {right_peak_price:.2f})")
            continue

        uptrend_move = left_peak_price - prior_data.min()
        cup_height = left_peak_price - cup_bottom_price
        cup_depth_ratio = cup_height / uptrend_move
        if not (0.2 <= cup_depth_ratio <= 0.6):
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Cup depth invalid ({cup_depth_ratio:.2%})")
            continue

        # Step 3: Detect Handle Formation
        handle_troughs = [t for t in troughs if t > right_peak_idx]
        if not handle_troughs:
            continue
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]

        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            print(f"Handle {right_peak_idx}-{handle_bottom_idx}: Retrace too deep ({handle_retrace:.2%})")
            continue

        # Find handle end
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, len(df)):
            if df['Close'].iloc[j] >= right_peak_price * 0.98:
                handle_end_idx = j
                break
        if not handle_end_idx:
            continue

        # Step 4: Confirm Breakout
        breakout_idx = None
        for j in range(handle_end_idx, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 1.02:
                breakout_idx = j
                break

        # Step 5: Calculate Metrics
        target_price = right_peak_price + cup_height
        confidence = 0.6
        if breakout_idx:
            confidence += 0.3
        confidence += (1 - abs(left_peak_price - right_peak_price) / left_peak_price / 0.10) * 0.1

        patterns.append({
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target': target_price,
            'cup_height': cup_height,
            'confidence': min(0.99, confidence),
            'status': 'confirmed' if breakout_idx else 'forming'
        })
        print(f"Pattern detected: Left {df['Date'].iloc[left_peak_idx]}, Bottom {df['Date'].iloc[cup_bottom_idx]}, Right {df['Date'].iloc[right_peak_idx]}, Handle {df['Date'].iloc[handle_bottom_idx]}, Confidence: {confidence:.2f}")

    print(f"Total Cup and Handle patterns detected: {len(patterns)}")
    return patterns

def plot_cup_and_handle(df, pattern_points, stock_name=""):
    """
    Plot Cup and Handle patterns with a curved cup line, distinct handle color, resistance, and breakout.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        pattern_points (list): List of detected Cup and Handle patterns.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Price", line=dict(color='#1E88E5')),
                  row=1, col=1)

    for idx, pattern in enumerate(pattern_points):
        cup_color = '#9C27B0'  # Purple for the cup
        handle_color = '#FF9800'  # Orange for the handle

        # Extract points
        left_peak_date = df['Date'].iloc[pattern['left_peak']]
        cup_bottom_date = df['Date'].iloc[pattern['cup_bottom']]
        right_peak_date = df['Date'].iloc[pattern['right_peak']]
        handle_bottom_date = df['Date'].iloc[pattern['handle_bottom']]
        handle_end_date = df['Date'].iloc[pattern['handle_end']]
        breakout_date = df['Date'].iloc[pattern['breakout']] if pattern['breakout'] else None

        # Plot key points
        fig.add_trace(go.Scatter(x=[left_peak_date], y=[df['Close'].iloc[pattern['left_peak']]],
                                mode="markers+text", text=["Left Rim"], textposition="top right",
                                marker=dict(size=12, color=cup_color), name=f"Left Rim ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[cup_bottom_date], y=[df['Close'].iloc[pattern['cup_bottom']]],
                                mode="markers+text", text=["Bottom"], textposition="bottom center",
                                marker=dict(size=12, color=cup_color), name=f"Bottom ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[right_peak_date], y=[df['Close'].iloc[pattern['right_peak']]],
                                mode="markers+text", text=["Right Rim"], textposition="top left",
                                marker=dict(size=12, color=cup_color), name=f"Right Rim ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[handle_bottom_date], y=[df['Close'].iloc[pattern['handle_bottom']]],
                                mode="markers+text", text=["Handle Low"], textposition="bottom right",
                                marker=dict(size=12, color=handle_color), name=f"Handle Low ({idx+1})"), row=1, col=1)

        # Create a smooth curve for the cup (left rim -> bottom -> right rim)
        num_points = 50  # Number of points for the curve
        cup_dates = [left_peak_date, cup_bottom_date, right_peak_date]
        cup_prices = [df['Close'].iloc[pattern['left_peak']], df['Close'].iloc[pattern['cup_bottom']],
                      df['Close'].iloc[pattern['right_peak']]]

        # Convert dates to numeric for interpolation
        cup_dates_numeric = [(d - cup_dates[0]).days for d in cup_dates]
        t = np.linspace(0, 1, num_points)
        t_orig = [0, 0.5, 1]  # Normalized positions of left rim, bottom, right rim

        # Quadratic interpolation for smooth curve
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_orig, cup_dates_numeric, kind='quadratic')
        interp_dates_numeric = interp_func(t)
        interp_prices = interp1d(t_orig, cup_prices, kind='quadratic')(t)

        # Convert numeric dates back to datetime
        interp_dates = [cup_dates[0] + pd.Timedelta(days=d) for d in interp_dates_numeric]

        fig.add_trace(go.Scatter(x=interp_dates, y=interp_prices, mode="lines",
                                line=dict(color=cup_color, width=2, dash='dot'), name=f"Cup Curve ({idx+1})"),
                      row=1, col=1)

        # Plot handle with a different color
        handle_x = df['Date'].iloc[pattern['right_peak']:pattern['handle_end'] + 1]
        handle_y = df['Close'].iloc[pattern['right_peak']:pattern['handle_end'] + 1]
        fig.add_trace(go.Scatter(x=handle_x, y=handle_y, mode="lines",
                                line=dict(color=handle_color, width=2), name=f"Handle ({idx+1})"), row=1, col=1)

        # Plot resistance line
        fig.add_trace(go.Scatter(x=[left_peak_date, df['Date'].iloc[-1]], y=[pattern['resistance']] * 2,
                                mode="lines", line=dict(color=cup_color, dash='dash'), name=f"Resistance ({idx+1})"),
                      row=1, col=1)

        # Plot breakout and target
        if breakout_date:
            fig.add_trace(go.Scatter(x=[breakout_date], y=[df['Close'].iloc[pattern['breakout']]],
                                    mode="markers+text", text=["Breakout"], textposition="top center",
                                    marker=dict(size=12, color=handle_color), name=f"Breakout ({idx+1})"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[breakout_date, df['Date'].iloc[-1]], y=[pattern['target']] * 2,
                                    mode="lines", line=dict(color=handle_color, dash='dot'), name=f"Target ({idx+1})"),
                          row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker=dict(color='#26A69A', opacity=0.7)),
                  row=2, col=1)

    fig.update_layout(
        title=f"Cup and Handle Patterns for {stock_name}",
        height=800,
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1)
    )
    return fig

def plot_pattern(df, pattern_points, pattern_name, stock_name=""):
    plotters = {
        'Double Bottom': plot_double_bottom,
        'Cup and Handle': plot_cup_and_handle,
        'Head and Shoulders': plot_head_and_shoulders
    }
    
    if pattern_name not in plotters:
        raise ValueError(f"Unsupported pattern type: {pattern_name}. Available: {list(plotters.keys())}")
    
    return plotters[pattern_name](df, pattern_points, stock_name)

def evaluate_pattern_detection(df, patterns, look_forward_window=10):
    """
    Evaluate the performance of detected patterns and calculate metrics per pattern type.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        patterns (dict): Dictionary of pattern types and their detected instances.
        look_forward_window (int): Number of bars to look forward for evaluation.
    
    Returns:
        dict: Metrics (Accuracy, Precision, Recall, F1) for each pattern type.
    """
    metrics = {}
    
    for pattern_type, pattern_list in patterns.items():
        TP = 0  # True Positives: Correctly predicted direction
        FP = 0  # False Positives: Incorrectly predicted direction
        FN = 0  # False Negatives: Missed patterns (approximated)
        TN = 0  # True Negatives: Correctly identified no pattern (approximated)

        if not pattern_list:
            metrics[pattern_type] = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Total Patterns": 0,
                "Correct Predictions": 0
            }
            continue

        total_patterns = len(pattern_list)

        for pattern in pattern_list:
            # Determine the last point of the pattern
            if pattern_type == "Head and Shoulders":
                last_point_idx = df.index.get_loc(max(pattern['left_shoulder'], pattern['head'], pattern['right_shoulder']))
            elif pattern_type == "Double Bottom":
                last_point_idx = max(pattern['trough1'], pattern['trough2'])
            elif pattern_type == "Cup and Handle":
                last_point_idx = pattern['handle_end']
            else:
                continue

            if last_point_idx + look_forward_window >= len(df):
                FN += 1  # Not enough data to evaluate
                continue

            last_price = df['Close'].iloc[last_point_idx]
            future_price = df['Close'].iloc[last_point_idx + look_forward_window]

            # Evaluate based on pattern type
            if pattern_type == "Head and Shoulders":  # Bearish
                if future_price < last_price:
                    TP += 1
                else:
                    FP += 1
            elif pattern_type in ["Double Bottom", "Cup and Handle"]:  # Bullish
                if future_price > last_price:
                    TP += 1
                else:
                    FP += 1

        # Approximate FN and TN (simplified approach)
        total_periods = len(df) - look_forward_window
        non_pattern_periods = total_periods - total_patterns
        FN = max(0, total_patterns - (TP + FP))
        TN = max(0, non_pattern_periods - FP)

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[pattern_type] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Total Patterns": total_patterns,
            "Correct Predictions": TP
        }

    return metrics

def is_trading_day(date):
    """Check if the given date is a trading day (Monday to Friday)."""
    return date.weekday() < 5 

def get_nearest_trading_day(date):
    """Adjust the date to the nearest previous trading day if it's a weekend or holiday."""
    while not is_trading_day(date):
        date -= datetime.timedelta(days=1)
    return date

# def forecast_future_prices(df, forecast_days=30):
#     """Forecast future prices using linear regression."""
#     # Prepare data for linear regression
#     X = np.array(range(len(df))).reshape(-1, 1)
#     y = df['Close'].values
    
#     # Train the model
#     model = LinearRegression()
#     model.fit(X, y)
    
#     # Predict future prices
#     future_X = np.array(range(len(df), len(df) + forecast_days)).reshape(-1, 1)
#     future_prices = model.predict(future_X)
    
#     # Create a DataFrame for the future data
#     future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days + 1, freq='B')[1:]
#     future_df = pd.DataFrame({
#         'Date': future_dates,
#         'Close': future_prices
#     })
    
#     # Combine historical and future data
#     combined_df = pd.concat([df, future_df], ignore_index=True)
    
#     return combined_df

def main():
    # App header with title
    st.markdown("# Stock Pattern Scanner (Yahoo Finance)")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <style>
            .block-container {
                margin-bottom: 0.5rem; /* Adjust this value to change the spacing */
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## ⚙️ Scanner Settings")
        
        # Date selection
        st.markdown("### 📅 Date Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date(2023, 1, 1),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.date(2024, 1, 1),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31)
            )

        if end_date < start_date:
            st.error("End Date must be after Start Date.")
            st.stop()
        if (end_date - start_date).days < 270:  # 9 months
            st.error("Date range must be at least 9 months for Head and Shoulders detection.")
            st.stop()
        if (end_date - start_date).days < 30:
            st.error("Date range must be at least 30 days")
            st.stop()

        if (end_date - start_date).days > 365 * 3:
            st.warning("Large date range may impact performance. Consider narrowing your range.")

        # Scan button
        scan_button = st.button("Scan the Stock", use_container_width=True)

    # Main content
    if 'scan_cancelled' not in st.session_state:
        st.session_state.scan_cancelled = False

    if scan_button:
        cancel_button = st.sidebar.button("Cancel Scan")
        if cancel_button:
            st.session_state.scan_cancelled = True
            st.warning("Scan cancelled by user")
            st.stop()

        try:
            with open("stock_symbols.txt", "r") as f:
                stock_symbols = [line.strip() for line in f]
        except FileNotFoundError:
            st.error("stock_symbols.txt not found. Please create the file with stock symbols, one per line.")
            return
        except Exception as e: 
            st.error(f"An error occurred while reading the stock symbols file: {e}")
            return
            
        with st.container():
            st.markdown("## 🔍 Scanning Stocks")
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_container = st.empty()

            stock_data = []
            for i, symbol in enumerate(stock_symbols):
                if st.session_state.scan_cancelled:
                    break

                status_container.info(f"Processing {symbol}... ({i+1}/{len(stock_symbols)})")

                try:
                    df = fetch_stock_data(symbol, start_date, end_date)
                    if df is None or df.empty:
                        print(f"{symbol}: No data returned")
                        continue
                    
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    patterns = {
                        "Head and Shoulders": detect_head_and_shoulders(df),
                        "Double Bottom": detect_double_bottom(df),
                        "Cup and Handle": detect_cup_and_handle(df),
                    }
                    
                    # Debug patterns detected
                    print(f"{symbol} - H&S patterns: {len(patterns['Head and Shoulders'])}")
                    
                    pattern_metrics = evaluate_pattern_detection(df, patterns)
                    
                    stock_info = yf.Ticker(symbol).info
                    current_price = stock_info.get('currentPrice', None)
                    volume = stock_info.get('volume', None)
                    percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    
                    # No confidence filter for simplicity
                    stock_data.append({
                        "Symbol": symbol,
                        "Patterns": patterns,
                        "Pattern Metrics": pattern_metrics,
                        "Data": df,
                        "Current Price": current_price,
                        "Volume": volume,
                        "Percent Change": percent_change,
                        "MA": df['MA'].iloc[-1] if 'MA' in df.columns and not pd.isna(df['MA'].iloc[-1]) else None,
                        "RSI": df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None,
                    })
                
                except Exception as e:
                    st.error(f"Error processing {symbol}: {e}")
                    continue
                
                # Update progress bar during actual scanning
                progress_bar.progress((i + 1) / len(stock_symbols))
            
            st.session_state.stock_data = stock_data
            st.session_state.selected_stock = None
            st.session_state.selected_pattern = None
            
            progress_container.empty()
            status_container.success("Scan completed successfully!")

    # Display results
    if 'stock_data' in st.session_state and st.session_state.stock_data:
        st.markdown("## 📊 Scan Results")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Stock List", "📈 Pattern Visualization"])
        
        with tab1:
            # Prepare data for the table
            table_data = []
            for stock in st.session_state.stock_data:
                pattern_counts = {
                    pattern: len(stock["Patterns"][pattern]) 
                    for pattern in stock["Patterns"]
                }
                
                row = {
                    "Symbol": stock["Symbol"],
                    "Current Price": stock['Current Price'] if stock['Current Price'] else None,
                    "Volume": stock['Volume'] if stock['Volume'] else None,
                    "% Change": stock['Percent Change'],
                    "MA (50)": stock['MA'],
                    "RSI (14)": stock['RSI'],
                    "Head and Shoulders": pattern_counts.get("Head and Shoulders", 0),
                    "Double Bottom": pattern_counts.get("Double Bottom", 0),
                    "Cup and Handle": pattern_counts.get("Cup and Handle", 0),
                }
                table_data.append(row)
            
            df_table = pd.DataFrame(table_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                pattern_filter = st.selectbox("Filter by Pattern", 
                                             ["All Patterns", "Head and Shoulders", "Double Bottom", "Cup and Handle"])
            
            filtered_df = df_table.copy()
            if pattern_filter != "All Patterns":
                filtered_df = filtered_df[filtered_df[pattern_filter] > 0]
            
            formatted_df = filtered_df.copy()
            formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Volume"] = formatted_df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")
            formatted_df["% Change"] = formatted_df["% Change"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            formatted_df["MA (50)"] = formatted_df["MA (50)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["RSI (14)"] = formatted_df["RSI (14)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(
                formatted_df,
                height=500,
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol"),
                    "Current Price": st.column_config.TextColumn("Price"),
                    "Volume": st.column_config.TextColumn("Volume"),
                    "% Change": st.column_config.TextColumn("Change (%)"),
                    "MA (50)": st.column_config.TextColumn("MA (50)"),
                    "RSI (14)": st.column_config.TextColumn("RSI (14)"),
                    "Head and Shoulders": st.column_config.NumberColumn("H&S", format="%d"),
                    "Double Bottom": st.column_config.NumberColumn("Double Bottom", format="%d"),
                    "Cup and Handle": st.column_config.NumberColumn("Cup & Handle", format="%d"),
                }
            )
            
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", len(filtered_df))
            with col2:
                total_patterns = filtered_df[["Head and Shoulders", "Double Bottom", "Cup and Handle"]].sum().sum()
                st.metric("Total Patterns", int(total_patterns))
        
        with tab2:
            st.markdown("### 🔍 Pattern Visualization")
            
            col1, col2 = st.columns(2)
            with col1:
                def format_stock_name(stock):
                    pattern_counts = {
                        'HS': len(stock['Patterns']['Head and Shoulders']),
                        'DB': len(stock['Patterns']['Double Bottom']),
                        'CH': len(stock['Patterns']['Cup and Handle'])
                    }
                    return f"{stock['Symbol']} [HS:{pattern_counts['HS']}, DB:{pattern_counts['DB']}, CH:{pattern_counts['CH']}]"
                
                selected_display = st.selectbox(
                    "Select Stock",
                    options=[format_stock_name(stock) for stock in st.session_state.stock_data],
                    key='stock_select'
                )
                
                selected_stock = selected_display.split(' [')[0]
            
            if selected_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = selected_stock
                st.session_state.selected_pattern = None
            
            selected_data = next((item for item in st.session_state.stock_data 
                                if item["Symbol"] == st.session_state.selected_stock), None)
            
            if selected_data:
                pattern_options = []
                for pattern_name, patterns in selected_data["Patterns"].items():
                    if patterns:
                        count = len(patterns)
                        display_name = f"{pattern_name} ({count})"
                        pattern_options.append((pattern_name, display_name))
                
                if pattern_options:
                    with col2:
                        selected_display = st.selectbox(
                            "Select Pattern",
                            options=[display for _, display in pattern_options],
                            key='pattern_select'
                        )
                        
                        selected_pattern = next(
                            pattern for pattern, display in pattern_options 
                            if display == selected_display
                        )
                    
                    if selected_pattern != st.session_state.selected_pattern:
                        st.session_state.selected_pattern = selected_pattern
                    
                    if st.session_state.selected_pattern:
                        pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                        
                        st.markdown("### 📊 Stock Metrics")
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric(
                                "Current Price", 
                                f"${selected_data['Current Price']:.2f}" if selected_data['Current Price'] else "N/A",
                                f"{selected_data['Percent Change']:.2f}%"
                            )
                        with metric_cols[1]:
                            st.metric("Volume", f"{int(selected_data['Volume']):,}" if selected_data['Volume'] else "N/A")
                        with metric_cols[2]:
                            st.metric("MA (50)", f"{selected_data['MA']:.2f}" if selected_data['MA'] else "N/A")
                        with metric_cols[3]:
                            st.metric("RSI (14)", f"{selected_data['RSI']:.2f}" if selected_data['RSI'] else "N/A")
                        
                        st.markdown("### 📈 Pattern Chart")
                        if not isinstance(pattern_points, list):
                            pattern_points = [pattern_points]
                        
                        st.plotly_chart(
                            plot_pattern(
                                selected_data["Data"],
                                pattern_points,
                                st.session_state.selected_pattern,
                                stock_name=selected_data["Symbol"]
                            ),
                            use_container_width=True,
                            height=600
                        )
                        
                        # Add pattern details (simplified)
                        with st.expander("📚 Pattern Details"):
                            if st.session_state.selected_pattern == "Head and Shoulders":
                                st.markdown("**Head and Shoulders**: Bearish reversal pattern with a peak (head) between two lower peaks (shoulders).")
                            elif st.session_state.selected_pattern == "Double Bottom":
                                st.markdown("**Double Bottom**: Bullish reversal pattern with two troughs at similar levels.")
                            elif st.session_state.selected_pattern == "Cup and Handle":
                                st.markdown("**Cup and Handle**: Bullish continuation pattern with a rounded bottom and consolidation.")
                else:
                    st.info(f"No patterns detected for {selected_stock}.")
            else:
                st.error("Selected stock data not found.")

if __name__ == "__main__":
    main()
