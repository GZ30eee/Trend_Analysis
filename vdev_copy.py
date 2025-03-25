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
    
    

def fetch_stock_data(symbol, start_date, end_date):
    try:
        # Convert dates to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_str, end=end_str)
        
        if df.empty:
            st.warning(f"No data found for {symbol} in the selected date range")
            return None
            
        # Reset index and ensure Date column exists
        df = df.reset_index()
        if 'Date' not in df.columns:
            st.error(f"Date column missing for {symbol}")
            return None
            
        # Handle missing data
        if df.isnull().values.any():
            df = df.ffill().bfill()
            
        # Forecast and indicators
        # df = forecast_future_prices(df, forecast_days)
        df = calculate_moving_average(df)
        df = calculate_rsi(df)
        
        return df
    
    except yf.YFinanceError as yf_error:
        st.error(f"Yahoo Finance error for {symbol}: {str(yf_error)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching {symbol}: {str(e)}")
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

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

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

def detect_head_and_shoulders(data, 
                              is_inverse=False,
                              tolerance=0.15,              # Loosened from 0.08
                              min_pattern_length=5,        # Loosened from 8
                              volume_ratio_head=1.1,       # Loosened from 1.2
                              volume_ratio_breakout=1.05,  # Loosened from 1.1
                              time_symmetry_threshold=0.8, # Loosened from 0.5
                              neckline_slope_threshold=0.01, # Loosened from 0.005
                              min_trend_strength=0.0001,   # Loosened from 0.0005
                              breakout_lookahead=15,
                              breakout_confirmation_bars=1): # Loosened from 2
    peaks = find_peaks(data) if not is_inverse else find_valleys(data)
    valleys = find_valleys(data) if not is_inverse else find_peaks(data)
    patterns = []
    
    if len(peaks) < 3:
        return patterns
    
    for i in range(1, len(peaks) - 1):
        LS, H, RS = peaks[i-1], peaks[i], peaks[i+1]
        
        # Basic structure check (unchanged)
        if not is_inverse:
            if not (data['Close'].iloc[LS] < data['Close'].iloc[H] > data['Close'].iloc[RS]):
                continue
        else:
            if not (data['Close'].iloc[LS] > data['Close'].iloc[H] < data['Close'].iloc[RS]):
                continue
        
        # Shoulder symmetry (looser tolerance)
        shoulder_diff = abs(data['Close'].iloc[LS] - data['Close'].iloc[RS]) / max(data['Close'].iloc[LS], data['Close'].iloc[RS])
        if shoulder_diff > tolerance:
            continue
        
        # Time symmetry (looser threshold)
        left_time = H - LS
        right_time = RS - H
        time_diff = abs(left_time - right_time) / max(left_time, right_time)
        if time_diff > time_symmetry_threshold:
            continue
        
        # Minimum pattern duration (looser)
        if (RS - LS) < min_pattern_length:
            continue
        
        # Find neckline points
        valley1 = data['Close'].iloc[LS:H].idxmin() if not is_inverse else data['Close'].iloc[LS:H].idxmax()
        valley2 = data['Close'].iloc[H:RS].idxmin() if not is_inverse else data['Close'].iloc[H:RS].idxmax()
        T1 = data.index.get_loc(valley1)
        T2 = data.index.get_loc(valley2)
        
        if pd.isna(valley1) or pd.isna(valley2):
            continue
        
        # Neckline slope (looser threshold)
        neckline_slope = (data['Close'].iloc[T2] - data['Close'].iloc[T1]) / (T2 - T1)
        if abs(neckline_slope) > neckline_slope_threshold:
            continue
        
        # Volume analysis (looser ratios, optional)
        if 'Volume' in data.columns:
            head_vol = data['Volume'].iloc[H]
            left_vol = data['Volume'].iloc[LS]
            right_vol = data['Volume'].iloc[RS]
            avg_prior_vol = data['Volume'].iloc[max(0, LS-10):LS].mean()
            vol_head_ok = head_vol > avg_prior_vol * volume_ratio_head or head_vol > right_vol  # OR instead of AND
        else:
            vol_head_ok = True
        
        # Prior trend validation (looser, optional)
        trend_lookback = min(30, LS)
        if trend_lookback >= 5:  # Only check if enough data
            X = np.arange(LS - trend_lookback, LS).reshape(-1, 1)
            y = data['Close'].iloc[LS - trend_lookback:LS].values
            trend_coef = LinearRegression().fit(X, y).coef_[0]
            if (not is_inverse and trend_coef < min_trend_strength) or (is_inverse and trend_coef > -min_trend_strength):
                continue
        else:
            trend_coef = 0  # Skip trend check if too little data
        
        # Breakout detection (looser, optional)
        neckline_at_rs = data['Close'].iloc[T1] + neckline_slope * (RS - T1)
        breakout_confirmed = False
        breakout_idx = None
        throwback_low_idx = None
        pullback_high_idx = None
        
        for j in range(RS, min(RS + breakout_lookahead, len(data) - breakout_confirmation_bars)):
            prices = data['Close'].iloc[j:j + breakout_confirmation_bars]
            vols = data['Volume'].iloc[j:j + breakout_confirmation_bars] if 'Volume' in data.columns else [1]
            neckline_at_j = data['Close'].iloc[T1] + neckline_slope * (j - T1)
            
            if (not is_inverse and all(p < neckline_at_j for p in prices)) or \
               (is_inverse and all(p > neckline_at_j for p in prices)):
                if 'Volume' in data.columns:
                    avg_breakout_vol = vols.mean()
                    if avg_breakout_vol < avg_prior_vol * volume_ratio_breakout:
                        continue
                breakout_idx = j
                breakout_confirmed = True
                break
        
        # V-formation and Pullback High (unchanged, optional)
        if breakout_confirmed:
            v_search_range = min(breakout_idx + breakout_lookahead, len(data))
            v_prices = data['Close'].iloc[breakout_idx:v_search_range]
            throwback_low_idx = v_prices.idxmin() if not is_inverse else v_prices.idxmax()
            throwback_low_idx = data.index.get_loc(throwback_low_idx)
            pullback_prices = data['Close'].iloc[throwback_low_idx:v_search_range]
            pullback_high_idx = pullback_prices.idxmax() if not is_inverse else pullback_prices.idxmin()
            pullback_high_idx = data.index.get_loc(pullback_high_idx)
            neckline_at_pullback = data['Close'].iloc[T1] + neckline_slope * (pullback_high_idx - T1)
            v_confirmed = (not is_inverse and pullback_prices.max() >= neckline_at_pullback * 0.98) or \
                          (is_inverse and pullback_prices.min() <= neckline_at_pullback * 1.02)
        else:
            v_confirmed = False
        
        # Pattern metrics
        neckline_at_break = data['Close'].iloc[T1] + neckline_slope * (breakout_idx - T1) if breakout_idx else neckline_at_rs
        pattern_height = abs(data['Close'].iloc[H] - neckline_at_break)
        target_price = (neckline_at_break - pattern_height) if not is_inverse else (neckline_at_break + pattern_height)
        neckline_end_idx = pullback_high_idx if v_confirmed and pullback_high_idx else (breakout_idx if breakout_confirmed else RS)
        neckline_at_end = data['Close'].iloc[T1] + neckline_slope * (neckline_end_idx - T1)
        
        # Loosened confidence calculation
        confidence = min(0.99,
                        (1 - (shoulder_diff / tolerance)) * 0.25 +  # Reduced weight
                        (1 - (time_diff / time_symmetry_threshold)) * 0.15 +  # Reduced weight
                        (min(1, abs(trend_coef) * 1000)) * 0.2 +  # Looser scaling
                        (0.2 if vol_head_ok else 0.15) +  # Smaller penalty
                        (0.2 if breakout_confirmed else 0.05) +  # Smaller penalty for no breakout
                        (0.15 if v_confirmed else 0))  # Smaller bonus
        
        pattern_data = {
            'left_shoulder': data.index[LS],
            'head': data.index[H],
            'right_shoulder': data.index[RS],
            'neckline_points': (data.index[T1], data.index[T2]),
            'breakout_point': data.index[breakout_idx] if breakout_idx else None,
            'throwback_low': data.index[throwback_low_idx] if throwback_low_idx else None,
            'pullback_high': data.index[pullback_high_idx] if pullback_high_idx else None,
            'neckline_end': data.index[neckline_end_idx],
            'neckline_price': neckline_at_end,
            'target_price': target_price,
            'pattern_height': pattern_height,
            'confidence': confidence,
            'status': 'confirmed' if breakout_confirmed else 'forming',
            'type': 'standard' if not is_inverse else 'inverse'
        }
        patterns.append(pattern_data)
    
    return sorted(patterns, key=lambda x: -x['confidence'])

#     """Find all peaks in the close price data with additional smoothing."""
#     peaks = []
#     for i in range(2, len(data) - 2):  # Extended window for better peak detection
#         if (data['Close'].iloc[i] > data['Close'].iloc[i-1] and 
#             data['Close'].iloc[i] > data['Close'].iloc[i+1] and
#             data['Close'].iloc[i] > data['Close'].iloc[i-2] and  # Additional checks
#             data['Close'].iloc[i] > data['Close'].iloc[i+2]):
#             peaks.append(i)
#     return peaks

# def find_valleys(data):
#     """Find all valleys in the close price data with additional smoothing."""
#     valleys = []
#     for i in range(2, len(data) - 2):  # Extended window for better valley detection
#         if (data['Close'].iloc[i] < data['Close'].iloc[i-1] and 
#             data['Close'].iloc[i] < data['Close'].iloc[i+1] and
#             data['Close'].iloc[i] < data['Close'].iloc[i-2] and  # Additional checks
#             data['Close'].iloc[i] < data['Close'].iloc[i+2]):
#             valleys.append(i)
#     return valleys

# def detect_head_and_shoulders(data, tolerance=0.03, min_pattern_length=20, volume_ratio=1.2):
#     """
#     Enhanced Head & Shoulders detection with:
#     - Volume analysis
#     - Trend confirmation
#     - Neckline validation
#     - Breakout confirmation
#     """
#     peaks = find_peaks(data)
#     valleys = find_valleys(data)
#     patterns = []
    
#     for i in range(len(peaks) - 2):
#         LS, H, RS = peaks[i], peaks[i+1], peaks[i+2]
        
#         # 1. Basic structure validation
#         if not (data['Close'].iloc[LS] < data['Close'].iloc[H] > data['Close'].iloc[RS]):
#             continue
            
#         # 2. Shoulder symmetry (price)
#         shoulder_diff = abs(data['Close'].iloc[LS] - data['Close'].iloc[RS]) / max(data['Close'].iloc[LS], data['Close'].iloc[RS])
#         if shoulder_diff > tolerance:
#             continue
            
#         # 3. Time symmetry
#         time_diff = abs((H - LS) - (RS - H)) / max(H - LS, RS - H)
#         if time_diff > 0.3:  # Allow 30% time difference
#             continue
            
#         # 4. Minimum pattern duration
#         if (RS - LS) < min_pattern_length:
#             continue
            
#         # 5. Neckline points
#         valley1 = min([v for v in valleys if LS < v < H], key=lambda x: data['Close'].iloc[x], default=None)
#         valley2 = min([v for v in valleys if H < v < RS], key=lambda x: data['Close'].iloc[x], default=None)
#         if not valley1 or not valley2:
#             continue
            
#         # 6. Neckline slope validation
#         neckline_slope = (data['Close'].iloc[valley2] - data['Close'].iloc[valley1]) / (valley2 - valley1)
#         if abs(neckline_slope) > 0.001:  # Filter steep necklines
#             continue
            
#         # 7. Volume analysis
#         # Left shoulder advance volume
#         left_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
#         # Right shoulder advance volume
#         right_advance_vol = data['Volume'].iloc[valley2+1:RS+1].mean()
#         # Head advance volume
#         head_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
        
#         if not (head_advance_vol > left_advance_vol * volume_ratio and 
#                 right_advance_vol < head_advance_vol):
#             continue
            
#         # 8. Prior uptrend validation
#         lookback = (RS - LS) // 2
#         X = np.arange(max(0, LS-lookback), LS).reshape(-1, 1)
#         y = data['Close'].iloc[max(0, LS-lookback):LS]
#         if LinearRegression().fit(X, y).coef_[0] <= 0:
#             continue
            
#         # 9. Breakout confirmation
#         neckline_at_break = data['Close'].iloc[valley1] + neckline_slope * (RS - valley1)
#         breakout_confirmed = False
#         breakout_idx = None
        
#         for j in range(RS, min(RS + 20, len(data) - 2)):  # Check next 20 candles
#             if all(data['Close'].iloc[j+k] < neckline_at_break for k in range(3)):  # 3 consecutive closes
#                 breakout_confirmed = True
#                 breakout_idx = j + 2
#                 break
                
#         if not breakout_confirmed:
#             continue
            
#         # 10. Breakout volume check
#         if data['Volume'].iloc[breakout_idx] < data['Volume'].iloc[RS] * 0.8:  # Should have decent volume
#             continue
            
#         # Calculate pattern metrics
#         pattern_height = data['Close'].iloc[H] - neckline_at_break
#         target_price = neckline_at_break - pattern_height
        
#         patterns.append({
#             'left_shoulder': data.index[LS],
#             'head': data.index[H],
#             'right_shoulder': data.index[RS],
#             'neckline_points': (data.index[valley1], data.index[valley2]),
#             'neckline_price': neckline_at_break,
#             'breakout_point': data.index[breakout_idx],
#             'target_price': target_price,
#             'pattern_height': pattern_height,
#             'confidence': min(0.99, (1 - shoulder_diff) * (1 - time_diff))
#         })
    
#     return sorted(patterns, key=lambda x: -x['confidence'])  # Return sorted by confidence

def plot_head_and_shoulders(df, patterns, stock_name=""):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.2, 0.15],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}]]
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Price', line=dict(color="#1E88E5", width=2),
        hoverinfo='x+y', hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Moving average (optional)
    if 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA'],
            mode='lines', name="MA (50)", line=dict(color="#FFB300", width=1.5),
            hoverinfo='x+y', hovertemplate='Date: %{x}<br>MA: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    pattern_colors = ['#FF5252', '#4CAF50', '#673AB7', '#FBC02D', '#0288D1']

    for i, pattern in enumerate(patterns):
        try:
            # Extract key points
            LS = pattern["left_shoulder"]
            H = pattern["head"]
            RS = pattern["right_shoulder"]
            T1, T2 = pattern["neckline_points"]
            breakout_point = pattern.get("breakout_point")
            throwback_low = pattern.get("throwback_low")
            pullback_high = pattern.get("pullback_high")
            neckline_end = pattern["neckline_end"]

            # Convert to indices
            ls_idx = df.index.get_loc(LS)
            h_idx = df.index.get_loc(H)
            rs_idx = df.index.get_loc(RS)
            t1_idx = df.index.get_loc(T1)
            t2_idx = df.index.get_loc(T2)
            breakout_idx = df.index.get_loc(breakout_point) if breakout_point else None
            throwback_low_idx = df.index.get_loc(throwback_low) if throwback_low else None
            pullback_high_idx = df.index.get_loc(pullback_high) if pullback_high else None
            neckline_end_idx = df.index.get_loc(neckline_end)

            # Validate sequence
            if not (ls_idx < t1_idx < h_idx < t2_idx < rs_idx):
                print(f"Pattern {i}: Invalid sequence - LS: {ls_idx}, T1: {t1_idx}, H: {h_idx}, T2: {t2_idx}, RS: {rs_idx}")
                continue

            # Get prices
            ls_price = df.loc[LS, 'Close']
            h_price = df.loc[H, 'Close']
            rs_price = df.loc[RS, 'Close']
            t1_price = df.loc[T1, 'Close']
            t2_price = df.loc[T2, 'Close']
            neckline_end_price = pattern["neckline_price"]
            target_price = pattern["target_price"]
            pattern_color = pattern_colors[i % len(pattern_colors)]

            # Calculate neckline slope
            neckline_slope = (t2_price - t1_price) / (t2_idx - t1_idx) if (t2_idx - t1_idx) != 0 else 0

            # Extend neckline leftward
            left_start_idx = max(0, ls_idx - int((rs_idx - ls_idx) * 0.2))  # 20% before LS
            left_neckline_price = t1_price + neckline_slope * (left_start_idx - t1_idx)

            # Find intersection point after RS where price touches neckline
            intersection_idx = rs_idx
            for j in range(rs_idx, min(rs_idx + 30, len(df))):  # Look ahead 30 days max
                neckline_at_j = t1_price + neckline_slope * (j - t1_idx)
                price_at_j = df.iloc[j]['Close']
                if (not pattern["type"] == "inverse" and price_at_j <= neckline_at_j * 1.02) or \
                   (pattern["type"] == "inverse" and price_at_j >= neckline_at_j * 0.98):
                    intersection_idx = j
                    break
            intersection_price = df.iloc[intersection_idx]['Close']

            # ========== PATTERN MARKERS ==========
            # Left shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[LS, 'Date']], y=[ls_price],
                mode="markers+text", text=["LS"], textposition="top center",
                marker=dict(size=12, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Left Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Left Shoulder: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Head
            fig.add_trace(go.Scatter(
                x=[df.loc[H, 'Date']], y=[h_price],
                mode="markers+text", text=["H"], textposition="top center",
                marker=dict(size=14, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Head {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Head: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Right shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[RS, 'Date']], y=[rs_price],
                mode="markers+text", text=["RS"], textposition="top center",
                marker=dict(size=12, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Right Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Right Shoulder: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Neckline points
            fig.add_trace(go.Scatter(
                x=[df.loc[T1, 'Date'], df.loc[T2, 'Date']],
                y=[t1_price, t2_price],
                mode="markers", marker=dict(size=8, color=pattern_color, symbol="diamond"),
                name=f"Neckline Points {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Neckline Point: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Breakout point
            if breakout_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[breakout_idx]['Date']], y=[df.iloc[breakout_idx]['Close']],
                    mode="markers+text", text=["Break"], textposition="bottom center",
                    marker=dict(size=10, color="#FF9800", symbol="triangle-down"),
                    name=f"Breakout {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Breakout: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Throwback Low
            if throwback_low_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[throwback_low_idx]['Date']], y=[df.iloc[throwback_low_idx]['Close']],
                    mode="markers+text", text=["Throw"], textposition="bottom center",
                    marker=dict(size=10, color="#F44336", symbol="star"),
                    name=f"Throwback Low {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Throwback Low: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Pullback High
            if pullback_high_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[pullback_high_idx]['Date']], y=[df.iloc[pullback_high_idx]['Close']],
                    mode="markers+text", text=["Pull"], textposition="top center",
                    marker=dict(size=10, color="#9C27B0", symbol="triangle-up"),
                    name=f"Pullback High {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Pullback High: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # ========== NECKLINE ==========
            neckline_x = [df.iloc[left_start_idx]['Date'], df.iloc[neckline_end_idx]['Date']]
            neckline_y = [left_neckline_price, neckline_end_price]
            fig.add_trace(go.Scatter(
                x=neckline_x, y=neckline_y,
                mode="lines", line=dict(color=pattern_color, width=2, dash="dash"),
                name=f"Neckline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== FULL PATTERN OUTLINE ==========
            # Connect to intersection point after RS
            pattern_x = [
                df.iloc[left_start_idx]['Date'],
                df.loc[LS, 'Date'],
                df.loc[T1, 'Date'],
                df.loc[H, 'Date'],
                df.loc[T2, 'Date'],
                df.loc[RS, 'Date'],
                df.iloc[intersection_idx]['Date']
            ]
            pattern_y = [
                left_neckline_price,
                ls_price,
                t1_price,
                h_price,
                t2_price,
                rs_price,
                intersection_price
            ]
            fig.add_trace(go.Scatter(
                x=pattern_x, y=pattern_y,
                mode="lines", line=dict(color=pattern_color, width=3, dash='solid'),
                opacity=0.6, name=f"Pattern Outline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== V-FORMATION ==========
            if breakout_idx and throwback_low_idx and pullback_high_idx:
                v_x = [
                    df.iloc[breakout_idx]['Date'],
                    df.iloc[throwback_low_idx]['Date'],
                    df.iloc[pullback_high_idx]['Date']
                ]
                v_y = [
                    df.iloc[breakout_idx]['Close'],
                    df.iloc[throwback_low_idx]['Close'],
                    df.iloc[pullback_high_idx]['Close']
                ]
                fig.add_trace(go.Scatter(
                    x=v_x, y=v_y,
                    mode="lines", line=dict(color="#F44336", width=2, dash="dot"),
                    name=f"V-Formation {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='none'
                ), row=1, col=1)

            # ========== TARGET PROJECTION ==========
            target_start_idx = breakout_idx if breakout_idx else rs_idx
            fig.add_trace(go.Scatter(
                x=[df.iloc[target_start_idx]['Date'], df.iloc[-1]['Date']],
                y=[target_price, target_price],
                mode="lines+text", text=["", f"Target: {target_price:.2f}"],
                textposition="middle right",
                line=dict(color="#E91E63", width=1.5, dash="dot"),
                name=f"Target {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # Annotations
            fig.add_annotation(
                x=df.loc[H, 'Date'],
                y=h_price + (pattern["pattern_height"] / 2 if not pattern["type"] == "inverse" else -pattern["pattern_height"] / 2),
                text=f"H: {pattern['pattern_height']:.2f}",
                showarrow=True, arrowhead=1, ax=0, ay=-30 if not pattern["type"] == "inverse" else 30,
                font=dict(size=10, color=pattern_color)
            )

        except Exception as e:
            print(f"Error plotting pattern {i}: {str(e)}")
            continue

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume", 
            marker=dict(color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'), opacity=0.7),
            hoverinfo='x+y', hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    # RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['RSI'],
                mode='lines', name="RSI", line=dict(color="#7B1FA2", width=1.5),
                hoverinfo='x+y', hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        fig.add_hline(y=70, row=3, col=1, line=dict(color="red", width=1, dash="dash"))
        fig.add_hline(y=30, row=3, col=1, line=dict(color="green", width=1, dash="dash"))

    # Final layout
    fig.update_layout(
        title={
            'text': f"Head & Shoulders Patterns for {stock_name} (Found: {len(patterns)})",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=900, template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v", groupclick="toggleitem",
            yanchor="top", y=1, xanchor="right", x=1.15,
            font=dict(size=10), bgcolor='rgba(255,255,255,0.8)', bordercolor='#CCCCCC', borderwidth=1
        ),
        margin=dict(l=50, r=150, t=100, b=50),
        plot_bgcolor='rgba(245,245,245,1)',
        paper_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_yaxes(title_text="Price", gridcolor="lightgray", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="lightgray", row=2, col=1)
    if 'RSI' in df.columns:
        fig.update_yaxes(title_text="RSI (14)", range=[0, 100], gridcolor="lightgray", row=3, col=1)

    return fig

def detect_double_bottom(df, 
                         order=5,              # Window for extrema detection
                         tolerance=0.05,      # Max % difference between bottoms
                         min_pattern_length=5, # Min days between troughs
                         max_patterns=3,       # Max patterns to return
                         min_height_ratio=0.2, # Min height relative to prior drop
                         downtrend_lookback=20, # Days to check prior downtrend
                         confirmation_bars=2,   # Bars to confirm breakout
                         volume_increase=1.5):  # Min volume spike on breakout

    # Helper function to find peaks and troughs
    def find_extrema(df, order):
        close = df['Close'].values
        peaks = argrelextrema(close, np.greater, order=order)[0].tolist()
        troughs = argrelextrema(close, np.less, order=order)[0].tolist()
        return peaks, troughs

    peaks, troughs = find_extrema(df, order=order)
    patterns = []
    
    if len(troughs) < 2:
        return patterns

    for i in range(len(troughs) - 1):
        trough1_idx = troughs[i]
        price1 = df['Close'].iloc[trough1_idx]

        # Check for preceding downtrend
        if trough1_idx < downtrend_lookback:
            continue
        prior_prices = df['Close'].iloc[trough1_idx - downtrend_lookback:trough1_idx]
        prior_high = prior_prices.max()
        if prior_high <= price1 or (prior_high - price1) / prior_high < 0.05:  # At least 5% drop
            continue

        for j in range(i + 1, len(troughs)):
            trough2_idx = troughs[j]
            
            # Basic validation
            if (trough2_idx - trough1_idx < min_pattern_length):
                continue
                
            price2 = df['Close'].iloc[trough2_idx]
            if abs(price1 - price2) / price1 > tolerance:
                continue

            # Neckline: Highest peak between troughs
            between_peaks = [p for p in peaks if trough1_idx < p < trough2_idx]
            if not between_peaks:
                continue
            neckline_idx = max(between_peaks, key=lambda p: df['Close'].iloc[p])
            neckline_price = df['Close'].iloc[neckline_idx]
            
            if neckline_price <= max(price1, price2) * 1.02:  # Must be >2% above bottoms
                continue

            # Pattern height validation
            min_trough_price = min(price1, price2)
            pattern_height = neckline_price - min_trough_price
            prior_drop = prior_high - min_trough_price
            if pattern_height < min_height_ratio * prior_drop:
                continue

            # Breakout and confirmation
            breakout_idx = None
            confirmation_idx = None
            for idx in range(trough2_idx, min(len(df), trough2_idx + 50)):
                current_price = df['Close'].iloc[idx]
                if current_price > neckline_price:
                    # Check volume spike on breakout
                    if 'Volume' in df.columns:
                        avg_prior_vol = df['Volume'].iloc[trough2_idx-10:trough2_idx].mean()
                        if df['Volume'].iloc[idx] < avg_prior_vol * volume_increase:
                            continue
                    breakout_idx = idx
                    
                    # Confirm price holds above neckline
                    if idx + confirmation_bars < len(df):
                        if all(df['Close'].iloc[idx+k] > neckline_price 
                               for k in range(1, confirmation_bars+1)):
                            confirmation_idx = idx + confirmation_bars
                            break
                    break

            # Invalidation check
            if breakout_idx:
                post_breakout = df['Close'].iloc[trough2_idx:breakout_idx+1]
                if post_breakout.min() < min_trough_price:
                    continue  # Pattern invalid if price falls below second bottom

            if not breakout_idx:
                continue

            # Target calculation
            target_price = neckline_price + pattern_height

            patterns.append({
                'trough1': trough1_idx,
                'trough2': trough2_idx,
                'neckline': neckline_idx,
                'neckline_price': neckline_price,
                'breakout': breakout_idx,
                'confirmation': confirmation_idx if confirmation_idx else breakout_idx,
                'target': target_price,
                'pattern_height': pattern_height,
                'trough_prices': (price1, price2),
                'status': 'confirmed' if confirmation_idx else 'forming'
            })

    # Filter overlapping patterns and sort by strength
    filtered_patterns = []
    last_end = -1
    for pattern in sorted(patterns, key=lambda x: x['pattern_height'], reverse=True):
        if pattern['trough1'] > last_end:
            filtered_patterns.append(pattern)
            last_end = pattern['confirmation'] if pattern['confirmation'] else pattern['breakout']
            if len(filtered_patterns) >= max_patterns:
                break

    return filtered_patterns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_double_bottom(df, pattern_points, stock_name=""):
    """
    Enhanced Double Bottom pattern visualization with clear W formation, neckline, 
    breakout, retest, and target levels.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        pattern_points (list): List of dictionaries containing pattern details.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Chart with Double Bottom Patterns", "Volume")
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name="Price",
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

    # Moving average (optional)
    if 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['MA'],
                mode='lines', name="MA (50)",
                line=dict(color="#FB8C00", width=2)
            ),
            row=1, col=1
        )
    
    pattern_colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']
    
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = pattern_colors[idx % len(pattern_colors)]
        
        # Extract pattern points
        trough1_idx = pattern['trough1']
        trough2_idx = pattern['trough2']
        neckline_idx = pattern['neckline']
        neckline_price = pattern['neckline_price']
        target_price = pattern['target']
        breakout_idx = pattern.get('breakout')
        confirmation_idx = pattern.get('confirmation')

        # Dates
        trough1_date = df['Date'].iloc[trough1_idx]
        trough2_date = df['Date'].iloc[trough2_idx]
        neckline_date = df['Date'].iloc[neckline_idx]
        
        # Markers for troughs
        fig.add_trace(
            go.Scatter(
                x=[trough1_date], y=[df['Close'].iloc[trough1_idx]],
                mode="markers+text", text=["Bottom 1"], textposition="bottom center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Bottom 1", legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[trough2_date], y=[df['Close'].iloc[trough2_idx]],
                mode="markers+text", text=["Bottom 2"], textposition="bottom center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Bottom 2", legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Neckline marker
        fig.add_trace(
            go.Scatter(
                x=[neckline_date], y=[neckline_price],
                mode="markers+text", text=["Neckline"], textposition="top center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="triangle-up", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Neckline", legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # W formation (trough1 -> neckline -> trough2)
        w_formation_x = [trough1_date, neckline_date, trough2_date]
        w_formation_y = [df['Close'].iloc[trough1_idx], neckline_price, df['Close'].iloc[trough2_idx]]
        fig.add_trace(
            go.Scatter(
                x=w_formation_x, y=w_formation_y,
                mode="lines", line=dict(color=color, width=2, dash='dot'),
                name=f"Pattern {idx+1}: W Formation", legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Neckline line (start at neckline_date, end at breakout or beyond)
        end_idx = confirmation_idx if confirmation_idx else (breakout_idx if breakout_idx else trough2_idx + int(0.5 * (trough2_idx - trough1_idx)))
        end_idx = min(len(df) - 1, end_idx)
        neckline_x = [neckline_date, df['Date'].iloc[end_idx]]
        neckline_y = [neckline_price, neckline_price]
        fig.add_trace(
            go.Scatter(
                x=neckline_x, y=neckline_y,
                mode="lines", line=dict(color=color, width=2),
                name=f"Pattern {idx+1}: Neckline", legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Breakout and confirmation
        if breakout_idx:
            breakout_date = df['Date'].iloc[breakout_idx]
            fig.add_trace(
                go.Scatter(
                    x=[breakout_date], y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text", text=["Breakout"], textposition="top right",
                    textfont=dict(size=12, color=color),
                    marker=dict(size=12, color=color, symbol="star", line=dict(width=2, color='white')),
                    name=f"Pattern {idx+1}: Breakout", legendgroup=f"pattern{idx+1}"
                ),
                row=1, col=1
            )
            
            if confirmation_idx:
                confirm_date = df['Date'].iloc[confirmation_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[confirm_date], y=[df['Close'].iloc[confirmation_idx]],
                        mode="markers+text", text=["Confirmation"], textposition="top right",
                        textfont=dict(size=12, color=color),
                        marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color='white')),
                        name=f"Pattern {idx+1}: Confirmation", legendgroup=f"pattern{idx+1}"
                    ),
                    row=1, col=1
                )
                
                # Check for retest (optional)
                post_breakout = df['Close'].iloc[breakout_idx:confirmation_idx+10]
                retest_idx = post_breakout.idxmin()
                if retest_idx and post_breakout[retest_idx] <= neckline_price * 1.02 and post_breakout[retest_idx] > min(df['Close'].iloc[trough1_idx], df['Close'].iloc[trough2_idx]):
                    retest_date = df['Date'].iloc[retest_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[retest_date], y=[df['Close'].iloc[retest_idx]],
                            mode="markers+text", text=["Retest"], textposition="bottom right",
                            textfont=dict(size=12, color=color),
                            marker=dict(size=10, color=color, symbol="x", line=dict(width=2, color='white')),
                            name=f"Pattern {idx+1}: Retest", legendgroup=f"pattern{idx+1}"
                        ),
                        row=1, col=1
                    )
            
            # Target line
            target_x = [breakout_date, df['Date'].iloc[-1]]
            target_y = [target_price, target_price]
            fig.add_trace(
                go.Scatter(
                    x=target_x, y=target_y,
                    mode="lines+text", text=["Target"], textposition="middle right",
                    textfont=dict(size=12, color=color),
                    line=dict(color=color, width=2, dash='dash'),
                    name=f"Pattern {idx+1}: Target", legendgroup=f"pattern{idx+1}"
                ),
                row=1, col=1
            )
        
        # Highlight downtrend (optional, assuming 20 days prior to trough1)
        downtrend_start = max(0, trough1_idx - 20)
        fig.add_trace(
            go.Scatter(
                x=df['Date'].iloc[downtrend_start:trough1_idx+1],
                y=df['Close'].iloc[downtrend_start:trough1_idx+1],
                mode="lines", line=dict(color="grey", width=2, dash='dash'),
                name=f"Pattern {idx+1}: Downtrend", legendgroup=f"pattern{idx+1}", opacity=0.5
            ),
            row=1, col=1
        )

    # Volume chart with breakout highlight
    colors = ['#26A69A' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF5350' for i in range(len(df))]
    if breakout_idx and 'Volume' in df.columns:
        colors[breakout_idx] = '#FFD700'  # Gold for breakout volume
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume", marker=dict(color=colors, opacity=0.8)
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': f"Double Bottom Pattern Detection for {stock_name}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800, template="plotly_white",
        legend=dict(
            orientation="v", groupclick="toggleitem",
            yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10), tracegroupgap=5
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified", xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text="Price", tickprefix="$", gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor='lightgray', row=2, col=1)
    
    return fig

def detect_cup_and_handle(df, order=15, cup_min_bars=20, handle_max_retrace=0.5):
    peaks, troughs = find_extrema(df, order=order)
    patterns = []
    
    if len(peaks) < 1 or len(troughs) < 1:
        return patterns
    
    for i in range(len(peaks) - 1):
        left_peak_idx = peaks[i]
        cup_troughs = [t for t in troughs if t > left_peak_idx]
        
        if not cup_troughs:
            continue
            
        cup_bottom_idx = cup_troughs[0]
        right_peaks = [p for p in peaks if p > cup_bottom_idx]
        
        if not right_peaks:
            continue
            
        right_peak_idx = right_peaks[0]
        
        # Validate cup formation
        if right_peak_idx - left_peak_idx < cup_min_bars:
            continue
            
        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]
        
        # Should be roughly equal peaks (within 5%)
        if abs(right_peak_price - left_peak_price) / left_peak_price > 0.05:
            continue
            
        # Find handle
        handle_troughs = [t for t in troughs if t > right_peak_idx]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]
        
        # Calculate handle end (first point above handle entry after bottom)
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 0.98:  # 2% tolerance
                handle_end_idx = j
                break
                
        if not handle_end_idx:  # Never found a valid handle end
            continue
            
        # Validate handle retracement
        cup_height = ((left_peak_price + right_peak_price) / 2) - cup_bottom_price
        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            continue
            
        # Find breakout if any
        breakout_idx = None
        for j in range(handle_end_idx, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 1.02:  # 2% above resistance
                breakout_idx = j
                break
                
        patterns.append({
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_start': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target': right_peak_price + cup_height,
            'cup_height': cup_height,
            'confidence': min(0.99, (1 - abs(left_peak_price-right_peak_price)/left_peak_price))
        })
    
    return patterns

def plot_cup_and_handle(df, pattern_points, stock_name=""):
    """
    Enhanced Cup and Handle pattern visualization with clear resistance, breakout, and proper cup/handle formation.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        pattern_points (list): List of dictionaries containing pattern details
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price Chart with Cup and Handle", "Volume", "RSI (14)")
    )
    
    # Add price line chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#26A69A', width=2)
        ),
        row=1, col=1
    )

    # Add moving average
    if 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['MA'], 
                mode='lines', 
                name="Moving Average (50)", 
                line=dict(color="#FB8C00", width=2)
            ),
            row=1, col=1
        )
    
    # Define colors for pattern visualization
    colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']
    
    # Add pattern-specific visualization
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = colors[idx % len(colors)]
        
        # Extract pattern points
        left_peak_idx = pattern['left_peak']
        cup_bottom_idx = pattern['cup_bottom']
        right_peak_idx = pattern['right_peak']
        handle_bottom_idx = pattern['handle_bottom']
        resistance_level = pattern['resistance']
        
        # Add markers for key points
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[left_peak_idx]],
                y=[df['Close'].iloc[left_peak_idx]],
                mode="markers+text",
                text=["Left Cup Lip"],
                textposition="top right",
                textfont=dict(size=10),
                marker=dict(color="#3F51B5", size=12, symbol="circle"),
                name="Left Cup Lip"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[cup_bottom_idx]],
                y=[df['Close'].iloc[cup_bottom_idx]],
                mode="markers+text",
                text=["Cup Bottom"],
                textposition="bottom center",
                textfont=dict(size=10),
                marker=dict(color="#4CAF50", size=12, symbol="circle"),
                name="Cup Bottom"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[right_peak_idx]],
                y=[df['Close'].iloc[right_peak_idx]],
                mode="markers+text",
                text=["Right Cup Lip"],
                textposition="top left",
                textfont=dict(size=10),
                marker=dict(color="#3F51B5", size=12, symbol="circle"),
                name="Right Cup Lip"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[handle_bottom_idx]],
                y=[df['Close'].iloc[handle_bottom_idx]],
                mode="markers+text",
                text=["Handle Bottom"],
                textposition="bottom right",
                textfont=dict(size=10),
                marker=dict(color="#FF9800", size=12, symbol="circle"),
                name="Handle Bottom"
            ),
            row=1, col=1
        )
        
        # Create a smooth arc for the cup
        # Generate points for the cup arc
        num_points = 100  # More points for a smoother arc
        
        # Create x values (dates) for the arc
        left_date = df['Date'].iloc[left_peak_idx]
        right_date = df['Date'].iloc[right_peak_idx]
        bottom_date = df['Date'].iloc[cup_bottom_idx]
        
        # Calculate time deltas for interpolation
        total_seconds = (right_date - left_date).total_seconds()
        
        # Generate dates for the arc
        arc_dates = []
        for i in range(num_points):
            # Calculate position (0 to 1)
            t = i / (num_points - 1)
            # Calculate seconds from left peak
            seconds_offset = total_seconds * t
            # Calculate the date
            current_date = left_date + pd.Timedelta(seconds=seconds_offset)
            arc_dates.append(current_date)
        
        # Create y values (prices) for the arc
        left_price = df['Close'].iloc[left_peak_idx]
        right_price = df['Close'].iloc[right_peak_idx]
        bottom_price = df['Close'].iloc[cup_bottom_idx]
        
        # Calculate the midpoint between left and right peaks
        mid_price = (left_price + right_price) / 2
        
        # Calculate the depth of the cup
        cup_depth = mid_price - bottom_price
        
        # Generate smooth arc using a quadratic function
        arc_prices = []
        for i in range(num_points):
            # Normalized position (0 to 1)
            t = i / (num_points - 1)
            
            # Parabolic function for U shape: y = a*x^2 + b*x + c
            # Where x is normalized from -1 to 1 for symmetry
            x = 2 * t - 1  # Map t from [0,1] to [-1,1]
            
            # For a symmetric cup, use:
            if abs(left_price - right_price) < 0.05 * left_price:  # If peaks are within 5%
                # Symmetric parabola
                price = mid_price - cup_depth * (1 - x*x)
            else:
                # Asymmetric parabola - linear interpolation with quadratic dip
                if x <= 0:
                    # Left side
                    price = left_price + (mid_price - left_price) * (x + 1) - cup_depth * (1 - x*x)
                else:
                    # Right side
                    price = mid_price + (right_price - mid_price) * x - cup_depth * (1 - x*x)
            
            arc_prices.append(price)
        
        # Add the smooth cup arc
        fig.add_trace(
            go.Scatter(
                x=arc_dates,
                y=arc_prices,
                mode="lines",
                name="Cup Formation",
                line=dict(color="#9C27B0", width=3)
            ),
            row=1, col=1
        )
        
        # Add handle visualization
        handle_indices = list(range(right_peak_idx, handle_bottom_idx + 1))
        if handle_bottom_idx < len(df) - 1:
            # Find where handle ends (either at breakout or at end of data)
            if pattern.get('breakout') is not None:
                handle_end_idx = pattern['breakout']
            else:
                # Find where price recovers to at least 50% of handle depth
                handle_depth = df['Close'].iloc[right_peak_idx] - df['Close'].iloc[handle_bottom_idx]
                recovery_level = df['Close'].iloc[handle_bottom_idx] + (handle_depth * 0.5)
                
                post_handle_indices = df.index[df.index > handle_bottom_idx]
                recovery_indices = [i for i in post_handle_indices if df['Close'].iloc[i] >= recovery_level]
                
                if recovery_indices:
                    handle_end_idx = recovery_indices[0]
                else:
                    handle_end_idx = len(df) - 1
            
            handle_indices.extend(range(handle_bottom_idx + 1, handle_end_idx + 1))
        
        # Add handle line
        handle_dates = df['Date'].iloc[handle_indices].tolist()
        handle_prices = df['Close'].iloc[handle_indices].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=handle_dates,
                y=handle_prices,
                mode="lines",
                name="Handle",
                line=dict(color="#FF9800", width=3)
            ),
            row=1, col=1
        )
        
        # Add resistance level
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[left_peak_idx],
            x1=df['Date'].iloc[-1],
            y0=resistance_level,
            y1=resistance_level,
            line=dict(color="#FF5722", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Add resistance annotation
        fig.add_annotation(
            x=df['Date'].iloc[right_peak_idx],
            y=resistance_level,
            text="Resistance",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF5722",
            ax=0,
            ay=-30,
            font=dict(size=10, color="#FF5722")
        )
        
        # Add breakout point and target if available
        if pattern.get('breakout') is not None:
            breakout_idx = pattern['breakout']
            
            # Add breakout marker
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text",
                    text=["Breakout"],
                    textposition="top right",
                    marker=dict(size=12, color="#4CAF50", symbol="triangle-up"),
                    name="Breakout"
                ),
                row=1, col=1
            )
            
            # Add target price line
            target_price = pattern['target']
            
            fig.add_shape(
                type="line",
                x0=df['Date'].iloc[breakout_idx],
                x1=df['Date'].iloc[-1],
                y0=target_price,
                y1=target_price,
                line=dict(color="#4CAF50", width=2, dash="dot"),
                row=1, col=1
            )
            
            # Add target annotation
            fig.add_annotation(
                x=df['Date'].iloc[-1],
                y=target_price,
                text=f"Target: {target_price:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#4CAF50",
                ax=-40,
                ay=0,
                font=dict(size=10, color="#4CAF50")
            )
            
            # Add measured move visualization
            cup_height = pattern['cup_height']
            
            # Add annotation for cup height (measured move)
            fig.add_annotation(
                x=df['Date'].iloc[breakout_idx],
                y=(resistance_level + target_price) / 2,
                text=f"Measured Move: {cup_height:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#4CAF50",
                ax=40,
                ay=0,
                font=dict(size=10, color="#4CAF50")
            )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name="Volume", 
            marker=dict(
                color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'),
                line=dict(color='rgba(0,0,0,0)', width=0)
            ),
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['RSI'], 
                mode='lines', 
                name="RSI (14)", 
                line=dict(color="#7B1FA2", width=2)
            ),
            row=3, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line", line=dict(dash="dash", color="red", width=2),
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=70, y1=70,
            row=3, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", color="green", width=2),
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=30, y1=30,
            row=3, col=1
        )
        
        # Add annotations for overbought/oversold
        fig.add_annotation(
            x=df['Date'].iloc[0], y=70,
            text="Overbought (70)",
            showarrow=False,
            xanchor="left",
            font=dict(color="red"),
            row=3, col=1
        )
        fig.add_annotation(
            x=df['Date'].iloc[0], y=30,
            text="Oversold (30)",
            showarrow=False,
            xanchor="left",
            font=dict(color="green"),
            row=3, col=1
        )
    
    # Add pattern explanation
    fig.add_annotation(
        x=df['Date'].iloc[0],
        y=df['Close'].max(),
        text="Cup and Handle: Bullish continuation pattern with target equal to cup depth projected above breakout",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="#0D47A1"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#0D47A1",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Cup and Handle Pattern Detection for {stock_name}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.4,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified"
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
    st.markdown("# Stock Pattern Scanner(Yahoo Finance)")
    
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

        if (end_date - start_date).days < 30:
            st.error("Date range must be at least 30 days")
            st.stop()

        if (end_date - start_date).days > 365 * 3:
            st.warning("Large date range may impact performance. Consider narrowing your range.")

        # Scan button
        scan_button = st.button("Scan the Stock", use_container_width=True)

    # Main content
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

            # Simulate progress
            for i in range(101):
                time.sleep(0.05)
                progress_bar.progress(i)

            status_container.write("Task Complete! 🎉")

            
            stock_data = []
            for i, symbol in enumerate(stock_symbols):
                if st.session_state.scan_cancelled:
                    break

                status_container.info(f"Processing {symbol}... ({i+1}/{len(stock_symbols)})")

                try:
                    df = fetch_stock_data(symbol, start_date, end_date)
                    if df is None or df.empty:
                        continue
                    
                    patterns = {
                        "Head and Shoulders": detect_head_and_shoulders(df),
                        "Double Bottom": detect_double_bottom(df),
                        "Cup and Handle": detect_cup_and_handle(df),
                    }
                    
                    # Calculate per-pattern metrics
                    pattern_metrics = evaluate_pattern_detection(df, patterns)
                    
                    stock_info = yf.Ticker(symbol).info
                    current_price = stock_info.get('currentPrice', None)
                    volume = stock_info.get('volume', None)
                    percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

                    stock_data.append({
                        "Symbol": symbol,
                        "Patterns": patterns,
                        "Pattern Metrics": pattern_metrics,  # New field for metrics
                        "Data": df,
                        "Current Price": current_price,
                        "Volume": volume,
                        "Percent Change": percent_change,
                        "MA": df['MA'].iloc[-1] if 'MA' in df.columns else None,
                        "RSI": df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
                    })
                
                except Exception as e:
                    st.error(f"Error processing {symbol}: {e}")
                    continue
                
                progress_bar.progress((i + 1) / len(stock_symbols))
            
            st.session_state.stock_data = stock_data
            st.session_state.selected_stock = None
            st.session_state.selected_pattern = None
            
            progress_container.empty()
            status_container.success("Scan completed successfully!")
    
    # Display results
    if st.session_state.stock_data:
        st.markdown("## 📊 Scan Results")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Stock List", "📈 Pattern Visualization"])
        
        with tab1:
            # Prepare data for the table
            table_data = []
            for stock in st.session_state.stock_data:
                # Count patterns
                pattern_counts = {
                    pattern: len(stock["Patterns"][pattern]) 
                    for pattern in stock["Patterns"]
                }
                
                # Create row
                row = {
                    "Symbol": stock["Symbol"],
                    "Current Price": stock['Current Price'] if stock['Current Price'] else None,
                    "Volume": stock['Volume'] if stock['Volume'] else None,
                    "% Change": stock['Percent Change'],
                    "MA (50)": stock['MA'],
                    "RSI (14)": stock['RSI'],
                    # "Accuracy": stock['Accuracy'],
                    # "Precision": stock['Precision'],
                    "Head and Shoulders": pattern_counts.get("Head and Shoulders", 0),
                    # "Double Top": pattern_counts.get("Double Top", 0),
                    "Double Bottom": pattern_counts.get("Double Bottom", 0),
                    "Cup and Handle": pattern_counts.get("Cup and Handle", 0),
                }
                table_data.append(row)
            
            # Create DataFrame
            df_table = pd.DataFrame(table_data)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            # with col1:
            #     min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0, step=1.0)
            # with col2:
            #     min_volume = st.number_input("Min Volume", min_value=0, value=0, step=1000)
            with col1:
                pattern_filter = st.selectbox("Filter by Pattern", 
                                             ["All Patterns", "Head and Shoulders", "Double Bottom", "Cup and Handle"])
            
            # Apply filters
            filtered_df = df_table.copy()
            # if min_price > 0:
            #     filtered_df = filtered_df[filtered_df["Current Price"] >= min_price]
            # if min_volume > 0:
            #     filtered_df = filtered_df[filtered_df["Volume"] >= min_volume]
            if pattern_filter != "All Patterns":
                filtered_df = filtered_df[filtered_df[pattern_filter] > 0]
            
            # Format the table
            formatted_df = filtered_df.copy()
            formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Volume"] = formatted_df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")
            formatted_df["% Change"] = formatted_df["% Change"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            formatted_df["MA (50)"] = formatted_df["MA (50)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["RSI (14)"] = formatted_df["RSI (14)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            # formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            # formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            # Display the table with custom styling
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
                    # "Accuracy": st.column_config.TextColumn("Accuracy"),
                    # "Precision": st.column_config.TextColumn("Precision"),
                    "Head and Shoulders": st.column_config.NumberColumn("H&S", format="%d"),
                    # "Double Top": st.column_config.NumberColumn("Double Top", format="%d"),
                    "Double Bottom": st.column_config.NumberColumn("Double Bottom", format="%d"),
                    "Cup and Handle": st.column_config.NumberColumn("Cup & Handle", format="%d"),
                }
            )
            
            # Show summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", len(filtered_df))
            with col2:
                total_patterns = filtered_df[["Head and Shoulders", "Double Bottom", "Cup and Handle"]].sum().sum()
                st.metric("Total Patterns", int(total_patterns))
            # with col3:
            #     avg_accuracy = filtered_df["Accuracy"].mean()
            #     st.metric("Avg. Accuracy", f"{avg_accuracy:.2f}")
            # with col4:
            #     avg_precision = filtered_df["Precision"].mean()
            #     st.metric("Avg. Precision", f"{avg_precision:.2f}")
        
        with tab2:
            st.markdown("### 🔍 Pattern Visualization")
            
            # Stock selection with pattern counts
            col1, col2 = st.columns(2)
            with col1:
                # Create a function to format the stock display name
                def format_stock_name(stock):
                    pattern_counts = {
                        'HS': len(stock['Patterns']['Head and Shoulders']),
                        'DB': len(stock['Patterns']['Double Bottom']),
                        'CH': len(stock['Patterns']['Cup and Handle'])
                    }
                    return f"{stock['Symbol']} [HS:{pattern_counts['HS']}, DB:{pattern_counts['DB']}, CH:{pattern_counts['CH']}]"
                
                # Create the selectbox with formatted names
                selected_display = st.selectbox(
                    "Select Stock",
                    options=[format_stock_name(stock) for stock in st.session_state.stock_data],
                    key='stock_select'
                )
                
                # Extract the symbol from the selected display name
                selected_stock = selected_display.split(' [')[0]
            
            if selected_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = selected_stock
                st.session_state.selected_pattern = None
            
            selected_data = next((item for item in st.session_state.stock_data 
                                if item["Symbol"] == st.session_state.selected_stock), None)
            
            if selected_data:
                # Pattern selection
                pattern_options = []
                for pattern_name, patterns in selected_data["Patterns"].items():
                    if patterns:  # Only include patterns that were detected
                        count = len(patterns)
                        display_name = f"{pattern_name} ({count})"
                        pattern_options.append((pattern_name, display_name))
                
                if pattern_options:
                    with col2:
                        # Show pattern selection with counts
                        selected_display = st.selectbox(
                            "Select Pattern",
                            options=[display for _, display in pattern_options],
                            key='pattern_select'
                        )
                        
                        # Get the actual pattern name
                        selected_pattern = next(
                            pattern for pattern, display in pattern_options 
                            if display == selected_display
                        )
                    
                    if selected_pattern != st.session_state.selected_pattern:
                        st.session_state.selected_pattern = selected_pattern
                    
                    if st.session_state.selected_pattern:
                        pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                        
                        # Display stock info
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
                        
                        # Plot the pattern
                        st.markdown("### 📈 Pattern Chart")
                        if selected_data:
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
                            
                            # Display pattern-specific metrics
                            st.markdown(f"### 📊 Metrics for {st.session_state.selected_pattern}")
                            metrics = selected_data["Pattern Metrics"][st.session_state.selected_pattern]
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
                            with metric_cols[1]:
                                st.metric("Precision", f"{metrics['Precision']:.2f}")
                            with metric_cols[2]:
                                st.metric("Recall", f"{metrics['Recall']:.2f}")
                            with metric_cols[3]:
                                st.metric("F1 Score", f"{metrics['F1']:.2f}")
                        
                        # Pattern explanation
                        with st.expander("📚 Pattern Details & Trading Guidance"):
                            if selected_pattern == "Head and Shoulders":
                                st.markdown("""
                                **Head and Shoulders Pattern (Bearish Reversal)**
                                
                                - **Structure**: Left shoulder (peak) → Head (higher peak) → Right shoulder (lower peak)
                                - **Neckline**: Support connecting the troughs after left shoulder and head
                                - **Confirmation**: Price closes below neckline with increased volume
                                
                                **Trading Implications**:
                                1. **Entry**: After neckline breakout (short position)
                                2. **Price Target**: Neckline price - pattern height
                                3. **Stop Loss**: Above right shoulder peak
                                """)
                            elif selected_pattern == "Double Bottom":
                                st.markdown("""
                                **Double Bottom Pattern (Bullish Reversal)**
                                
                                - **Structure**: Two distinct troughs at similar price levels
                                - **Confirmation**: Break above resistance (peak between troughs)
                                
                                **Trading Implications**:
                                1. **Entry**: After resistance breakout (long position)
                                2. **Price Target**: Resistance price + pattern height
                                3. **Stop Loss**: Below second trough
                                """)
                            elif selected_pattern == "Cup and Handle":
                                st.markdown("""
                                **Cup and Handle Pattern (Bullish Continuation)**
                                
                                - **Structure**: 
                                - Cup: Rounded bottom (U-shape) 
                                - Handle: Small downward drift/pullback
                                - **Confirmation**: Break above handle resistance
                                
                                **Trading Implications**:
                                1. **Entry**: After handle breakout (long position)
                                2. **Price Target**: Cup height added to breakout point
                                3. **Stop Loss**: Below handle low
                                """)
                            
                            st.markdown("---")
                            st.markdown("**Pattern Statistics**")
                            st.metric("Pattern Confidence", f"{pattern_points[0].get('confidence', 0)*100:.1f}%")
                            if 'target_price' in pattern_points[0]:
                                current_price = selected_data['Current Price']
                                if current_price:
                                    potential_pct = ((pattern_points[0]['target_price'] - current_price)/current_price)*100
                                    st.metric("Target Price", 
                                            f"${pattern_points[0]['target_price']:.2f}",
                                            f"{potential_pct:.1f}%")
                else:
                    st.info(f"No patterns detected for {selected_stock}.")
            else:
                st.error("Selected stock data not found.")
if __name__ == "__main__":
    main()
