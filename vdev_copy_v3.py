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
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# =========================LSTM
def prepare_lstm_data(df, sequence_length=60, prediction_window=10, debug=False):
    """
    Enhanced LSTM data preparation with trading-day awareness and balanced classes
    """
    # Validate input
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        if debug: print("Invalid DataFrame - missing 'Close' column")
        return None, None, None
    
    if len(df) < sequence_length + prediction_window + 10:  # Extra buffer
        if debug: 
            print(f"Insufficient data: {len(df)} < {sequence_length + prediction_window + 10}")
        return None, None, None
    
    try:
        # Filter only trading days (Mon-Fri)
        if 'Date' in df.columns:
            df = df[df['Date'].dt.weekday < 5].copy()
        
        # Normalize with MinMax
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(prices)
        
        X, y = [], []
        volatility = df['Close'].pct_change().std()
        
        for i in range(sequence_length, len(scaled_data) - prediction_window):
            X.append(scaled_data[i-sequence_length:i, 0])
            
            # Enhanced labeling considering volatility
            future_price = scaled_data[i+prediction_window-1, 0]
            current_price = scaled_data[i-1, 0]
            move = future_price - current_price
            
            # Label 1 if significant upward move (>0.5*volatility)
            y.append(1 if move > (0.5 * volatility) else 0)
        
        # Balance classes
        X, y = balance_classes(X, y)
        
        if len(X) == 0:
            if debug: print("No valid sequences after balancing")
            return None, None, None
            
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(-1)
        y_tensor = torch.FloatTensor(np.array(y)).unsqueeze(-1)
        
        if debug:
            print(f"Created {len(X_tensor)} sequences (Bullish: {y_tensor.sum().item()}, Bearish: {len(y_tensor)-y_tensor.sum().item()})")
        
        return X_tensor, y_tensor, scaler
        
    except Exception as e:
        if debug: print(f"LSTM prep error: {str(e)}")
        return None, None, None

def balance_classes(X, y):
    """Ensure equal bull/bear examples for better training"""
    y_np = np.array(y).flatten()
    bull_idx = np.where(y_np == 1)[0]
    bear_idx = np.where(y_np == 0)[0]
    
    min_samples = min(len(bull_idx), len(bear_idx))
    if min_samples == 0:
        return X, y
        
    balanced_idx = np.concatenate([
        np.random.choice(bull_idx, min_samples, replace=False),
        np.random.choice(bear_idx, min_samples, replace=False)
    ])
    np.random.shuffle(balanced_idx)
    
    return [X[i] for i in balanced_idx], [y[i] for i in balanced_idx]
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last timestep
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

@st.cache_resource
def train_lstm_model(_X_train, _y_train, epochs=20, batch_size=32):
    """
    Train a PyTorch LSTM model for price direction prediction.
    
    Args:
        _X_train (torch.Tensor): Training input sequences (unhashed).
        _y_train (torch.Tensor): Training labels (unhashed).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        model: Trained PyTorch LSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(_X_train, _y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def predict_with_lstm(model, X, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X = X.to(device)  # Shape: [1, sequence_length, 1]
        predictions = model(X).cpu().numpy()  # Shape: [1, 1]
    return predictions

def get_lstm_confirmation(df, breakout_idx, model, scaler, pattern_type, sequence_length=60):
    if breakout_idx < sequence_length or breakout_idx >= len(df):
        return 0.0
    
    try:
        sequence = df['Close'].iloc[breakout_idx-sequence_length:breakout_idx].values.reshape(-1, 1)
        scaled_sequence = scaler.transform(sequence)
        X = torch.FloatTensor(scaled_sequence).unsqueeze(0)  # Shape: (1, sequence_length, 1)
        
        pred = predict_with_lstm(model, X, scaler)[0][0]
        
        if pattern_type == "Head and Shoulders":
            return 0.2 * (1 - pred)
        elif pattern_type == "Double Bottom":
            return 0.2 * pred
        elif pattern_type == "Cup and Handle":
            return 0.15 * pred
        else:
            return 0.0
    except Exception as e:
        print(f"LSTM confirmation failed for {pattern_type}: {str(e)}")
        return 0.0  # Fallback to no boost
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
        
        # Get data from Yahoo Finance
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_str, end=end_str)
        
        if df.empty or len(df) < 50:
            print(f"{symbol}: Insufficient data - {len(df)} rows")
            return None
            
        # Reset index and ensure proper column names
        df = df.reset_index()
        if 'Date' not in df.columns:
            print(f"{symbol}: No Date column in data")
            return None
            
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Check if we have valid Close prices
        if df['Close'].isnull().all():
            print(f"{symbol}: No valid Close prices")
            return None
            
        # Calculate indicators
        df['Close_Smoothed'] = savgol_filter(df['Close'], window_length=11, polyorder=2)
        df = calculate_moving_average(df, window=50)
        df = calculate_rsi(df, window=50)
        
        print(f"First date: {df['Date'].min()}, Last date: {df['Date'].max()}")
        print(f"Missing dates: {pd.date_range(start=start_date, end=end_date).difference(df['Date'])}")
        
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

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def detect_head_and_shoulders(df, depth=3, min_pattern_separation=10, lstm_model=None, lstm_scaler=None, debug=False):
    try:
        if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
            if debug: print("Invalid DataFrame - missing 'Close' column")
            return []
        
        data = df.copy()
        if 'Date' not in data.columns:
            data['Date'] = data.index
        
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isna().any():
            if debug: print("Non-numeric values found in Close prices")
            return []

        # Lower depth for more extrema
        peaks = argrelextrema(data['Close'].values, np.greater, order=depth)[0]
        troughs = argrelextrema(data['Close'].values, np.less, order=depth)[0]
        
        if len(peaks) < 3:
            if debug: print(f"Not enough peaks ({len(peaks)}) to form H&S pattern")
            return []
        
        patterns = []
        used_indices = set()
        
        for i in range(len(peaks) - 2):
            if peaks[i] in used_indices or peaks[i+1] in used_indices or peaks[i+2] in used_indices:
                continue
            
            ls_idx, h_idx, rs_idx = peaks[i], peaks[i+1], peaks[i+2]
            
            try:
                ls_price = float(data['Close'].iloc[ls_idx])
                h_price = float(data['Close'].iloc[h_idx])
                rs_price = float(data['Close'].iloc[rs_idx])
                
                if not (h_price > ls_price and h_price > rs_price):
                    if debug: print(f"Invalid structure at {data['Date'].iloc[h_idx]}")
                    continue
                
                # Relax shoulder difference to 10%
                shoulder_diff = abs(ls_price - rs_price) / min(ls_price, rs_price)
                if shoulder_diff > 0.10:
                    if debug: print(f"Shoulders not balanced at {data['Date'].iloc[h_idx]}")
                    continue
                
                # Relax time symmetry to 50%
                ls_to_head_duration = h_idx - ls_idx
                rs_to_head_duration = rs_idx - h_idx
                duration_diff = abs(ls_to_head_duration - rs_to_head_duration) / min(ls_to_head_duration, rs_to_head_duration)
                if duration_diff > 0.5:
                    if debug: print(f"Shoulders not symmetric in time at {data['Date'].iloc[h_idx]}")
                    continue
                
                t1_candidates = [t for t in troughs if ls_idx < t < h_idx]
                t2_candidates = [t for t in troughs if h_idx < t < rs_idx]
                
                if not t1_candidates or not t2_candidates:
                    if debug: print(f"No troughs found between peaks at {data['Date'].iloc[h_idx]}")
                    continue
                
                t1_idx = t1_candidates[np.argmin(data['Close'].iloc[t1_candidates])]
                t2_idx = t2_candidates[np.argmin(data['Close'].iloc[t2_candidates])]
                
                # Relax neckline trough difference to 10%
                trough_diff = abs(data['Close'].iloc[t1_idx] - data['Close'].iloc[t2_idx]) / min(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
                if trough_diff > 0.10:
                    if debug: print(f"Neckline troughs not balanced at {data['Date'].iloc[h_idx]}")
                    continue
                
                neckline_slope = (data['Close'].iloc[t2_idx] - data['Close'].iloc[t1_idx]) / (t2_idx - t1_idx)
                neckline_price_at_rs = data['Close'].iloc[t1_idx] + neckline_slope * (rs_idx - t1_idx)
                
                breakout_idx = None
                min_breakout_distance = 3
                # Extend look-ahead to 60 periods and lower drop threshold to 1%
                for j in range(rs_idx + min_breakout_distance, min(rs_idx + 60, len(data))):
                    neckline_at_j = data['Close'].iloc[t1_idx] + neckline_slope * (j - t1_idx)
                    if data['Close'].iloc[j] < neckline_at_j and (neckline_at_j - data['Close'].iloc[j]) / neckline_at_j > 0.01:
                        breakout_idx = j
                        break
                
                if breakout_idx is not None:
                    if any(abs(breakout_idx - p['breakout_idx']) < min_pattern_separation for p in patterns):
                        if debug: print(f"Breakout too close at {data['Date'].iloc[breakout_idx]}")
                        continue
                
                if breakout_idx is None:
                    if debug: print(f"No breakout below neckline after RS at {data['Date'].iloc[rs_idx]}")
                    continue
                
                valid_pattern = True
                for k in range(i + 3, len(peaks)):
                    next_peak_idx = peaks[k]
                    if next_peak_idx > rs_idx + 30:
                        break
                    if float(data['Close'].iloc[next_peak_idx]) > h_price:
                        valid_pattern = False
                        if debug: print(f"Pattern invalidated by higher peak at {data['Date'].iloc[next_peak_idx]}")
                        break
                
                if not valid_pattern:
                    continue
                
                pattern_height = h_price - max(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
                target_price = neckline_price_at_rs - pattern_height
                duration_days = (data['Date'].iloc[rs_idx] - data['Date'].iloc[ls_idx]).days
                
                base_confidence = min(0.99, 0.6 + (0.3 * (1 - shoulder_diff / 0.10)))  # Adjusted base confidence
                lstm_boost = 0.0
                if lstm_model and lstm_scaler:
                    lstm_boost = get_lstm_confirmation(data, breakout_idx, lstm_model, lstm_scaler, "Head and Shoulders")
                
                pattern = {
                    'left_shoulder_idx': ls_idx,
                    'head_idx': h_idx,
                    'right_shoulder_idx': rs_idx,
                    'neckline_trough1_idx': t1_idx,
                    'neckline_trough2_idx': t2_idx,
                    'breakout_idx': breakout_idx,
                    'left_shoulder_price': ls_price,
                    'head_price': h_price,
                    'right_shoulder_price': rs_price,
                    'neckline_slope': neckline_slope,
                    'target_price': target_price,
                    'pattern_height': pattern_height,
                    'duration_days': duration_days,
                    'confidence': min(0.99, base_confidence + lstm_boost)
                }
                
                patterns.append(pattern)
                used_indices.update([ls_idx, h_idx, rs_idx])
                
            except Exception as e:
                if debug: print(f"Skipping pattern due to error: {e}")
                continue
        
        final_patterns = []
        sorted_patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
        for pattern in sorted_patterns:
            if not any(
                (pattern['left_shoulder_idx'] <= existing['right_shoulder_idx'] and 
                 pattern['right_shoulder_idx'] >= existing['left_shoulder_idx'])
                for existing in final_patterns
            ):
                final_patterns.append(pattern)
        
        return final_patterns
        
    except Exception as e:
        if debug: print(f"Error in detect_head_and_shoulders: {e}")
        return []

def plot_head_and_shoulders(df, pattern_points, stock_name=""):
    """Plot Head and Shoulders patterns with detailed statistics and improved visualization"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create figure with 3 rows: price chart, volume, and statistics table
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    # Price line (row 1)
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            mode='lines', 
            name="Price",
            line=dict(color='#1E88E5', width=2),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    pattern_stats = []  # Store statistics for each pattern

    for idx, pattern in enumerate(pattern_points):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        ls_idx = pattern['left_shoulder_idx']
        h_idx = pattern['head_idx']
        rs_idx = pattern['right_shoulder_idx']
        t1_idx = pattern['neckline_trough1_idx']
        t2_idx = pattern['neckline_trough2_idx']
        breakout_idx = pattern['breakout_idx']

        # Improved Neckline Positioning - Extending the shorter neckline to the left
        # Calculate neckline points starting from before the left shoulder
        neckline_start_price = df['Close'].iloc[t1_idx]
        neckline_end_price = df['Close'].iloc[t1_idx] + pattern['neckline_slope'] * (t2_idx - t1_idx)
        
        # Extend neckline to the left (e.g., 10 periods before the first trough) and to the breakout
        neckline_x = [
            df['Date'].iloc[max(0, t1_idx - 10)],  # Start 10 periods before the first trough
            df['Date'].iloc[breakout_idx]           # End at Breakout point
        ]
        
        # Calculate neckline y-values using the slope
        neckline_y = [
            neckline_start_price + pattern['neckline_slope'] * (max(0, t1_idx - 10) - t1_idx),  # Left extension
            neckline_start_price + pattern['neckline_slope'] * (breakout_idx - t1_idx)           # Breakout point
        ]
        
        # Ensure neckline stays below Right Shoulder price at rs_idx
        rs_price = pattern['right_shoulder_price']
        neckline_at_rs = neckline_start_price + pattern['neckline_slope'] * (rs_idx - t1_idx)
        if neckline_at_rs >= rs_price:
            adjustment_factor = 0.98  # Place neckline at 98% of RS price
            adjusted_neckline_slope = (rs_price * adjustment_factor - neckline_start_price) / (rs_idx - t1_idx)
            neckline_y = [
                neckline_start_price + adjusted_neckline_slope * (max(0, t1_idx - 10) - t1_idx),
                neckline_start_price + adjusted_neckline_slope * (breakout_idx - t1_idx)
            ]

        # Add extended neckline trace
        fig.add_trace(
            go.Scatter(
                x=neckline_x,
                y=neckline_y,
                mode="lines",
                line=dict(color=color, dash='dash', width=2),
                name=f"Neckline {idx+1}"
            ),
            row=1, col=1
        )

        # Calculate detailed pattern metrics
        left_trough_price = min(df['Close'].iloc[t1_idx], df['Close'].iloc[ls_idx])
        right_trough_price = min(df['Close'].iloc[t2_idx], df['Close'].iloc[rs_idx])
        
        left_shoulder_height = pattern['left_shoulder_price'] - left_trough_price
        head_height = pattern['head_price'] - max(df['Close'].iloc[t1_idx], df['Close'].iloc[t2_idx])
        right_shoulder_height = pattern['right_shoulder_price'] - right_trough_price
        
        left_duration = (df['Date'].iloc[h_idx] - df['Date'].iloc[ls_idx]).days
        right_duration = (df['Date'].iloc[rs_idx] - df['Date'].iloc[h_idx]).days
        total_duration = (df['Date'].iloc[rs_idx] - df['Date'].iloc[ls_idx]).days
        
        neckline_at_breakout = df['Close'].iloc[t1_idx] + pattern['neckline_slope'] * (breakout_idx - t1_idx)
        neckline_break = neckline_at_breakout - df['Close'].iloc[breakout_idx]
        break_percentage = (neckline_break / neckline_at_breakout) * 100
        
        # Calculate symmetry metrics
        price_symmetry = abs(pattern['left_shoulder_price'] - pattern['right_shoulder_price']) / min(pattern['left_shoulder_price'], pattern['right_shoulder_price'])
        time_symmetry = abs(left_duration - right_duration) / min(left_duration, right_duration)
        
        # Store statistics for table
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{pattern['left_shoulder_price']:.2f}",
            f"{left_shoulder_height:.2f}",
            f"{pattern['head_price']:.2f}",
            f"{head_height:.2f}",
            f"{pattern['right_shoulder_price']:.2f}",
            f"{right_shoulder_height:.2f}",
            f"{left_duration}/{right_duration} days",
            f"{price_symmetry*100:.1f}%/{time_symmetry*100:.1f}%",
            f"{neckline_break:.2f} ({break_percentage:.1f}%)",
            f"{pattern['target_price']:.2f}",
            f"{pattern['confidence']*100:.1f}%"
        ])

        # Plot Left Shoulder
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[ls_idx]], 
                y=[pattern['left_shoulder_price']],
                mode="markers+text",
                text=["LS"],
                textposition="bottom center",
                marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Head
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[h_idx]], 
                y=[pattern['head_price']],
                mode="markers+text",
                text=["H"],
                textposition="top center",
                marker=dict(size=14, symbol="diamond", color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Right Shoulder
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[rs_idx]], 
                y=[pattern['right_shoulder_price']],
                mode="markers+text",
                text=["RS"],
                textposition="bottom center",
                marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Breakout
        if breakout_idx:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text",
                    text=["Breakout"],
                    textposition="top right",
                    marker=dict(size=10, symbol="x", color=color, line=dict(width=2))
                ),
                row=1, col=1
            )
            
            # Target line
            # fig.add_trace(
            #     go.Scatter(
            #         x=[df['Date'].iloc[breakout_idx], df['Date'].iloc[-1]],
            #         y=[pattern['target_price']] * 2,
            #         mode="lines",
            #         line=dict(color=color, dash='dot', width=1.5)
            #     ),
            #     row=1, col=1
            # )

    # Volume (row 2)
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'],
            name="Volume",
            marker=dict(color='#26A69A', opacity=0.7),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )

    # Statistics table (row 3)
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Pattern", "LS Price", "LS Height", "Head Price", "Head Height", 
                    "RS Price", "RS Height", "Duration", "Symmetry (P/T)", 
                    "Neckline Break", "Target", "Confidence"
                ],
                font=dict(size=10, color='white'),
                fill_color='#1E88E5',
                align=['left', 'center', 'center', 'center', 'center', 
                      'center', 'center', 'center', 'center', 'center', 'center'],
                height=30
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=10),
                align=['left', 'center', 'center', 'center', 'center', 
                      'center', 'center', 'center', 'center', 'center', 'center'],
                height=25,
                fill_color=['rgba(245,245,245,0.8)', 'white']
            )
        ),
        row=3, col=1
    )

    # Update layout with enhanced legend
    fig.update_layout(
        title=dict(
            text=f"<b>Head & Shoulders Analysis for {stock_name}</b>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1100,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            title=dict(
                text='<b>Pattern Details</b>',
                font=dict(size=12)
            ),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.1,
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            itemsizing='constant',
            itemwidth=40,
            traceorder='normal',
            itemclick='toggle',
            itemdoubleclick='toggleothers'
        ),
        hovermode='x unified',
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False,
        margin=dict(r=300, t=100, b=20, l=50)
    )

    # Add custom legend annotations with extra spacing
    for idx, pattern in enumerate(pattern_points):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        fig.add_annotation(
            x=1.38,
            y=0.98 - (idx * 0.15),
            xref="paper",
            yref="paper",
            text=(
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b><br>"
                f"LS: {pattern['left_shoulder_price']:.2f}<br>"
                f"H: {pattern['head_price']:.2f}<br>"
                f"RS: {pattern['right_shoulder_price']:.2f}<br>"
                f"Neckline Slope: {pattern['neckline_slope']:.4f}<br>"
                f"Breakout: {df['Close'].iloc[pattern['breakout_idx']]:.2f}<br>"
                f"Target: {pattern['target_price']:.2f}<br>"
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ),
            showarrow=False,
            font=dict(size=10),
            align='left',
            bordercolor="#AAAAAA",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(255,255,255,0.8)"
        )

    return fig

def detect_double_bottom(df, order=7, tolerance=0.1, min_pattern_length=20, 
                        max_patterns=10, min_retracement=0.3, max_retracement=0.7,
                        min_volume_ratio=0.6, lookback_period=90, lstm_model=None,
                        lstm_scaler=None, debug=False):
    """Enhanced Double Bottom detection with LSTM confirmation"""
    # Input validation
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        if debug: print("Invalid DataFrame")
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    # Data quality checks
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    if data['Close'].isna().any():
        if debug: print("NaN values in Close prices")
        return []
    
    # Volume indicators
    if 'Volume' in data.columns:
        data['Volume_5MA'] = data['Volume'].rolling(5).mean()
        data['Volume_20MA'] = data['Volume'].rolling(20).mean()
    
    # Find extrema
    troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    
    if len(troughs) < 2:
        if debug: print(f"Insufficient troughs ({len(troughs)})")
        return []

    patterns = []
    used_indices = set()

    # First pass: Find potential bottoms
    potential_bottoms = []
    for i in range(len(troughs) - 1):
        t1_idx, t2_idx = troughs[i], troughs[i+1]
        
        # Pattern structure validation
        if (t2_idx - t1_idx) < min_pattern_length:
            continue
            
        price1 = data['Close'].iloc[t1_idx]
        price2 = data['Close'].iloc[t2_idx]
        price_diff = abs(price1 - price2) / min(price1, price2)
        
        if price_diff > tolerance:
            continue
            
        potential_bottoms.append((t1_idx, t2_idx))

    # Second pass: Validate patterns
    for t1_idx, t2_idx in potential_bottoms:
        if t1_idx in used_indices or t2_idx in used_indices:
            continue
            
        # Neckline identification
        between_points = data.iloc[t1_idx:t2_idx+1]
        neckline_idx = between_points['Close'].idxmax()
        
        if neckline_idx in [t1_idx, t2_idx] or neckline_idx in used_indices:
            continue
            
        neckline_price = data['Close'].iloc[neckline_idx]
        price1 = data['Close'].iloc[t1_idx]
        price2 = data['Close'].iloc[t2_idx]
        
        # Retracement validation
        move_up = neckline_price - price1
        move_down = neckline_price - price2
        if move_up <= 0:
            continue
            
        retracement = move_down / move_up
        if not (min_retracement <= retracement <= max_retracement):
            continue
            
        # Volume validation
        vol_ok = True
        if 'Volume' in data.columns:
            vol1 = data['Volume_5MA'].iloc[t1_idx]
            vol2 = data['Volume_5MA'].iloc[t2_idx]
            if vol2 < vol1 * min_volume_ratio:
                vol_ok = False
        
        # Breakout detection
        breakout_idx = None
        breakout_strength = 0
        breakout_volume_confirmation = False
        
        for j in range(t2_idx, min(len(data), t2_idx + lookback_period)):
            if j in used_indices:
                break
                
            current_price = data['Close'].iloc[j]
            if current_price > neckline_price * 1.02:  # 2% breakout threshold
                breakout_strength += (current_price - neckline_price) / neckline_price
                if breakout_idx is None:
                    breakout_idx = j
                if 'Volume' in data.columns and data['Volume'].iloc[j] > data['Volume_20MA'].iloc[j]:
                    breakout_volume_confirmation = True
                if j > breakout_idx + 2 and breakout_strength > 0.05:
                    break
            elif breakout_idx and current_price < neckline_price:
                breakout_idx = None
                breakout_strength = 0
        
        if not breakout_idx:
            continue
        
        # Confidence calculation
        base_confidence = 0.5
        base_confidence += 0.2 * min(1.0, breakout_strength / 0.1)
        base_confidence += (1 - price_diff / tolerance) * 0.2
        base_confidence += 0.1 if vol_ok else 0
        base_confidence += 0.1 if breakout_volume_confirmation else 0
        
        # LSTM confirmation
        lstm_boost = 0.0
        if lstm_model and lstm_scaler:
            lstm_boost = get_lstm_confirmation(
                data, breakout_idx, lstm_model, lstm_scaler, "Double Bottom"
            )
        
        # Final validation
        pattern_indices = set(range(t1_idx, t2_idx + 1)) | {neckline_idx, breakout_idx}
        if not (pattern_indices & used_indices):
            patterns.append({
                'trough1_idx': t1_idx,
                'trough2_idx': t2_idx,
                'neckline_idx': neckline_idx,
                'neckline_price': neckline_price,
                'breakout_idx': breakout_idx,
                'breakout_strength': breakout_strength,
                'target_price': neckline_price + (neckline_price - min(price1, price2)),
                'pattern_height': neckline_price - min(price1, price2),
                'trough_prices': (float(price1), float(price2)),
                'confidence': min(0.95, base_confidence + lstm_boost),
                'lstm_boost': lstm_boost,
                'status': 'confirmed',
                'retracement': retracement,
                'volume_ratio': vol2 / vol1 if 'Volume' in data.columns else 1.0,
                'breakout_volume_confirmed': breakout_volume_confirmation,
                'time_span': (data['Date'].iloc[t2_idx] - data['Date'].iloc[t1_idx]).days
            })
            used_indices.update(pattern_indices)
    
    return sorted(patterns, key=lambda x: -x['confidence'])[:max_patterns]

def plot_double_bottom(df, pattern_points, stock_name=""):
    """Plot confirmed Double Bottom patterns with scrollable pattern info."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name="Price",
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    valid_patterns = []
    used_indices = set()
    
    # Filter for confirmed patterns and ensure no overlap
    for pattern in sorted(pattern_points, key=lambda x: x.get('confidence', 0), reverse=True):
        if not all(k in pattern for k in ['trough1_idx', 'neckline_idx', 'trough_prices', 'neckline_price', 'breakout_idx']):
            continue  # Skip if missing required keys or no breakout (unconfirmed)
            
        t1_idx = pattern['trough1_idx']
        nl_idx = pattern['neckline_idx']
        breakout_idx = pattern['breakout_idx']
        t2_idx = pattern['trough2_idx']
        
        # Define the range of indices this pattern occupies
        pattern_indices = set(range(t1_idx, breakout_idx + 1)) | {nl_idx}
        if pattern_indices & used_indices:
            continue  # Skip if overlaps with any previously plotted pattern
        
        # Validate second bottom (B2) after neckline
        search_end = min(len(df) - 1, breakout_idx)
        b2_segment = df.iloc[nl_idx:search_end + 1]
        b2_idx = b2_segment['Close'].idxmin()
        b2_price = df['Close'].iloc[b2_idx]
        
        if b2_idx <= nl_idx or b2_price >= pattern['neckline_price']:
            continue
        
        # Add B2 to used indices
        pattern_indices.add(b2_idx)
        
        # Find prior peak (PP) before first bottom
        pp_segment = df.iloc[max(0, t1_idx - 30):t1_idx + 1]
        pp_idx = pp_segment['Close'].idxmax()
        pp_price = df['Close'].iloc[pp_idx]

        # Calculate risk/reward since we have a breakout
        risk = pattern['neckline_price'] - min(pattern['trough_prices'][0], b2_price)
        reward = (pattern['neckline_price'] + risk) - pattern['neckline_price']
        rr_ratio = reward / risk if risk > 0 else 0

        valid_patterns.append({
            **pattern,
            'pp_idx': pp_idx,
            'pp_price': pp_price,
            'b2_idx': b2_idx,
            'b2_price': b2_price,
            'rr_ratio': rr_ratio
        })
        used_indices.update(pattern_indices)

    pattern_stats = []
    
    for idx, pattern in enumerate(valid_patterns):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        
        pp_idx = pattern['pp_idx']
        pp_price = pattern['pp_price']
        t1_idx = pattern['trough1_idx']
        t1_price = pattern['trough_prices'][0]
        nl_idx = pattern['neckline_idx']
        nl_price = pattern['neckline_price']
        b2_idx = pattern['b2_idx']
        b2_price = pattern['b2_price']
        breakout_idx = pattern['breakout_idx']
        breakout_strength = pattern['breakout_strength']
        rr_ratio = pattern['rr_ratio']
        
        w_x = [df['Date'].iloc[pp_idx], 
               df['Date'].iloc[t1_idx],
               df['Date'].iloc[nl_idx],
               df['Date'].iloc[b2_idx],
               df['Date'].iloc[breakout_idx]]
        w_y = [pp_price, t1_price, nl_price, b2_price, df['Close'].iloc[breakout_idx]]
        
        fig.add_trace(
            go.Scatter(
                x=w_x,
                y=w_y,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10, color=color, line=dict(width=1, color='white')),
                name=f'Pattern {idx+1}',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        fig.add_annotation(
            x=df['Date'].iloc[breakout_idx],
            y=df['Close'].iloc[breakout_idx],
            text=f"BO<br>{breakout_strength:.1%}",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=10, color=color)
        )
        
        point_labels = [
            ('PP', pp_idx, pp_price, 'top center'),
            ('B1', t1_idx, t1_price, 'bottom center'),
            ('MP', nl_idx, nl_price, 'top center'),
            ('B2', b2_idx, b2_price, 'bottom center'),
            ('BO', breakout_idx, df['Close'].iloc[breakout_idx], 'top right')
        ]
        
        for label, p_idx, p_price, pos in point_labels:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[p_idx]],
                    y=[p_price],
                    mode="markers+text",
                    text=[label],
                    textposition=pos,
                    marker=dict(
                        size=12 if label in ['PP', 'BO'] else 10,
                        color=color,
                        symbol='diamond' if label in ['PP', 'BO'] else 'circle',
                        line=dict(width=1, color='white')
                    ),
                    textfont=dict(size=10, color='white'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        neckline_x = [df['Date'].iloc[max(0, t1_idx - 10)],
                      df['Date'].iloc[min(len(df) - 1, breakout_idx + 20)]]
        fig.add_trace(
            go.Scatter(
                x=neckline_x,
                y=[nl_price, nl_price],
                mode="lines",
                line=dict(color=color, dash='dash', width=1.5),
                showlegend=False,
                opacity=0.7,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        duration = (df['Date'].iloc[b2_idx] - df['Date'].iloc[t1_idx]).days
        price_diff = abs(t1_price - b2_price) / t1_price if t1_price != 0 else 0
        
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{t1_price:.2f}",
            f"{b2_price:.2f}",
            f"{nl_price:.2f}",
            f"{price_diff:.2%}",
            f"{pattern['retracement']:.1%}",
            f"{pattern['volume_ratio']:.1f}x",
            f"{duration} days",
            "Yes",
            f"{df['Close'].iloc[breakout_idx]:.2f}",
            f"{pattern['target_price']:.2f}",
            f"{rr_ratio:.1f}",
            f"{pattern['confidence']*100:.1f}%",
            f"{breakout_strength:.1%}"
        ])
    
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume",
            marker=dict(color='#7f7f7f', opacity=0.4),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    for pattern in valid_patterns:
        start_idx = max(0, pattern['trough1_idx'] - 10)
        end_idx = min(len(df) - 1, pattern['breakout_idx'] + 10)
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[start_idx], df['Date'].iloc[end_idx]],
                y=[df['Volume'].max() * 1.1] * 2,
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor=f'hsla({(valid_patterns.index(pattern) * 60) % 360}, 70%, 50%, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Pattern", "Bottom 1", "Bottom 2", "Neckline", "Price Diff",
                        "Retrace", "Vol Ratio", "Duration", "Confirmed", "Breakout",
                        "Target", "R/R", "Confidence", "BO Strength"],
                font=dict(size=10, color='white'),
                fill_color='#1f77b4',
                align='center'
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=9),
                align='center',
                fill_color=['rgba(245,245,245,0.8)', 'white'],
                height=25
            ),
            columnwidth=[0.7] + [0.6] * 13
        ),
        row=3, col=1
    )
    
    # Create scrollable pattern annotations
    num_patterns = len(valid_patterns)
    if num_patterns > 0:
        # Add each pattern annotation in a vertical stack
        for idx, pattern in enumerate(valid_patterns):
            color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
            breakout_idx = pattern['breakout_idx']
            rr_ratio = pattern['rr_ratio']
            
            annotation_text = [
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b>",
                f"PP: {pattern['pp_price']:.2f} | B1: {pattern['trough_prices'][0]:.2f}",
                f"MP: {pattern['neckline_price']:.2f} | B2: {pattern['b2_price']:.2f}",
                f"BO: {df['Close'].iloc[breakout_idx]:.2f} (Strength: {pattern['breakout_strength']:.1%})",
                f"Target: {pattern['target_price']:.2f} | R/R: {rr_ratio:.1f}",
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ]
            
            fig.add_annotation(
                x=1.30,
                y=0.98 - (idx * 0.12),
                xref="paper",
                yref="paper",
                text="<br>".join(annotation_text),
                showarrow=False,
                font=dict(size=9),
                align='left',
                bordercolor="#AAAAAA",
                borderwidth=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                yanchor="top"
            )

    fig.update_layout(
        title=dict(
            text=f"<b>Double Bottom Analysis for {stock_name}</b><br>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(r=250, t=120, b=20, l=50),  # Reduced right margin
        legend=dict(
            title=dict(text='<b>Pattern Legend</b>'),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.05,  # Moved legend closer
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10)
        ),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False
    )

    return fig
def detect_cup_and_handle(df, order=10, cup_min_bars=20, handle_max_retrace=0.5, 
                         lstm_model=None, lstm_scaler=None, debug=False):
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        if debug: print("Invalid DataFrame")
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    if data['Close'].isna().any():
        if debug: print("NaN values in Close prices")
        return []
    
    peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    
    if len(peaks) < 2 or len(troughs) < 1:
        if debug: print(f"Insufficient peaks ({len(peaks)}) or troughs ({len(troughs)})")
        return []

    patterns = []
    used_indices = set()  # Track used indices to prevent overlap
    
    for i in range(len(peaks) - 1):
        left_peak_idx = peaks[i]
        
        # Skip if this peak is already part of another pattern
        if left_peak_idx in used_indices:
            continue
            
        # Uptrend validation
        uptrend_lookback = min(30, left_peak_idx)
        prior_data = data['Close'].iloc[left_peak_idx - uptrend_lookback:left_peak_idx]
        if prior_data.iloc[-1] <= prior_data.iloc[0] * 1.05:
            continue
            
        cup_troughs = [t for t in troughs if left_peak_idx < t < (left_peak_idx + 200)]
        if not cup_troughs:
            continue
            
        cup_bottom_idx = min(cup_troughs, key=lambda x: data['Close'].iloc[x])
        if cup_bottom_idx in used_indices:
            continue
            
        cup_bottom_price = data['Close'].iloc[cup_bottom_idx]
        
        right_peaks = [p for p in peaks if cup_bottom_idx < p < (cup_bottom_idx + 100)]
        if not right_peaks:
            continue
            
        right_peak_idx = right_peaks[0]
        if right_peak_idx in used_indices:
            continue
            
        right_peak_price = data['Close'].iloc[right_peak_idx]
        
        if right_peak_idx - left_peak_idx < cup_min_bars:
            continue
            
        if abs(right_peak_price - data['Close'].iloc[left_peak_idx]) / data['Close'].iloc[left_peak_idx] > 0.10:
            continue
            
        handle_troughs = [t for t in troughs if right_peak_idx < t < (right_peak_idx + 50)]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        if handle_bottom_idx in used_indices:
            continue
            
        handle_bottom_price = data['Close'].iloc[handle_bottom_idx]
        
        cup_height = data['Close'].iloc[left_peak_idx] - cup_bottom_price
        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            continue
            
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, min(handle_bottom_idx + 30, len(data))):
            if j in used_indices:
                break
            if data['Close'].iloc[j] >= right_peak_price * 0.98:
                handle_end_idx = j
                break
                
        if not handle_end_idx:
            continue
            
        breakout_idx = None
        for j in range(handle_end_idx, min(handle_end_idx + 30, len(data))):
            if j in used_indices:
                break
            if data['Close'].iloc[j] > right_peak_price * 1.02:
                breakout_idx = j
                break
                
        # Define pattern range
        pattern_start = left_peak_idx
        pattern_end = breakout_idx if breakout_idx else handle_end_idx
        pattern_range = set(range(pattern_start, pattern_end + 1))
        
        # Check for overlap with existing patterns
        if pattern_range & used_indices:
            if debug: print(f"Pattern at {data['Date'].iloc[left_peak_idx]} overlaps with existing pattern")
            continue
        
        base_confidence = 0.6
        if breakout_idx:
            base_confidence += 0.3
        base_confidence += (1 - abs(data['Close'].iloc[left_peak_idx] - right_peak_price) / data['Close'].iloc[left_peak_idx] / 0.10) * 0.1
        
        lstm_boost = 0.0
        if lstm_model and lstm_scaler and breakout_idx:
            lstm_boost = get_lstm_confirmation(data, breakout_idx, lstm_model, lstm_scaler, "Cup and Handle")
        
        pattern = {
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target_price': right_peak_price + cup_height,
            'cup_height': cup_height,
            'confidence': min(0.95, base_confidence + lstm_boost),
            'lstm_boost': lstm_boost,
            'status': 'confirmed' if breakout_idx else 'forming'
        }
        
        patterns.append(pattern)
        used_indices.update(pattern_range)
    
    return sorted(patterns, key=lambda x: -x['confidence'])
def plot_cup_and_handle(df, pattern_points, stock_name=""):
    """
    Enhanced Cup and Handle plot with statistics, pattern indexing, and improved visualization.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    from scipy.interpolate import interp1d
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name="Price",
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    pattern_stats = []
    
    for idx, pattern in enumerate(pattern_points):
        # Assign unique color to each pattern
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        handle_color = f'hsl({(idx * 60 + 30) % 360}, 70%, 50%)'  # Slightly different hue for handle
        
        # Extract points
        left_peak_idx = pattern['left_peak']
        cup_bottom_idx = pattern['cup_bottom']
        right_peak_idx = pattern['right_peak']
        handle_bottom_idx = pattern['handle_bottom']
        handle_end_idx = pattern['handle_end']
        breakout_idx = pattern.get('breakout')
        
        left_peak_date = df['Date'].iloc[left_peak_idx]
        cup_bottom_date = df['Date'].iloc[cup_bottom_idx]
        right_peak_date = df['Date'].iloc[right_peak_idx]
        handle_bottom_date = df['Date'].iloc[handle_bottom_idx]
        handle_end_date = df['Date'].iloc[handle_end_idx]
        breakout_date = df['Date'].iloc[breakout_idx] if breakout_idx else None
        
        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]
        handle_end_price = df['Close'].iloc[handle_end_idx]
        breakout_price = df['Close'].iloc[breakout_idx] if breakout_idx else None
        
        # Create smooth cup curve
        num_points = 50
        cup_dates = [left_peak_date, cup_bottom_date, right_peak_date]
        cup_prices = [left_peak_price, cup_bottom_price, right_peak_price]
        
        cup_dates_numeric = [(d - cup_dates[0]).days for d in cup_dates]
        t = np.linspace(0, 1, num_points)
        t_orig = [0, 0.5, 1]
        
        interp_func = interp1d(t_orig, cup_dates_numeric, kind='quadratic')
        interp_dates_numeric = interp_func(t)
        interp_prices = interp1d(t_orig, cup_prices, kind='quadratic')(t)
        interp_dates = [cup_dates[0] + pd.Timedelta(days=d) for d in interp_dates_numeric]

        # Plot cup curve
        # Plot cup curve
        fig.add_trace(
            go.Scatter(
                x=interp_dates, y=interp_prices,
                mode="lines",
                line=dict(color=color, width=3, dash='dot'),  # Added dash='dot'
                name=f'Pattern {idx+1} Cup',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Plot handle
        handle_x = df['Date'].iloc[right_peak_idx:handle_end_idx + 1]
        handle_y = df['Close'].iloc[right_peak_idx:handle_end_idx + 1]
        fig.add_trace(
            go.Scatter(
                x=handle_x, y=handle_y,
                mode="lines",
                line=dict(color=handle_color, width=3),
                name=f'Pattern {idx+1} Handle',
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot resistance line
        resistance_x = [left_peak_date, df['Date'].iloc[-1]]
        fig.add_trace(
            go.Scatter(
                x=resistance_x,
                y=[right_peak_price] * 2,
                mode="lines",
                line=dict(color=color, dash='dash', width=1.5),
                name=f'Resistance {idx+1}',
                opacity=0.7,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Plot breakout and target if exists
        if breakout_idx:
            # Breakout line
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[handle_end_idx], df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[handle_end_idx], breakout_price],
                    mode="lines",
                    line=dict(color=handle_color, width=3),
                    name=f'Breakout {idx+1}',
                    hovertemplate='Breakout: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Target line
            # target_end_idx = min(breakout_idx + 60, len(df)-1)
            # fig.add_trace(
            #     go.Scatter(
            #         x=[df['Date'].iloc[breakout_idx], df['Date'].iloc[target_end_idx]],
            #         y=[pattern['target']] * 2,
            #         mode="lines",
            #         line=dict(color=handle_color, dash='dot', width=1.5),
            #         name=f"Target {idx+1}",
            #         opacity=0.8,
            #         hoverinfo='skip'
            #     ),
            #     row=1, col=1
            # )
            
            # Breakout annotation
            fig.add_annotation(
                x=breakout_date,
                y=breakout_price,
                text=f"BO<br>{pattern.get('breakout_strength', 'N/A')}",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=10, color=handle_color)
            )

        # Plot key points with labels
        point_labels = [
            ('LR', left_peak_idx, left_peak_price, 'top right'),
            ('CB', cup_bottom_idx, cup_bottom_price, 'bottom center'),
            ('RR', right_peak_idx, right_peak_price, 'top left'),
            ('HL', handle_bottom_idx, handle_bottom_price, 'bottom right'),
            ('HE', handle_end_idx, handle_end_price, 'top center')
        ]
        
        if breakout_idx:
            point_labels.append(('BO', breakout_idx, breakout_price, 'top right'))
        
        for label, p_idx, p_price, pos in point_labels:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[p_idx]],
                    y=[p_price],
                    mode="markers+text",
                    text=[label],
                    textposition=pos,
                    marker=dict(
                        size=12 if label in ['LR','RR','BO'] else 10,
                        color=color if label in ['LR','RR','CB'] else handle_color,
                        symbol='diamond' if label in ['LR','RR','BO'] else 'circle',
                        line=dict(width=1, color='white')
                    ),
                    textfont=dict(size=10, color='white'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # Calculate pattern statistics
        cup_duration = (right_peak_date - left_peak_date).days
        handle_duration = (handle_end_date - right_peak_date).days
        total_duration = cup_duration + handle_duration
        cup_depth_pct = (left_peak_price - cup_bottom_price) / left_peak_price * 100
        handle_retrace_pct = (right_peak_price - handle_bottom_price) / (right_peak_price - cup_bottom_price) * 100
        
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{left_peak_price:.2f}",
            f"{cup_bottom_price:.2f}",
            f"{right_peak_price:.2f}",
            f"{handle_bottom_price:.2f}",
            f"{cup_depth_pct:.1f}%",
            f"{handle_retrace_pct:.1f}%",
            f"{cup_duration} days",
            f"{handle_duration} days",
            f"{total_duration} days",
            "Yes" if breakout_idx else "No",
            f"{breakout_price:.2f}" if breakout_idx else "Pending",
            f"{pattern.get('target', 'N/A'):.2f}" if 'target' in pattern else "N/A",  # Modified this line
            f"{pattern.get('rr_ratio', 'N/A')}",
            f"{pattern['confidence']*100:.1f}%"
        ])

        # Add volume highlighting
        start_idx = max(0, left_peak_idx - 10)
        end_idx = min(len(df)-1, (breakout_idx if breakout_idx else handle_end_idx) + 10)
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[start_idx], df['Date'].iloc[end_idx]],
                y=[df['Volume'].max() * 1.1] * 2,
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor=f'hsla({(idx * 60) % 360}, 70%, 50%, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume",
            marker=dict(color='#7f7f7f', opacity=0.4),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )

    # Pattern statistics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Pattern", "Left Rim", "Cup Bottom", "Right Rim", "Handle Low",
                       "Cup Depth", "Handle Retrace", "Cup Days", "Handle Days", "Total Days",
                       "Confirmed", "Breakout", "Target", "R/R", "Confidence"],
                font=dict(size=10, color='white'),
                fill_color='#1f77b4',
                align='center'
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=9),
                align='center',
                fill_color=['rgba(245,245,245,0.8)', 'white'],
                height=25
            ),
            columnwidth=[0.7]+[0.6]*13
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Cup and Handle Analysis for {stock_name}</b><br>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(r=350, t=120, b=20, l=50),
        legend=dict(
            title=dict(text='<b>Pattern Legend</b>'),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.07,
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10)
        ),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False
    )

    # Add pattern annotations
    num_patterns = len(pattern_points)
    if num_patterns > 0:
        available_space = 0.9
        spacing = available_space / max(num_patterns, 1)
        start_y = 0.95
        
        for idx, pattern in enumerate(pattern_points):
            color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
            handle_color = f'hsl({(idx * 60 + 30) % 360}, 70%, 50%)'
            breakout_idx = pattern.get('breakout')
            
            annotation_text = [
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b>",
                f"Type: {'Confirmed' if breakout_idx else 'Potential'}",
                f"Left Rim: {df['Close'].iloc[pattern['left_peak']]:.2f}",
                f"Cup Bottom: {df['Close'].iloc[pattern['cup_bottom']]:.2f}",
                f"Right Rim: {df['Close'].iloc[pattern['right_peak']]:.2f}",
                f"Handle Low: {df['Close'].iloc[pattern['handle_bottom']]:.2f}",
                f"Breakout: {df['Close'].iloc[breakout_idx]:.2f}" if breakout_idx else "Breakout: Pending",
                f"Target: {pattern.get('target', 'N/A'):.2f}" if 'target' in pattern else "Target: N/A",  # Modified this line
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ]
            
            y_position = start_y - (idx * spacing)
            
            fig.add_annotation(
                x=1.38,
                y=0.98 - (idx * 0.15),
                xref="paper",
                yref="paper",
                text="<br>".join(annotation_text),
                showarrow=False,
                font=dict(size=10),
                align='left',
                bordercolor="#AAAAAA",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                yanchor="top"
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


import streamlit as st
import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

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
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from st_aggrid import AgGrid, GridOptionsBuilder

# =========================LSTM Functions (keep all your existing LSTM functions here)
# [Previous LSTM functions remain unchanged...]

def main():
    st.header("📈 Stock Pattern Scanner (Yahoo Finance)")

    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'selected_pattern' not in st.session_state:
        st.session_state.selected_pattern = None
    if 'scan_cancelled' not in st.session_state:
        st.session_state.scan_cancelled = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "📋 Stock List"

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Scanner Settings")
        
        # Date selection
        # In your main() function, modify the date_input elements like this:

        # Date selection
        st.markdown("### 📅 Date Selection")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date(2020, 1, 1),
                min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date.today(),
                key="start_date_input"  # Add unique key
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.date.today() - datetime.timedelta(days=1),
                min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date.today(),
                key="end_date_input"  # Add unique key
            )

        if end_date < start_date:
            st.error("⚠️ End Date must be after Start Date.")
            st.stop()
        if (end_date - start_date).days < 90:
            st.error("⚠️ Date range must be at least 90 days for reliable patterns.")
            st.stop()

        # LSTM Settings
        st.markdown("### 🧠 LSTM Settings")
        train_lstm = st.checkbox("Enable LSTM Pattern Enhancement", value=False)
        sequence_length = 60  # Default value
        epochs = 20  # Default value
        if train_lstm:
            sequence_length = st.slider("Sequence Length", 30, 100, 60)
            epochs = st.slider("Training Epochs", 5, 50, 20)

        # Scan button
        scan_button = st.button("🚀 Scan the Stock")

        # Cancel button - placed below scan button
        if scan_button:
            cancel_button_placeholder = st.empty()
            if cancel_button_placeholder.button("❌ Cancel Scan"):
                st.session_state.scan_cancelled = True
                st.warning("Scan cancelled by user")
                st.stop()

        # High Accuracy Patterns section - only shown after scan completes
        if 'stock_data' in st.session_state and st.session_state.stock_data and not st.session_state.get('scan_cancelled', False):
            st.markdown("---")
            st.markdown("### 🎯 High Accuracy Patterns (90%+)")
            
            # Collect all high accuracy patterns
            high_acc_patterns = []
            for stock in st.session_state.stock_data:
                for pattern_name, patterns in stock["Patterns"].items():
                    for pattern in patterns:
                        if isinstance(pattern, dict) and pattern.get('confidence', 0) >= 0.9:
                            high_acc_patterns.append({
                                'stock': stock['Symbol'],
                                'pattern': pattern_name,
                                'confidence': pattern['confidence'],
                                'pattern_points': [pattern]
                            })
            
            if high_acc_patterns:
                # Shuffle the patterns to get random selection
                import random
                random.shuffle(high_acc_patterns)
                
                # Display up to 10 random patterns (or all if less than 10)
                display_count = min(10, len(high_acc_patterns))
                for i, pattern in enumerate(high_acc_patterns[:display_count]):
                    confidence_pct = pattern['confidence'] * 100
                    color = "#4CAF50" if confidence_pct >= 95 else "#FFC107"
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{pattern['stock']}**")
                            st.caption(pattern['pattern'])
                        
                        with col2:
                            st.markdown(
                                f"""<div style="
                                    background-color: {color};
                                    color: white;
                                    padding: 4px 8px;
                                    border-radius: 12px;
                                    font-size: 0.8em;
                                    text-align: center;
                                ">{confidence_pct:.1f}%</div>""",
                                unsafe_allow_html=True
                            )
                        
                        # Clicking the card will select the stock and pattern in the visualization tab
                        # if st.button("View", key=f"view_{pattern['stock']}_{i}"):
                        #     st.session_state.selected_stock = pattern['stock']
                        #     st.session_state.selected_pattern = pattern['pattern']
                        #     st.session_state.active_tab = "📈 Pattern Visualization"
                        #     st.rerun()
                
                if len(high_acc_patterns) > display_count:
                    st.caption(f"Showing {display_count} of {len(high_acc_patterns)} high accuracy patterns")
            else:
                st.info("No patterns with 90%+ confidence found")

    # Load stock symbols with validation
    try:
        with open("stock_symbols.txt", "r") as f:
            stock_symbols = [line.strip() for line in f if line.strip()]
        
        if not stock_symbols:
            st.error("No valid stock symbols found in stock_symbols.txt")
            return
            
        # Create sample file if empty
        if len(stock_symbols) == 0:
            sample_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
            with open("stock_symbols.txt", "w") as f:
                f.write("\n".join(sample_symbols))
            stock_symbols = sample_symbols
            st.info("Created sample stock_symbols.txt with default symbols")
            
    except FileNotFoundError:
        st.error("stock_symbols.txt not found. Creating sample file...")
        sample_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        with open("stock_symbols.txt", "w") as f:
            f.write("\n".join(sample_symbols))
        stock_symbols = sample_symbols
        st.info("Created stock_symbols.txt with sample symbols")
    except Exception as e:
        st.error(f"Error loading stock symbols: {str(e)}")
        return

    if scan_button:
        # Initialize LSTM components
        lstm_model = None
        lstm_scaler = None
        
        # LSTM Training
        if train_lstm:
            st.info("Training PyTorch LSTM model...")
            all_X, all_y = [], []
            valid_stocks = 0
            
            # Process ALL stocks
            for symbol in stock_symbols:
                try:
                    df = fetch_stock_data(symbol, start_date, end_date)
                    if df is not None and len(df) >= sequence_length + 10:
                        X, y, scaler = prepare_lstm_data(df, sequence_length=sequence_length, debug=True)
                        if X is not None and y is not None:
                            all_X.append(X)
                            all_y.append(y)
                            lstm_scaler = scaler
                            valid_stocks += 1
                            print(f"Prepared LSTM data for {symbol}")
                except Exception as e:
                    print(f"Error preparing LSTM data for {symbol}: {e}")
            
            if valid_stocks == 0:
                st.error("No valid data for LSTM training. Check date range or sequence length.")
                train_lstm = False
            else:
                print(f"Training on {valid_stocks} stocks...")
                try:
                    X_train = torch.cat(all_X, dim=0)
                    y_train = torch.cat(all_y, dim=0)
                    lstm_model = train_lstm_model(X_train, y_train, epochs=epochs)
                    st.success(f"LSTM trained on {valid_stocks} stocks!")
                except Exception as e:
                    st.error(f"LSTM training failed: {str(e)}")
                    train_lstm = False
        
        # Stock Scanning Process
        st.info("Starting stock pattern scan...")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        stock_data = []
        failed_symbols = []

        for i, symbol in enumerate(stock_symbols):
            if st.session_state.get('scan_cancelled', False):
                st.warning("Scan cancelled by user")
                break

            status_text.text(f"Processing {symbol} ({i+1}/{len(stock_symbols)})")
            progress_bar.progress((i + 1) / len(stock_symbols))

            try:
                df = fetch_stock_data(symbol, start_date, end_date)
                if df is None or df.empty:
                    failed_symbols.append(symbol)
                    continue

                # Pattern detection
                patterns = {
                    "Head and Shoulders": detect_head_and_shoulders(
                        df, depth=3, min_pattern_separation=10,
                        lstm_model=lstm_model, lstm_scaler=lstm_scaler, debug=False
                    ),
                    "Double Bottom": detect_double_bottom(
                        df, order=7, lstm_model=lstm_model,
                        lstm_scaler=lstm_scaler, debug=False
                    ),
                    "Cup and Handle": detect_cup_and_handle(
                        df, order=10, cup_min_bars=20,
                        handle_max_retrace=0.5, lstm_model=lstm_model,
                        lstm_scaler=lstm_scaler, debug=False
                    )
                }

                # Get current market data
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    volume = ticker.history(period="1d")['Volume'].iloc[-1]
                except Exception as e:
                    print(f"Error getting current data for {symbol}: {str(e)}")
                    current_price = None
                    volume = None

                # Calculate performance metrics
                percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

                stock_data.append({
                    "Symbol": symbol,
                    "Patterns": patterns,
                    "Data": df,
                    "Current Price": current_price,
                    "Volume": volume,
                    "Percent Change": percent_change,
                    "MA": df['MA'].iloc[-1] if 'MA' in df.columns and not pd.isna(df['MA'].iloc[-1]) else None,
                    "RSI": df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None,
                })

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue

        # Store results
        st.session_state.stock_data = stock_data
        st.session_state.selected_stock = None
        st.session_state.selected_pattern = None
        st.session_state.scan_cancelled = False

        # Show completion message
        progress_text.empty()
        progress_bar.empty()

        if st.session_state.get('scan_cancelled', False):
            status_text.warning("Scan was cancelled by user")
        else:
            if failed_symbols:
                status_text.warning(
                    f"Scan completed with {len(stock_data)} successful stocks. "
                    f"Failed on {len(failed_symbols)} symbols."
                )
                with st.expander("Show failed symbols"):
                    st.write(", ".join(failed_symbols))
            else:
                status_text.success("Scan completed successfully!")

            # Show summary statistics
            total_patterns = sum(
                len(stock['Patterns'][pattern])
                for stock in stock_data
                for pattern in stock['Patterns']
            )
            st.metric("Total Patterns Found", total_patterns)

    # Display results if available
    if 'stock_data' in st.session_state and st.session_state.stock_data:
        display_results()

def display_results():
    """Display the scan results in a user-friendly format"""
    st.markdown("## Scan Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📋 Stock List", "📈 Pattern Visualization"])
    
    # Set active tab based on session state
    if st.session_state.active_tab == "📈 Pattern Visualization":
        # Force the visualization tab to be active
        with tab1:
            st.empty()
        with tab2:
            display_visualization_tab()
    else:
        with tab1:
            display_stock_list_tab()
        with tab2:
            display_visualization_tab()

    # Add JavaScript to switch tabs if needed
    if st.session_state.get('force_tab_switch', False):
        st.session_state.force_tab_switch = False
        st.write("""
        <script>
            const tabButtons = parent.document.querySelectorAll('[data-testid="stTabButton"]');
            if (tabButtons.length >= 2) {
                tabButtons[1].click();
            }
        </script>
        """, unsafe_allow_html=True)

def display_stock_list_tab():
    """Display the stock list tab content"""
    # Prepare data for the table
    table_data = []
    for stock in st.session_state.stock_data:
        pattern_counts = {
            pattern: len(stock["Patterns"][pattern]) 
            for pattern in stock["Patterns"]
        }
        
        row = {
            "Symbol": stock["Symbol"],
            "Current Price": stock['Current Price'],
            "Volume": stock['Volume'],
            "Head and Shoulders": pattern_counts.get("Head and Shoulders", 0),
            "Double Bottom": pattern_counts.get("Double Bottom", 0),
            "Cup and Handle": pattern_counts.get("Cup and Handle", 0),
        }
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        pattern_filter = st.selectbox(
            "Filter by Pattern", 
            ["All Patterns", "Head and Shoulders", "Double Bottom", "Cup and Handle"]
        )
    
    # Apply filters
    filtered_df = df_table.copy()
    if pattern_filter != "All Patterns":
        filtered_df = filtered_df[filtered_df[pattern_filter] > 0]
    
    # Format the table
    formatted_df = filtered_df.copy()
    formatted_df["Current Price"] = formatted_df.apply(
        lambda row: f"₹{row['Current Price']:.2f}" if row["Symbol"].endswith(".NS") else f"${row['Current Price']:.2f}",
        axis=1
    )

    formatted_df["Volume"] = formatted_df["Volume"].apply(
        lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A"
    )

    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(formatted_df)
    gb.configure_default_column(
        cellStyle={'textAlign': 'center', 'verticalAlign': 'middle'},
        wrapText=True,
        resizable=True,
        sortable=True,
        filter=True
    )
    gb.configure_column("Symbol", width=100, cellStyle={'textAlign': 'left', 'verticalAlign': 'middle'})
    gb.configure_column("Current Price", width=120)
    gb.configure_column("Volume", width=120)
    gb.configure_column("Head and Shoulders", width=150)
    gb.configure_column("Double Bottom", width=150)
    gb.configure_column("Cup and Handle", width=150)
    grid_options = gb.build()
    grid_options['rowStyle'] = {'cursor': 'pointer'}

    # Display AgGrid
    AgGrid(formatted_df, gridOptions=grid_options, height=500, fit_columns_on_grid_load=True)
    
    # Show summary stats
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(filtered_df))
    with col2:
        total_patterns = filtered_df[["Head and Shoulders", "Double Bottom", "Cup and Handle"]].sum().sum()
        st.metric("Total Patterns", int(total_patterns))

def display_visualization_tab():
    """Display the visualization tab content"""
    st.markdown("### 📊 Pattern Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        def format_stock_name(stock):
            pattern_counts = {
                'HS': len(stock['Patterns']['Head and Shoulders']) if 'Head and Shoulders' in stock['Patterns'] else 0,
                'DB': len(stock['Patterns']['Double Bottom']) if 'Double Bottom' in stock['Patterns'] else 0,
                'CH': len(stock['Patterns']['Cup and Handle']) if 'Cup and Handle' in stock['Patterns'] else 0
            }
            return f"{stock['Symbol']} [HS:{pattern_counts['HS']}, DB:{pattern_counts['DB']}, CH:{pattern_counts['CH']}]"
        
        # Set default value based on selected_stock
        default_index = 0
        if 'selected_stock' in st.session_state and st.session_state.selected_stock is not None:
            stock_names = [format_stock_name(stock) for stock in st.session_state.stock_data]
            try:
                # Find the first matching stock name
                matching_names = [
                    i for i, name in enumerate(stock_names)
                    if st.session_state.selected_stock and name.startswith(st.session_state.selected_stock)
                ]
                if matching_names:
                    default_index = matching_names[0]
            except (ValueError, StopIteration):
                pass
        
        selected_display = st.selectbox(
            "Select Stock",
            options=[format_stock_name(stock) for stock in st.session_state.stock_data],
            index=default_index,
            key='stock_select'
        )
        
        selected_stock = selected_display.split(' [')[0]
        st.session_state.selected_stock = selected_stock
    
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
                # Set default value based on selected_pattern
                default_pattern_index = 0
                if 'selected_pattern' in st.session_state and st.session_state.selected_pattern is not None:
                    try:
                        default_pattern_index = next(
                            i for i, (pattern, display) in enumerate(pattern_options)
                            if pattern == st.session_state.selected_pattern
                        )
                    except StopIteration:
                        pass
                
                selected_display = st.selectbox(
                    "Select Pattern",
                    options=[display for _, display in pattern_options],
                    index=default_pattern_index,
                    key='pattern_select'
                )
                
                selected_pattern = next(
                    pattern for pattern, display in pattern_options 
                    if display == selected_display
                )
                st.session_state.selected_pattern = selected_pattern
            
            if st.session_state.selected_pattern:
                pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                
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
                
                with st.expander("📚 Pattern Details"):
                    if st.session_state.selected_pattern == "Head and Shoulders":
                        st.markdown("""
                        **Head and Shoulders**: Bearish reversal pattern with:
                        - A peak (head) between two lower peaks (shoulders)
                        - Neckline support that, when broken, confirms the pattern
                        - Price target estimated by the height of the pattern
                        """)
                    elif st.session_state.selected_pattern == "Double Bottom":
                        st.markdown("""
                        **Double Bottom**: Bullish reversal pattern with:
                        - Two distinct troughs at approximately the same price level
                        - Confirmation when price breaks above the intermediate peak
                        - Price target estimated by the height of the pattern
                        """)
                    elif st.session_state.selected_pattern == "Cup and Handle":
                        st.markdown("""
                        **Cup and Handle**: Bullish continuation pattern with:
                        - Rounded bottom (cup) formation
                        - Small downward drift (handle) before breakout
                        - Price target estimated by the depth of the cup
                        """)
        else:
            st.info(f"No patterns detected for {selected_stock}.")
    else:
        st.error("Selected stock data not found.")
if __name__ == "__main__":
    main()
