import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm
from plotly.subplots import make_subplots
import datetime

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

# Helper functions
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        if not df.empty:
            df = calculate_moving_average(df)
            df = calculate_rsi(df)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def find_extrema(df, order=5):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

# Improved Head and Shoulders pattern detection as provided
def detect_head_and_shoulders(df):
    prices = df['Close']
    peaks = argrelextrema(prices.values, np.greater, order=10)[0]
    patterns = []

    for i in range(len(peaks) - 2):
        LS, H, RS = peaks[i], peaks[i + 1], peaks[i + 2]

        # Check if the head is higher than the shoulders
        if prices.iloc[H] > prices.iloc[LS] and prices.iloc[H] > prices.iloc[RS]:
            # Check if the shoulders are roughly equal (within 5% tolerance)
            shoulder_diff = abs(prices.iloc[LS] - prices.iloc[RS]) / max(prices.iloc[LS], prices.iloc[RS])
            if shoulder_diff <= 0.05:  # 5% tolerance
                # Find neckline (troughs between shoulders and head)
                T1 = prices.iloc[LS:H + 1].idxmin()  # Trough between left shoulder and head
                T2 = prices.iloc[H:RS + 1].idxmin()  # Trough between head and right shoulder
                patterns.append({
                    "left_shoulder": LS,
                    "head": H,
                    "right_shoulder": RS,
                    "neckline": [T1, T2]
                })

    return patterns

def detect_double_top(df):
    peaks, _ = find_extrema(df, order=10)
    if len(peaks) < 2:
        return []

    patterns = []
    for i in range(len(peaks) - 1):
        if abs(df['Close'][peaks[i]] - df['Close'][peaks[i + 1]]) < df['Close'][peaks[i]] * 0.03:
            patterns.append({'peak1': peaks[i], 'peak2': peaks[i + 1]})
    return patterns

def detect_double_bottom(df):
    _, troughs = find_extrema(df, order=10)
    if len(troughs) < 2:
        return []

    patterns = []
    for i in range(len(troughs) - 1):
        if abs(df['Close'][troughs[i]] - df['Close'][troughs[i + 1]]) < df['Close'][troughs[i]] * 0.03:
            patterns.append({'trough1': troughs[i], 'trough2': troughs[i + 1]})
    return patterns

def detect_cup_and_handle(df):
    _, troughs = find_extrema(df, order=20)
    if len(troughs) < 1:
        return []

    patterns = []
    min_idx = troughs[0] 
    left_peak = df.iloc[:min_idx]['Close'].idxmax()
    right_peak = df.iloc[min_idx:]['Close'].idxmax()

    if not left_peak or not right_peak:
        return []

    handle_start_idx = right_peak
    handle_end_idx = None
    
    handle_retracement_level = 0.5 
    
    handle_start_price = df['Close'][handle_start_idx]
    
    potential_handle_bottom_idx = df.iloc[handle_start_idx:]['Close'].idxmin()

    if potential_handle_bottom_idx is not None:
        handle_bottom_price = df['Close'][potential_handle_bottom_idx]
        
        handle_top_price = handle_start_price - (handle_start_price - handle_bottom_price) * handle_retracement_level

        for i in range(potential_handle_bottom_idx + 1, len(df)):
            if df['Close'][i] > handle_top_price:
                handle_end_idx = i
                break

    if handle_end_idx:
         patterns.append({
            "left_peak": left_peak,
            "min_idx": min_idx,
            "right_peak": right_peak,
            "handle_start": handle_start_idx,
            "handle_end": handle_end_idx
        })
    return patterns

def calculate_moving_average(df, window=50):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):  # Changed from 50 to standard 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Improved plot function for Head and Shoulders pattern
def plot_head_and_shoulders(df, patterns):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}]]
    )

    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Stock Price', line=dict(color="#1E88E5", width=2)
    ), row=1, col=1)

    if 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA'],
            mode='lines', name="Moving Average (50)", line=dict(color="#FB8C00", width=2)
        ), row=1, col=1)

    for i, pattern in enumerate(patterns):
        LS, H, RS = pattern["left_shoulder"], pattern["head"], pattern["right_shoulder"]
        T1, T2 = pattern["neckline"]

        # Add markers for left shoulder, head, and right shoulder
        try:
            fig.add_trace(go.Scatter(
                x=[df.loc[int(LS), 'Date']], y=[df.loc[int(LS), 'Close']],
                mode="markers+text", text=["Left Shoulder"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"), name=f"Left Shoulder {i + 1}"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[df.loc[int(H), 'Date']], y=[df.loc[int(H), 'Close']],
                mode="markers+text", text=["Head"], textposition="top center",
                marker=dict(size=14, color="#4CAF50", symbol="circle"), name=f"Head {i + 1}"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[df.loc[int(RS), 'Date']], y=[df.loc[int(RS), 'Close']],
                mode="markers+text", text=["Right Shoulder"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"), name=f"Right Shoulder {i + 1}"
            ), row=1, col=1)

            # Add neckline
            fig.add_trace(go.Scatter(
                x=[df.loc[int(T1), 'Date'], df.loc[int(T2), 'Date']],
                y=[df.loc[int(T1), 'Close'], df.loc[int(T2), 'Close']],
                mode="lines", name=f"Neckline {i + 1}", line=dict(color="#673AB7", width=2, dash="dash")
            ), row=1, col=1)
        except KeyError:
            # Skip this pattern if any points are not in the dataframe
            continue

        # Shade the pattern area
        shoulder_points = [LS, H, RS]
        area_dates = df.loc[shoulder_points, 'Date'].tolist()
        area_prices = df.loc[shoulder_points, 'Close'].tolist()
        fig.add_trace(go.Scatter(
            x=area_dates, y=area_prices, fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.2)', line=dict(color="rgba(0,0,0,0)"), name=f"Pattern Area {i + 1}"
        ), row=1, col=1)

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

    fig.update_layout(
        title={
            'text': "Head & Shoulders Pattern Detection",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color="#0D47A1")
        },
        xaxis_title="Date",
        xaxis=dict(visible=False, showticklabels=False, showgrid=False),
        xaxis2=dict(visible=False, showticklabels=False, showgrid=False),
        xaxis3=dict(title="Date"),
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI",
        showlegend=True,
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=120, t=100, b=40),
        hovermode="x unified"
    )
    return fig

def plot_pattern(df, pattern_points, pattern_name):
    # Create a modern color palette
    colors = {
        'background': '#ffffff',
        'price': '#2962FF',
        'price_area': 'rgba(41, 98, 255, 0.1)',
        'ma': '#FF6D00',
        'volume': 'rgba(158, 158, 158, 0.5)',
        'rsi': '#6200EA',
        'overbought': '#F44336',
        'oversold': '#4CAF50',
        'pattern': '#E91E63',
        'cup': '#9C27B0',
        'handle': '#FF9800',
        'base': '#4CAF50'
    }
    
    # Use the specialized H&S plotting function if applicable
    if pattern_name == "Head and Shoulders":
        return plot_head_and_shoulders(df, pattern_points)
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price", "Volume", "RSI")
    )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            mode='lines', 
            name="Price", 
            line=dict(color=colors['price'], width=2)
        ),
        row=1, col=1
    )

    # Add price area
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            fill='tozeroy', 
            fillcolor=colors['price_area'], 
            line=dict(color='rgba(255,255,255,0)'), 
            name="Price Area",
            showlegend=False
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
                name="MA (50)", 
                line=dict(color=colors['ma'], width=1.5)
            ),
            row=1, col=1
        )
    
    # Add pattern points
    if not isinstance(pattern_points, list):
        pattern_points = [pattern_points]
    
    pattern_colors = [
        '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', 
        '#3F51B5', '#CDDC39', '#795548', '#607D8B'
    ]
    
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = pattern_colors[idx % len(pattern_colors)]
        
        for key, point_idx in pattern.items():
            if isinstance(point_idx, (np.int64, int)):  
                fig.add_trace(
                    go.Scatter(
                        x=[df.loc[point_idx, 'Date']], 
                        y=[df.loc[point_idx, 'Close']],
                        mode="markers+text", 
                        text=[key], 
                        textposition="top center",
                        textfont=dict(size=10, color="#424242"),
                        marker=dict(size=8, color=color, line=dict(width=1, color='#ffffff')),
                        name=f"{key} {idx + 1}"
                    ),
                    row=1, col=1
                )
            elif isinstance(point_idx, list):
                for i in range(len(point_idx) - 1):
                    fig.add_trace(
                        go.Scatter(
                            x=[df.loc[point_idx[i], 'Date'], df.loc[point_idx[i + 1], 'Date']],
                            y=[df.loc[point_idx[i], 'Close'], df.loc[point_idx[i + 1], 'Close']],
                            mode="lines", 
                            name=f"{key} Line {idx + 1}", 
                            line=dict(color=color, width=1.5)
                        ),
                        row=1, col=1
                    )
            
        # Cup and Handle specific visualization
        if pattern_name == "Cup and Handle" and "handle_start" in pattern and "handle_end" in pattern:
            cup_dates = df['Date'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            cup_prices = df['Close'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            handle_dates = df['Date'][pattern['handle_start']:pattern['handle_end']+1].tolist()
            handle_prices = df['Close'][pattern['handle_start']:pattern['handle_end']+1].tolist()

            fig.add_trace(
                go.Scatter(
                    x=cup_dates, 
                    y=cup_prices, 
                    mode="lines", 
                    name="Cup", 
                    line=dict(color=colors['cup'], width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=handle_dates, 
                    y=handle_prices, 
                    mode="lines", 
                    name="Handle", 
                    line=dict(color=colors['handle'], width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['left_peak'], 'Date']], 
                    y=[df.loc[pattern['left_peak'], 'Close']],
                    mode="markers+text", 
                    text="Left Cup Lip", 
                    textposition="top right",
                    textfont=dict(size=10, color="#424242"),
                    marker=dict(size=8, color="#2196F3", line=dict(width=1, color='#ffffff')),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['right_peak'], 'Date']], 
                    y=[df.loc[pattern['right_peak'], 'Close']],
                    mode="markers+text", 
                    text="Right Cup Lip", 
                    textposition="top left",
                    textfont=dict(size=10, color="#424242"),
                    marker=dict(size=8, color="#2196F3", line=dict(width=1, color='#ffffff')),
                    showlegend=False
                ),
                row=1, col=1
            )

            min_cup_price = min(cup_prices)
            fig.add_trace(
                go.Scatter(
                    x=[cup_dates[0], cup_dates[-1]], 
                    y=[min_cup_price, min_cup_price],
                    mode="lines", 
                    name="Base", 
                    line=dict(color=colors['base'], width=1.5, dash="dash")
                ),
                row=1, col=1
            )
    
        # Connect pattern points for other patterns
        if pattern_name in ["Double Top", "Double Bottom"]:
            x_values = [df.loc[pattern[key], 'Date'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            y_values = [df.loc[pattern[key], 'Close'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode="lines", 
                    line=dict(color=color, width=1.5, dash="dash"),
                    name=f"Pattern {idx + 1}"
                ),
                row=1, col=1
            )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name="Volume", 
            marker_color=colors['volume'],
            marker_line_width=0
        ),
        row=2, col=1
    )
    
    # Add RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['RSI'], 
                mode='lines', 
                name="RSI (14)", 
                line=dict(color=colors['rsi'], width=1.5)
            ),
            row=3, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=70,
            x1=df['Date'].iloc[-1],
            y1=70,
            line=dict(color=colors['overbought'], width=1, dash="dash"),
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=30,
            x1=df['Date'].iloc[-1],
            y1=30,
            line=dict(color=colors['oversold'], width=1, dash="dash"),
            row=3, col=1
        )
        
        # Add annotations for overbought/oversold
        fig.add_annotation(
            x=df['Date'].iloc[0],
            y=70,
            text="Overbought (70)",
            showarrow=False,
            xshift=50,
            yshift=10,
            font=dict(size=10, color=colors['overbought']),
            row=3, col=1
        )
        
        fig.add_annotation(
            x=df['Date'].iloc[0],
            y=30,
            text="Oversold (30)",
            showarrow=False,
            xshift=50,
            yshift=-10,
            font=dict(size=10, color=colors['oversold']),
            row=3, col=1
        )
    
    # Update layout for a modern look
    fig.update_layout(
        title={
            'text': f"{pattern_name} Pattern for {df['Date'].iloc[0].strftime('%b %Y')} - {df['Date'].iloc[-1].strftime('%b %Y')}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#212121')
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=120, t=100, b=40),
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.05)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.1)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.05)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.1)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.05)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.1)',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.05)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.1)',
        row=3, col=1
    )
    
    return fig

def evaluate_pattern_detection(df, patterns):
    total_patterns = 0
    correct_predictions = 0
    false_positives = 0
    look_forward_window = 10

    for pattern_type, pattern_list in patterns.items():
        total_patterns += len(pattern_list)
        for pattern in pattern_list:
            if pattern_type == "Head and Shoulders":
                last_point_idx = max(pattern['left_shoulder'], pattern['head'], pattern['right_shoulder'])
            elif pattern_type == "Double Top":
                last_point_idx = max(pattern['peak1'], pattern['peak2'])
            elif pattern_type == "Double Bottom":
                last_point_idx = max(pattern['trough1'], pattern['trough2'])
            elif pattern_type == "Cup and Handle":
                last_point_idx = pattern['handle_end']
            else:
                continue 

            if last_point_idx + look_forward_window < len(df):
                if pattern_type in ["Double Bottom", "Cup and Handle"]:
                    if df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        false_positives += 1
                elif pattern_type in ["Head and Shoulders", "Double Top"]:
                    if df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        false_positives += 1

    if total_patterns > 0:
        accuracy = correct_predictions / total_patterns
        precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0
    else:
        accuracy = 0.0
        precision = 0.0

    return accuracy, precision

def is_trading_day(date):
    """Check if the given date is a trading day (Monday to Friday)."""
    return date.weekday() < 5 

def get_nearest_trading_day(date):
    """Adjust the date to the nearest previous trading day if it's a weekend or holiday."""
    while not is_trading_day(date):
        date -= datetime.timedelta(days=1)
    return date

# Main app
def main():
    # App header with title
    st.markdown("# Stock Pattern Scanner(Dyanmic)")
    st.markdown("Identify technical chart patterns with precision")

    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <style>
            .block-container {
                margin-bottom: 0.5rem; /* Adjust this value to change the spacing */
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("# Scanner Settings")
        # st.markdown("---")
        
        # Date selection
        st.markdown("### 📅 Date Selection")
        date_option = st.radio(
            "Select Date Option",
            options=["Date Range"],
            index=0,
            help="Choose between a date range or a specific trading day"
        )

        start_date = None
        end_date = None
        single_date = None

        if date_option == "Date Range":
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
                return

        elif date_option == "Single Day":
            single_date = st.date_input(
                "Select Date",
                value=datetime.date(2023, 1, 1),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31)
            )
            
            if not is_trading_day(single_date):
                st.warning(f"{single_date} is not a trading day. Adjusting to the nearest trading day.")
                single_date = get_nearest_trading_day(single_date)
            
            start_date = single_date
            end_date = single_date

        # st.markdown("---")
        
        # Scan button
        # st.markdown("### 🚀 Start Scanning")
        scan_button = st.button("Scan Stocks", use_container_width=True)
        
        if date_option == "Date Range":
            st.info(f"Selected: **{start_date}** to **{end_date}**")
        elif date_option == "Single Day":
            st.info(f"Selected: **{single_date}**")
            
        # st.markdown("---")
        
        # About section
        # with st.expander("ℹ️ About This App"):
        #     st.markdown("""
        #     This app scans stocks for technical chart patterns including:
            
        #     - Head and Shoulders
        #     - Double Top
        #     - Double Bottom
        #     - Cup and Handle
            
        #     The scanner evaluates pattern accuracy and provides visualization tools.
        #     """)
    
    # Main content
    if scan_button:
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
                status_container.info(f"Processing {symbol}... ({i+1}/{len(stock_symbols)})")
                
                try:
                    df = fetch_stock_data(symbol, start_date, end_date)
                    if df is None or df.empty:
                        continue
                    
                    patterns = {
                        "Head and Shoulders": detect_head_and_shoulders(df),
                        "Double Top": detect_double_top(df),
                        "Double Bottom": detect_double_bottom(df),
                        "Cup and Handle": detect_cup_and_handle(df),
                    }
                    
                    accuracy, precision = evaluate_pattern_detection(df, patterns)
                    
                    stock_info = yf.Ticker(symbol).info
                    current_price = stock_info.get('currentPrice', None)
                    volume = stock_info.get('volume', None)
                    percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

                    stock_data.append({
                        "Symbol": symbol, 
                        "Patterns": patterns, 
                        "Data": df,
                        "Current Price": current_price,
                        "Volume": volume,
                        "Percent Change": percent_change,
                        "Accuracy": accuracy,
                        "Precision": precision,
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
                    "Accuracy": stock['Accuracy'],
                    "Precision": stock['Precision'],
                    "Head and Shoulders": pattern_counts.get("Head and Shoulders", 0),
                    "Double Top": pattern_counts.get("Double Top", 0),
                    "Double Bottom": pattern_counts.get("Double Bottom", 0),
                    "Cup and Handle": pattern_counts.get("Cup and Handle", 0),
                }
                table_data.append(row)
            
            # Create DataFrame
            df_table = pd.DataFrame(table_data)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0, step=1.0)
            with col2:
                min_volume = st.number_input("Min Volume", min_value=0, value=0, step=1000)
            with col3:
                pattern_filter = st.selectbox("Filter by Pattern", 
                                             ["All Patterns", "Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle"])
            
            # Apply filters
            filtered_df = df_table.copy()
            if min_price > 0:
                filtered_df = filtered_df[filtered_df["Current Price"] >= min_price]
            if min_volume > 0:
                filtered_df = filtered_df[filtered_df["Volume"] >= min_volume]
            if pattern_filter != "All Patterns":
                filtered_df = filtered_df[filtered_df[pattern_filter] > 0]
            
            # Format the table
            formatted_df = filtered_df.copy()
            formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Volume"] = formatted_df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")
            formatted_df["% Change"] = formatted_df["% Change"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            formatted_df["MA (50)"] = formatted_df["MA (50)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["RSI (14)"] = formatted_df["RSI (14)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
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
                    "Accuracy": st.column_config.TextColumn("Accuracy"),
                    "Precision": st.column_config.TextColumn("Precision"),
                    "Head and Shoulders": st.column_config.NumberColumn("H&S", format="%d"),
                    "Double Top": st.column_config.NumberColumn("Double Top", format="%d"),
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
                total_patterns = filtered_df[["Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle"]].sum().sum()
                st.metric("Total Patterns", int(total_patterns))
            with col3:
                avg_accuracy = filtered_df["Accuracy"].mean()
                st.metric("Avg. Accuracy", f"{avg_accuracy:.2f}")
            with col4:
                avg_precision = filtered_df["Precision"].mean()
                st.metric("Avg. Precision", f"{avg_precision:.2f}")
        
        with tab2:
            st.markdown("### 🔍 Pattern Visualization")
            
            # Stock selection
            col1, col2 = st.columns(2)
            with col1:
                selected_stock = st.selectbox(
                    "Select Stock",
                    options=[stock["Symbol"] for stock in st.session_state.stock_data],
                    key='stock_select'
                )
            
            if selected_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = selected_stock
                st.session_state.selected_pattern = None
            
            selected_data = next((item for item in st.session_state.stock_data 
                                if item["Symbol"] == st.session_state.selected_stock), None)
            
            if selected_data:
                # Pattern selection
                pattern_options = [p for p, v in selected_data["Patterns"].items() if v]
                
                if pattern_options:
                    with col2:
                        selected_pattern = st.selectbox(
                            "Select Pattern",
                            options=pattern_options,
                            key='pattern_select'
                        )
                    
                    if selected_pattern != st.session_state.selected_pattern:
                        st.session_state.selected_pattern = selected_pattern
                    
                    if st.session_state.selected_pattern:
                        pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                        
                        # Display stock info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${selected_data['Current Price']:.2f}" if selected_data['Current Price'] else "N/A")
                        with col2:
                            st.metric("Volume", f"{int(selected_data['Volume']):,}" if selected_data['Volume'] else "N/A")
                        with col3:
                            st.metric("% Change", f"{selected_data['Percent Change']:.2f}%")
                        with col4:
                            st.metric("RSI (14)", f"{selected_data['RSI']:.2f}" if selected_data['RSI'] else "N/A")
                        
                        # Plot the pattern
                        if not isinstance(pattern_points, list):
                            pattern_points = [pattern_points]
                        
                        st.plotly_chart(
                            plot_pattern(
                                selected_data["Data"],
                                pattern_points,
                                st.session_state.selected_pattern
                            ),
                            use_container_width=True
                        )
                        
                        # Pattern explanation
                        with st.expander("📚 Pattern Explanation"):
                            if selected_pattern == "Head and Shoulders":
                                st.markdown("""
                                **Head and Shoulders Pattern**
                                
                                A bearish reversal pattern that signals a potential trend change from bullish to bearish. It consists of:
                                - Left shoulder: A peak followed by a decline
                                - Head: A higher peak followed by another decline
                                - Right shoulder: A lower peak similar to the left shoulder
                                
                                **Trading Implications**: When the price breaks below the neckline (support level), it often indicates a strong sell signal.
                                """)
                            elif selected_pattern == "Double Top":
                                st.markdown("""
                                **Double Top Pattern**
                                
                                A bearish reversal pattern that forms after an extended upward trend. It consists of:
                                - Two peaks at approximately the same price level
                                - A valley (trough) between the peaks
                                
                                **Trading Implications**: When the price falls below the valley between the two tops, it signals a potential downtrend.
                                """)
                            elif selected_pattern == "Double Bottom":
                                st.markdown("""
                                **Double Bottom Pattern**
                                
                                A bullish reversal pattern that forms after an extended downward trend. It consists of:
                                - Two troughs at approximately the same price level
                                - A peak between the troughs
                                
                                **Trading Implications**: When the price rises above the peak between the two bottoms, it signals a potential uptrend.
                                """)
                            elif selected_pattern == "Cup and Handle":
                                st.markdown("""
                                **Cup and Handle Pattern**
                                
                                A bullish continuation pattern that signals a pause in an uptrend before continuing higher. It consists of:
                                - Cup: A rounded bottom formation (U-shape)
                                - Handle: A slight downward drift forming a consolidation
                                
                                **Trading Implications**: The pattern completes when the price breaks above the resistance level formed by the cup's rim.
                                """)
                else:
                    st.info(f"No patterns detected for {selected_stock}.")
            else:
                st.error("Selected stock data not found.")

if __name__ == "__main__":
    main()