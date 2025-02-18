import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_pattern' not in st.session_state:
    st.session_state.selected_pattern = None

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    print(f"Fetched data for {symbol} from {start_date} to {end_date}:")
    print(df)
    df.reset_index(inplace=True)
    if not df.empty:
        df = calculate_moving_average(df)  # Add Moving Average
        df = calculate_rsi(df)  # Add RSI
    return df if not df.empty else None

def find_extrema(df, order=5):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

def detect_head_and_shoulders(df):
    peaks, troughs = find_extrema(df, order=10)
    patterns = []
    
    if len(peaks) < 3:
        return patterns
    
    for i in range(1, len(peaks) - 1):
        left_shoulder, head, right_shoulder = peaks[i - 1], peaks[i], peaks[i + 1]
        if (df['Close'][left_shoulder] < df['Close'][head]) and \
           (df['Close'][right_shoulder] < df['Close'][head]) and \
           (abs(df['Close'][left_shoulder] - df['Close'][right_shoulder]) < df['Close'][head] * 0.05):
            patterns.append({'left_shoulder': left_shoulder, 'head': head, 'right_shoulder': right_shoulder})
    
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
    min_idx = troughs[0]  # Lowest point of the cup
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
    
def detect_ascending_triangle(df):
    peaks, troughs = find_extrema(df, order=10)
    if len(peaks) < 2 or len(troughs) < 2:
        return []
    patterns = []
    if abs(df['Close'][peaks[-1]] - df['Close'][peaks[-2]]) < df['Close'][peaks[-1]] * 0.03:
        patterns.append({"resistance": peaks[-1], "support": troughs[-1]})
    return patterns

def detect_descending_triangle(df):
    peaks, troughs = find_extrema(df, order=10)
    if len(peaks) < 2 or len(troughs) < 2:
        return []
    patterns = []
    if abs(df['Close'][troughs[-1]] - df['Close'][troughs[-2]]) < df['Close'][troughs[-1]] * 0.03:
        patterns.append({"support": troughs[-1], "resistance": peaks[-1]})
    return patterns

def detect_symmetrical_triangle(df, order=10):
    """
    Detects symmetrical triangle patterns in the stock data.
    Returns a list of dictionaries containing the pattern details.
    """
    peaks, troughs = find_extrema(df, order=order)
    patterns = []
    
    if len(peaks) < 2 or len(troughs) < 2:
        return patterns 
    
    for i in range(len(peaks) - 1):
        for j in range(len(troughs) - 1):
            upper_trend = [peaks[i], peaks[i + 1]]
            lower_trend = [troughs[j], troughs[j + 1]]
            
            if (df['Close'][upper_trend[1]] < df['Close'][upper_trend[0]]) and \
               (df['Close'][lower_trend[1]] > df['Close'][lower_trend[0]]):
                
                start_date = df['Date'].iloc[min(upper_trend[0], lower_trend[0])]
                end_date = df['Date'].iloc[max(upper_trend[1], lower_trend[1])]
                duration = (end_date - start_date).days
                

                volume_trend = df['Volume'].iloc[min(upper_trend[0], lower_trend[0]):max(upper_trend[1], lower_trend[1]) + 1]
                volume_decreasing = volume_trend.is_monotonic_decreasing
                
                breakout_point = None
                breakout_direction = None
                for k in range(max(upper_trend[1], lower_trend[1]) + 1, len(df)):
                    if df['Close'][k] > df['Close'][upper_trend[1]]:
                        breakout_point = k
                        breakout_direction = "up"
                        break
                    elif df['Close'][k] < df['Close'][lower_trend[1]]:
                        breakout_point = k
                        breakout_direction = "down"
                        break
                
                if breakout_point:
                    widest_part = abs(df['Close'][upper_trend[0]] - df['Close'][lower_trend[0]])
                    if breakout_direction == "up":
                        price_target = df['Close'][breakout_point] + widest_part
                    else:
                        price_target = df['Close'][breakout_point] - widest_part
                else:
                    price_target = None
                
                patterns.append({
                    "upper_trend": upper_trend,
                    "lower_trend": lower_trend,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration": duration,
                    "volume_decreasing": volume_decreasing,
                    "breakout_point": breakout_point,
                    "breakout_direction": breakout_direction,
                    "price_target": price_target
                })
    
    return patterns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_moving_average(df, window=14):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def plot_pattern(df, pattern_points, pattern_name):
    # Create subplots: 3 rows (Price + Volume + RSI), 1 column
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.2, 0.2]
    )
    
    # Plot Stock Price
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Stock Price", line=dict(color="blue")),
        row=1, col=1
    )

    # Plot Moving Average (MA) if available
    if 'MA' in df.columns:
        print("MA column exists.")
        print(df[['Date', 'Close', 'MA']].head())  # Print first few rows of MA
    else:
        print("MA column does not exist.")
    if 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA'], mode='lines', name="Moving Average (14)", line=dict(color="orange")),
            row=1, col=1
        )
    
    # Plot Price Area (optional)
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'], fill='tozeroy', 
            fillcolor='rgba(0, 100, 255, 0.2)', line=dict(color='rgba(255,255,255,0)'), 
            name="Price Area"
        ),
        row=1, col=1
    )
    
    # Convert pattern_points to a list if it's not already
    if not isinstance(pattern_points, list):
        pattern_points = [pattern_points]
    
    # List of colors and shapes for different patterns
    colors = [
        'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 
        'yellow', 'pink', 'brown', 'lime', 'teal', 
        'gold', 'indigo', 'violet', 'navy', 'turquoise', 
        'coral', 'darkred', 'darkgreen', 'darkblue', 'silver', 
        'maroon', 'olive', 'chocolate', 'crimson', 'darkcyan', 
        'darkmagenta', 'deepskyblue', 'firebrick', 'hotpink', 
        'lightcoral', 'lightskyblue', 'mediumseagreen', 
        'orangered', 'royalblue', 'saddlebrown', 'slateblue', 
        'springgreen', 'tomato'
    ]
    shapes = ['circle']
    
    # Plot each pattern
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = colors[idx % len(colors)]
        shape = shapes[idx % len(shapes)]
        
        for key, point_idx in pattern.items():
            if isinstance(point_idx, (np.int64, int)):  
                fig.add_trace(
                    go.Scatter(
                        x=[df.loc[point_idx, 'Date']], y=[df.loc[point_idx, 'Close']],
                        mode="markers+text", text=[key], textposition="top center",
                        textfont=dict(size=10),  # Reduced text size
                        marker=dict(size=6, color=color, symbol=shape),  # Reduced marker size
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
                            mode="lines", name=f"{key} Line {idx + 1}", line=dict(color=color)
                        ),
                        row=1, col=1
                    )
            
        #------CUP HANDLE-------#
        if pattern_name == "Cup and Handle" and "handle_start" in pattern and "handle_end" in pattern:
            cup_dates = df['Date'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            cup_prices = df['Close'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            handle_dates = df['Date'][pattern['handle_start']:pattern['handle_end']+1].tolist()
            handle_prices = df['Close'][pattern['handle_start']:pattern['handle_end']+1].tolist()

            fig.add_trace(
                go.Scatter(x=cup_dates, y=cup_prices, mode="lines", name="Cup", line=dict(color="purple")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=handle_dates, y=handle_prices, mode="lines", name="Handle", line=dict(color="orange")),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['left_peak'], 'Date']], y=[df.loc[pattern['left_peak'], 'Close']],
                    mode="markers+text", text="Left Cup Lip", textposition="top right",
                    textfont=dict(size=10),  # Reduced text size
                    marker=dict(color="blue", size=6)  # Reduced marker size
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['right_peak'], 'Date']], y=[df.loc[pattern['right_peak'], 'Close']],
                    mode="markers+text", text="Right Cup Lip", textposition="top left",
                    textfont=dict(size=10),  # Reduced text size
                    marker=dict(color="blue", size=6)  # Reduced marker size
                ),
                row=1, col=1
            )

            min_cup_price = min(cup_prices)
            fig.add_trace(
                go.Scatter(
                    x=[cup_dates[0], cup_dates[-1]], y=[min_cup_price, min_cup_price],
                    mode="lines", name="Base", line=dict(color="green")
                ),
                row=1, col=1
            )
    
        # Encircle the pattern with a circle
        if pattern_name in ["Head and Shoulders", "Double Top", "Double Bottom"]:
            x_values = [df.loc[pattern[key], 'Date'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            y_values = [df.loc[pattern[key], 'Close'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, y=y_values,
                    mode="lines", line=dict(color=color, width=2, dash="dash"),
                    name=f"Pattern {idx + 1} Circle"
                ),
                row=1, col=1
            )
    
    # Add Volume Plot
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker_color='rgba(100, 100, 100, 0.5)'),
        row=2, col=1
    )
    
    # Add RSI Plot if available
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name="RSI (14)", line=dict(color="purple")),
            row=3, col=1
        )
        # Add RSI overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold (30)")
    
    # Update layout
    fig.update_layout(
        title=f"{pattern_name} Pattern with MA and RSI",
        xaxis_title="Date",
        height=800,
        showlegend=True
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def evaluate_pattern_detection(df, patterns):
    total_patterns = 0
    correct_predictions = 0
    false_positives = 0
    look_forward_window = 10

    for pattern_type, pattern_list in patterns.items():
        total_patterns += len(pattern_list)
        for pattern in pattern_list:
            # Logic to determine the "end" of the pattern

            if pattern_type == "Head and Shoulders":
                last_point_idx = max(pattern['left_shoulder'], pattern['head'], pattern['right_shoulder'])
            elif pattern_type == "Double Top":
                last_point_idx = max(pattern['peak1'], pattern['peak2'])
            elif pattern_type == "Double Bottom":
                last_point_idx = max(pattern['trough1'], pattern['trough2'])
            elif pattern_type == "Cup and Handle":
                last_point_idx = pattern['handle_end']
            elif pattern_type in ["Ascending Triangle", "Descending Triangle"]:
                # Use the resistance or support index as the last point
                last_point_idx = pattern.get("resistance") or pattern.get("support")
            elif pattern_type == "Symmetrical Triangle":
                last_point_idx = max(pattern['upper_trend'][1], pattern['lower_trend'][1])
            else:
                continue  # Skip unknown pattern types


            if last_point_idx + look_forward_window < len(df):
                # Check for bullish patterns
                if pattern_type in ["Double Bottom", "Cup and Handle", "Ascending Triangle", "Symmetrical Triangle"]:
                    if df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:  # Check for *opposite* movement
                        false_positives += 1
                # Check for bearish patterns
                elif pattern_type in ["Head and Shoulders", "Double Top", "Descending Triangle"]:
                    if df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]: # Check for *opposite* movement
                        false_positives += 1

    if total_patterns > 0:
        accuracy = correct_predictions / total_patterns
        precision = correct_predictions / (correct_predictions + false_positives)
    else:
        accuracy = 0.0
        precision = 0.0

    return accuracy, precision


try:
    with open("stock_symbols.txt", "r") as f:
        stock_symbols = [line.strip() for line in f]
except FileNotFoundError:
    st.error("stock_symbols.txt not found. Please create the file with stock symbols, one per line.")
    st.stop() 
except Exception as e: 
    st.error(f"An error occurred while reading the stock symbols file: {e}")
    st.stop()
    
st.title("📊Stock Pattern Scanner")
st.sidebar.header("🔍 Select Date Range or Single Day")

# Option to choose between date range and single day
date_option = st.sidebar.radio(
    "Select Date Option",
    options=["Date Range", "Single Day"],
    index=0  # Default to "Date Range"
)

# Initialize start_date and end_date
start_date = None
end_date = None
single_date = None

import datetime

def is_trading_day(date):
    """
    Check if the given date is a trading day (Monday to Friday).
    """
    return date.weekday() < 5  # Monday = 0, Sunday = 6

def get_nearest_trading_day(date):
    """
    Adjust the date to the nearest previous trading day if it's a weekend or holiday.
    """
    while not is_trading_day(date):
        date -= datetime.timedelta(days=1)
    return date
# Date Range Option
if date_option == "Date Range":
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime("2023-01-01"),
        min_value=pd.to_datetime("1900-01-01"),
        max_value=pd.to_datetime("2100-12-31")
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime("2024-01-01"),
        min_value=pd.to_datetime("1900-01-01"),
        max_value=pd.to_datetime("2100-12-31")
    )
    # Ensure end_date is after start_date
    if end_date < start_date:
        st.sidebar.error("End Date must be after Start Date.")
        st.stop()

# Single Day Option
# Single Day Option
# Single Day Option

elif date_option == "Single Day":
    single_date = st.sidebar.date_input(
        "Select Date",
        value=pd.to_datetime("2023-01-01"),
        min_value=pd.to_datetime("1900-01-01"),
        max_value=pd.to_datetime("2100-12-31")
    )
    
    # Check if the selected date is a trading day
    if not is_trading_day(single_date):
        st.sidebar.warning(f"{single_date} is not a trading day (weekend or holiday). Adjusting to the nearest trading day.")
        single_date = get_nearest_trading_day(single_date)
    
    # Set start_date and end_date to the same day for single day selection
    start_date = single_date
    end_date = single_date  # Set end_date to the same day as start_date

# Display selected dates
if date_option == "Date Range":
    st.sidebar.write(f"Selected Date Range: **{start_date}** to **{end_date}**")
elif date_option == "Single Day":
    st.sidebar.write(f"Selected Single Day: **{single_date}**")

if st.sidebar.button("Scan Stocks"):
    stock_data = []
    progress_bar = st.progress(0)
    
    for i, symbol in tqdm(enumerate(stock_symbols), total=len(stock_symbols)):
        try:
            # Fetch stock data and calculate MA and RSI
            df = fetch_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                st.warning(f"No data found for {symbol}. Skipping...")
                continue
            
            # Print fetched data for debugging
            print(f"Data for {symbol}:")
            print(df)
            
            # Detect patterns
            patterns = {
                "Head and Shoulders": detect_head_and_shoulders(df),
                "Double Top": detect_double_top(df),
                "Double Bottom": detect_double_bottom(df),
                "Cup and Handle": detect_cup_and_handle(df),
                # "Ascending Triangle": detect_ascending_triangle(df),
                # "Descending Triangle": detect_descending_triangle(df),
                # "Symmetrical Triangle": detect_symmetrical_triangle(df),
            }
            
            # Print detected patterns for debugging
            print(f"Patterns for {symbol}:")
            print(patterns)
            
            # Evaluate patterns before using MA and RSI
            accuracy_before, precision_before = evaluate_pattern_detection(df, patterns)
            
            # Evaluate patterns after using MA and RSI
            accuracy_after, precision_after = evaluate_pattern_detection(df, patterns)
            
            # Fetch additional stock info
            stock_info = yf.Ticker(symbol).info
            current_price = stock_info.get('currentPrice', None)
            volume = stock_info.get('volume', None)
            percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            accuracy, precision = evaluate_pattern_detection(df, patterns)

            # Append stock data to the list
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
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(stock_symbols))
    
    # Store stock data in session state
    st.session_state.stock_data = stock_data
    st.session_state.selected_stock = None
    st.session_state.selected_pattern = None
    
    st.success("Stock scanning completed!") 
if st.session_state.stock_data:
    table_data = []
    for stock in st.session_state.stock_data:
        row = {
            "Symbol": stock["Symbol"],
            "Current Price": f"${stock['Current Price']:.2f}" if stock['Current Price'] else "N/A",
            "Volume": f"{stock['Volume']:,}" if stock['Volume'] else "N/A",
            "Percent Change": f"{stock['Percent Change']:.2f}%",
            "MA (14)": f"{stock['Data']['MA'].iloc[-1]:.2f}" if 'MA' in stock['Data'].columns else "N/A",
            "RSI (14)": f"{stock['Data']['RSI'].iloc[-1]:.2f}" if 'RSI' in stock['Data'].columns else "N/A",
            "Accuracy": f"{stock['Accuracy']:.2f}" if 'Accuracy' in stock else "N/A",
            "Precision": f"{stock['Precision']:.2f}" if 'Precision' in stock else "N/A",
        }

        for pattern in stock["Patterns"]:
            row[pattern] = "✅" if stock["Patterns"][pattern] else "❌"
        table_data.append(row)
    
    st.dataframe(
        pd.DataFrame(table_data),
        height=600,
        column_config={
            "Symbol": "Symbol",
            "Current Price": "Price",
            "Volume": "Volume",
            "Percent Change": "Change (%)",
            "MA (14)": "Moving Average (14)",
            "RSI (14)": "RSI (14)",
            "Accuracy Before": "Accuracy (Before)",
            "Precision Before": "Precision (Before)",
            "Accuracy After": "Accuracy (After)",
            "Precision After": "Precision (After)",
        },
        use_container_width=True
    )
    
    selected_stock = st.selectbox(
        "Select a Stock",
        options=[stock["Symbol"] for stock in st.session_state.stock_data],
        key='stock_select'
    )
    
    if selected_stock != st.session_state.selected_stock:
        st.session_state.selected_stock = selected_stock
        st.session_state.selected_pattern = None 
    
    selected_data = next((item for item in st.session_state.stock_data 
                         if item["Symbol"] == st.session_state.selected_stock), None)
    
    if selected_data:
        pattern_options = [p for p, v in selected_data["Patterns"].items() if v]
        selected_pattern = st.selectbox(
            "Select Pattern",
            options=pattern_options,
            key='pattern_select'
        )
        
        if selected_pattern != st.session_state.selected_pattern:
            st.session_state.selected_pattern = selected_pattern
        
        if st.session_state.selected_pattern:
            pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
            if not isinstance(pattern_points, list):
                pattern_points = [pattern_points]
            
            st.plotly_chart(plot_pattern(
                selected_data["Data"],
                pattern_points,
                st.session_state.selected_pattern
            ))