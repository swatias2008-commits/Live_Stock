import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# CONFIG
# -----------------------------

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "V", "JNJ", "WMT", "NFLX", "DIS", "T", "KO", "BAC"
]

# -----------------------------
# CHECK MARKET STATUS
# -----------------------------

def check_market_status(ticker_symbol):
    """Fetches and displays the current market status for a ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Use a short timeout for the info call to prevent long hangs
        info = ticker.info
        market_state = info.get('marketState', 'UNKNOWN')

        if market_state in ['PRE', 'REGULAR', 'POST']:
            return "Market is LIVE", "green"

        elif market_state == 'CLOSED':
            return "Market is CLOSED", "red"

        else:
            # Fallback check for last update time
            ts = info.get('regularMarketTime')
            if ts:
                # Convert the timestamp to a datetime object
                if isinstance(ts, int):
                    dt = datetime.datetime.fromtimestamp(ts)
                else:
                    # Handle cases where regularMarketTime might be a different format
                    return f"Status: {market_state} (Time data format error)", "orange"
                    
                diff = datetime.datetime.now() - dt
                # Consider 'LIVE' if the last trade was within the last 5 minutes (300 seconds)
                if diff.total_seconds() < 300:
                    return f"Market is LIVE (Updated {dt.strftime('%H:%M:%S')})", "green"

            return f"Status: {market_state}", "orange"

    except Exception:
        # Catch yfinance errors (e.g., bad connection, invalid ticker)
        return "Could not check market status", "red"

# -----------------------------
# MOVING AVERAGE PREDICTION
# -----------------------------

def predict_moving_average(data, days=14):
    """Calculates the Moving Average and predicts the next day's close."""
    if len(data) < days:
        return None, data # Return None for prediction if insufficient data

    # Ensure the dataframe is a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Calculate the Moving Average
    df['MA'] = df['Close'].rolling(window=days).mean()
    
    # Get the last valid index and its moving average value
    last_day = df.index[-1]
    last_ma = df['MA'].iloc[-1]
    
    # Calculate the next business day for the prediction
    next_day = last_day + pd.Timedelta(days=1)
    # Skip weekends (Saturday=5, Sunday=6)
    while next_day.weekday() >= 5:  
        next_day += pd.Timedelta(days=1)

    # Create the prediction DataFrame
    prediction_df = pd.DataFrame({'Close': [last_ma]}, index=[next_day])
    return prediction_df, df

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Live Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Dashboard & Predictor")

st.sidebar.header("Select Stock")
selected_ticker = st.sidebar.selectbox("Ticker", TICKERS)

# Set default start date to 2 years ago
default_start_date = date.today() - datetime.timedelta(days=365 * 2)

start_date = st.sidebar.date_input(
    "Start Date",
    default_start_date
)

end_date = st.sidebar.date_input("End Date", date.today())

if st.sidebar.button("Get Data & Predict"):

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
    else:
        try:
            st.subheader(f"Data for {selected_ticker}")

            # Market Status
            status_text, color = check_market_status(selected_ticker)
            st.markdown(f"### Market Status: :{color}[{status_text}]")

            # Download Data
            stock_data = yf.download(selected_ticker, start=start_date, end=end_date)
            
            if stock_data.empty:
                st.error("No data found for this ticker in the selected date range.")
            else:
                # --- Metrics ---
                # Safely get info, providing default values for keys that might be missing
                ticker_info = yf.Ticker(selected_ticker).info
                current_price = ticker_info.get("currentPrice", stock_data["Close"].iloc[-1])
                volume = ticker_info.get("volume", 0)
                week52_high = ticker_info.get("fiftyTwoWeekHigh", 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:,.2f}")
                col2.metric("Volume", f"{volume:,}")
                col3.metric("52 Week High", f"${week52_high:,.2f}")

                # --- Prediction ---
                ma_days = 50
                result_df, chart_data = predict_moving_average(stock_data, ma_days)

                if result_df is not None:
                    predicted_price = result_df["Close"].iloc[0]

                    st.success(
                        f"Predicted Close for **{result_df.index[0].date()}**: "
                        f"**${predicted_price:.2f}** (Using {ma_days}-Day MA)"
                    )
                    
                    # --- Chart ---
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05
                    )
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Price'
                    ), row=1, col=1)

                    # Moving Average
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['MA'],
                        mode='lines',
                        line=dict(color='orange', width=2),
                        name=f"{ma_days}-Day MA"
                    ), row=1, col=1)

                    # Prediction Point
                    fig.add_trace(go.Scatter(
                        x=result_df.index,
                        y=result_df["Close"],
                        mode='markers',
                        marker=dict(size=10, symbol="star", color='red'),
                        name="MA Prediction"
                    ), row=1, col=1)

                    # Volume Bar Chart
                    fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data["Volume"],
                        name="Volume",
                        marker_color='lightblue'
                    ), row=2, col=1)

                    # Update Layout
                    fig.update_layout(
                        title_text=f"**{selected_ticker}** Stock Price, {ma_days}-Day MA, & Prediction",
                        height=700,
                        xaxis_rangeslider_visible=False,
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    
                    # Clean up axis labels
                    fig.update_yaxes(title_text="Stock Price (USD)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning(
                        f"Not enough data to calculate the {ma_days}-Day Moving Average. "
                        f"Need at least {ma_days} data points. Please extend the date range."
                    )
                    # Show a basic price chart even without prediction
                    fig_basic = go.Figure(data=[go.Candlestick(
                         x=stock_data.index,
                         open=stock_data['Open'],
                         high=stock_data['High'],
                         low=stock_data['Low'],
                         close=stock_data['Close'],
                         name='Price'
                     )])
                    fig_basic.update_layout(title=f"{selected_ticker} Candlestick Chart", template="plotly_white")
                    st.plotly_chart(fig_basic, use_container_width=True)


                # --- Raw Data ---
                st.subheader("Raw Data (Last 20 Entries)")
                st.dataframe(stock_data.tail(20).style.format("${:,.2f}"))

        except Exception as e:
            # Catch all unexpected errors during data fetching or processing
            st.error(f"An unexpected error occurred during data fetching: {e}")
