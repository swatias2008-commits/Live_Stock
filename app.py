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
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        market_state = info.get('marketState', 'UNKNOWN')

        if market_state in ['PRE', 'REGULAR', 'POST']:
            return "Market is LIVE", "green"

        elif market_state == 'CLOSED':
            return "Market is CLOSED", "red"

        else:
            ts = info.get('regularMarketTime')
            if ts:
                dt = datetime.datetime.fromtimestamp(ts)
                diff = datetime.datetime.now() - dt
                if diff.total_seconds() < 300:
                    return f"Market is LIVE (Updated {dt.strftime('%H:%M:%S')})", "green"

            return f"Status: {market_state}", "orange"

    except:
        return "Could not check market status", "red"

# -----------------------------
# MOVING AVERAGE PREDICTION
# -----------------------------

def predict_moving_average(data, days=14):
    if len(data) < days:
        return None

    data['MA'] = data['Close'].rolling(window=days).mean()
    last_day = data.index[-1]
    next_day = last_day + pd.Timedelta(days=1)

    while next_day.weekday() >= 5:  
        next_day += pd.Timedelta(days=1)

    prediction_df = pd.DataFrame({'Close': [data['MA'].iloc[-1]]}, index=[next_day])
    return prediction_df, data

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Live Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Dashboard & Predictor")

st.sidebar.header("Select Stock")
selected_ticker = st.sidebar.selectbox("Ticker", TICKERS)

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.datetime.now() - datetime.timedelta(days=365 * 2)
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
                st.error("No data found for this ticker.")
            else:
                # Metrics
                ticker_info = yf.Ticker(selected_ticker).info
                current_price = ticker_info.get("currentPrice", stock_data["Close"].iloc[-1])
                volume = ticker_info.get("volume", 0)
                week52_high = ticker_info.get("fiftyTwoWeekHigh", 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Volume", f"{volume:,}")
                col3.metric("52 Week High", f"${week52_high:.2f}")

                # Prediction
                ma_days = 50
                result = predict_moving_average(stock_data.copy(), ma_days)

                if result:
                    prediction_df, chart_data = result
                    predicted_price = prediction_df["Close"].iloc[0]

                    st.success(
                        f"Predicted Close for {prediction_df.index[0].date()}: "
                        f"**${predicted_price:.2f}** (Using {ma_days}-Day MA)"
                    )

                    # Chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3]
                    )

                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Candlestick'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['MA'],
                        mode='lines',
                        name=f"{ma_days}-Day MA"
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=prediction_df.index,
                        y=prediction_df["Close"],
                        mode='markers',
                        marker=dict(size=12, symbol="star"),
                        name="Prediction"
                    ), row=1, col=1)

                    fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data["Volume"],
                        name="Volume"
                    ), row=2, col=1)

                    fig.update_layout(
                        title=f"{selected_ticker} Stock Price & Prediction",
                        height=700,
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Raw Data")
                st.dataframe(stock_data.tail(20))

        except Exception as e:
            st.error(f"Error: {e}")
