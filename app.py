import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from src.tickers import POPULAR_TICKERS

cache = {}

def fetch_and_prepare(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None, None, None, None, None, None, None
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test, scaler, data, df.index

def train_model(X_train, y_train):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=60, batch_size=64, verbose=0)
    return model

def predict_future(model, scaler, data, future_days):
    # start with last 60 days of real data as the initial sequence
    last_60 = data[-60:]
    scaled = scaler.transform(last_60)
    current_sequence = list(scaled)
    future_predictions = []

    for _ in range(future_days):
        # reshape for model input
        X = np.array(current_sequence[-60:]).reshape(1, 60, 1)
        # predict next day and feed it back into the sequence
        next_pred = model.predict(X, verbose=0)
        current_sequence.append(next_pred[0])
        future_predictions.append(next_pred[0])

    # reverse normalization to get real prices
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

tab1, tab2 = st.tabs(["Stock Predictor", "Popular Tickers"])

with tab1:
    st.title("Stock Price Predictor")
    st.write("Enter any stock ticker to fetch its data, train an LSTM, and see predictions.")

    ticker = st.text_input("Stock ticker symbol", value="AAPL").upper()
    start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2024-01-01"))

    if st.button("Predict"):
        st.markdown("---")
        if ticker in cache:
            st.info(f"Using cached model for {ticker}")
            model, scaler, X_test, y_test, data, dates = cache[ticker]
        else:
            with st.spinner(f"Fetching data and training model for {ticker}... this takes a few minutes"):
                X_train, y_train, X_test, y_test, scaler, data, dates = fetch_and_prepare(ticker, start_date, end_date)
                if X_train is None:
                    st.error(f"No data found for {ticker}. Please try another ticker.")
                    st.stop()
                model = train_model(X_train, y_train)
                cache[ticker] = (model, scaler, X_test, y_test, data, dates)
            st.success(f"Model trained for {ticker}")

        # generate predictions on test set
        predictions = model.predict(X_test)
        if len(predictions.shape) == 3:
            predictions = predictions[:, -1, :]
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = math.sqrt(mean_squared_error(actual_test, predictions))
        st.metric(label="RMSE", value=f"${rmse:.2f}")

        # build date ranges for plotting
        full_dates = dates[60:]
        split = int(len(full_dates) * 0.8)
        test_dates = full_dates[split:]
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        # pick x-axis tick format based on date range
        if total_days <= 180:
            tick_format = "%b %d"
            freq = "2W"
        elif total_days <= 365:
            tick_format = "%b %Y"
            freq = "MS"
        elif total_days <= 730:
            tick_format = "%b %Y"
            freq = "2MS"
        else:
            tick_format = "%Y"
            freq = "6MS"

        # create the plot — fig and ax exist from here onwards
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(full_dates, data[60:], label="Actual Price (full)", color="blue", alpha=0.5)
        ax.plot(test_dates, predictions, label="Predicted Price (test set)", color="orange")
        ax.axvline(x=test_dates[0], color='gray', linestyle='--', linewidth=1, label="Train/Test split")

        # if end date is in the future, extend predictions forward
        today = pd.Timestamp.today()
        end_dt = pd.Timestamp(end_date)

        if end_dt > today:
            future_days = len(pd.bdate_range(start=today, end=end_dt))
            with st.spinner("Predicting future prices..."):
                future_preds = predict_future(model, scaler, data, future_days)
            future_dates = pd.bdate_range(start=today, periods=future_days)
            ax.plot(future_dates, future_preds, label="Future Prediction", color="green", linestyle="--")
            ax.axvline(x=today, color='green', linestyle=':', linewidth=1, label="Today")

        # set x-axis ticks across the full range including future
        tick_end = end_dt if end_dt > today else full_dates[-1]
        ticks = pd.date_range(start=full_dates[0], end=tick_end, freq=freq)
        ax.set_xticks(ticks)
        ax.set_xticklabels([t.strftime(tick_format) for t in ticks], rotation=45, ha='right')
        ax.set_title(f"{ticker} — Actual vs Predicted Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

with tab2:
    st.title("Top 50 Popular Tickers")
    st.write("Reference list of the most commonly traded stocks and their ticker symbols.")
    st.markdown("---")
    df_tickers = pd.DataFrame(POPULAR_TICKERS)
    df_tickers.index += 1
    df_tickers.columns = ["Company", "Ticker"]
    st.dataframe(df_tickers, use_container_width=True)