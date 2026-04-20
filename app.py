import math
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

cache = {}

def fetch_and_prepare(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    
    if df.empty:
        return None, None

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

    return (X_train, y_train, X_test, y_test, scaler, data, df.index)

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

st.title("Stock Price Predictor")
st.write("Enter any stock ticker to fetch its data, train an LSTM, and see predictions.")

ticker = st.text_input("Stock ticker symbol", value="AAPL").upper()
start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2024-01-01"))

if st.button("Predict"):
    if ticker in cache:
        st.info(f"Using cached model for {ticker}")
        model, scaler, X_test, y_test, actual, dates = cache[ticker]
    else:
        with st.spinner(f"Fetching data and training model for {ticker}... this takes a few minutes"):
            result = fetch_and_prepare(ticker, start_date, end_date)
            X_train, y_train, X_test, y_test, scaler, data, dates = result

            if X_train is None:
                st.error(f"No data found for {ticker}. Please try another ticker.")
                st.stop()

            model = train_model(X_train, y_train)
            actual = data[60:]

            cache[ticker] = (model, scaler, X_test, y_test, actual, dates)

        st.success(f"Model trained for {ticker}")

    predictions = model.predict(X_test)
    if len(predictions.shape) == 3:
        predictions = predictions[:, -1, :]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(actual_test, predictions))
    st.metric(label="RMSE", value=f"${rmse:.2f}")

    full_dates = dates[60:]  # skip first 60 days used for first sequence
    split = int(len(full_dates) * 0.8)
    train_dates = full_dates[:split]
    test_dates = full_dates[split:]

    fig, ax = plt.subplots(figsize=(14, 5))

    # plot full actual price history
    ax.plot(full_dates, data[60:], label="Actual Price (full)", color="blue", alpha=0.5)

    # overlay predictions on the test portion only
    ax.plot(test_dates, predictions, label="Predicted Price (test set)", color="orange")

    # vertical line separating train and test
    ax.axvline(x=test_dates[0], color='gray', linestyle='--', linewidth=1, label="Train/Test split")

    ax.set_title(f"{ticker} — Actual vs Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)