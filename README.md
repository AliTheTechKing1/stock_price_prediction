# Stock Price Predictor

An interactive web app that uses an LSTM neural network to predict stock prices for any ticker symbol. Built as my first data science / finance / AI project.

---

## What it does

- Fetches historical stock data for any ticker using yfinance
- Trains a custom LSTM model on that ticker's closing prices
- Displays actual vs predicted prices on an interactive chart
- Predicts future prices beyond today's date
- Caches trained models so repeated predictions are instant
- Includes a reference table of the top 50 most popular stocks and their ticker symbols

---

## Tech stack

- **Python** — core language
- **Keras / TensorFlow** — LSTM model
- **yfinance** — stock data fetching
- **scikit-learn** — data normalization and evaluation
- **Pandas / NumPy** — data manipulation
- **Matplotlib** — charting
- **Streamlit** — web app interface

---

## Model architecture

- 2 stacked LSTM layers (64 and 32 units)
- Dropout layers (20%) to prevent overfitting
- Trained on 80% of historical data, tested on 20%
- 60-day sliding window — the model looks at 60 days of prices to predict day 61
- Optimized with Adam optimizer and MSE loss

---

## Limitations

- **Single feature** — the model only uses closing price. Adding technical indicators (RSI, moving averages, volume) would improve accuracy
- **Smooth future predictions** — iterative future forecasting compounding means the model captures trend direction but loses short-term volatility the further out it predicts
- **No external signals** — news, earnings reports, and macroeconomic events are not factored in, which are major drivers of real price movement

---

## What I'd improve next

- Add technical indicators as additional features (RSI, MACD, Bollinger Bands)
- Train on multiple tickers simultaneously for a more generalised model
- Add a confidence interval band around future predictions
- Deploy publicly on Streamlit Community Cloud

---
