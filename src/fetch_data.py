import yfinance as yf
import pandas as pd

df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
df = df[["Close"]]

df = df.ffill()
print(df.isnull().sum())

df.to_csv("..\\data\\AAPL.csv")
