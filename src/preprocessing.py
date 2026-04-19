import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/AAPL.csv", skiprows=2, index_col="Date" , parse_dates=True)


Scaler = MinMaxScaler()
Scaled = Scaler.fit_transform(df)

X, y = [], []

for i in range(60, len(Scaled)):
    X.append(Scaled[i-60:i])
    y.append(Scaled[i])
    
    
X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]  

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#import joblib

#joblib.dump(Scaler, "..\\data\\scaler.pkl")

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)