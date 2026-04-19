import math 
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


model = load_model('data/lstm_model.keras')
scaler = joblib.load('data/scaler.pkl')


X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')


predictions = model.predict(X_test)

print("raw predictions shape:", predictions.shape)

# if model returned full sequences, take only the last timestep
if len(predictions.shape) == 3:
    predictions = predictions[:, -1, :]

predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print("predictions shape:", predictions.shape)
print("y_test shape:", y_test.shape)

rmse = math.sqrt(mean_squared_error(actual, predictions))
print(f"RMSE: ${rmse:.2f}")


plt.figure(figsize=(14, 5))
plt.plot(actual, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="orange")
plt.title("AAPL — Actual vs Predicted Price")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("data/prediction_plot.png")  
plt.show()
print("Plot saved.")
