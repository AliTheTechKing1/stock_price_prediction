import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout        
import joblib



X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")


model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        Dense(1)
        
    ])

model.compile(optimizer='adam', loss="mse")

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=64,
    validation_split=0.1
)
model.save("data/lstm_model.keras")
print("Model trained and saved.")