

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Streamlit UI
st.title("📈 Bitcoin Price Prediction (Next 7 Days)")
st.write("Using LSTM Neural Network")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = yf.download('BTC-USD', start='2020-01-01', end='2025-04-20', progress=False)
        
        # Handle multi-index if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]

        # Retry if empty
        if df.empty:
            st.warning("⚠️ Empty data. Retrying in 10 seconds...")
            time.sleep(10)
            df = yf.download('BTC-USD', start='2020-01-01', end='2025-04-20', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            else:
                df = df[['Close']]

        # Save to backup or raise error
        if df.empty:
            raise ValueError("Downloaded DataFrame is empty.")
        
        df.to_csv("btc_data_backup.csv")
        st.success("✅ Live data loaded successfully.")
    
    except Exception as e:
        st.warning(f"⚠️ Live data load failed: {e}")
        if os.path.exists("btc_data_backup.csv"):
            df = pd.read_csv("btc_data_backup.csv", index_col=0, parse_dates=True)
            st.info("📁 Loaded data from local backup.")
        else:
            st.error("❌ No local backup found. Cannot proceed.")
            st.stop()

    df.dropna(inplace=True)
    return df

df = load_data()
st.line_chart(df)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build and train LSTM model
with st.spinner("🔄 Training LSTM model..."):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Predict next 7 days
future_input = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_prices = []
for _ in range(7):
    pred = model.predict(future_input)[0][0]
    predicted_prices.append(pred)
    future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

# Inverse transform predictions
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Show predictions
st.subheader("📊 Predicted Bitcoin Prices for Next 7 Days:")
pred_df = pd.DataFrame(predicted_prices, columns=['Predicted Price (USD)'])
pred_df.index = [f'Day {i+1}' for i in range(7)]
st.dataframe(pred_df)

# Plot predictions
st.line_chart(pred_df)
