# app.py - The Streamlit UI for Stock price prediction 
# Import all essential libraries 
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# The data path where the given csv file is stored - TSLA.csv file. 
DEFAULT_DATA_PATH = "data/TSLA.csv"

st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
'''
Stock prices are time-dependent — today’s price depends on yesterday’s price.
If missing values are handled incorrectly, 
the model learns wrong patterns and produces unreliable forecasts.
This function ensures:
- Chronological order is preserved
- No data leakage from the future
- Smooth, continuous input for ARIMA, RNN, and LSTM
'''
# Creates a reusable function to clean missing values in a time-series column (like closing price)
def handle_missing_time_series(s: pd.Series, method="ffill"): # forward fill - ffill is the safest for stocks 
    s = s.sort_index()
    if method == "ffill":
        return s.ffill().bfill()
    elif method == "interpolate":
        return s.interpolate(method="time").ffill().bfill()
    else:
        return s.dropna()

def make_sequences(data_scaled: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(lookback, len(data_scaled) - horizon + 1):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i:i + horizon, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def split_sequences_by_train_size(all_scaled, lookback, horizon, train_size):
    X_all, y_all = make_sequences(all_scaled, lookback, horizon)
    pred_start_indices = np.arange(lookback, lookback + len(X_all))
    train_mask = pred_start_indices < train_size
    test_mask  = ~train_mask
    return X_all[train_mask], y_all[train_mask], X_all[test_mask], y_all[test_mask]

def inverse_scale_preds(y_scaled, scaler: MinMaxScaler):
    flat = y_scaled.reshape(-1, 1)
    inv = scaler.inverse_transform(flat).reshape(y_scaled.shape)
    return inv

def evaluate_multi_step(y_true_inv, y_pred_inv):
    mse = mean_squared_error(y_true_inv.flatten(), y_pred_inv.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_inv.flatten(), y_pred_inv.flatten())
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}

def build_rnn_model(units=64, dropout=0.2, learning_rate=1e-3, lookback=60, horizon=1):
    model = Sequential([
        SimpleRNN(units, input_shape=(lookback, 1)),
        Dropout(dropout),
        Dense(horizon)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model

def build_lstm_model(units=64, dropout=0.2, learning_rate=1e-3, lookback=60, horizon=1):
    model = Sequential([
        LSTM(units, input_shape=(lookback, 1)),
        Dropout(dropout),
        Dense(horizon)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model

def plot_series(actual, predicted, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.legend()
    st.pyplot(fig)

# -----------------------------
# UI - streamlit based for models - ARIMA, LSTM and Simple RNN 
# -----------------------------
st.title("🚗 Tesla Stock Price Prediction ")
st.write("Models: **SimpleRNN**, **LSTM**, and **ARIMA** | Horizons: **1 / 5 / 10 days**")

col1, col2, col3 = st.columns(3)

with col1:
    model_choice = st.selectbox("Choose Model", ["SimpleRNN", "LSTM", "ARIMA"])
with col2:
    horizon = st.selectbox("Forecast Horizon (days)", [1, 5, 10], index=0)
with col3:
    lookback = st.slider("Lookback Window (days)", min_value=20, max_value=120, value=60, step=5)

st.sidebar.header("Training Settings (DL)")
units = st.sidebar.selectbox("Units", [32, 64, 128], index=1)
dropout = st.sidebar.selectbox("Dropout", [0.1, 0.2, 0.3], index=1)
lr = st.sidebar.selectbox("Learning Rate", [1e-3, 5e-4, 1e-4], index=0)
epochs = st.sidebar.slider("Epochs", 5, 50, 15, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload TSLA CSV (optional)", type=["csv"])

# -----------------------------
# Load Data
# -----------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(DEFAULT_DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

target_col = "Close"
series = handle_missing_time_series(df[target_col], method="ffill")

st.subheader("📌 Quick EDA")
c1, c2 = st.columns(2)
with c1:
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isna().sum().to_dict())
with c2:
    st.write(series.describe())

fig = plt.figure()
plt.title("Close Price")
plt.plot(series)
st.pyplot(fig)

# -----------------------------
# Train/Test Split
# -----------------------------
values = series.values.reshape(-1, 1)
train_size = int(len(values) * 0.8)
train_vals = values[:train_size]
test_vals  = values[train_size:]

# -----------------------------
# Run Model
# -----------------------------
if st.button("Run Forecast"):
    if model_choice in ["SimpleRNN", "LSTM"]:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_vals)
        test_scaled  = scaler.transform(test_vals)
        all_scaled = np.vstack([train_scaled, test_scaled])

        X_train, y_train, X_test, y_test = split_sequences_by_train_size(all_scaled, lookback, horizon, train_size)

        if len(X_test) == 0:
            st.error("Not enough data for chosen lookback/horizon. Reduce lookback or horizon.")
            st.stop()

        if model_choice == "SimpleRNN":
            model = build_rnn_model(units=units, dropout=dropout, learning_rate=lr, lookback=lookback, horizon=horizon)
        else:
            model = build_lstm_model(units=units, dropout=dropout, learning_rate=lr, lookback=lookback, horizon=horizon)

        cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cb],
            verbose=0
        )

        y_pred_scaled = model.predict(X_test)
        y_test_inv = inverse_scale_preds(y_test, scaler)
        y_pred_inv = inverse_scale_preds(y_pred_scaled, scaler)

        metrics = evaluate_multi_step(y_test_inv, y_pred_inv)
        st.subheader("✅ Metrics (Test)")
        st.json(metrics)

        # Plot 1-step for clean visualization, else plot first horizon component
        if horizon == 1:
            plot_series(y_test_inv.flatten(), y_pred_inv.flatten(), f"{model_choice} - 1 Day Forecast (Test)")
        else:
            plot_series(y_test_inv[:, 0], y_pred_inv[:, 0], f"{model_choice} - {horizon} Day Forecast (Showing Day+1 component)")

        # Show next forecast using last lookback window
        last_window = all_scaled[-lookback:].reshape(1, lookback, 1)
        future_scaled = model.predict(last_window)
        future_inv = inverse_scale_preds(future_scaled, scaler).flatten()

        st.subheader(f"📈 Next {horizon} Day Forecast (from latest available day)")
        st.write(future_inv)

    else:
        # ARIMA
        train_series = series.iloc[:train_size]
        test_series  = series.iloc[train_size:]

        # Small order search
        best_aic = np.inf
        best_order = None
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        m = ARIMA(train_series, order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p,d,q)
                    except:
                        continue

        arima_model = ARIMA(train_series, order=best_order).fit()
        forecast = arima_model.forecast(steps=len(test_series))

        mse = mean_squared_error(test_series.values, forecast.values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_series.values, forecast.values)

        st.subheader("✅ Metrics (Test)")
        st.json({"MSE": mse, "RMSE": rmse, "MAE": mae, "Best_Order(p,d,q)": best_order})

        fig = plt.figure()
        plt.title("ARIMA Forecast vs Actual (Test)")
        plt.plot(test_series.index, test_series.values, label="Actual")
        plt.plot(test_series.index, forecast.values, label="Forecast")
        plt.legend()
        st.pyplot(fig)

        # Next horizon forecast
        future = arima_model.forecast(steps=horizon).values
        st.subheader(f"📈 Next {horizon} Day Forecast (ARIMA)")
        st.write(future)
