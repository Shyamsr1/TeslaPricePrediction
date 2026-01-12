# 🚗📈 Tesla Stock Price Prediction using Deep Learning

This project builds an end-to-end **time series forecasting system** to predict **Tesla (TSLA) stock closing prices** using **Deep Learning models** such as **Simple RNN** and **LSTM**, and compares their performance with a classical **ARIMA** baseline.

---

## 📌 Problem Statement

Stock prices are **sequential and time-dependent**, making them suitable for **Recurrent Neural Networks (RNNs)**.  
The goal of this project is to **predict Tesla’s closing stock price** for:
- **1-day ahead**
- **5-days ahead**
- **10-days ahead**

and compare the effectiveness of **ARIMA vs SimpleRNN vs LSTM**.

---

## 🎯 Business Objective

- Assist investors and analysts in **short-term trend anticipation**
- Demonstrate how **Deep Learning outperforms traditional models** on non-linear time-series data
- Provide a **scalable forecasting framework** usable for other stocks

---

## 🧠 Models Implemented

- **ARIMA** – Classical statistical time-series baseline
- **Simple RNN** – Basic recurrent neural network
- **LSTM** – Long Short-Term Memory network for long-range dependencies

---

## Project Structure 
``` bash
TeslaStockPricePredictionProject/
├── data/
│   └── (stock CSV files / processed data)
│
├── models/
│   └── (saved models: .h5 / .pkl / scalers)
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
└── tesla_stock_prediction.ipynb
```
---


## 🗂 Dataset Details

- **Source:** Historical Tesla stock price data
- **Target Variable:** `Close` price
- **Frequency:** Daily
- **Nature:** Sequential time-series data

---

## 🧹 Data Preprocessing

- Sorted data by **date index** (no shuffling)
- Checked and handled **missing values** using time-series-safe methods:
  - Forward fill
  - Backward fill
  - Time interpolation (optional)
- Applied **MinMax scaling**
- Created **lookback windows** for sequence modeling

---

## 🔁 Forecasting Horizons

| Horizon | Description |
|------|-----------|
| 1 Day | Very short-term movement |
| 5 Days | Weekly trend |
| 10 Days | Short-term investment signal |

---

## 📊 Evaluation Metrics

- Mean Absolute Error (**MAE**)
- Root Mean Squared Error (**RMSE**)
- Visual comparison of **Actual vs Predicted prices**

---

## 📈 Key Insights

- **LSTM consistently outperforms SimpleRNN** due to better handling of long-term dependencies
- **SimpleRNN learns patterns**, but struggles with volatile price movements
- **ARIMA works as a baseline**, but fails to adapt to regime shifts and non-linearity
- Model performance is highly sensitive to:
  - Lookback window size
  - Scaling
  - Epochs and batch size

---

## 🧪 Experiment Flow

1. Data Loading & Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Missing Value Handling (Time-Series Safe)  
4. Train-Test Split (Chronological)  
5. Model Training (ARIMA, RNN, LSTM)  
6. Multi-horizon Forecasting  
7. Model Comparison & Visualization  

---

## 🧾 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **TensorFlow / Keras**
- **Statsmodels (ARIMA)**
- **Streamlit (for UI, optional)**

---

## 🚀 How to Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app (if applicable)
streamlit run app.py

---
## 📌 Conclusion

- LSTM is the best-performing model for Tesla stock price prediction
- Deep Learning models handle non-linearity and volatility better than classical approaches
- Stock prices are influenced by external events, which are not captured in price-only models

---
## 🔮 Future Enhancements

- Add technical indicators (RSI, MACD, Moving Averages)
- Include trading volume
- Integrate news and sentiment analysis
- Extend forecasting to other stocks
- Deploy as a production-ready web app

---
## 👨‍💻 Author
- Shyam Sirugudi Ramaswamy
- 📌 Data Scientist | AI/ML Engineer
- 🔗 GitHub: https://github.com/Shyamsr1

---
## License 
- MIT License
- Copyright (c) 2026 Shyam Sirugudi Ramaswamy
