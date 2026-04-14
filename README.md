# 📈 AAPL Stock Price Prediction using LSTM

An end-to-end **time series machine learning pipeline** that predicts the **next trading day's closing price of Apple Inc. (AAPL)** using an LSTM (Long Short-Term Memory) neural network.

> ⚠️ This project is built for **learning and experimentation purposes only** and is NOT intended for real trading decisions.

---

# 🚀 Project Highlights

- 📊 End-to-end ML pipeline (data → features → model → prediction → evaluation)
- 🧠 LSTM-based time series forecasting
- ⚡ Real-world evaluation strategy (T+1 prediction)
- 🛡️ Data leakage prevention
- 📅 Trading-day-aware logic (handles weekends correctly)
- 📝 Logging system for monitoring pipeline behavior

---

# 🧠 Problem Statement

Given historical stock data, predict:

👉 **Next trading day's closing price**

This is a **T+1 forecasting problem**, where:
- Prediction is made using past data
- Evaluation happens only after actual price is available

---

# 📂 Project Structure

.
├── pipeline.py              # Main prediction & evaluation pipeline
├── lstm_model.h5            # Trained model (excluded from repo)
├── scaler_X.pkl             # Feature scaler (excluded)
├── scaler_y.pkl             # Target scaler (excluded)
├── predictions.csv          # Stored predictions
├── logs/
│   └── aapl_predictions.log # Execution logs
├── requirements.txt         # Dependencies
└── README.md

---

# 📊 Features Engineered

The model uses both **price-based** and **derived technical indicators**:

### 📉 Price-Based
- Close, Open, High, Low, Volume

### 📈 Derived Features
- Returns
- Momentum
- Volatility
- Rate of Change (ROC)
- High-Low Spread
- Open-Close Difference

### 📅 Time Features
- Day of Week
- Cyclical Month Encoding (sin/cos)

---

# ⚙️ Pipeline Workflow

## 🔁 Prediction

1. Download latest stock data
2. Identify **last available trading day**
3. Generate features
4. Use last **60 time steps**
5. Predict next trading day price
6. Save prediction

---

## 📉 Evaluation

1. Fetch actual closing price of last trading day
2. Compare with stored prediction
3. Compute error:
Error (%) = |Actual - Predicted| / Actual × 100

---

# ▶️ How to Run

## 1. Install dependencies

pip install -r requirements.txt

---

## 2. Run pipeline manually

---

## 3. Check outputs

### 📜 Logs
logs/aapl_predictions.log

### 📊 Predictions
predictions.csv

---

# 🧪 Example Output

Predicted 2026-04-14: $248.47
Actual close for 2026-04-13: $247.90
Prediction error: 0.23%

---

# 📦 Model & Scalers

The trained model and scalers are **not included** in this repository.

## To reproduce:

1. Train the LSTM model
2. Save:
   - `lstm_model.h5`
   - `scaler_X.pkl`
   - `scaler_y.pkl`

---

# 🚀 Future Improvements

- 📊 Add visualization dashboard (Streamlit)
- 📅 Integrate NYSE holiday calendar
- 🧠 Try advanced models:
  - GRU
  - Transformer-based models
- ⚡ Hyperparameter tuning automation
- 📉 Add more indicators (RSI, MACD)

---

# 📌 Key Learnings

- Time series ML requires **strict temporal awareness**
- Evaluation must mimic **real-world prediction flow**
- Feature engineering is critical for financial data
- Model accuracy ≠ real-world usefulness

---

# 🙌 Author

**Vijai S**

---

# ⭐ If you like this project

Feel free to ⭐ the repo and build on top of it!