#!/usr/bin/env python3
"""
AAPL Stock Price Prediction Pipeline - SIMPLIFIED VERSION
Uses compile=False to bypass custom object issues entirely.
"""

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
import sys
import os
from pathlib import Path
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING
# ============================================================================
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "aapl_predictions.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD MODEL & SCALERS - SIMPLE APPROACH
# ============================================================================

try:
    from tensorflow.keras.models import load_model
    
    model_path = Path(__file__).parent / "lstm_model.h5"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # THE KEY: Load with compile=False to skip custom object issues
    logger.info("Loading model with compile=False...")
    model = load_model(str(model_path), compile=False)
    logger.info("✓ Model loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

try:
    scaler_x_path = Path(__file__).parent / "scaler_X.pkl"
    scaler_y_path = Path(__file__).parent / "scaler_y.pkl"
    
    if not scaler_x_path.exists() or not scaler_y_path.exists():
        logger.error("Scaler files not found")
        sys.exit(1)
    
    scaler_X = joblib.load(str(scaler_x_path))
    scaler_y = joblib.load(str(scaler_y_path))
    logger.info("✓ Scalers loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load scalers: {e}")
    sys.exit(1)

# ============================================================================
# CONSTANTS
# ============================================================================
FEATURES = [
    'Close', 'Returns', 'Momentum', 'Volatility', 'Vol_20',
    'High_Low_Spread', 'Open_Close_Diff', 'Volume',
    'Month_sin', 'Month_cos', 'DayOfWeek', 'Momentum_prev', 'ROC'
]
TIME_STEPS = 60
PREDICTIONS_CSV = Path(__file__).parent / "predictions.csv"

# ============================================================================
# FEATURE PREPARATION
# ============================================================================
def prepare_features(df):
    """Calculate all features"""
    df = df.copy()
    
    df["Month"] = df.index.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek'] = df.index.dayofweek
    df["Vol_20"] = df["Volume"].rolling(window=20).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = df['Close'].pct_change(periods=10)
    df['Volatility'] = df['Returns'].rolling(10).std()
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Open_Close_Diff'] = df['Open'] - df['Close']
    df['Momentum_prev'] = df['Momentum'].shift(1)
    
    return df

def get_next_trading_day(date):
    next_day = date + timedelta(days=1)
    
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    
    return next_day

# ============================================================================
# PREDICTION
# ============================================================================
def predict_next_day(df):
    """Predict tomorrow's price"""
    try:
        latest_data = df[FEATURES].tail(TIME_STEPS)
        
        if len(latest_data) < TIME_STEPS:
            logger.warning(f"Insufficient data: {len(latest_data)}/{TIME_STEPS}")
            return None
        
        if latest_data.isnull().any().any():
            latest_data = latest_data.fillna(method='bfill').fillna(method='ffill')
        
        latest_scaled = scaler_X.transform(latest_data)
        X_input = np.reshape(latest_scaled, (1, TIME_STEPS, latest_scaled.shape[1]))
        
        pred_scaled = model.predict(X_input, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)
        
        return float(pred[0][0])
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return None

# ============================================================================
# SAVE/CALCULATE PREDICTIONS
# ============================================================================
def save_prediction(pred, pred_date):
    """Save prediction"""
    try:
        new_row = pd.DataFrame({
            "Prediction_Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Target_Date": [str(pred_date)],
            "Predicted_Price": [round(pred, 2)]
        })
        
        if PREDICTIONS_CSV.exists():
            old_df = pd.read_csv(PREDICTIONS_CSV)
            df = pd.concat([old_df, new_row], ignore_index=True)
        else:
            df = new_row
        
        df.to_csv(PREDICTIONS_CSV, index=False)
        logger.info(f"✓ Saved prediction for {pred_date}: ${pred:.2f}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return False

def calculate_error(actual_price, pred_date):
    """Calculate error"""
    try:
        if not PREDICTIONS_CSV.exists():
            return None
        
        df = pd.read_csv(PREDICTIONS_CSV)
        row = df[df["Target_Date"] == str(pred_date)]
        
        if row.empty:
            return None
        
        pred_price = row.iloc[0]["Predicted_Price"]
        error = abs((actual_price - pred_price) / actual_price) * 100
        
        return round(error, 2)
    
    except Exception as e:
        logger.error(f"Error calculating error: {e}")
        return None

# ============================================================================
# DATA FETCHING
# ============================================================================
def get_actual_close(date):
    """Get actual closing price safely"""
    try:
        end_date = date + timedelta(days=1)
        
        data = yf.download(
            "AAPL",
            start=date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )

        if data is None or len(data) == 0:
            logger.warning(f"No data for {date} (weekend/holiday?)")
            return None

        # ✅ Handle both cases (Series / DataFrame)
        close_value = data["Close"]

        # If it's a DataFrame (multi-index case)
        if isinstance(close_value, pd.DataFrame):
            close_value = close_value.iloc[0, 0]
        else:
            close_value = close_value.iloc[0]

        close_price = float(close_value)

        logger.info(f"✓ Actual close for {date}: ${close_price:.2f}")
        return close_price

    except Exception as e:
        logger.warning(f"Error fetching close for {date}: {e}")
        return None
def download_aapl_data(retries=3):
    """Download AAPL data with retries"""
    for attempt in range(retries):
        try:
            logger.info(f"Downloading AAPL data (attempt {attempt + 1}/{retries})...")
            df = yf.download("AAPL", period="2y", progress=False, timeout=30)
            
            if len(df) == 0:
                raise ValueError("Empty dataframe")
            
            logger.info(f"✓ Downloaded {len(df)} rows")
            return df.sort_index()
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 5)
    
    logger.error(f"Failed after {retries} attempts")
    return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Main pipeline (data-driven trading day logic)"""
    logger.info("=" * 70)
    logger.info("AAPL PREDICTION PIPELINE STARTED")
    logger.info("=" * 70)
    
    try:
        today = datetime.now().date()

        # =========================
        # DOWNLOAD DATA FIRST (IMPORTANT FIX)
        # =========================
        df = download_aapl_data()
        if df is None:
            logger.error("Data download failed")
            return

        df = df.sort_index()

        # =========================
        # 🔥 DATA-DRIVEN TRADING DAYS (KEY FIX)
        # =========================
        last_trading_day = df.index[-1].date()
        prev_trading_day = df.index[-2].date()
        next_trading_day = get_next_trading_day(last_trading_day)

        logger.info(f"System Date: {today}")
        logger.info(f"Last Trading Day (from data): {last_trading_day}")
        logger.info(f"Predicting for: {next_trading_day}")
        logger.info(f"Evaluating: {last_trading_day}")

        # =========================
        # FEATURE ENGINEERING
        # =========================
        logger.info("Preparing features...")
        df = prepare_features(df)
        df = df.dropna()

        if len(df) < TIME_STEPS:
            logger.error(f"Insufficient data: {len(df)}/{TIME_STEPS}")
            return

        # =========================
        # PREDICTION
        # =========================
        logger.info("Making prediction...")
        pred = predict_next_day(df)

        if pred is None:
            logger.error("Prediction failed")
            return

        logger.info(f"Predicted {next_trading_day}: ${pred:.2f}")

        # Save prediction for NEXT trading day
        save_prediction(pred, next_trading_day)

        # =========================
        # EVALUATION (FIXED LOGIC)
        # =========================
        logger.info(f"Checking actual for {last_trading_day}...")
        actual_price = get_actual_close(last_trading_day)

        if actual_price is not None:
            error = calculate_error(actual_price, last_trading_day)
            if error is not None:
                logger.info(f"Prediction error for {last_trading_day}: {error:.2f}%")
            else:
                logger.info("No prediction found for evaluation day")
        else:
            logger.info("No actual price available (holiday/weekend)")

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    choice = input("Run pipeline? (y/n): ")
    if choice.lower() == 'y':
        run_pipeline()