import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 3])
    return np.array(x), np.array(y)

def train_and_eval(ticker, file_path, notebook_dir):
    print(f"\n==================== Training for {ticker} (Expanded Features) ====================")
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Expanded Feature Set
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions',
                'avg_intensity', 'avg_credibility', 'avg_relevance']
    
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    seq_length = 10
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # --- XGBoost (Price Diff Approach) ---
    print(f"[{ticker}] Training XGBoost (Target: Price Difference)...")
    y_train_diff = y_train - X_train[:, -1, 3]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model_xgb.fit(X_train_flat, y_train_diff)
    y_pred_xgb_scaled = X_test[:, -1, 3] + model_xgb.predict(X_test_flat)

    # --- BiLSTM ---
    print(f"[{ticker}] Training BiLSTM...")
    model_bilstm = Sequential([
        Input(shape=(seq_length, len(features))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(1)
    ])
    model_bilstm.compile(optimizer='adam', loss='mse')
    model_bilstm.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    y_pred_bilstm_scaled = model_bilstm.predict(X_test).flatten()

    # Evaluation Metrics
    r2_xgb = r2_score(y_test, y_pred_xgb_scaled)
    r2_bilstm = r2_score(y_test, y_pred_bilstm_scaled)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb_scaled))
    rmse_bilstm = np.sqrt(mean_squared_error(y_test, y_pred_bilstm_scaled))
    
    print(f"  => XGBoost Result - R2: {r2_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
    print(f"  => BiLSTM Result  - R2: {r2_bilstm:.4f}, RMSE: {rmse_bilstm:.4f}")
    
    # Plotting (English Labels)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price (Scaled)', color='black', alpha=0.5, linewidth=2)
    plt.plot(y_pred_xgb_scaled, label=f'XGBoost (R2: {r2_xgb:.3f})', alpha=0.8, linestyle='--')
    plt.plot(y_pred_bilstm_scaled, label=f'BiLSTM (R2: {r2_bilstm:.3f})', alpha=0.8)
    plt.title(f"Stock Price Prediction with Aggregated Sentiment (Max/Min/Avg/Sum/Count) - {ticker}")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(notebook_dir, f"{ticker}_english_results.png"))
    plt.close()
    
    return {"XGBoost": {"R2": r2_xgb, "RMSE": rmse_xgb}, "BiLSTM": {"R2": r2_bilstm, "RMSE": rmse_bilstm}}

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    processed_dir = os.path.join(base_dir, "data", "processed")
    notebook_dir = os.path.join(base_dir, "notebooks")
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "BABA", "VNM"]
    all_res = {}

    for t in tickers:
        file_p = os.path.join(processed_dir, f"{t}_final_expanded.csv")
        if os.path.exists(file_p):
            all_res[t] = train_and_eval(t, file_p, notebook_dir)

    # Save final English metrics
    with open(os.path.join(processed_dir, "final_metrics_report.txt"), "w") as f:
        f.write("FINAL PERFORMANCE REPORT (ALL TICKERS)\n")
        f.write("======================================\n")
        for t, m in all_res.items():
            f.write(f"\nTicker: {t}\n")
            f.write(f"  XGBoost -> R2: {m['XGBoost']['R2']:.4f}, RMSE: {m['XGBoost']['RMSE']:.4f}\n")
            f.write(f"  BiLSTM  -> R2: {m['BiLSTM']['R2']:.4f}, RMSE: {m['BiLSTM']['RMSE']:.4f}\n")
    print("\n>>> All training tasks completed with English outputs!")
