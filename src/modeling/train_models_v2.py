import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 3]) # Close price (normalized index 3)
    return np.array(x), np.array(y)

def train_and_eval(ticker, file_path, output_dir):
    print(f"\n===== HUẤN LUYỆN: {ticker} (Dữ liệu V2 Robust) =====")
    df = pd.read_csv(file_path)
    
    # Chuẩn hóa tên cột giá
    col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
    df = df.rename(columns=col_map)
    
    # Danh sách 8 đặc trưng cảm xúc + 5 đặc trưng giá
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 
                'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    seq_length = 10
    if len(test_data) <= seq_length: return None

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # --- XGBOOST ---
    print(f"  > Đang chạy XGBoost...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model_xgb.fit(X_train_flat, y_train)
    y_pred_xgb = model_xgb.predict(X_test_flat)

    # --- BiLSTM ---
    print(f"  > Đang chạy BiLSTM...")
    model_bilstm = Sequential([
        Input(shape=(seq_length, len(available_features))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(1)
    ])
    model_bilstm.compile(optimizer='adam', loss='mse')
    model_bilstm.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    y_pred_bilstm = model_bilstm.predict(X_test).flatten()

    # METRICS
    r2_xgb = r2_score(y_test, y_pred_xgb)
    r2_bilstm = r2_score(y_test, y_pred_bilstm)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    rmse_bilstm = np.sqrt(mean_squared_error(y_test, y_pred_bilstm))
    
    print(f"    RESULT {ticker}:")
    print(f"    - XGBoost: R2={r2_xgb:.4f}, RMSE={rmse_xgb:.4f}")
    print(f"    - BiLSTM : R2={r2_bilstm:.4f}, RMSE={rmse_bilstm:.4f}")
    
    # PLOT
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price', color='black', alpha=0.6)
    plt.plot(y_pred_xgb, label=f'XGBoost (R2: {r2_xgb:.3f})', linestyle='--')
    plt.plot(y_pred_bilstm, label=f'BiLSTM (R2: {r2_bilstm:.3f})')
    plt.title(f"Comparison: XGBoost vs BiLSTM with Robust Sentiment - {ticker}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{ticker}_v2_comparison.png"))
    plt.close()
    
    return {"XGBoost": {"R2": r2_xgb, "RMSE": rmse_xgb}, "BiLSTM": {"R2": r2_bilstm, "RMSE": rmse_bilstm}}

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    processed_dir = os.path.join(base_dir, "data", "processed")
    notebook_dir = os.path.join(base_dir, "notebooks")
    
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "BABA", "VNM"]
    results = {}
    
    for t in tickers:
        f_p = os.path.join(processed_dir, f"{t}_final_dataset_v2.csv")
        if os.path.exists(f_p):
            res = train_and_eval(t, f_p, notebook_dir)
            if res: results[t] = res

    # Final Report
    with open(os.path.join(notebook_dir, "final_v2_report.txt"), "w", encoding='utf-8') as f:
        f.write("BÁO CÁO SO SÁNH MÔ HÌNH V2 (DỮ LIỆU ĐÃ CHUẨN HÓA)\n")
        f.write("================================================\n")
        for t, m in results.items():
            f.write(f"\nTicker: {t}\n")
            f.write(f"  XGBoost -> R2: {m['XGBoost']['R2']:.4f}, RMSE: {m['XGBoost']['RMSE']:.4f}\n")
            f.write(f"  BiLSTM  -> R2: {m['BiLSTM']['R2']:.4f}, RMSE: {m['BiLSTM']['RMSE']:.4f}\n")
            winner = "XGBoost" if m['XGBoost']['R2'] > m['BiLSTM']['R2'] else "BiLSTM"
            f.write(f"  => Người thắng: {winner}\n")
    
    print(f"\n>>> Hoàn thành! Kết quả và đồ thị đã được lưu tại {notebook_dir}")
