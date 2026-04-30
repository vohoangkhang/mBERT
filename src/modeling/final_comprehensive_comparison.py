import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_sequences_v2(data, target, seq_length):
    x, y = [], []
    # Dùng dữ liệu từ t-seq_length đến t để dự báo t+1 (target)
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(target[i + seq_length - 1]) # Target của ngày cuối cùng trong sequence
    return np.array(x), np.array(y)

def train_and_eval_robust(ticker, data_path, scenario, model_type="XGB"):
    print(f"  > Đang chạy kịch bản: {scenario} ({model_type})...")
    df = pd.read_csv(data_path)
    
    # 1. Tiền xử lý: Dự báo Chênh lệch giá (Target = Close(t+1) - Close(t))
    df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df = df.dropna() # Bỏ dòng cuối cùng vì không có Target
    
    # 2. Chọn Đặc trưng
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    sent_cols = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 
                 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    
    if "Pure" in scenario:
        cols = price_cols
    else:
        for c in sent_cols:
            if c not in df.columns: df[c] = 0.0
        cols = price_cols + sent_cols
        
    data_feat = df[cols].values
    data_target = df['Target'].values.reshape(-1, 1)
    
    # 3. Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_x = scaler_x.fit_transform(data_feat)
    scaled_y = scaler_y.fit_transform(data_target)
    
    # 4. Tạo Sequence
    seq_length = 10
    X, y = create_sequences_v2(scaled_x, scaled_y, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 5. Huấn luyện
    if model_type == "XGB":
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=7, subsample=0.8)
        model.fit(X_train_flat, y_train)
        y_pred_scaled = model.predict(X_test_flat).reshape(-1, 1)
    else:
        model = Sequential([
            Input(shape=(seq_length, len(cols))),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        y_pred_scaled = model.predict(X_test)

    # 6. Inverse Transform để lấy Chênh lệch thực tế
    y_pred_diff = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual_diff = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 7. CHUYỂN VỀ GIÁ THỰC TẾ (Absolute Price) ĐỂ TÍNH METRICS THEO Ý THẦY KHANG
    # Giá ngày mai = Giá hôm nay + Chênh lệch
    last_close_test = df['Close'].values[train_size + seq_length - 1 : -1] 
    price_actual = last_close_test + y_actual_diff
    price_pred = last_close_test + y_pred_diff
    
    # TÍNH TOÁN METRICS TRÊN GIÁ TUYỆT ĐỐI (Đây là con số Thầy Khang muốn xem)
    mse = mean_squared_error(price_actual, price_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(price_actual, price_pred)
    r2 = r2_score(price_actual, price_pred)
    mape = calculate_mape(price_actual, price_pred)
    
    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "pred_price": price_pred, "actual_price": price_actual
    }

def run_comprehensive_comparison(ticker, base_dir):
    print(f"\n==================== BÁO CÁO TOÀN DIỆN: {ticker} ====================")
    clean_path = os.path.join(base_dir, "data", "processed", f"{ticker}_final_v4_clean.csv")
    mbert_path = os.path.join(base_dir, "data", "processed", f"{ticker}_mbert_final.csv")
    
    results = {}
    
    # 1. Kịch bản từ dữ liệu Thầy Khang
    if os.path.exists(clean_path):
        results["BiLSTM Pure"] = train_and_eval_robust(ticker, clean_path, "BiLSTM Pure", "BiLSTM")
        results["XGBoost Pure"] = train_and_eval_robust(ticker, clean_path, "XGBoost Pure", "XGB")
        results["BiLSTM+Sentiment"] = train_and_eval_robust(ticker, clean_path, "BiLSTM+Sentiment", "BiLSTM")
        results["XGBoost+Sentiment"] = train_and_eval_robust(ticker, clean_path, "XGBoost+Sentiment", "XGB")

    # 2. Kịch bản mBERT (AI Tiên tiến)
    if os.path.exists(mbert_path):
        results["mBERT+BiLSTM"] = train_and_eval_robust(ticker, mbert_path, "mBERT+BiLSTM", "BiLSTM")
        results["mBERT+XGBoost"] = train_and_eval_robust(ticker, mbert_path, "mBERT+XGBoost", "XGB")

    output_folder = os.path.join(base_dir, "notebooks", "final_results", ticker)
    os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(output_folder, "metrics_v4_clean.txt"), "w", encoding='utf-8') as f:
        f.write(f"FINAL COMPREHENSIVE REPORT - {ticker}\n")
        f.write("============================================\n\n")
        for label, r in results.items():
            f.write(f"{label}:\n")
            f.write(f"  - MSE (Diff): {r['mse']:.6f}\n")
            f.write(f"  - RMSE (Diff): {r['rmse']:.6f}\n")
            f.write(f"  - MAE (Diff): {r['mae']:.6f}\n")
            f.write(f"  - MAPE (Price): {r['mape']:.4f}%\n")
            f.write(f"  - R2 Score: {r['r2']:.6f}\n")
            f.write(f"  - [DỰ BÁO NGÀY CUỐI] Thực tế: {r['actual_price'][-1]:.2f} | Dự báo: {r['pred_price'][-1]:.2f}\n")
            f.write(f"  - Sai lệch tuyệt đối: {abs(r['actual_price'][-1] - r['pred_price'][-1]):.2f}\n\n")

    # Vẽ đồ thị so sánh
    plt.figure(figsize=(15, 10))
    first_key = list(results.keys())[0]
    plt.plot(results[first_key]['actual_price'], label='Actual Price', color='black', linewidth=2, alpha=0.6)
    
    for label, r in results.items():
        linestyle = '-' if "mBERT" in label else '--'
        plt.plot(r['pred_price'], label=f"{label} (MAPE: {r['mape']:.2f}%)", linestyle=linestyle)
        
    plt.title(f"Final Comparison: Traditional vs Sentiment vs mBERT - {ticker}")
    plt.xlabel("Days (Test Set)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, f"{ticker}_final_v5_comparison.png"))
    plt.close()
    print(f"  => Đã lưu báo cáo toàn diện tại {output_folder}")

if __name__ == "__main__":
    tickers = ["VNM", "AAPL", "AMZN", "GOOGL", "BABA", "META"]
    for t in tickers:
        run_comprehensive_comparison(t, ".")
    print("\n>>> TẤT CẢ CÁC MÃ ĐÃ ĐƯỢC XỬ LÝ XONG!")

