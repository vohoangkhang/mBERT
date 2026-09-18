import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# --- CẤU HÌNH ---
TICKERS = ["AAPL", "VNM", "AMZN", "GOOGL", "META", "BABA"]
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HÀM TÍNH METRICS ---
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# --- HÀM TẠO DỮ LIỆU CHUỖI ---
def create_sequences(data, seq_length=10):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(x), np.array(y)

# --- HUẤN LUYỆN VÀ DỰ BÁO ---
def train_and_predict(ticker, df, features, model_type="XGB"):
    target = 'Close'
    data_cols = features + [target]
    df_clean = df[data_cols].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    if model_type == "XGB":
        # XGBoost dùng dữ liệu dạng bảng
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == "BiLSTM":
        # BiLSTM dùng dữ liệu dạng chuỗi (Sequence)
        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)
        
        if len(X_test) == 0: return None, None
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(X_test).flatten()
        
    # Inverse scale
    dummy_test = np.zeros((len(y_test), len(data_cols)))
    dummy_test[:, -1] = y_test
    y_test_inv = scaler.inverse_transform(dummy_test)[:, -1]
    
    dummy_pred = np.zeros((len(y_pred), len(data_cols)))
    dummy_pred[:, -1] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]
    
    return y_test_inv, y_pred_inv

# --- CHƯƠNG TRÌNH CHÍNH ---
for ticker in TICKERS:
    print(f"\n>>> ĐANG XỬ LÝ {ticker}...")
    f_clean = os.path.join(DATA_DIR, f"{ticker}_final_v4_clean.csv")
    f_mbert = os.path.join(DATA_DIR, f"{ticker}_mbert_final.csv")
    
    df_c = pd.read_csv(f_clean)
    df_m = pd.read_csv(f_mbert)
    
    # 1-2. Pure (Giá)
    feat_pure = ['Open', 'High', 'Low', 'Volume']
    y_true_pure, y_p_xgb_pure = train_and_predict(ticker, df_c, feat_pure, "XGB")
    _, y_p_lstm_pure = train_and_predict(ticker, df_c, feat_pure, "BiLSTM")
    
    # 4-5. Sentiment (Giá + Sentiment Thầy)
    feat_sent = feat_pure + ['sentiment'] if 'sentiment' in df_c.columns else feat_pure
    y_true_sent, y_p_xgb_sent = train_and_predict(ticker, df_c, feat_sent, "XGB")
    _, y_p_lstm_sent = train_and_predict(ticker, df_c, feat_sent, "BiLSTM")
    
    # 6-7. mBERT (Giá + 8 mBERT)
    feat_mbert = feat_pure + ['avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance', 
                             'count_mentions', 'sum_polarity', 'max_polarity', 'min_polarity']
    y_true_mbert, y_p_xgb_mbert = train_and_predict(ticker, df_m, feat_mbert, "XGB")
    _, y_p_lstm_mbert = train_and_predict(ticker, df_m, feat_mbert, "BiLSTM")
    
    # 3. mBERT Label Visualization (Đặc biệt)
    # So sánh sentiment_score (từ nhãn 1-5 sao) với giá
    # Lấy dữ liệu mBERT thô (chỉ cột sentiment)
    df_m['sentiment_mapped'] = df_m['avg_polarity'] # Chính là (stars-3)/2
    
    # --- ĐỒ THỊ 1: SO SÁNH 7 KỊCH BẢN ---
    plt.figure(figsize=(15, 8))
    # Đồng bộ hóa chiều dài
    min_len = min(len(y_true_pure), len(y_p_xgb_pure), len(y_p_lstm_pure), len(y_p_xgb_sent), len(y_p_lstm_sent), len(y_p_xgb_mbert), len(y_p_lstm_mbert))
    
    plt.plot(y_true_pure[-min_len:], label='Giá Thực Tế (Actual)', color='black', linewidth=2)
    plt.plot(y_p_xgb_pure[-min_len:], label='1. XGBoost Pure', linestyle='--')
    plt.plot(y_p_lstm_pure[-min_len:], label='2. BiLSTM Pure', linestyle='--')
    plt.plot(y_p_xgb_sent[-min_len:], label='4. XGBoost + Sentiment', alpha=0.7)
    plt.plot(y_p_lstm_sent[-min_len:], label='5. BiLSTM + Sentiment', alpha=0.7)
    plt.plot(y_p_xgb_mbert[-min_len:], label='6. XGBoost + mBERT', linewidth=2)
    plt.plot(y_p_lstm_mbert[-min_len:], label='7. BiLSTM + mBERT', linewidth=2)
    
    plt.title(f"Comparison of 7 Scenarios for Stock Prediction - {ticker}")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_7_scenarios_comparison.png"))
    plt.close()
    
    # --- ĐỒ THỊ 2: mBERT TIN TỨC -> NHÃN (Scenario 3) ---
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    last_100 = df_m.tail(100)
    ax1.plot(last_100['Date'], last_100['Close'], label='Giá Đóng Cửa', color='blue')
    # Vẽ cột sentiment (Nhãn đã quy đổi từ 1-5 sao)
    ax2.bar(last_100['Date'], last_100['avg_polarity'], alpha=0.3, color='orange', label='mBERT Sentiment Score (-1 to 1)')
    
    ax1.set_ylabel("Stock Price", color='blue')
    ax2.set_ylabel("mBERT Sentiment (from 1-5 stars)", color='orange')
    plt.title(f"3. mBERT Sentiment Label Analysis - {ticker}")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_scenario_3_mbert_labels.png"))
    plt.close()
    
    # --- BẢNG METRICS ---
    metrics_list = []
    scenarios = {
        "1. XGB Pure": (y_true_pure, y_p_xgb_pure),
        "2. BiLSTM Pure": (y_true_pure, y_p_lstm_pure),
        "4. XGB + Sentiment": (y_true_sent, y_p_xgb_sent),
        "5. BiLSTM + Sentiment": (y_true_sent, y_p_lstm_sent),
        "6. XGB + mBERT": (y_true_mbert, y_p_xgb_mbert),
        "7. BiLSTM + mBERT": (y_true_mbert, y_p_lstm_mbert)
    }
    
    for name, (yt, yp) in scenarios.items():
        if yt is not None and yp is not None:
            m = calculate_metrics(yt[-min_len:], yp[-min_len:])
            m["Scenario"] = name
            metrics_list.append(m)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_metrics_table.csv"), index=False)
    print(f"  => Đã lưu đồ thị và bảng metrics cho {ticker}")

print("\n>>> HOÀN TẤT TOÀN BỘ BÁO CÁO 7 KỊCH BẢN!")
