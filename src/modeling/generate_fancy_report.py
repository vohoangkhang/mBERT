import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# Thiết lập style cho đồ thị chuyên nghiệp
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_sequences_v2(data, target, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(target[i + seq_length - 1])
    return np.array(x), np.array(y)

def print_fancy_table(label, metrics):
    """In bảng metrics ra console đẹp mắt"""
    print(f"\n[ BÁO CÁO KẾT QUẢ: {label} ]")
    print("-" * 65)
    print(f"{'Metric':<20} | {'Giá trị':<20}")
    print("-" * 65)
    print(f"{'MSE':<20} | {metrics['mse']:.6f}")
    print(f"{'RMSE':<20} | {metrics['rmse']:.6f}")
    print(f"{'MAE':<20} | {metrics['mae']:.6f}")
    print(f"{'MAPE':<20} | {metrics['mape']:.4f}%")
    print(f"{'R2 Score':<20} | {metrics['r2']:.6f}")
    print("-" * 65)
    print(f"Dự báo cuối: Thực tế {metrics['actual_price'][-1]:.2f} | Dự báo {metrics['pred_price'][-1]:.2f}")
    print(f"Sai lệch: {abs(metrics['actual_price'][-1] - metrics['pred_price'][-1]):.2f}")
    print("-" * 65)

def train_and_eval_robust(ticker, data_path, scenario, model_type="XGB"):
    df = pd.read_csv(data_path)
    df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df = df.dropna()
    
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
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data_feat)
    scaled_y = scaler_y.fit_transform(data_target)
    
    seq_length = 10
    X, y = create_sequences_v2(scaled_x, scaled_y, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if model_type == "XGB":
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=7, subsample=0.8, verbosity=0)
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

    y_pred_diff = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual_diff = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    last_close_test = df['Close'].values[train_size + seq_length - 1 : -1] 
    price_actual = last_close_test + y_actual_diff
    price_pred = last_close_test + y_pred_diff
    
    metrics = {
        "mse": mean_squared_error(price_actual, price_pred),
        "rmse": np.sqrt(mean_squared_error(price_actual, price_pred)),
        "mae": mean_absolute_error(price_actual, price_pred),
        "r2": r2_score(price_actual, price_pred),
        "mape": calculate_mape(price_actual, price_pred),
        "pred_price": price_pred, 
        "actual_price": price_actual
    }
    return metrics

def generate_report_for_ticker(ticker, base_dir):
    print(f"\n{'='*80}")
    print(f"🚀 BẮT ĐẦU QUY TRÌNH PHÂN TÍCH & BÁO CÁO CHO MÃ: {ticker}")
    print(f"{'='*80}")
    
    clean_path = os.path.join(base_dir, "data", "processed", f"{ticker}_final_v4_clean.csv")
    mbert_path = os.path.join(base_dir, "data", "processed", f"{ticker}_mbert_final.csv")
    output_dir = os.path.join(base_dir, "reports", "final_visuals", ticker)
    os.makedirs(output_dir, exist_ok=True)

    scenarios = [
        {"name": "XGBoost (Chỉ dùng Giá)", "path": clean_path, "type": "XGB", "desc": "Mô hình XGBoost cơ bản sử dụng các chỉ số kỹ thuật truyền thống."},
        {"name": "BiLSTM (Chỉ dùng Giá)", "path": clean_path, "type": "BiLSTM", "desc": "Mô hình BiLSTM mạnh mẽ trong việc học chuỗi thời gian từ giá chứng khoán."},
        {"name": "BiLSTM + Sentiment (Thầy Khang)", "path": clean_path, "type": "BiLSTM", "desc": "Kết hợp phân tích cảm xúc từ tin tức (Logic Backward) vào mô hình BiLSTM."},
        {"name": "mBERT + BiLSTM (Tiên tiến nhất)", "path": mbert_path, "type": "BiLSTM", "desc": "Sử dụng mô hình ngôn ngữ mBERT (Multilingual BERT) để trích xuất đặc trưng cảm xúc sâu sắc."}
    ]

    all_metrics = []

    plt.figure(figsize=(15, 8))
    actual_plotted = False

    for sc in scenarios:
        if not os.path.exists(sc['path']): continue
        
        print(f"\n🔹 GIỚI THIỆU MÔ HÌNH: {sc['name']}")
        print(f"   > {sc['desc']}")
        print(f"   [ ĐANG THỰC THI... ]")
        
        res = train_and_eval_robust(ticker, sc['path'], sc['name'], sc['type'])
        print_fancy_table(sc['name'], res)
        
        # Vẽ từng đường đồ thị để lưu vào file riêng lẻ và file tổng
        plt.plot(res['pred_price'], label=f"{sc['name']} (MAPE: {res['mape']:.2f}%)")
        if not actual_plotted:
            plt.plot(res['actual_price'], label='Giá Thực Tế (Actual)', color='black', linewidth=3, alpha=0.5)
            actual_plotted = True
            
        # Lưu metrics vào danh sách tổng hợp để in bảng cuối
        all_metrics.append({
            "Mô hình": sc['name'],
            "MAPE (%)": f"{res['mape']:.4f}",
            "R2 Score": f"{res['r2']:.4f}",
            "RMSE": f"{res['rmse']:.4f}"
        })
        
        # Vẽ và lưu đồ thị riêng cho kịch bản này
        plt_sc = plt.figure(figsize=(12, 6))
        plt.plot(res['actual_price'], label='Thực tế', color='black', alpha=0.6)
        plt.plot(res['pred_price'], label='Dự báo', color='red')
        plt.title(f"{ticker} - {sc['name']}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{sc['name'].replace(' ', '_').lower()}.png"), dpi=300)
        plt.close(plt_sc)

    # Đồ thị tổng hợp
    plt.title(f"TỔNG HỢP SO SÁNH CÁC MÔ HÌNH - {ticker}", fontsize=16)
    plt.xlabel("Ngày dự báo (Tập Test)", fontsize=12)
    plt.ylabel("Giá đóng cửa (Close Price)", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{ticker}_master_comparison.png"), dpi=300)
    plt.close()

    # Bảng tổng kết cuối cùng
    print(f"\n📊 BẢNG TỔNG HỢP METRICS CHO {ticker}:")
    summary_df = pd.DataFrame(all_metrics)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    
    print(f"\n✅ Đã hoàn thành báo cáo cho {ticker}. File lưu tại: {output_dir}")

if __name__ == "__main__":
    # Ưu tiên chạy VNM trước để bạn kiểm tra
    tickers = ["VNM", "AAPL", "AMZN", "GOOGL", "BABA", "META"]
    for t in tickers:
        generate_report_for_ticker(t, ".")
    print("\n🎉 TẤT CẢ CÁC MÃ ĐÃ ĐƯỢC XỬ LÝ XONG VÀ LƯU VÀO THƯ MỤC 'reports/final_visuals/'!")
