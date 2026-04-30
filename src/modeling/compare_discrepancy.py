import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

def get_predictions(df, features):
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    seq_length = 10
    def create_seq(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:(i + seq_length), :])
            y.append(data[i + seq_length, 3]) # Close price index
        return np.array(x), np.array(y)
    
    X_train, y_train = create_seq(train_data, seq_length)
    X_test, y_test = create_seq(test_data, seq_length)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_diff = y_train - X_train[:, -1, 3]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X_train_flat, y_train_diff)
    preds = X_test[:, -1, 3] + model.predict(X_test_flat)
    
    return y_test, preds

def analyze_discrepancy(ticker, processed_dir, notebook_dir):
    print(f"\n>>> Phân tích chênh lệch cho {ticker}...")
    
    # Files paths
    f_mbert = os.path.join(processed_dir, f"{ticker}_final_8_features_mbert.csv")
    f_xlmr = os.path.join(processed_dir, f"{ticker}_final_8_features_xlmr.csv")
    f_thay = os.path.join(processed_dir, f"{ticker}_final_expanded.csv")
    
    # Trường hợp đặc biệt cho VNM
    if ticker == "VNM":
        f_mbert = os.path.join(processed_dir, "VNM_final_compare_mbert_test.csv")
        f_xlmr = os.path.join(processed_dir, "VNM_final_compare_xlmr_test.csv")
        f_thay = os.path.join(processed_dir, "VNM_final_expanded.csv")

    if not (os.path.exists(f_mbert) and os.path.exists(f_xlmr) and os.path.exists(f_thay)):
        print(f"  ! Thiếu dữ liệu so sánh cho {ticker}")
        return

    def load_clean(path):
        df = pd.read_csv(path)
        df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
        return df

    df_mbert = load_clean(f_mbert)
    df_xlmr = load_clean(f_xlmr)
    df_thay = load_clean(f_thay)

    # Features for 8-attribute models
    feat_8 = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance', 'volume_mentions', 'weighted_sentiment_index', 'sentiment_p_n_score', 'temporal_factor']
    
    # Features for Thay (Expanded) - check available columns
    feat_thay = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    
    # Filter features that exist in the dataframe
    f_mbert_final = [f for f in feat_8 if f in df_mbert.columns]
    f_xlmr_final = [f for f in feat_8 if f in df_xlmr.columns]
    f_thay_final = [f for f in feat_thay if f in df_thay.columns]

    y_true, p_mbert = get_predictions(df_mbert, f_mbert_final)
    _, p_xlmr = get_predictions(df_xlmr, f_xlmr_final)
    _, p_thay = get_predictions(df_thay, f_thay_final)

    min_len = min(len(y_true), len(p_mbert), len(p_xlmr), len(p_thay))
    y_true, p_mbert, p_xlmr, p_thay = y_true[:min_len], p_mbert[:min_len], p_xlmr[:min_len], p_thay[:min_len]

    # Directional Accuracy
    actual_move = (y_true[1:] > y_true[:-1]).astype(int)
    mbert_move = (p_mbert[1:] > y_true[:-1]).astype(int)
    xlmr_move = (p_xlmr[1:] > y_true[:-1]).astype(int)
    thay_move = (p_thay[1:] > y_true[:-1]).astype(int)

    # Absolute Error
    err_mbert = np.abs(y_true - p_mbert)
    err_xlmr = np.abs(y_true - p_xlmr)
    err_thay = np.abs(y_true - p_thay)

    # Plot
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='Giá thực tế', color='black', linewidth=2)
    plt.plot(p_mbert, label='mBERT + 8 Chỉ số', alpha=0.7)
    plt.plot(p_xlmr, label='XLM-RoBERTa + 8 Chỉ số', alpha=0.7)
    plt.plot(p_thay, label='Mô hình Thầy (Expanded)', alpha=0.7)
    plt.title(f"So sánh dự báo 3 mô hình khi có tin tức thị trường - {ticker}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(err_mbert, label='Sai số mBERT', alpha=0.6)
    plt.plot(err_xlmr, label='Sai số XLM-R', alpha=0.6)
    plt.plot(err_thay, label='Sai số Thầy', alpha=0.6)
    plt.title(f"Phân tích chênh lệch sai số tuyệt đối - {ticker}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(notebook_dir, f"{ticker}_discrepancy_analysis.png"))
    plt.close()

    print(f"[{ticker}] Độ chính xác hướng (Up/Down):")
    print(f"  - mBERT: {np.mean(actual_move == mbert_move):.2%}")
    print(f"  - XLM-R: {np.mean(actual_move == xlmr_move):.2%}")
    print(f"  - Thầy : {np.mean(actual_move == thay_move):.2%}")

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    processed_dir = os.path.join(base_dir, "data", "processed")
    notebook_dir = os.path.join(base_dir, "notebooks")
    
    tickers = ["AAPL", "AMZN", "GOOGL", "VNM"]
    for t in tickers:
        analyze_discrepancy(t, processed_dir, notebook_dir)
