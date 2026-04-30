import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_variant(df, features, label):
    # Ensure all features exist in df
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].values
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
            y.append(data[i + seq_length, 3]) # Close price
        return np.array(x), np.array(y)
    
    X_train, y_train = create_seq(train_data, seq_length)
    X_test, y_test = create_seq(test_data, seq_length)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_diff = y_train - X_train[:, -1, 3]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X_train_flat, y_train_diff)
    preds = X_test[:, -1, 3] + model.predict(X_test_flat)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return y_test, preds, r2, rmse

def analyze_ticker_all(ticker, processed_dir, notebook_dir):
    print(f"\n>>> Đang so sánh toàn bộ các biến thể cho {ticker}...")
    
    # Files
    f_mbert = os.path.join(processed_dir, f"{ticker}_final_8_features_mbert.csv")
    f_xlmr = os.path.join(processed_dir, f"{ticker}_final_8_features_xlmr.csv")
    f_thay = os.path.join(processed_dir, f"{ticker}_final_expanded.csv")
    
    # Special VNM case
    if ticker == "VNM":
        f_mbert = os.path.join(processed_dir, "VNM_final_compare_mbert_test.csv")
        f_xlmr = os.path.join(processed_dir, "VNM_final_compare_xlmr_test.csv")
        
    if not (os.path.exists(f_mbert) and os.path.exists(f_xlmr) and os.path.exists(f_thay)):
        print(f"  ! Thiếu dữ liệu cho {ticker}")
        return

    def load_clean(path):
        df = pd.read_csv(path)
        df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
        return df

    df_mbert = load_clean(f_mbert)
    df_xlmr = load_clean(f_xlmr)
    df_thay = load_clean(f_thay)

    # Feature Sets
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    single_sent = price_cols + ['avg_polarity']
    full_8 = price_cols + ['avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance', 'volume_mentions', 'weighted_sentiment_index', 'sentiment_p_n_score', 'temporal_factor']
    
    results = {}
    
    # 1. mBERT + 8 attributes
    _, p_m8, r2_m8, rmse_m8 = train_variant(df_mbert, full_8, "mBERT+8")
    results['mBERT+8'] = {'preds': p_m8, 'R2': r2_m8, 'RMSE': rmse_m8}
    
    # 2. XLM-R Single
    _, p_x1, r2_x1, rmse_x1 = train_variant(df_xlmr, single_sent, "XLM-R Single")
    results['XLM-R Single'] = {'preds': p_x1, 'R2': r2_x1, 'RMSE': rmse_x1}
    
    # 3. XLM-R + 8 attributes
    _, p_x8, r2_x8, rmse_x8 = train_variant(df_xlmr, full_8, "XLM-R+8")
    results['XLM-R+8'] = {'preds': p_x8, 'R2': r2_x8, 'RMSE': rmse_x8}
    
    # 4. Thay Single
    _, p_t1, r2_t1, rmse_t1 = train_variant(df_thay, single_sent, "Thay Single")
    results['Thay Single'] = {'preds': p_t1, 'R2': r2_t1, 'RMSE': rmse_t1}
    
    # 5. Thay + 8 attributes
    feat_thay8 = price_cols + ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    y_test, p_t8, r2_t8, rmse_t8 = train_variant(df_thay, feat_thay8, "Thay+8")
    results['Thay+8'] = {'preds': p_t8, 'R2': r2_t8, 'RMSE': rmse_t8}

    # Plot Comparison
    plt.figure(figsize=(15, 7))
    plt.plot(y_test, label='Giá thực tế', color='black', linewidth=2)
    for label, res in results.items():
        plt.plot(res['preds'], label=f"{label} (R2: {res['R2']:.3f})", alpha=0.7)
    
    plt.title(f"So sánh tất cả các biến thể mô hình - {ticker}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(notebook_dir, f"{ticker}_all_variants_comparison.png"))
    plt.close()

    print(f"--- Kết quả cho {ticker} ---")
    for label, res in results.items():
        print(f"  {label:15}: R2 = {res['R2']:.4f}, RMSE = {res['RMSE']:.4f}")

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    processed_dir = os.path.join(base_dir, "data", "processed")
    notebook_dir = os.path.join(base_dir, "notebooks")
    
    tickers = ["AAPL", "AMZN", "GOOGL", "VNM"]
    for t in tickers:
        analyze_ticker_all(t, processed_dir, notebook_dir)
