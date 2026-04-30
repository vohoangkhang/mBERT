import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

def train_and_eval_source(ticker, source_name, file_path):
    print(f"\n>>> Huấn luyện {ticker} với nguồn Sentiment: {source_name}...")
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'avg_polarity', 'avg_intensity', 'avg_credibility',
                'avg_relevance', 'volume_mentions', 'weighted_sentiment_index',
                'sentiment_p_n_score', 'temporal_factor']

    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    seq_length = 10
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:(i + seq_length), :])
            y.append(data[i + seq_length, 3])
        return np.array(x), np.array(y)

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # XGBoost (Price Diff)
    y_train_diff = y_train - X_train[:, -1, 3]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model_xgb.fit(X_train_flat, y_train_diff)
    y_pred_xgb = X_test[:, -1, 3] + model_xgb.predict(X_test_flat)

    r2 = r2_score(y_test, y_pred_xgb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    print(f"  => {source_name} Result - R2: {r2:.4f}, RMSE: {rmse:.4f}")
    return r2, rmse

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    processed_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    # Chuẩn bị 3 file dataset để gộp
    price_file = os.path.join(raw_dir, "VNM_prices.csv")
    df_price = pd.read_csv(price_file)
    df_price['Date'] = pd.to_datetime(df_price['time']).dt.date

    sources = ["thay", "mbert_test", "xlmr_test"]
    results = {}

    for src in sources:
        sentiment_file = os.path.join(processed_dir, f"VNM_8_features_{src}.csv")
        df_sent = pd.read_csv(sentiment_file)
        df_sent['Date'] = pd.to_datetime(df_sent['Date']).dt.date
        
        final_df = pd.merge(df_price, df_sent, on='Date', how='left').fillna(0)
        final_path = os.path.join(processed_dir, f"VNM_final_compare_{src}.csv")
        final_df.to_csv(final_path, index=False)
        
        r2, rmse = train_and_eval_source("VNM", src, final_path)
        results[src] = {"R2": r2, "RMSE": rmse}

    print("\n--- BẢNG SO SÁNH HIỆU QUẢ SENTIMENT (mBERT vs XLM-R vs Thầy) ---")
    for k, v in results.items():
        print(f"Model {k.upper()}: R2 = {v['R2']:.4f}")
