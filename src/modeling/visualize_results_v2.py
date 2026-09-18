import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# Tắt warning TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_sequences(data, target, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(target[i + seq_length - 1])
    return np.array(x), np.array(y)

def train_eval_model(ticker, data_path, scenario_type, model_name):
    """
    scenario_type: 'Pure', 'Sentiment', 'mBERT', 'mBERT_Only'
    model_name: 'XGBoost', 'BiLSTM', 'mBERT'
    """
    df = pd.read_csv(data_path)
    df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df = df.dropna()
    
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Danh sách các cột sentiment tiềm năng
    potential_sent_cols = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 
                          'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance', 'max_intensity']
    
    # Chỉ lấy những cột thực sự tồn tại trong dataframe
    available_sent_cols = [c for c in potential_sent_cols if c in df.columns]
    
    if scenario_type == 'Pure':
        cols = price_cols
    elif scenario_type == 'Sentiment' or scenario_type == 'mBERT':
        cols = price_cols + available_sent_cols
    elif scenario_type == 'mBERT_Only':
        cols = available_sent_cols
        
    data_feat = df[cols].values
    data_target = df['Target'].values.reshape(-1, 1)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_x = scaler_x.fit_transform(data_feat)
    scaled_y = scaler_y.fit_transform(data_target)
    
    seq_length = 10
    X, y = create_sequences(scaled_x, scaled_y, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if 'XGBoost' in model_name or model_name == 'mBERT': # mBERT standalone dùng XGB cho nhanh hoặc Linear
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = XGBRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(X_train_flat, y_train)
        y_pred_scaled = model.predict(X_test_flat).reshape(-1, 1)
    else: # BiLSTM
        model = Sequential([
            Input(shape=(seq_length, len(cols))),
            Bidirectional(LSTM(32, return_sequences=True)),
            Bidirectional(LSTM(16)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred_scaled = model.predict(X_test)

    y_pred_diff = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual_diff = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    last_close_test = df['Close'].values[train_size + seq_length - 1 : -1] 
    price_actual = last_close_test + y_actual_diff
    price_pred = last_close_test + y_pred_diff
    
    return price_actual, price_pred

def plot_correlation_matrix(ticker, data_path, output_path):
    df = pd.read_csv(data_path)
    cols_to_corr = ['Close', 'Volume', 'avg_polarity', 'max_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity']
    # Lọc các cột tồn tại
    cols_to_corr = [c for c in cols_to_corr if c in df.columns]
    
    plt.figure(figsize=(10, 8))
    corr = df[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix: Price vs Sentiment - {ticker}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    tickers = ["AAPL", "VNM", "AMZN", "GOOGL", "META", "BABA"]
    base_dir = "."
    output_dir = "visualizations_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Vẽ Ma trận tương quan (Lấy AAPL làm đại diện)
    print("Vẽ Ma trận tương quan...")
    plot_correlation_matrix("AAPL", f"data/processed/AAPL_mbert_final.csv", f"{output_dir}/correlation_matrix.png")

    # Danh sách các Model cần so sánh theo yêu cầu
    models_to_run = [
        ("XGBoost", "Pure", "XGBoost"),
        ("BiLSTM", "Pure", "BiLSTM"),
        ("XGBoost + Sentiment", "Sentiment", "XGBoost"),
        ("BiLSTM + Sentiment", "Sentiment", "BiLSTM"),
        ("XGBoost + mBERT", "mBERT", "XGBoost"),
        ("BiLSTM + mBERT", "mBERT", "BiLSTM"),
        ("mBERT", "mBERT_Only", "mBERT")
    ]
    
    for model_label, scenario_type, engine in models_to_run:
        print(f"Đang xử lý model: {model_label}...")
        plt.figure(figsize=(20, 12))
        
        for i, ticker in enumerate(tickers):
            # Chọn file phù hợp
            if "mBERT" in model_label:
                path = f"data/processed/{ticker}_mbert_final.csv"
            else:
                path = f"data/processed/{ticker}_final_v4_clean.csv"
            
            if not os.path.exists(path): continue
                
            actual, pred = train_eval_model(ticker, path, scenario_type, engine)
            
            # Vẽ subplot 2x3
            plt.subplot(2, 3, i+1)
            plt.plot(actual, label='Actual', color='blue', alpha=0.6)
            plt.plot(pred, label='Predicted', color='red', linestyle='--')
            plt.title(f"{ticker}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.suptitle(f"Model Comparison: {model_label}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        file_name = model_label.replace(" + ", "_").replace(" ", "_").lower()
        plt.savefig(f"{output_dir}/{file_name}_comparison.png")
        plt.close()
        print(f"  => Đã lưu {file_name}_comparison.png")

if __name__ == "__main__":
    main()
