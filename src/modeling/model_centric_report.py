import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# --- CONFIGURATION ---
TICKERS = ["AAPL", "VNM", "AMZN", "GOOGL", "META", "BABA"]
MODELS = [
    "XGBoost Pure", 
    "BiLSTM Pure", 
    "mBERT Sentiment Analysis",
    "XGBoost + Sentiment", 
    "BiLSTM + Sentiment", 
    "XGBoost + mBERT", 
    "BiLSTM + mBERT"
]
OUTPUT_DIR = "visualizations_model_centric"
DATA_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sequences(data, target, seq_length=10):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(target[i + seq_length - 1])
    return np.array(x), np.array(y)

def get_predictions(ticker, scenario):
    f_clean = os.path.join(DATA_DIR, f"{ticker}_final_v4_clean.csv")
    f_mbert = os.path.join(DATA_DIR, f"{ticker}_mbert_final.csv")
    
    if "mBERT" in scenario and "Analysis" not in scenario:
        df = pd.read_csv(f_mbert)
        feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_polarity', 'max_polarity', 'min_polarity', 
                     'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    else:
        df = pd.read_csv(f_clean)
        df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]
        if "Sentiment" in scenario:
            feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment'] if 'sentiment' in df.columns else ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    feat_cols = [c for c in feat_cols if c in df.columns]
    
    # Delta Prediction (V2 Logic)
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df = df.dropna()
    
    data_feat = df[feat_cols].values
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

    if "XGBoost" in scenario:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train_flat, y_train)
        y_pred_scaled = model.predict(X_test_flat).reshape(-1, 1)
    else: # BiLSTM
        model = Sequential([
            Input(shape=(seq_length, len(feat_cols))),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        y_pred_scaled = model.predict(X_test)

    y_pred_diff = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual_diff = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    last_close_test = df['Close'].values[train_size + seq_length - 1 : -1] 
    price_actual = last_close_test + y_actual_diff
    price_pred = last_close_test + y_pred_diff
    
    return price_actual, price_pred

# --- RUN AND PLOT ---
for scenario in MODELS:
    print(f"\n>>> Generating plot for: {scenario}")
    
    if scenario == "mBERT Sentiment Analysis":
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle(scenario.upper(), fontsize=24, fontweight='bold')
        for i, ticker in enumerate(TICKERS):
            ax = axes[i//2, i%2]
            df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}_mbert_final.csv")).tail(100)
            ax2 = ax.twinx()
            ax.plot(df['Date'], df['Close'], color='blue', label='Stock Price')
            ax2.bar(df['Date'], df['avg_polarity'], color='orange', alpha=0.4, label='mBERT Sentiment')
            ax.set_title(ticker, fontsize=16)
            if i % 2 == 0: ax.set_ylabel("Price (USD)")
            ax.tick_params(axis='x', labelbottom=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, "mBERT_Sentiment_Correlation_All.png"))
        plt.close()
        continue

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(scenario.upper(), fontsize=24, fontweight='bold')
    
    for i, ticker in enumerate(TICKERS):
        print(f"  - Processing {ticker}...")
        ax = axes[i//2, i%2]
        try:
            y_true, y_pred = get_predictions(ticker, scenario)
            ax.plot(y_true, label='Actual', color='black', alpha=0.7, linewidth=1.5)
            ax.plot(y_pred, label='Predicted', color='red', linestyle='--', alpha=0.8)
            ax.set_title(ticker, fontsize=16)
            ax.legend()
            if i % 2 == 0: ax.set_ylabel("Price (USD)")
        except Exception as e:
            print(f"  ! Error at {ticker}: {e}")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_name = scenario.replace(" ", "_").replace("+", "plus")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{file_name}_All_Stocks.png"))
    plt.close()

print("\n>>> COMPLETED! All plots generated in English with clean titles.")
