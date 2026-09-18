
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import glob

# Thiết lập thư mục và danh sách mã cổ phiếu
DATA_DIR = "data/processed"
OUTPUT_DIR = "statistical_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "VNM", "AMZN", "GOOGL", "META", "BABA"]

def get_errors(ticker):
    # Đọc dữ liệu
    df_c = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}_final_v4_clean.csv"))
    df_m = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}_mbert_final.csv"))
    
    # Định nghĩa features
    feat_pure = ['Open', 'High', 'Low', 'Volume']
    feat_sent = feat_pure + ['sentiment'] if 'sentiment' in df_c.columns else feat_pure
    feat_mbert = feat_pure + ['avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance', 
                             'count_mentions', 'sum_polarity', 'max_polarity', 'min_polarity']
    
    # Hàm dự báo nhanh (Train-Test Split 80-20)
    def predict_errors(df, features, model_type="XGB"):
        X = df[features].values
        y = df['Close'].values
        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        if model_type == "XGB":
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else: # BiLSTM
            # Scaling
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_s = scaler_x.fit_transform(X)
            y_s = scaler_y.fit_transform(y.reshape(-1, 1))
            X_train_s, X_test_s = X_s[:split], X_s[split:]
            y_train_s, y_test_s = y_s[:split], y_s[split:]
            
            # Reshape for LSTM
            X_train_s = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
            X_test_s = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(1, X_train_s.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_s, y_train_s, epochs=10, batch_size=32, verbose=0)
            preds_s = model.predict(X_test_s)
            preds = scaler_y.inverse_transform(preds_s).flatten()
            
        # Trả về sai số tuyệt đối hàng ngày (Daily Absolute Errors)
        return np.abs(y_test - preds)

    errs = {}
    errs['XGB_Pure'] = predict_errors(df_c, feat_pure, "XGB")
    errs['BiLSTM_Pure'] = predict_errors(df_c, feat_pure, "BiLSTM")
    errs['XGB_Sentiment'] = predict_errors(df_c, feat_sent, "XGB")
    errs['BiLSTM_Sentiment'] = predict_errors(df_c, feat_sent, "BiLSTM")
    errs['XGB_mBERT'] = predict_errors(df_m, feat_mbert, "XGB")
    errs['BiLSTM_mBERT'] = predict_errors(df_m, feat_mbert, "BiLSTM")
    
    return errs

# Thu thập sai số từ tất cả cổ phiếu
all_errors = {k: [] for k in ['XGB_Pure', 'BiLSTM_Pure', 'XGB_Sentiment', 'BiLSTM_Sentiment', 'XGB_mBERT', 'BiLSTM_mBERT']}

for ticker in TICKERS:
    print(f"Đang xử lý {ticker}...")
    ticker_errs = get_errors(ticker)
    for k in all_errors:
        all_errors[k].extend(ticker_errs[k])

# Chuyển thành DataFrame
error_df = pd.DataFrame(all_errors)

# --- DESCRIPTIVE STATISTICS ---
stats_summary = []
for col in error_df.columns:
    stats_summary.append({
        'Model': col,
        'Mean (MAE)': np.mean(error_df[col]),
        'Median': np.median(error_df[col]),
        'Std Dev': np.std(error_df[col]),
        'Max Error': np.max(error_df[col])
    })
summary_df = pd.DataFrame(stats_summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"), index=False)

# --- STATISTICAL SIGNIFICANCE TESTS ---
results = []
comparisons = [
    ('XGB_Pure', 'XGB_mBERT'),
    ('BiLSTM_Pure', 'BiLSTM_mBERT'),
    ('XGB_Sentiment', 'XGB_mBERT'),
    ('BiLSTM_Sentiment', 'BiLSTM_mBERT')
]

for a, b in comparisons:
    t_stat, p_val = stats.ttest_rel(error_df[a], error_df[b])
    wilcoxon_stat, w_p_val = stats.wilcoxon(error_df[a], error_df[b])
    results.append({
        'Comparison': f"{a} vs {b}",
        'Mean Error A': np.mean(error_df[a]),
        'Mean Error B': np.mean(error_df[b]),
        'Improvement (%)': ((np.mean(error_df[a]) - np.mean(error_df[b])) / np.mean(error_df[a])) * 100,
        'p-value (Wilcoxon)': w_p_val,
        'Significant (p<0.05)': "YES" if w_p_val < 0.05 else "NO"
    })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_statistical_test.csv"), index=False)

# --- VISUALIZATION ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 12))

# 1. Boxplot of Errors
ax1 = plt.subplot(3, 1, 1)
plot_df = error_df.melt(var_name='Model', value_name='Daily Absolute Error')
sns.boxplot(x='Model', y='Daily Absolute Error', data=plot_df, palette='Set2', ax=ax1)
ax1.set_title("Daily Absolute Error Distribution (All 6 Tickers)", fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylabel("Error (Log Scale)")

# 2. Descriptive Statistics Table
ax2 = plt.subplot(3, 1, 2)
ax2.axis('off')
table_data = summary_df.copy()
# Format numbers for display
for col in ['Mean (MAE)', 'Median', 'Std Dev', 'Max Error']:
    table_data[col] = table_data[col].map('{:.4f}'.format)

tbl = ax2.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)
ax2.set_title("Descriptive Statistics Summary", fontsize=14, fontweight='bold', pad=20)

# 3. Wilcoxon Test Results Table
ax3 = plt.subplot(3, 1, 3)
ax3.axis('off')
p_vals_display = res_df.copy()
# Format p-values
p_vals_display['p-value (Wilcoxon)'] = p_vals_display['p-value (Wilcoxon)'].apply(lambda x: f"{x:.4e}")
p_vals_display['Mean Error A'] = p_vals_display['Mean Error A'].map('{:.4f}'.format)
p_vals_display['Mean Error B'] = p_vals_display['Mean Error B'].map('{:.4f}'.format)
p_vals_display['Improvement (%)'] = p_vals_display['Improvement (%)'].map('{:.2f}%'.format)

tbl2 = ax3.table(cellText=p_vals_display.values, colLabels=p_vals_display.columns, loc='center', cellLoc='center')
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10)
tbl2.scale(1.2, 1.5)
ax3.set_title("Wilcoxon Signed-Rank Test Results", fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "statistical_analysis_report.png"), dpi=300, bbox_inches='tight')
print(f"Report saved at: {OUTPUT_DIR}/statistical_analysis_report.png")

# Display to console
print("\n--- DETAILED STATISTICAL TEST RESULTS ---")
print(res_df.to_string())
