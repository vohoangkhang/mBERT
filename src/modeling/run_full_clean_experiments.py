import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
import pmdarima as pm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = r'd:\HUTECH\thayKhang\stock-nlp-prediction'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
METRICS_DIR = os.path.join(BASE_DIR, 'result_final', '1_Metrics')
os.makedirs(METRICS_DIR, exist_ok=True)

TICKERS = ['AAPL', 'AMZN', 'BABA', 'GOOGL', 'META', 'VNM']

FEAT_PURE = ['Open', 'High', 'Low', 'Volume']
FEAT_SENT = FEAT_PURE + ['sentiment']
FEAT_MBERT = FEAT_PURE + [
    'avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance',
    'count_mentions', 'sum_polarity', 'max_polarity', 'min_polarity'
]

def calculate_all_metrics(y_true, y_pred, y_prev):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    actual_direction = np.sign(y_true - y_prev)
    predicted_direction = np.sign(y_pred - y_prev)
    da = float(np.mean(actual_direction == predicted_direction) * 100.0)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R2': r2,
        'DA (%)': da
    }

def train_eval_tabular(df, features, model_class, **kwargs):
    target = 'Close'
    data_cols = features + [target]
    df_clean = df[data_cols].dropna().reset_index(drop=True)
    
    train_size = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:train_size]
    df_test = df_clean.iloc[train_size:].reset_index(drop=True)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(df_train[features])
    scaler_y.fit(df_train[[target]])
    
    X_train = scaler_X.transform(df_train[features])
    y_train = scaler_y.transform(df_train[[target]]).ravel()
    
    X_test = scaler_X.transform(df_test[features])
    y_test = df_test[target].values
    
    all_close = df_clean[target].values
    y_prev = all_close[train_size - 1 : len(all_close) - 1]
    
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    metrics = calculate_all_metrics(y_test, y_pred, y_prev)
    return metrics, y_test, y_pred

def train_eval_bilstm(df, features, seq_length=10):
    target = 'Close'
    data_cols = features + [target]
    df_clean = df[data_cols].dropna().reset_index(drop=True)
    
    train_size = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:train_size]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(df_train[features])
    scaler_y.fit(df_train[[target]])
    
    scaled_X = scaler_X.transform(df_clean[features])
    scaled_y = scaler_y.transform(df_clean[[target]]).ravel()
    
    X_all, y_all = [], []
    for i in range(len(df_clean) - seq_length):
        X_all.append(scaled_X[i : i + seq_length])
        y_all.append(scaled_y[i + seq_length])
    X_all, y_all = np.array(X_all), np.array(y_all)
    
    split_idx = train_size - seq_length
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test = X_all[split_idx:]
    
    y_test = df_clean[target].values[train_size:]
    y_prev = df_clean[target].values[train_size - 1 : len(df_clean) - 1]
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    val_split_len = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-val_split_len], y_train[:-val_split_len]
    X_val, y_val = X_train[-val_split_len:], y_train[-val_split_len:]
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    min_len = min(len(y_test), len(y_pred), len(y_prev))
    y_test_cut = y_test[-min_len:]
    y_pred_cut = y_pred[-min_len:]
    y_prev_cut = y_prev[-min_len:]
    
    metrics = calculate_all_metrics(y_test_cut, y_pred_cut, y_prev_cut)
    return metrics, y_test_cut, y_pred_cut

def run_naive_and_arima(df):
    target = 'Close'
    series = df[target].dropna().values
    train_size = int(len(series) * 0.8)
    
    train_series = series[:train_size]
    test_series = series[train_size:]
    y_prev = series[train_size - 1 : len(series) - 1]
    
    y_pred_naive = y_prev.copy()
    naive_metrics = calculate_all_metrics(test_series, y_pred_naive, y_prev)
    
    try:
        arima_model = pm.auto_arima(
            train_series,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=1,
            seasonal=False,
            suppress_warnings=True,
            error_action='ignore',
            stepwise=True
        )
        order_str = 'ARIMA' + str(arima_model.order)
        y_pred_arima = []
        for t in range(len(test_series)):
            pred = arima_model.predict(n_periods=1)[0]
            y_pred_arima.append(pred)
            arima_model.update([test_series[t]])
        y_pred_arima = np.array(y_pred_arima)
    except Exception as e:
        order_str = 'ARIMA(0,1,0)'
        y_pred_arima = y_prev.copy()
        
    arima_metrics = calculate_all_metrics(test_series, y_pred_arima, y_prev)
    return naive_metrics, test_series, y_pred_naive, arima_metrics, test_series, y_pred_arima, order_str

def main():
    print('='*80)
    print('BAT DAU CHAY TOAN BO 14 MO HINH THUC NGHIEM CHUAN HOA (CLEAN PIPELINE)')
    print('='*80)
    
    all_metrics_records = []
    predictions_dict = {}
    
    for ticker in TICKERS:
        print('\n>>> DANG XU LY MA:', ticker, '...')
        f_clean = os.path.join(DATA_DIR, ticker + '_final_v4_clean.csv')
        f_mbert = os.path.join(DATA_DIR, ticker + '_mbert_final.csv')
        
        df_c = pd.read_csv(f_clean)
        df_m = pd.read_csv(f_mbert)
        
        predictions_dict[ticker] = {}
        
        # 1. Naive Persistence and 2. ARIMA
        m_naive, y_t_n, y_p_n, m_arima, y_t_a, y_p_a, arima_name = run_naive_and_arima(df_c)
        m_naive['Model'] = '1. Naive Persistence'
        m_naive['Ticker'] = ticker
        all_metrics_records.append(m_naive)
        predictions_dict[ticker]['Naive'] = (y_t_n, y_p_n)
        
        m_arima['Model'] = '2. ' + arima_name
        m_arima['Ticker'] = ticker
        all_metrics_records.append(m_arima)
        predictions_dict[ticker]['ARIMA'] = (y_t_a, y_p_a)
        print('  [OK] Naive MAE:', round(m_naive['MAE'], 3), 'DA:', round(m_naive['DA (%)'], 1), '| ARIMA MAE:', round(m_arima['MAE'], 3), 'DA:', round(m_arima['DA (%)'], 1))
        
        # 3-5. Decision Tree
        m_dt_p, yt, yp = train_eval_tabular(df_c, FEAT_PURE, DecisionTreeRegressor, max_depth=5, random_state=42)
        m_dt_p['Model'] = '3. Decision Tree Pure'; m_dt_p['Ticker'] = ticker; all_metrics_records.append(m_dt_p)
        predictions_dict[ticker]['DT_Pure'] = (yt, yp)
        
        m_dt_s, yt, yp = train_eval_tabular(df_c, FEAT_SENT, DecisionTreeRegressor, max_depth=5, random_state=42)
        m_dt_s['Model'] = '4. Decision Tree + Sentiment'; m_dt_s['Ticker'] = ticker; all_metrics_records.append(m_dt_s)
        predictions_dict[ticker]['DT_Sent'] = (yt, yp)
        
        m_dt_m, yt, yp = train_eval_tabular(df_m, FEAT_MBERT, DecisionTreeRegressor, max_depth=5, random_state=42)
        m_dt_m['Model'] = '5. Decision Tree + mBERT'; m_dt_m['Ticker'] = ticker; all_metrics_records.append(m_dt_m)
        predictions_dict[ticker]['DT_mBERT'] = (yt, yp)
        print('  [OK] Decision Tree Pure MAE:', round(m_dt_p['MAE'], 3), '| Sent MAE:', round(m_dt_s['MAE'], 3), '| mBERT MAE:', round(m_dt_m['MAE'], 3), 'DA:', round(m_dt_m['DA (%)'], 1))
        
        # 6-8. Random Forest
        m_rf_p, yt, yp = train_eval_tabular(df_c, FEAT_PURE, RandomForestRegressor, n_estimators=100, max_depth=6, random_state=42)
        m_rf_p['Model'] = '6. Random Forest Pure'; m_rf_p['Ticker'] = ticker; all_metrics_records.append(m_rf_p)
        predictions_dict[ticker]['RF_Pure'] = (yt, yp)
        
        m_rf_s, yt, yp = train_eval_tabular(df_c, FEAT_SENT, RandomForestRegressor, n_estimators=100, max_depth=6, random_state=42)
        m_rf_s['Model'] = '7. Random Forest + Sentiment'; m_rf_s['Ticker'] = ticker; all_metrics_records.append(m_rf_s)
        predictions_dict[ticker]['RF_Sent'] = (yt, yp)
        
        m_rf_m, yt, yp = train_eval_tabular(df_m, FEAT_MBERT, RandomForestRegressor, n_estimators=100, max_depth=6, random_state=42)
        m_rf_m['Model'] = '8. Random Forest + mBERT'; m_rf_m['Ticker'] = ticker; all_metrics_records.append(m_rf_m)
        predictions_dict[ticker]['RF_mBERT'] = (yt, yp)
        print('  [OK] Random Forest Pure MAE:', round(m_rf_p['MAE'], 3), '| Sent MAE:', round(m_rf_s['MAE'], 3), '| mBERT MAE:', round(m_rf_m['MAE'], 3), 'DA:', round(m_rf_m['DA (%)'], 1))
        
        # 9-11. XGBoost
        m_xgb_p, yt, yp = train_eval_tabular(df_c, FEAT_PURE, XGBRegressor, n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        m_xgb_p['Model'] = '9. XGBoost Pure'; m_xgb_p['Ticker'] = ticker; all_metrics_records.append(m_xgb_p)
        predictions_dict[ticker]['XGB_Pure'] = (yt, yp)
        
        m_xgb_s, yt, yp = train_eval_tabular(df_c, FEAT_SENT, XGBRegressor, n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        m_xgb_s['Model'] = '10. XGBoost + Sentiment'; m_xgb_s['Ticker'] = ticker; all_metrics_records.append(m_xgb_s)
        predictions_dict[ticker]['XGB_Sent'] = (yt, yp)
        
        m_xgb_m, yt, yp = train_eval_tabular(df_m, FEAT_MBERT, XGBRegressor, n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        m_xgb_m['Model'] = '11. XGBoost + mBERT'; m_xgb_m['Ticker'] = ticker; all_metrics_records.append(m_xgb_m)
        predictions_dict[ticker]['XGB_mBERT'] = (yt, yp)
        print('  [OK] XGBoost Pure MAE:', round(m_xgb_p['MAE'], 3), '| Sent MAE:', round(m_xgb_s['MAE'], 3), '| mBERT MAE:', round(m_xgb_m['MAE'], 3), 'DA:', round(m_xgb_m['DA (%)'], 1))
        
        # 12-14. BiLSTM
        m_lstm_p, yt, yp = train_eval_bilstm(df_c, FEAT_PURE, seq_length=10)
        m_lstm_p['Model'] = '12. BiLSTM Pure'; m_lstm_p['Ticker'] = ticker; all_metrics_records.append(m_lstm_p)
        predictions_dict[ticker]['BiLSTM_Pure'] = (yt, yp)
        
        m_lstm_s, yt, yp = train_eval_bilstm(df_c, FEAT_SENT, seq_length=10)
        m_lstm_s['Model'] = '13. BiLSTM + Sentiment'; m_lstm_s['Ticker'] = ticker; all_metrics_records.append(m_lstm_s)
        predictions_dict[ticker]['BiLSTM_Sent'] = (yt, yp)
        
        m_lstm_m, yt, yp = train_eval_bilstm(df_m, FEAT_MBERT, seq_length=10)
        m_lstm_m['Model'] = '14. BiLSTM + mBERT'; m_lstm_m['Ticker'] = ticker; all_metrics_records.append(m_lstm_m)
        predictions_dict[ticker]['BiLSTM_mBERT'] = (yt, yp)
        print('  [OK] BiLSTM Pure MAE:', round(m_lstm_p['MAE'], 3), '| Sent MAE:', round(m_lstm_s['MAE'], 3), '| mBERT MAE:', round(m_lstm_m['MAE'], 3), 'DA:', round(m_lstm_m['DA (%)'], 1))

    df_all_metrics = pd.DataFrame(all_metrics_records)
    cols_order = ['Ticker', 'Model', 'MAE', 'RMSE', 'MAPE (%)', 'R2', 'DA (%)']
    df_all_metrics = df_all_metrics[cols_order]
    
    out_master_csv = os.path.join(METRICS_DIR, 'master_clean_all_models_metrics.csv')
    df_all_metrics.to_csv(out_master_csv, index=False)
    print('\n[DONE] DA LUU BANG METRICS TONG HOP:', out_master_csv)
    
    print('\n>>> DANG THUC HIEN KIEM DINH WILCOXON SIGNED-RANK TEST ...')
    test_pairs = [
        ('BiLSTM_Pure', 'BiLSTM_mBERT', 'BiLSTM Pure vs BiLSTM + mBERT'),
        ('XGB_Pure', 'XGB_mBERT', 'XGBoost Pure vs XGBoost + mBERT'),
        ('RF_Pure', 'RF_mBERT', 'Random Forest Pure vs Random Forest + mBERT'),
        ('DT_Pure', 'DT_mBERT', 'Decision Tree Pure vs Decision Tree + mBERT'),
        ('Naive', 'BiLSTM_mBERT', 'Naive Persistence vs BiLSTM + mBERT')
    ]
    
    wilcoxon_records = []
    for m1_key, m2_key, label in test_pairs:
        all_err1, all_err2 = [], []
        for ticker in TICKERS:
            if m1_key in predictions_dict[ticker] and m2_key in predictions_dict[ticker]:
                yt1, yp1 = predictions_dict[ticker][m1_key]
                yt2, yp2 = predictions_dict[ticker][m2_key]
                min_l = min(len(yt1), len(yp1), len(yt2), len(yp2))
                err1 = np.abs(yt1[-min_l:] - yp1[-min_l:])
                err2 = np.abs(yt2[-min_l:] - yp2[-min_l:])
                all_err1.extend(err1)
                all_err2.extend(err2)
        
        all_err1 = np.array(all_err1)
        all_err2 = np.array(all_err2)
        
        mean_err1 = np.mean(all_err1)
        mean_err2 = np.mean(all_err2)
        imp = ((mean_err1 - mean_err2) / mean_err1) * 100.0
        
        diff = all_err1 - all_err2
        diff_non_zero = diff[diff != 0]
        if len(diff_non_zero) > 0:
            res = stats.wilcoxon(all_err1, all_err2, zero_method='pratt')
            w_stat, p_val = res.statistic, res.pvalue
        else:
            w_stat, p_val = 0, 1.0
            
        wilcoxon_records.append({
            'Comparison': label,
            'Mean Error A': round(mean_err1, 4),
            'Mean Error B': round(mean_err2, 4),
            'Improvement (%)': round(imp, 2),
            'W-Statistic': float(w_stat),
            'p-value': f'{p_val:.4e}',
            'Conclusion': 'Reject H0 (p < 0.05)' if p_val < 0.05 else 'Fail to reject H0'
        })
        
    df_wilcoxon = pd.DataFrame(wilcoxon_records)
    out_wilcoxon_csv = os.path.join(METRICS_DIR, 'wilcoxon_clean_results.csv')
    df_wilcoxon.to_csv(out_wilcoxon_csv, index=False)
    print('[DONE] DA LUU BANG WILCOXON TEST:', out_wilcoxon_csv)
    print('\n' + df_wilcoxon.to_string())
    
    print('\n' + '='*80)
    print('HOAN TAT THUC NGHIEM THANH CONG 100%!')
    print('='*80)

if __name__ == '__main__':
    main()
