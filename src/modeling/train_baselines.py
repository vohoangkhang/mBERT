"""
Train baseline models: Decision Tree + Random Forest
Cả 2 chế độ: Pure (chỉ giá) và mBERT (giá + 8 đặc trưng sentiment).
Cấu trúc giống hệt XGBoost trong master_7_report.py để kết quả so sánh công bằng.
"""
import pandas as pd
import numpy as np
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations_v3")
FIG_DIR = os.path.join(BASE_DIR, "figures_baseline")
os.makedirs(FIG_DIR, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "BABA", "GOOGL", "META", "VNM"]

FEAT_PURE = ['Open', 'High', 'Low', 'Volume']
FEAT_MBERT = FEAT_PURE + [
    'avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance',
    'count_mentions', 'sum_polarity', 'max_polarity', 'min_polarity'
]


def calculate_metrics(y_true, y_pred):
    """Tính 4 metrics giống master_7_report.py."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_tabular_model(df, features, model_class, **model_kwargs):
    """
    Train model dạng tabular (giống XGBoost trong master_7_report.py).
    - Split 80/20 tuần tự
    - MinMaxScaler
    - Inverse transform để trả về giá thực
    """
    target = 'Close'
    data_cols = features + [target]
    df_clean = df[data_cols].dropna()

    if len(df_clean) < 20:
        return None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse scale
    dummy_test = np.zeros((len(y_test), len(data_cols)))
    dummy_test[:, -1] = y_test
    y_test_inv = scaler.inverse_transform(dummy_test)[:, -1]

    dummy_pred = np.zeros((len(y_pred), len(data_cols)))
    dummy_pred[:, -1] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]

    return y_test_inv, y_pred_inv


def run_baselines():
    """Chạy toàn bộ baseline cho tất cả tickers."""
    all_results = []

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f">>> TRAINING BASELINES FOR {ticker}")
        print(f"{'='*50}")

        f_clean = os.path.join(DATA_DIR, f"{ticker}_final_v4_clean.csv")
        f_mbert = os.path.join(DATA_DIR, f"{ticker}_mbert_final.csv")

        if not os.path.exists(f_clean) or not os.path.exists(f_mbert):
            print(f"  ! Missing data files for {ticker}, skipping...")
            continue

        df_c = pd.read_csv(f_clean)
        df_m = pd.read_csv(f_mbert)

        # Định nghĩa 4 baseline scenarios
        scenarios = {
            "8. Decision Tree Pure": {
                "df": df_c, "features": FEAT_PURE,
                "model_class": DecisionTreeRegressor,
                "kwargs": {"max_depth": 10, "random_state": 42}
            },
            "9. Random Forest Pure": {
                "df": df_c, "features": FEAT_PURE,
                "model_class": RandomForestRegressor,
                "kwargs": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
            },
            "10. Decision Tree + mBERT": {
                "df": df_m, "features": FEAT_MBERT,
                "model_class": DecisionTreeRegressor,
                "kwargs": {"max_depth": 10, "random_state": 42}
            },
            "11. Random Forest + mBERT": {
                "df": df_m, "features": FEAT_MBERT,
                "model_class": RandomForestRegressor,
                "kwargs": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
            },
        }

        ticker_metrics = []
        predictions = {}

        for name, cfg in scenarios.items():
            print(f"  Training {name}...")
            y_true, y_pred = train_tabular_model(
                cfg["df"], cfg["features"], cfg["model_class"], **cfg["kwargs"]
            )

            if y_true is not None:
                metrics = calculate_metrics(y_true, y_pred)
                metrics["Scenario"] = name
                metrics["Ticker"] = ticker
                ticker_metrics.append(metrics)
                predictions[name] = (y_true, y_pred)
                print(f"    MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                      f"R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")
            else:
                print(f"    ! Failed (insufficient data)")

        # Lưu metrics baseline
        if ticker_metrics:
            df_baseline = pd.DataFrame(ticker_metrics)
            df_baseline.to_csv(os.path.join(VIZ_DIR, f"{ticker}_baseline_metrics.csv"), index=False)

            # Gộp với metrics cũ (nếu có)
            old_metrics_file = os.path.join(VIZ_DIR, f"{ticker}_metrics_table.csv")
            if os.path.exists(old_metrics_file):
                df_old = pd.read_csv(old_metrics_file)
                if 'Ticker' not in df_old.columns:
                    df_old['Ticker'] = ticker
                df_all = pd.concat([df_old, df_baseline], ignore_index=True)
            else:
                df_all = df_baseline
            df_all.to_csv(os.path.join(VIZ_DIR, f"{ticker}_all_metrics.csv"), index=False)

            all_results.extend(ticker_metrics)

        # Vẽ biểu đồ so sánh predictions
        if predictions:
            plot_predictions(ticker, predictions)

    # Lưu tổng hợp toàn bộ
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(os.path.join(VIZ_DIR, "baseline_summary_all_tickers.csv"), index=False)
        print(f"\n>>> Saved summary: baseline_summary_all_tickers.csv")
        print("\n" + "=" * 60)
        print("BẢNG KẾT QUẢ TỔNG HỢP BASELINES")
        print("=" * 60)
        print(df_summary.to_string(index=False))

    return all_results


def plot_predictions(ticker, predictions):
    """Vẽ biểu đồ so sánh predictions cho 4 baselines."""
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300})

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Baseline Model Predictions - {ticker}', fontsize=14, fontweight='bold')

    colors = {'Decision Tree': '#e74c3c', 'Random Forest': '#27ae60'}

    for idx, (name, (y_true, y_pred)) in enumerate(predictions.items()):
        ax = axes[idx // 2][idx % 2]
        ax.plot(y_true, label='Actual', color='black', linewidth=1.5)

        # Lấy màu theo loại model
        model_type = 'Decision Tree' if 'Decision Tree' in name else 'Random Forest'
        ax.plot(y_pred, label='Predicted', color=colors[model_type],
                linewidth=1.2, linestyle='--', alpha=0.8)

        short_name = name.split('. ')[1] if '. ' in name else name
        ax.set_title(short_name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{ticker}_baseline_predictions.png"))
    plt.close()
    print(f"  => Saved plot: {ticker}_baseline_predictions.png")


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING BASELINE MODELS: Decision Tree + Random Forest")
    print("=" * 60)
    run_baselines()
    print("\n>>> HOÀN TẤT TRAINING BASELINES!")
