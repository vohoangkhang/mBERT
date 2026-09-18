"""
Train ARIMA baseline (univariate, chỉ dùng Close price).
ARIMA là statistical baseline kinh điển để so sánh với ML models.
Chạy SAU khi DT + RF đã xong.
"""
import pandas as pd
import numpy as np
import os
import sys, io
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations_v3")
FIG_DIR = os.path.join(BASE_DIR, "figures_baseline")
os.makedirs(FIG_DIR, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "BABA", "GOOGL", "META", "VNM"]


def calculate_metrics(y_true, y_pred):
    """Tính 4 metrics giống master_7_report.py."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_arima(ticker, df):
    """
    Train ARIMA trên chuỗi Close price.
    - Split 80/20 tuần tự (giống các model khác)
    - Dùng auto_arima (pmdarima) hoặc ARIMA(5,1,0) mặc định
    - Walk-forward prediction trên test set
    """
    close = df['Close'].values.astype(float)

    train_size = int(len(close) * 0.8)
    train = close[:train_size]
    test = close[train_size:]

    print(f"  Train size: {len(train)}, Test size: {len(test)}")

    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Fit ARIMA trên training data
        # Sử dụng ARIMA(5,1,0) - phổ biến cho chuỗi giá cổ phiếu
        # order (p,d,q): p=5 (lag), d=1 (differencing), q=0 (MA)
        predictions = []
        history = list(train)

        print(f"  Đang chạy ARIMA walk-forward ({len(test)} bước)...")

        for t in range(len(test)):
            try:
                model = ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                yhat = model_fit.forecast(steps=1)[0]
                predictions.append(yhat)
                history.append(test[t])

                if (t + 1) % 50 == 0:
                    print(f"    Step {t+1}/{len(test)}")
            except Exception:
                # Fallback: dùng giá trước đó
                predictions.append(history[-1])
                history.append(test[t])

        predictions = np.array(predictions)
        return test, predictions

    except ImportError:
        print("  ! statsmodels chưa được cài. Đang cài...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'statsmodels', '-q'])
        # Recursive call sau khi cài
        return train_arima(ticker, df)


def try_auto_arima(ticker, df):
    """
    Thử dùng auto_arima để tìm order tối ưu.
    Nếu pmdarima không có, fallback về ARIMA(5,1,0).
    """
    close = df['Close'].values.astype(float)
    train_size = int(len(close) * 0.8)
    train = close[:train_size]
    test = close[train_size:]

    try:
        from pmdarima import auto_arima

        print(f"  Tìm ARIMA order tối ưu bằng auto_arima...")
        auto_model = auto_arima(
            train, start_p=1, start_q=0, max_p=7, max_q=3,
            d=1, seasonal=False, trace=False, stepwise=True,
            suppress_warnings=True, error_action='ignore'
        )
        best_order = auto_model.order
        print(f"  Best ARIMA order: {best_order}")

        # Walk-forward với best order
        from statsmodels.tsa.arima.model import ARIMA
        predictions = []
        history = list(train)

        for t in range(len(test)):
            try:
                model = ARIMA(history, order=best_order)
                model_fit = model.fit()
                yhat = model_fit.forecast(steps=1)[0]
                predictions.append(yhat)
            except Exception:
                predictions.append(history[-1])
            history.append(test[t])

            if (t + 1) % 50 == 0:
                print(f"    Step {t+1}/{len(test)}")

        return test, np.array(predictions), best_order

    except ImportError:
        print("  pmdarima chưa cài, dùng ARIMA(5,1,0) mặc định")
        y_true, y_pred = train_arima(ticker, df)
        return y_true, y_pred, (5, 1, 0)


def run_arima():
    """Chạy ARIMA cho tất cả tickers."""
    all_results = []

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f">>> ARIMA FOR {ticker}")
        print(f"{'='*50}")

        f_clean = os.path.join(DATA_DIR, f"{ticker}_final_v4_clean.csv")
        if not os.path.exists(f_clean):
            print(f"  ! Missing {f_clean}, skipping...")
            continue

        df = pd.read_csv(f_clean)
        print(f"  Data: {len(df)} rows")

        # Train ARIMA
        y_true, y_pred, order = try_auto_arima(ticker, df)

        if y_true is not None and y_pred is not None:
            metrics = calculate_metrics(y_true, y_pred)
            metrics["Scenario"] = f"12. ARIMA{order}"
            metrics["Ticker"] = ticker
            all_results.append(metrics)

            print(f"  ARIMA{order}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                  f"R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")

            # Vẽ biểu đồ
            plt.figure(figsize=(12, 5))
            plt.plot(y_true, label='Actual', color='black', linewidth=1.5)
            plt.plot(y_pred, label=f'ARIMA{order}', color='#3498db', linewidth=1.2, linestyle='--')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.xlabel('Trading Days (Test Set)')
            plt.ylabel('Close Price')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"{ticker}_arima_prediction.png"), dpi=300)
            plt.close()
            print(f"  => Saved: {ticker}_arima_prediction.png")

            # Cập nhật bảng metrics tổng hợp
            arima_df = pd.DataFrame([metrics])
            arima_file = os.path.join(VIZ_DIR, f"{ticker}_arima_metrics.csv")
            arima_df.to_csv(arima_file, index=False)

            # Gộp vào all_metrics
            all_file = os.path.join(VIZ_DIR, f"{ticker}_all_metrics.csv")
            if os.path.exists(all_file):
                df_all = pd.read_csv(all_file)
                # Bỏ ARIMA cũ nếu có
                df_all = df_all[~df_all['Scenario'].str.contains('ARIMA', na=False)]
                df_all = pd.concat([df_all, arima_df], ignore_index=True)
            else:
                df_all = arima_df
            df_all.to_csv(all_file, index=False)

    # Lưu tổng hợp ARIMA
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(os.path.join(VIZ_DIR, "arima_summary_all_tickers.csv"), index=False)
        print("\n" + "=" * 60)
        print("BẢNG KẾT QUẢ ARIMA")
        print("=" * 60)
        print(df_summary.to_string(index=False))

    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING ARIMA BASELINE")
    print("=" * 60)
    run_arima()
    print("\n>>> HOÀN TẤT TRAINING ARIMA!")
