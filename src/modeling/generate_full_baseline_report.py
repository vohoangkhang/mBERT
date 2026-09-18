"""
Script tổng hợp toàn bộ kết quả thí nghiệm (6 Scenarios gốc + 5 Baselines mới: DT Pure, RF Pure, DT+mBERT, RF+mBERT, ARIMA).
Xuất ra file CSV tổng hợp và tạo biểu đồ so sánh cho báo cáo phản biện.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r'd:\HUTECH\thayKhang\stock-nlp-prediction'
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations_v3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures_baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "BABA", "GOOGL", "META", "VNM"]

# Cấu hình font chuyên nghiệp
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

all_dfs = []
for t in TICKERS:
    f = os.path.join(VIZ_DIR, f"{t}_all_metrics.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        if 'Ticker' not in df.columns:
            df['Ticker'] = t
        all_dfs.append(df)

if all_dfs:
    full_df = pd.concat(all_dfs, ignore_index=True)
    summary_csv = os.path.join(VIZ_DIR, "master_all_models_metrics.csv")
    full_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved master summary CSV: {summary_csv}")

    # 1. Bảng tổng hợp MAE trung bình theo Model/Scenario trên toàn bộ 6 cổ phiếu
    # Chuẩn hóa tên scenario ngắn gọn
    def clean_scenario_name(s):
        s = str(s)
        if 'ARIMA' in s: return 'ARIMA'
        if 'Decision Tree Pure' in s: return 'DT Pure'
        if 'Random Forest Pure' in s: return 'RF Pure'
        if 'Decision Tree + mBERT' in s: return 'DT + mBERT'
        if 'Random Forest + mBERT' in s: return 'RF + mBERT'
        if 'XGB Pure' in s: return 'XGB Pure'
        if 'BiLSTM Pure' in s: return 'BiLSTM Pure'
        if 'XGB + Sentiment' in s: return 'XGB + Sent'
        if 'BiLSTM + Sentiment' in s: return 'BiLSTM + Sent'
        if 'XGB + mBERT' in s: return 'XGB + mBERT'
        if 'BiLSTM + mBERT' in s: return 'BiLSTM + mBERT'
        return s

    full_df['Short_Scenario'] = full_df['Scenario'].apply(clean_scenario_name)

    # Pivot table MAE & R2
    pivot_mae = full_df.pivot(index='Short_Scenario', columns='Ticker', values='MAE')
    pivot_r2 = full_df.pivot(index='Short_Scenario', columns='Ticker', values='R2')
    pivot_mape = full_df.pivot(index='Short_Scenario', columns='Ticker', values='MAPE')

    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP MAE THEO MÃ CỔ PHIẾU VÀ SCENARIO")
    print("="*80)
    print(pivot_mae.to_string())

    # Vẽ biểu đồ so sánh MAE cho 6 mã cổ phiếu (Grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 7))
    scenarios_order = [
        'ARIMA', 'DT Pure', 'RF Pure', 'XGB Pure', 'BiLSTM Pure',
        'DT + mBERT', 'RF + mBERT', 'XGB + mBERT', 'BiLSTM + mBERT'
    ]
    
    # Filter scenarios exist in pivot
    present_scenarios = [s for s in scenarios_order if s in pivot_mae.index]
    pivot_mae_sub = pivot_mae.loc[present_scenarios]

    pivot_mae_sub.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xlabel('Model / Scenario')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_models_mae_comparison.png"))
    plt.close()
    print("✓ Saved chart: all_models_mae_comparison.png")
