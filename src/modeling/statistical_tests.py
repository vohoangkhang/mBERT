
import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

def run_tests():
    all_files = glob.glob('visualizations_v3/*_metrics_table.csv')
    
    data_list = []
    for f in all_files:
        stock_name = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f)
        df['Stock'] = stock_name
        data_list.append(df)
        
    full_df = pd.concat(data_list)
    
    # Danh sách các Scenarios quan trọng
    scenarios = {
        'XGB_Pure': '1. XGB Pure',
        'BiLSTM_Pure': '2. BiLSTM Pure',
        'XGB_Sentiment': '4. XGB + Sentiment',
        'BiLSTM_Sentiment': '5. BiLSTM + Sentiment',
        'XGB_mBERT': '6. XGB + mBERT',
        'BiLSTM_mBERT': '7. BiLSTM + mBERT'
    }
    
    results = []
    
    def perform_comparison(group_a_name, group_b_name, metric='MAE'):
        a = full_df[full_df['Scenario'] == scenarios[group_a_name]][metric].values
        b = full_df[full_df['Scenario'] == scenarios[group_b_name]][metric].values
        
        if len(a) == 0 or len(b) == 0:
            return
            
        # T-test (Paired) - vì cùng dự báo trên cùng tập dữ liệu (6 mã cổ phiếu)
        t_stat, t_p = stats.ttest_rel(a, b)
        
        # Wilcoxon Signed-Rank Test (Non-parametric)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a, b)
        
        results.append({
            'Comparison': f"{group_a_name} vs {group_b_name}",
            'Metric': metric,
            f'Mean_{group_a_name}': np.mean(a),
            f'Mean_{group_b_name}': np.mean(b),
            'T-test p-value': t_p,
            'Wilcoxon p-value': wilcoxon_p,
            'Significant (0.05)': 'Yes' if t_p < 0.05 or wilcoxon_p < 0.05 else 'No'
        })

    # So sánh Pure vs mBERT
    perform_comparison('XGB_Pure', 'XGB_mBERT', 'MAE')
    perform_comparison('BiLSTM_Pure', 'BiLSTM_mBERT', 'MAE')
    
    # So sánh Sentiment thường vs mBERT
    perform_comparison('XGB_Sentiment', 'XGB_mBERT', 'MAE')
    perform_comparison('BiLSTM_Sentiment', 'BiLSTM_mBERT', 'MAE')
    
    # So sánh RMSE
    perform_comparison('XGB_Pure', 'XGB_mBERT', 'RMSE')
    perform_comparison('BiLSTM_Pure', 'BiLSTM_mBERT', 'RMSE')

    report_df = pd.DataFrame(results)
    print("\n--- KẾT QUẢ KIỂM ĐỊNH THỐNG KÊ (MAE & RMSE) ---")
    print(report_df.to_string(index=False))
    
    report_df.to_csv('statistical_test_results.csv', index=False)
    print("\nKết quả đã được lưu vào file 'statistical_test_results.csv'")

if __name__ == "__main__":
    run_tests()
