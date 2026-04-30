import pandas as pd
import numpy as np
import os

def adjust_weekends(date):
    # 5 là Thứ 7, 6 là Chủ nhật
    if date.weekday() == 5: # Thứ 7 -> chuyển sang Thứ 2 (+2 ngày)
        return date + pd.Timedelta(days=2)
    elif date.weekday() == 6: # Chủ nhật -> chuyển sang Thứ 2 (+1 ngày)
        return date + pd.Timedelta(days=1)
    return date

def process_thay_sentiment(ticker_name, input_file, output_file):
    print(f">>> Đang gộp và tính toán đặc trưng (Max, Min, Avg, Sum, Count) cho {ticker_name}...")
    if not os.path.exists(input_file): return None

    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # ĐIỀU CHỈNH NGÀY CUỐI TUẦN
        df['Date'] = df['Date'].apply(adjust_weekends)
        
        # Gom nhóm và tính toán 5 hàm gộp chính theo chỉ dẫn
        # (Giữ nguyên logic của thầy nhưng gộp theo ngày đã điều chỉnh)
        daily = df.groupby('Date').agg({
            'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
            'impact_score': ['mean', 'max', 'sum'],
            'confidence': 'mean',
            'relevance_score': 'mean',
            'sentiment_momentum': 'sum'
        })
        
        # Làm phẳng MultiIndex columns
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index()
        
        # Đổi tên cho khớp với logic 8+ đặc trưng
        daily = daily.rename(columns={
            'sentiment_score_mean': 'avg_polarity',
            'sentiment_score_max': 'max_polarity',
            'sentiment_score_min': 'min_polarity',
            'sentiment_score_sum': 'sum_polarity',
            'sentiment_score_count': 'count_mentions',
            'impact_score_mean': 'avg_intensity',
            'confidence_mean': 'avg_credibility',
            'relevance_score_mean': 'avg_relevance'
        })
        
        daily.to_csv(output_file, index=False)
        return daily
    except Exception as e:
        print(f"  ! Lỗi: {e}")
        return None

def merge_with_prices(ticker, sentiment_df, price_file, final_output):
    print(f">>> Thực hiện LEFT JOIN (Prices + Sentiment) cho {ticker}...")
    if not os.path.exists(price_file): return

    df_price = pd.read_csv(price_file)
    date_col = 'Date' if 'Date' in df_price.columns else 'time' if 'time' in df_price.columns else df_price.index.name
    df_price['Date'] = pd.to_datetime(df_price[date_col]).dt.date
    
    # LỌC DỮ LIỆU TỪ NĂM 2023 ĐỂ ĐẢM BẢO SỰ TRÙNG KHỚP VỚI TIN TỨC
    start_date = pd.to_datetime("2023-01-01").date()
    df_price = df_price[df_price['Date'] >= start_date]
    
    # THỰC HIỆN LEFT JOIN
    final_df = pd.merge(df_price, sentiment_df, on='Date', how='left')
    
    # ĐIỀN 0 CHO NGÀY KHÔNG CÓ TIN TỨC
    sent_cols = [c for c in sentiment_df.columns if c != 'Date']
    final_df[sent_cols] = final_df[sent_cols].fillna(0)
    
    final_df.to_csv(final_output, index=False)
    print(f"  => Đã lưu bộ dữ liệu gộp tại {final_output} (Dải ngày: {final_df['Date'].min()} đến {final_df['Date'].max()})")

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    mapping = {"APPLE": "AAPL", "AMAZON": "AMZN", "GOOGLE": "GOOGL", "VNM": "VNM", "BABA": "BABA", "META": "META"}
    
    for thay_name, ticker in mapping.items():
        # 1. Gộp dữ liệu "Thay" truyền thống
        s_in = os.path.join(base_dir, "data", "data_thay", f"{thay_name}_sentiment.csv")
        s_out = os.path.join(base_dir, "data", "processed", f"{ticker}_features_expanded.csv")
        p_in = os.path.join(base_dir, "data", "raw", f"{ticker}_prices.csv")
        f_out = os.path.join(base_dir, "data", "processed", f"{ticker}_final_expanded.csv")
        
        daily_s = process_thay_sentiment(ticker, s_in, s_out)
        if daily_s is not None:
            merge_with_prices(ticker, daily_s, p_in, f_out)

        # 2. Gộp dữ liệu mBERT mới (nếu file đã được tạo từ analyze_multi_models.py)
        mbert_in = os.path.join(base_dir, "data", "processed", f"{ticker}_8_features_mbert_test.csv")
        mbert_out = os.path.join(base_dir, "data", "processed", f"{ticker}_final_8_features_mbert.csv")
        if os.path.exists(mbert_in):
            print(f">>> Đang xử lý mBERT đặc trưng cho {ticker}...")
            df_mbert = pd.read_csv(mbert_in)
            df_mbert['Date'] = pd.to_datetime(df_mbert['Date']).dt.date
            # Điều chỉnh ngày cuối tuần cho mBERT luôn
            df_mbert['Date'] = df_mbert['Date'].apply(adjust_weekends)
            merge_with_prices(ticker, df_mbert, p_in, mbert_out)
