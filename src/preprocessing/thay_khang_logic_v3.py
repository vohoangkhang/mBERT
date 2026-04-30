import pandas as pd
import numpy as np
import os

def adjust_backward(date):
    """Tin T7, CN dồn về Thứ 6 trước đó theo ý thầy Khang."""
    # weekday() : Monday=0, Tuesday=1, ..., Saturday=5, Sunday=6
    if date.weekday() == 5: # Thứ 7 -> lùi về Thứ 6 (-1 ngày)
        return date - pd.Timedelta(days=1)
    elif date.weekday() == 6: # Chủ nhật -> lùi về Thứ 6 (-2 ngày)
        return date - pd.Timedelta(days=2)
    return date

def process_robust_sentiment(ticker_name, input_file):
    print(f">>> Đang xử lý Sentiment (Backward Logic) cho {ticker_name}...")
    if not os.path.exists(input_file): return None

    try:
        df = pd.read_csv(input_file)
        # Chuyển về định dạng datetime và dồn ngược ngày
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Date_Adjusted'] = df['Date'].apply(adjust_backward)
        
        # GOM NHÓM THEO NGÀY ĐÃ ĐIỀU CHỈNH
        daily = df.groupby('Date_Adjusted').agg({
            'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
            'impact_score': ['mean', 'max', 'sum'],
            'confidence': 'mean',
            'relevance_score': 'mean',
            'sentiment_momentum': 'sum'
        })
        
        # Làm phẳng MultiIndex columns
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index().rename(columns={'Date_Adjusted': 'Date'})
        
        # Áp dụng tên cột chuẩn cho 8 đặc trưng
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
        return daily
    except Exception as e:
        print(f"  ! Lỗi xử lý {ticker_name}: {e}")
        return None

def merge_final(ticker, sentiment_df, price_file, output_file):
    if sentiment_df is None or not os.path.exists(price_file): return
    
    df_price = pd.read_csv(price_file)
    date_col = 'Date' if 'Date' in df_price.columns else 'time'
    df_price['Date'] = pd.to_datetime(df_price[date_col]).dt.date
    
    # LỌC DỮ LIỆU TỪ 2023 ĐỂ TRÙNG KHỚP
    start_date = pd.to_datetime("2023-01-01").date()
    df_price = df_price[df_price['Date'] >= start_date]
    
    # Merge LEFT JOIN
    final_df = pd.merge(df_price, sentiment_df, on='Date', how='left')
    
    # ĐIỀN 0 CHO NGÀY KHÔNG CÓ TIN
    sent_cols = [c for c in sentiment_df.columns if c != 'Date']
    final_df[sent_cols] = final_df[sent_cols].fillna(0)
    
    final_df.to_csv(output_file, index=False)
    print(f"  => Đã lưu: {output_file} ({len(final_df)} dòng)")

if __name__ == "__main__":
    # SỬ DỤNG ĐƯỜNG DẪN TƯƠNG ĐỐI
    data_dir = "data"
    mapping = {"APPLE": "AAPL", "AMAZON": "AMZN", "GOOGLE": "GOOGL", "VNM": "VNM", "ALIBABA": "BABA", "META": "META"}
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    
    for thay_name, ticker in mapping.items():
        # 1. Xử lý dữ liệu THAY
        s_in = os.path.join(data_dir, "data_thay", f"{thay_name}_sentiment.csv")
        p_in = os.path.join(data_dir, "raw", f"{ticker}_prices.csv")
        f_out = os.path.join(data_dir, "processed", f"{ticker}_robust_thay.csv")
        
        daily_thay = process_robust_sentiment(thay_name, s_in)
        merge_final(ticker, daily_thay, p_in, f_out)
        
        # 2. Xử lý dữ liệu mBERT
        mbert_in = os.path.join(data_dir, "processed", f"{ticker}_news_mbert.csv")
        if not os.path.exists(mbert_in):
            mbert_in = s_in 
            
        daily_mbert = process_robust_sentiment(f"{ticker}_mBERT", mbert_in)
        f_out_mbert = os.path.join(data_dir, "processed", f"{ticker}_robust_mbert.csv")
        merge_final(ticker, daily_mbert, p_in, f_out_mbert)
