import pandas as pd
import numpy as np
import os

def adjust_weekends_backward(date):
    """Logic thầy Khang: Tin T7, CN đẩy về Thứ 6 trước đó."""
    # 5 là Thứ 7, 6 là Chủ nhật
    if date.weekday() == 5: # Thứ 7 -> chuyển về Thứ 6 (-1 ngày)
        return date - pd.Timedelta(days=1)
    elif date.weekday() == 6: # Chủ nhật -> chuyển về Thứ 6 (-2 ngày)
        return date - pd.Timedelta(days=2)
    return date

def robust_process_sentiment(ticker, input_file):
    print(f">>> Đang xử lý tin tức cho {ticker} (Logic: News T7,CN -> Fri)...")
    if not os.path.exists(input_file):
        print(f"  ! Không tìm thấy file: {input_file}")
        return None

    try:
        df = pd.read_csv(input_file)
        
        # CHUẨN HÓA NGÀY THÁNG
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
        except:
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # ĐIỀU CHỈNH ĐẨY LÙI NGÀY CUỐI TUẦN (Theo ý thầy Khang)
        df['Date'] = df['Date'].apply(adjust_weekends_backward)
        
        # KIỂM TRA TÊN CỘT (Đảm bảo có đủ cột cần thiết)
        col_map = {
            'sentiment_score': 'sentiment_score',
            'impact_score': 'impact_score',
            'relevance_score': 'relevance_score',
            'confidence': 'confidence',
            'sentiment_momentum': 'sentiment_momentum'
        }
        for col in col_map.keys():
            if col not in df.columns:
                df[col] = 0.0 # Điền 0 nếu thiếu cột
        
        # GOM NHÓM VÀ TÍNH TOÁN 8 ĐẶC TRƯNG THỰC TẾ
        daily = df.groupby('Date').agg(
            avg_polarity=('sentiment_score', 'mean'),
            max_polarity=('sentiment_score', 'max'),
            min_polarity=('sentiment_score', 'min'),
            sum_polarity=('sentiment_score', 'sum'),
            count_mentions=('sentiment_score', 'count'),
            avg_intensity=('impact_score', 'mean'),
            avg_credibility=('confidence', 'mean'),
            avg_relevance=('relevance_score', 'mean')
        ).reset_index()
        
        return daily
    except Exception as e:
        print(f"  ! Lỗi xử lý {ticker}: {e}")
        return None

def merge_and_finalize(ticker, sentiment_df, price_file, output_file):
    if sentiment_df is None or not os.path.exists(price_file):
        return
    
    print(f">>> Gộp dữ liệu Giá và Tin tức cho {ticker}...")
    df_price = pd.read_csv(price_file)
    
    # Chuẩn hóa cột ngày của Giá
    date_col = 'Date' if 'Date' in df_price.columns else 'time'
    df_price['Date'] = pd.to_datetime(df_price[date_col]).dt.date
    
    # LỌC DỮ LIỆU TỪ 2023 (Thời điểm tin tức bắt đầu dồi dào)
    start_date = pd.to_datetime("2023-01-01").date()
    df_price = df_price[df_price['Date'] >= start_date]
    
    # LEFT JOIN (Giữ lại tất cả ngày giao dịch)
    final_df = pd.merge(df_price, sentiment_df, on='Date', how='left')
    
    # ĐIỀN 0 CHO NGÀY KHÔNG CÓ TIN TỨC
    sent_cols = [c for c in sentiment_df.columns if c != 'Date']
    final_df[sent_cols] = final_df[sent_cols].fillna(0)
    
    # Lưu file
    final_df.to_csv(output_file, index=False)
    print(f"  => Đã lưu: {output_file} ({len(final_df)} dòng, dải ngày: {final_df['Date'].min()} đến {final_df['Date'].max()})")

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    data_dir = os.path.join(base_dir, "data")
    
    mapping = {
        "APPLE": "AAPL",
        "AMAZON": "AMZN",
        "GOOGLE": "GOOGL",
        "META": "META",
        "ALIBABA": "BABA",
        "VNM": "VNM"
    }
    
    for thay_name, ticker in mapping.items():
        s_in = os.path.join(data_dir, "data_thay", f"{thay_name}_sentiment.csv")
        p_in = os.path.join(data_dir, "raw", f"{ticker}_prices.csv")
        f_out = os.path.join(data_dir, "processed", f"{ticker}_final_dataset_v2.csv")
        
        daily_s = robust_process_sentiment(ticker, s_in)
        merge_and_finalize(ticker, daily_s, p_in, f_out)
