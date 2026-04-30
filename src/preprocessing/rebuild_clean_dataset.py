import pandas as pd
import numpy as np
import os

def apply_backward_logic(df):
    """Tin T7, CN dồn về Thứ 6 trước đó theo ý thầy Khang."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Adjusted'] = df['Date'].copy()
    mask_sat = df['Date'].dt.weekday == 5
    mask_sun = df['Date'].dt.weekday == 6
    df.loc[mask_sat, 'Date_Adjusted'] = df.loc[mask_sat, 'Date'] - pd.Timedelta(days=1)
    df.loc[mask_sun, 'Date_Adjusted'] = df.loc[mask_sun, 'Date'] - pd.Timedelta(days=2)
    return df

def aggregate_sentiment(df):
    daily = df.groupby(df['Date_Adjusted'].dt.date).agg({
        'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
        'impact_score': ['mean', 'max'],
        'confidence': 'mean'
    })
    daily.columns = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'max_intensity', 'avg_credibility']
    return daily.reset_index().rename(columns={'Date_Adjusted': 'Date'})

def rebuild_clean_v6(ticker, price_file, sentiment_file, output_file):
    print(f"\n>>> Đang xử lý V6 (Giá RAW + Sentiment Thay) cho mã: {ticker}...")
    
    if not os.path.exists(price_file) or not os.path.exists(sentiment_file):
        print(f"  ! Thiếu file cho {ticker}, bỏ qua.")
        return

    # 1. Xử lý Giá từ RAW
    df_price = pd.read_csv(price_file)
    df_price.columns = [c.lower() for c in df_price.columns]
    date_col = [c for c in df_price.columns if c in ['date', 'time', 'timestamp']][0]
    df_price['Date'] = pd.to_datetime(df_price[date_col]).dt.date
    
    # Mapping chuẩn hóa
    cols_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    for old, new in cols_map.items():
        if old in df_price.columns: df_price = df_price.rename(columns={old: new})
    
    df_price = df_price[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_price = df_price[df_price['Close'] > 0] # Đảm bảo không có giá 0

    # 2. Xử lý Sentiment từ THAY
    df_sent = pd.read_csv(sentiment_file)
    df_sent = apply_backward_logic(df_sent)
    df_daily_sent = aggregate_sentiment(df_sent)
    df_daily_sent['Date'] = pd.to_datetime(df_daily_sent['Date']).dt.date

    # 3. Merge Left Join
    final_df = pd.merge(df_price, df_daily_sent, on='Date', how='left')
    
    # Điền 0 cho các ngày không có tin tức
    sent_cols = [c for c in df_daily_sent.columns if c != 'Date']
    final_df[sent_cols] = final_df[sent_cols].fillna(0)
    
    # Lọc lấy từ năm 2021 để đồng bộ
    final_df = final_df[final_df['Date'] >= pd.to_datetime("2021-01-01").date()]

    final_df.to_csv(output_file, index=False)
    print(f"  => Thành công! Đã tạo: {output_file} ({len(final_df)} dòng)")

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw")
    src_dir = os.path.join("data", "data_thay")
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    mapping = {"APPLE": "AAPL", "AMAZON": "AMZN", "GOOGLE": "GOOGL", "ALIBABA": "BABA", "META": "META", "VNM": "VNM"}
    
    for thay_name, ticker in mapping.items():
        p_file = os.path.join(raw_dir, f"{ticker}_prices.csv")
        s_file = os.path.join(src_dir, f"{thay_name}_sentiment.csv")
        o_file = os.path.join(out_dir, f"{ticker}_final_v4_clean.csv")
        
        rebuild_clean_v6(ticker, p_file, s_file, o_file)
    
    print("\n>>> QUÁ TRÌNH LÀM SẠCH DỮ LIỆU V6 HOÀN TẤT!")
