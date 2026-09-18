"""
Pipeline chuẩn bị dữ liệu từ data_src/ → data/processed/
Tạo file _final_v4_clean.csv và _mbert_final.csv cho mỗi ticker.
"""
import pandas as pd
import numpy as np
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_SRC = os.path.join(BASE_DIR, "data_src")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping tên thầy → ticker
MAPPING = {
    "APPLE": "AAPL",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "META": "META",
    "ALIBABA": "BABA",
    "VNM": "VNM"
}

def adjust_weekends_backward(date):
    """Logic thầy Khang: Tin T7/CN dồn về Thứ 6 trước đó."""
    if date.weekday() == 5:  # Thứ 7 → lùi 1 ngày
        return date - pd.Timedelta(days=1)
    elif date.weekday() == 6:  # Chủ nhật → lùi 2 ngày
        return date - pd.Timedelta(days=2)
    return date


def process_sentiment(ticker_name, input_file):
    """Đọc và xử lý sentiment data, trả về 2 DataFrame: clean (1 cột) và mbert (8 cột)."""
    if not os.path.exists(input_file):
        print(f"  ! Không tìm thấy: {input_file}")
        return None, None

    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = df['Date'].apply(adjust_weekends_backward)

    # Đảm bảo các cột tồn tại
    for col in ['sentiment_score', 'impact_score', 'relevance_score', 'confidence', 'sentiment_momentum']:
        if col not in df.columns:
            df[col] = 0.0

    # Clean: chỉ 1 cột sentiment (mean daily)
    daily_clean = df.groupby('Date').agg(
        sentiment=('sentiment_score', 'mean')
    ).reset_index()

    # mBERT: 8 đặc trưng thống kê
    daily_mbert = df.groupby('Date').agg(
        avg_polarity=('sentiment_score', 'mean'),
        max_polarity=('sentiment_score', 'max'),
        min_polarity=('sentiment_score', 'min'),
        sum_polarity=('sentiment_score', 'sum'),
        count_mentions=('sentiment_score', 'count'),
        avg_intensity=('impact_score', 'mean'),
        avg_credibility=('confidence', 'mean'),
        avg_relevance=('relevance_score', 'mean')
    ).reset_index()

    return daily_clean, daily_mbert


def merge_price_sentiment(ticker, sentiment_df, price_file, output_file):
    """Merge giá cổ phiếu với sentiment, lọc từ 2023."""
    if sentiment_df is None or not os.path.exists(price_file):
        print(f"  ! Thiếu dữ liệu cho {ticker}")
        return

    df_price = pd.read_csv(price_file)

    # Chuẩn hóa thứ tự cột giá
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_price = df_price[price_cols]

    # Lọc từ 2023
    start_date = pd.to_datetime("2023-01-01").date()
    df_price = df_price[df_price['Date'] >= start_date].copy()

    # Left join
    final_df = pd.merge(df_price, sentiment_df, on='Date', how='left')

    # Điền 0 cho ngày không có tin
    sent_cols = [c for c in sentiment_df.columns if c != 'Date']
    final_df[sent_cols] = final_df[sent_cols].fillna(0)

    final_df.to_csv(output_file, index=False)
    print(f"  => Saved: {output_file} ({len(final_df)} rows, {final_df['Date'].min()} to {final_df['Date'].max()})")


if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE CHUẨN BỊ DỮ LIỆU CHO BASELINE MODELS")
    print("=" * 60)

    for thay_name, ticker in MAPPING.items():
        print(f"\n>>> Processing {ticker} ({thay_name})...")

        # Đường dẫn input
        sentiment_file = os.path.join(DATA_SRC, "data_thay", "sentiment", f"{thay_name}_sentiment.csv")
        price_file = os.path.join(DATA_SRC, "raw_clean", "prices", f"{ticker}_prices.csv")

        # Xử lý sentiment
        daily_clean, daily_mbert = process_sentiment(thay_name, sentiment_file)

        # Merge và lưu: file clean (1 cột sentiment)
        clean_out = os.path.join(OUTPUT_DIR, f"{ticker}_final_v4_clean.csv")
        merge_price_sentiment(ticker, daily_clean, price_file, clean_out)

        # Merge và lưu: file mBERT (8 cột)
        mbert_out = os.path.join(OUTPUT_DIR, f"{ticker}_mbert_final.csv")
        merge_price_sentiment(ticker, daily_mbert, price_file, mbert_out)

    print("\n" + "=" * 60)
    print("HOÀN TẤT PIPELINE DỮ LIỆU!")
    print("=" * 60)
