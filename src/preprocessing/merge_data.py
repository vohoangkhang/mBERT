import pandas as pd
import os

def merge_stock_news(price_file, news_file, output_path):
    print(f"  - Gộp {price_file} và {news_file}...")
    
    # 1. Đọc dữ liệu giá
    try:
        df_price = pd.read_csv(price_file, header=[0, 1], index_col=0)
        df_price = df_price.reset_index()
        df_price.columns = ['Date'] + [col[0] for col in df_price.columns[1:]]
    except:
        df_price = pd.read_csv(price_file)
        if 'Date' not in df_price.columns and 'date' in df_price.columns:
            df_price = df_price.rename(columns={'date': 'Date'})
            
    # Chuẩn hóa Date cho df_price
    df_price['Date'] = pd.to_datetime(df_price['Date'], utc=True).dt.date
    
    # 2. Đọc dữ liệu tin tức sentiment
    if not os.path.exists(news_file):
        print(f"    ! Cảnh báo: Không tìm thấy file tin tức {news_file}. Điền sentiment = 0.")
        df_price['Avg_Sentiment'] = 0.0
        df_price.to_csv(output_path, index=False)
        return
        
    df_news = pd.read_csv(news_file)
    # Chuẩn hóa Date cho df_news (Dùng utc=True để xử lý mixed timezones)
    df_news['date'] = pd.to_datetime(df_news['date'], utc=True, errors='coerce').dt.date
    df_news = df_news.dropna(subset=['date'])
    
    # 3. Gom nhóm tin tức theo ngày
    daily_news = df_news.groupby('date')['sentiment_score'].mean().reset_index()
    daily_news.columns = ['Date', 'Avg_Sentiment']
    
    # 4. Gộp (Merge)
    final_df = pd.merge(df_price, daily_news, on='Date', how='left')
    final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(0)
    
    # 5. Sắp xếp và lưu
    final_df = final_df.sort_values(by='Date')
    final_df.to_csv(output_path, index=False)
    print(f"    => Đã tạo {output_path} ({len(final_df)} dòng)")

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "GOOGL", "VNM"]
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    print(">>> Bắt đầu quy trình gộp dữ liệu cho toàn bộ danh mục...")
    for ticker in tickers:
        price_p = os.path.join(raw_dir, f"{ticker}_prices.csv")
        news_p = os.path.join(processed_dir, f"{ticker}_news_sentiment.csv")
        final_p = os.path.join(processed_dir, f"{ticker}_final_dataset.csv")
        
        if os.path.exists(price_p):
            merge_stock_news(price_p, news_p, final_p)
        else:
            print(f"  ! Bỏ qua {ticker} vì không có dữ liệu giá.")
