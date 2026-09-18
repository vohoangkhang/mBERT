import pandas as pd
import os

def analyze_dates(file_path, ticker):
    print(f"\n--- Phân tích file: {file_path} ---")
    try:
        # Đọc theo chunk nếu file quá lớn
        chunksize = 10**5
        min_date = None
        max_date = None
        count = 0
        
        for chunk in pd.read_csv(file_path, usecols=['date', 'stock'], chunksize=chunksize):
            # Lọc theo mã cổ phiếu
            ticker_news = chunk[chunk['stock'] == ticker]
            if not ticker_news.empty:
                # Chuyển sang datetime (không dùng UTC để tránh lỗi lệch múi giờ khi so sánh)
                ticker_news['date'] = pd.to_datetime(ticker_news['date'], errors='coerce')
                current_min = ticker_news['date'].min()
                current_max = ticker_news['date'].max()
                
                if min_date is None or current_min < min_date:
                    min_date = current_min
                if max_date is None or current_max > max_date:
                    max_date = current_max
                count += len(ticker_news)
        
        print(f"Mã cổ phiếu: {ticker}")
        print(f"Ngày bắt đầu: {min_date}")
        print(f"Ngày kết thúc: {max_date}")
        print(f"Tổng số tin tức: {count}")
        return min_date, max_date
    except Exception as e:
        print(f"Lỗi: {e}")
        return None, None

if __name__ == "__main__":
    files = ["data/raw_analyst_ratings.csv", "data/raw_partner_headlines.csv"]
    ticker = "AAPL"
    
    for f in files:
        if os.path.exists(f):
            analyze_dates(f, ticker)
