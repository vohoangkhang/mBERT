import yfinance as yf
import pandas as pd
import os
from vnstock import Vnstock

def fetch_us_stock(ticker, start_date, end_date):
    print(f"Đang lấy dữ liệu Mỹ cho mã: {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            # Xử lý MultiIndex header của yfinance mới
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.reset_index()
            # Đảm bảo cột Date là kiểu datetime string đơn giản
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        return data
    except Exception as e:
        print(f"Lỗi yfinance cho {ticker}: {e}")
        return None

def fetch_vn_stock(ticker, start_date, end_date):
    print(f"Đang lấy dữ liệu Việt Nam cho mã: {ticker}...")
    try:
        df = Vnstock().stock(symbol=ticker, source='VCI').quote.history(start=start_date, end=end_date)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        print(f"Lỗi vnstock cho {ticker}: {e}")
        return None

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    if not os.path.exists(raw_data_dir): os.makedirs(raw_data_dir)

    start = "2020-01-01"
    end = "2026-03-16"

    us_tickers = ["AAPL", "AMZN", "GOOGL", "META", "BABA"]
    for ticker in us_tickers:
        df = fetch_us_stock(ticker, start, end)
        if df is not None and not df.empty:
            path = os.path.join(raw_data_dir, f"{ticker}_prices.csv")
            df.to_csv(path, index=False)
            print(f"Đã lưu {ticker} vào {path}")

    df_vnm = fetch_vn_stock("VNM", start, end)
    if df_vnm is not None and not df_vnm.empty:
        path = os.path.join(raw_data_dir, "VNM_prices.csv")
        df_vnm.to_csv(path, index=False)
        print(f"Đã lưu VNM vào {path}")
