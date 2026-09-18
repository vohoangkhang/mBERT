import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def fetch_rss_news(rss_url):
    """
    Cào dữ liệu tin tức từ RSS Feed.
    """
    print(f"Đang lấy tin tức từ RSS: {rss_url}...")
    try:
        # Giả lập User-Agent để tránh bị chặn
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(rss_url, headers=headers)
        soup = BeautifulSoup(response.content, features="xml")
        
        items = soup.find_all('item')
        news_list = []
        
        for item in items:
            # Xử lý các trường thông tin cơ bản từ RSS
            news = {
                'title': item.title.text if item.title else "",
                'link': item.link.text if item.link else "",
                'description': item.description.text if item.description else "",
                'pubDate': item.pubDate.text if item.pubDate else ""
            }
            news_list.append(news)
            
        return pd.DataFrame(news_list)
    except Exception as e:
        print(f"Lỗi khi cào RSS: {e}")
        return None

if __name__ == "__main__":
    # URL RSS mẫu từ CafeF (Thị trường chứng khoán)
    # Chú ý: Cần tìm đúng URL RSS cụ thể cho mã cổ phiếu nếu có
    rss_urls = {
        "cafef_thi_truong": "https://cafef.vn/thi-truong-chung-khoan.rss",
        "vietstock_tin_tuc": "https://vietstock.vn/rss/tin-tuc-su-kien.rss"
    }
    
    raw_data_dir = os.path.join("data", "raw")
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    
    for name, url in rss_urls.items():
        df_news = fetch_rss_news(url)
        if df_news is not None and not df_news.empty:
            file_path = os.path.join(raw_data_dir, f"{name}_news.csv")
            df_news.to_csv(file_path, index=False)
            print(f"Đã lưu {len(df_news)} tin tức vào {file_path}")
