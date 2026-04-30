import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

def scrape_vnm_news_cafef(max_pages=15):
    """Cào tin tức VNM từ CafeF."""
    print(">>> Đang cào tin tức VNM từ CafeF...")
    news_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for page in range(1, max_pages + 1):
        # URL tin tức mã VNM trên CafeF
        url = f"https://cafef.vn/hose/VNM-cong-ty-co-phan-sua-viet-nam/tin-tuc-su-kien.chn?page={page}"
        print(f"  - Đang cào trang {page}...")
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm danh sách tin tức trong div .listNews
            items = soup.select('ul.listNews li')
            for item in items:
                title_tag = item.find('a')
                date_tag = item.find('span', class_='time')
                
                if title_tag and date_tag:
                    title = title_tag.text.strip()
                    date_str = date_tag.text.strip()
                    
                    # CafeF thường hiển thị dạng "DD/MM/YYYY HH:mm"
                    try:
                        publish_date = pd.to_datetime(date_str, format='%d/%m/%Y %H:%M')
                    except:
                        publish_date = None
                    
                    # Lọc tin trong năm 2020
                    if publish_date and publish_date.year == 2020:
                        news_list.append({
                            'title': title,
                            'publish_date': publish_date,
                            'stock': 'VNM'
                        })
            
            # Dừng lại nếu đã qua năm 2020
            if news_list and news_list[-1]['publish_date'] and news_list[-1]['publish_date'].year < 2020:
                break
                
            time.sleep(1) # Tránh bị chặn IP
        except Exception as e:
            print(f"Lỗi ở trang {page}: {e}")
            
    return pd.DataFrame(news_list)

if __name__ == "__main__":
    raw_data_dir = os.path.join("data", "raw")
    df_vnm_news = scrape_vnm_news_cafef(max_pages=50) # Tăng số trang để tìm tin cũ
    
    if not df_vnm_news.empty:
        # Lọc lại chính xác năm 2020
        df_vnm_news = df_vnm_news[df_vnm_news['publish_date'].dt.year == 2020]
        file_path = os.path.join(raw_data_dir, "VNM_news_raw.csv")
        df_vnm_news.to_csv(file_path, index=False)
        print(f"Đã lưu {len(df_vnm_news)} tin tức VNM vào {file_path}")
    else:
        print("Không tìm thấy tin tức nào.")
