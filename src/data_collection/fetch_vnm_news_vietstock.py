import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

def scrape_vnm_news_vietstock(max_pages=20):
    """Cào tin tức VNM từ Vietstock."""
    print(">>> Đang cào tin tức VNM từ Vietstock (Dữ liệu lịch sử)...")
    news_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://vietstock.vn/'
    }
    
    for page in range(1, max_pages + 1):
        # Vietstock URL cho tin tức doanh nghiệp mã VNM
        url = f"https://vietstock.vn/doanh-nghiep/VNM/tin-tuc.htm?page={page}"
        print(f"  - Đang cào trang {page}...")
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm danh sách tin trong class .channel-detail
            items = soup.select('.channel-detail .mb20')
            if not items:
                print("    ! Không tìm thấy thêm tin tức. Dừng lại.")
                break
                
            for item in items:
                title_tag = item.find('h4')
                date_tag = item.find('span', class_='date')
                
                if title_tag and date_tag:
                    title = title_tag.text.strip()
                    date_str = date_tag.text.strip() # Định dạng thường là "DD/MM/YYYY"
                    
                    try:
                        publish_date = pd.to_datetime(date_str, format='%d/%m/%Y')
                    except:
                        publish_date = None
                    
                    if publish_date and publish_date.year == 2020:
                        news_list.append({
                            'title': title,
                            'publish_date': publish_date,
                            'stock': 'VNM'
                        })
                    elif publish_date and publish_date.year < 2020:
                        # Nếu đã qua năm 2020 thì dừng cào trang tiếp theo
                        return pd.DataFrame(news_list)
            
            time.sleep(1.5)
        except Exception as e:
            print(f"Lỗi ở trang {page}: {e}")
            
    return pd.DataFrame(news_list)

if __name__ == "__main__":
    raw_data_dir = os.path.join("data", "raw")
    # Chúng ta cần cào khá sâu để tới được năm 2020 (khoảng trang 40-60)
    df_vnm_news = scrape_vnm_news_vietstock(max_pages=80) 
    
    if not df_vnm_news.empty:
        df_vnm_news = df_vnm_news[df_vnm_news['publish_date'].dt.year == 2020]
        file_path = os.path.join(raw_data_dir, "VNM_news_raw.csv")
        df_vnm_news.to_csv(file_path, index=False)
        print(f"Đã lưu {len(df_vnm_news)} tin tức VNM vào {file_path}")
    else:
        print("Không tìm thấy tin tức VNM nào trong năm 2020.")
