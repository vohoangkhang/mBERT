import pandas as pd
import numpy as np
import os
from transformers import pipeline
import torch
from tqdm import tqdm

def apply_backward_logic(df):
    """Tin T7, CN dồn về Thứ 6 trước đó theo ý thầy Khang."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Adjusted'] = df['Date'].copy()
    mask_sat = df['Date'].dt.weekday == 5
    mask_sun = df['Date'].dt.weekday == 6
    df.loc[mask_sat, 'Date_Adjusted'] = df.loc[mask_sat, 'Date'] - pd.Timedelta(days=1)
    df.loc[mask_sun, 'Date_Adjusted'] = df.loc[mask_sun, 'Date'] - pd.Timedelta(days=2)
    return df

def get_mbert_sentiment(ticker, s_in, price_clean_file, out_path):
    print(f"\n>>> [mBERT Inference] Đang xử lý cho {ticker}...")
    if not os.path.exists(s_in) or not os.path.exists(price_clean_file):
        print(f"  ! Thiếu file cho {ticker}, bỏ qua.")
        return

    df_sent = pd.read_csv(s_in)
    
    # 1. Khởi tạo mô hình mBERT
    device = 0 if torch.cuda.is_available() else -1
    print(f"  > Đang chạy trên thiết bị: {'GPU' if device == 0 else 'CPU'}")
    analyzer = pipeline("sentiment-analysis", 
                        model="nlptown/bert-base-multilingual-uncased-sentiment", 
                        device=device, 
                        framework="pt")
    
    # 2. Xử lý theo lô (Batch Processing)
    texts = df_sent['Content'].astype(str).str[:512].tolist()
    batch_size = 64
    scores, intensities = [], []
    
    print(f"  > Đang tính toán cảm xúc cho {len(texts)} dòng tin tức...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_results = analyzer(texts[i:i+batch_size])
        for res in batch_results:
            stars = int(res['label'].split()[0])
            scores.append((stars - 3) / 2) # Chuyển từ (1-5 sao) về (-1 đến 1)
            intensities.append(res['score'])
            
    df_sent['sentiment_score'] = scores
    df_sent['intensity'] = intensities
    
    # 3. Áp dụng logic Thầy Khang (Backward)
    df_sent = apply_backward_logic(df_sent)
    
    # 4. Gom nhóm theo ngày
    daily_sent = df_sent.groupby(df_sent['Date_Adjusted'].dt.date).agg({
        'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
        'intensity': ['mean', 'max', 'std']
    })
    daily_sent.columns = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    daily_sent = daily_sent.reset_index().rename(columns={'Date_Adjusted': 'Date'}).fillna(0)
    daily_sent['Date'] = pd.to_datetime(daily_sent['Date']).dt.date

    # 5. Merge với Giá Sạch V6
    df_price = pd.read_csv(price_clean_file)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    
    # Chỉ giữ lại các cột giá trong df_price
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_price = df_price[price_cols]
    
    final_df = pd.merge(df_price, daily_sent, on='Date', how='left')
    final_df = final_df.fillna(0)
    
    final_df.to_csv(out_path, index=False)
    print(f"  => Đã lưu bộ dữ liệu mBERT Sạch tại: {out_path} ({len(final_df)} dòng)")

if __name__ == "__main__":
    src_dir = os.path.join("data", "data_thay")
    proc_dir = os.path.join("data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    mapping = {"APPLE": "AAPL", "VNM": "VNM", "AMAZON": "AMZN", "GOOGLE": "GOOGL", "META": "META", "ALIBABA": "BABA"}
    
    for thay_name, ticker in mapping.items():
        s_in = os.path.join(src_dir, f"{thay_name}_sentiment.csv")
        p_clean = os.path.join(proc_dir, f"{ticker}_final_v4_clean.csv")
        out_path = os.path.join(proc_dir, f"{ticker}_mbert_final.csv")
        
        get_mbert_sentiment(ticker, s_in, p_clean, out_path)
    
    print("\n>>> QUÁ TRÌNH TRÍCH XUẤT mBERT HOÀN TẤT!")
