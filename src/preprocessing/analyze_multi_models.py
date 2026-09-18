import pandas as pd
import numpy as np
import os
from transformers import pipeline
import torch

def get_sentiment_model(model_name):
    print(f">>> Đang tải mô hình {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model_name, device=device)

def analyze_news(input_file, model_name, model_label):
    if not os.path.exists(input_file): return None
    
    df = pd.read_csv(input_file)
    analyzer = get_sentiment_model(model_name)
    
    print(f">>> Đang phân tích cảm xúc bằng {model_label} cho {input_file} (Tổng {len(df)} tin)...")
    texts = df['Content'].astype(str).str[:512].tolist()
    
    # Chạy inference theo batch để nhanh hơn và tránh lỗi bộ nhớ
    batch_size = 32
    scores = []
    intensities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        results = analyzer(batch_texts)
        for res in results:
            if "star" in res['label'].lower():
                stars = int(res['label'].split()[0])
                score = (stars - 3) / 2
                intensity = res['score']
            else:
                score = res['score'] if 'POS' in res['label'].upper() else -res['score'] if 'NEG' in res['label'].upper() else 0
                intensity = res['score']
            scores.append(score)
            intensities.append(intensity)
    
    df['sentiment_score'] = scores
    df['intensity_score'] = intensities
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # GOM NHÓM THEO NGÀY VÀ TÍNH 8 THÀNH PHẦN THỰC TẾ
    daily = df.groupby('Date').agg(
        avg_polarity=('sentiment_score', 'mean'),
        max_polarity=('sentiment_score', 'max'),
        min_polarity=('sentiment_score', 'min'),
        sum_polarity=('sentiment_score', 'sum'),
        count_mentions=('sentiment_score', 'count'),
        avg_intensity=('intensity_score', 'mean'),
        avg_credibility=('intensity_score', 'max'), # Dùng max intensity như một chỉ số tin cậy tạm thời
        avg_relevance=('intensity_score', 'std')    # Độ lệch chuẩn thể hiện sự phân tán nội dung
    ).reset_index()
    
    # Điền 0 cho các giá trị NaN (std có thể là NaN nếu chỉ có 1 tin)
    daily = daily.fillna(0)
    
    # Các đặc trưng mở rộng (tính toán dựa trên dữ liệu thật)
    daily['weighted_sentiment_index'] = daily['avg_polarity'] * daily['count_mentions']
    daily['sentiment_p_n_score'] = daily['sum_polarity'] / (daily['count_mentions'] + 1)
    daily['temporal_factor'] = 1.0 # Có thể cải thiện sau
    
    return daily

if __name__ == "__main__":
    base_dir = r"D:\HUTECH\thayKhang\stock-nlp-prediction"
    vnm_raw = os.path.join(base_dir, "data", "data_thay", "VNM.csv")
    apple_raw = os.path.join(base_dir, "data", "data_thay", "APPLE.csv")
    output_dir = os.path.join(base_dir, "data", "processed")

    # 1. Chạy với mBERT cho VNM và APPLE
    print("\n--- PHÂN TÍCH MBERT ---")
    vnm_mbert = analyze_news(vnm_raw, "nlptown/bert-base-multilingual-uncased-sentiment", "mBERT")
    if vnm_mbert is not None:
        vnm_mbert.to_csv(os.path.join(output_dir, "VNM_8_features_mbert_test.csv"), index=False)
        
    apple_mbert = analyze_news(apple_raw, "nlptown/bert-base-multilingual-uncased-sentiment", "mBERT")
    if apple_mbert is not None:
        # Lưu vào file mà train_mbert_models.py đang chờ đợi
        apple_mbert.to_csv(os.path.join(output_dir, "AAPL_8_features_mbert_test.csv"), index=False)
