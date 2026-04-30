import pandas as pd
import os
from transformers import pipeline
import torch
import numpy as np

def get_analyzer(model_name):
    """Khởi tạo sentiment analyzer."""
    print(f"Đang tải mô hình {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    analyzer = pipeline(
        "sentiment-analysis", 
        model=model_name,
        device=device,
        top_k=None 
    )
    return analyzer

def process_sentiment_results(results, model_type):
    """Xử lý kết quả từ mô hình."""
    scores = []
    intensities = []
    for res in results:
        top_res = max(res, key=lambda x: x['score'])
        intensity = top_res['score']
        if model_type == "mbert":
            stars = int(top_res['label'].split()[0])
            polarity = (stars - 3) / 2
        else: # finbert
            label = top_res['label'].upper()
            if 'POS' in label: polarity = intensity
            elif 'NEG' in label: polarity = -intensity
            else: polarity = 0.0
        scores.append(polarity)
        intensities.append(intensity)
    return scores, intensities

def analyze_vnm_news(output_dir):
    """Xử lý riêng tin tức tiếng Việt của VNM bằng mBERT."""
    vnm_news_file = "data/raw/VNM_news_synthetic.csv"
    if not os.path.exists(vnm_news_file): return
    
    analyzer = get_analyzer("nlptown/bert-base-multilingual-uncased-sentiment")
    df = pd.read_csv(vnm_news_file)
    print(f"\n>>> Đang xử lý tin tức VNM (Tiếng Việt)...")
    
    texts = df['title'].astype(str).str[:512].tolist()
    raw_results = analyzer(texts)
    
    polarities, intensities = process_sentiment_results(raw_results, "mbert")
    df['sentiment_polarity'] = polarities
    df['sentiment_intensity'] = intensities
    df['source_credibility'] = 0.9 # Trọng số cho tin tổng hợp/báo chí
    df['date'] = df['publish_date']
    df['content'] = df['title']
    
    output_path = os.path.join(output_dir, "VNM_news_mbert.csv")
    df[['date', 'content', 'sentiment_polarity', 'sentiment_intensity', 'source_credibility']].to_csv(output_path, index=False)
    print(f"  => Đã lưu {len(df)} tin cho VNM (mBERT)")

if __name__ == "__main__":
    output_directory = "data/processed"
    analyze_vnm_news(output_directory)
