import pandas as pd
import numpy as np
import os

# Danh sách từ khóa tài chính để tính Topic Relevance
FINANCIAL_KEYWORDS = [
    "profit", "loss", "growth", "crisis", "dividend", "revenue", "earnings", 
    "merger", "acquisition", "shares", "stock", "market", "trade", "bullish", 
    "bearish", "inflation", "interest", "fed", "economy", "fiscal", "quarterly"
]

def calculate_topic_relevance(text):
    """Tính độ liên quan chủ đề dựa trên từ khóa."""
    words = str(text).lower().split()
    count = sum(1 for word in words if word in FINANCIAL_KEYWORDS)
    return min(count / 5.0, 1.0) # Chuẩn hóa về [0, 1]

def finalize_ticker_features(ticker, sentiment_file, price_file, output_path):
    print(f">>> Đang xử lý thuộc tính cho {ticker}...")
    
    # 1. Đọc dữ liệu sentiment (từng tin tức)
    df_news = pd.read_csv(sentiment_file)
    df_news['date'] = pd.to_datetime(df_news['date'], utc=True)
    df_news['day'] = df_news['date'].dt.date
    
    # Chuẩn hóa tên cột
    if 'sentiment_score' in df_news.columns and 'sentiment_polarity' not in df_news.columns:
        df_news['sentiment_polarity'] = df_news['sentiment_score']
    
    # Giả lập các cột thiếu nếu cần
    if 'sentiment_intensity' not in df_news.columns:
        df_news['sentiment_intensity'] = df_news['sentiment_polarity'].abs()
    if 'source_credibility' not in df_news.columns:
        df_news['source_credibility'] = 0.8
    if 'content' not in df_news.columns and 'Content' in df_news.columns:
        df_news['content'] = df_news['Content']
    elif 'content' not in df_news.columns:
        df_news['content'] = ""

    # 2. Tính các thuộc tính bổ sung cho từng tin
    df_news['topic_relevance'] = df_news['content'].apply(calculate_topic_relevance)
    
    # Tính Weighted Score cho từng tin: Polarity * Intensity * Credibility
    df_news['weighted_score'] = (
        df_news['sentiment_polarity'] * 
        df_news['sentiment_intensity'] * 
        df_news['source_credibility']
    )
    
    # 3. Gom nhóm theo ngày để tính 8 thuộc tính
    daily_features = df_news.groupby('day').agg(
        avg_polarity=('sentiment_polarity', 'mean'),
        avg_intensity=('sentiment_intensity', 'mean'),
        avg_credibility=('source_credibility', 'mean'),
        avg_relevance=('topic_relevance', 'mean'),
        volume_mentions=('content', 'count'),
        weighted_sentiment_index=('weighted_score', 'sum') # Tổng hợp ảnh hưởng trong ngày
    ).reset_index()
    
    # Tính Sentiment Score (P-N)/(P+N)
    def calc_p_n_score(group):
        pos = len(group[group['sentiment_polarity'] > 0])
        neg = len(group[group['sentiment_polarity'] < 0])
        if (pos + neg) == 0: return 0
        return (pos - neg) / (pos + neg)
    
    p_n_scores = df_news.groupby('day').apply(calc_p_n_score).reset_index()
    p_n_scores.columns = ['day', 'sentiment_p_n_score']
    
    daily_features = pd.merge(daily_features, p_n_scores, on='day')
    
    # 4. Đọc dữ liệu giá
    try:
        df_price = pd.read_csv(price_file, header=[0, 1], index_col=0)
        df_price = df_price.reset_index()
        df_price.columns = ['Date'] + [col[0] for col in df_price.columns[1:]]
    except:
        df_price = pd.read_csv(price_file)
        if 'Date' not in df_price.columns: df_price = df_price.rename(columns={'date': 'Date'})
    
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    
    # 5. Gộp Giá và 8 thuộc tính Cảm xúc
    final_df = pd.merge(df_price, daily_features, left_on='Date', right_on='day', how='left')
    
    # Điền 0 cho những ngày không có tin tức
    feature_cols = [
        'avg_polarity', 'avg_intensity', 'avg_credibility', 'avg_relevance', 
        'volume_mentions', 'weighted_sentiment_index', 'sentiment_p_n_score'
    ]
    for col in feature_cols:
        final_df[col] = final_df[col].fillna(0)
        
    # Thêm Temporal Factor (giả định đơn giản: tin tức trong ngày có trọng số 1.0)
    final_df['temporal_factor'] = final_df['volume_mentions'].apply(lambda x: 1.0 if x > 0 else 0.0)
    
    # 6. Lưu kết quả
    final_df.to_csv(output_path, index=False)
    print(f"  => Đã lưu bộ dữ liệu 8 thuộc tính tại {output_path} ({len(final_df)} dòng)")

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "VNM"]
    models = ["mbert", "xlmr"]
    processed_dir = "data/processed"
    raw_dir = "data/raw"
    
    for ticker in tickers:
        for model in models:
            sentiment_f = os.path.join(processed_dir, f"{ticker}_news_{model}.csv")
            price_f = os.path.join(raw_dir, f"{ticker}_prices.csv")
            output_f = os.path.join(processed_dir, f"{ticker}_final_8_features_{model}.csv")
            
            if os.path.exists(sentiment_f) and os.path.exists(price_f):
                finalize_ticker_features(ticker, sentiment_f, price_f, output_f)
            else:
                # Đặc biệt cho VNM và model test
                alt_sent = os.path.join(processed_dir, f"VNM_news_sentiment.csv") if ticker == "VNM" else None
                if alt_sent and os.path.exists(alt_sent) and model == "mbert":
                     finalize_ticker_features(ticker, alt_sent, price_f, output_f)
                else:
                     print(f"  ! Thiếu file cho {ticker} model {model}, vui lòng kiểm tra lại.")
