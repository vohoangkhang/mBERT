import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm
from pyvi import ViTokenizer

def apply_backward_logic(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Adjusted'] = df['Date'].copy()
    mask_sat = df['Date'].dt.weekday == 5
    mask_sun = df['Date'].dt.weekday == 6
    df.loc[mask_sat, 'Date_Adjusted'] = df.loc[mask_sat, 'Date'] - pd.Timedelta(days=1)
    df.loc[mask_sun, 'Date_Adjusted'] = df.loc[mask_sun, 'Date'] - pd.Timedelta(days=2)
    return df

def process_finbert(ticker, src_file, price_file, out_file, device=0):
    print(f'\n>>> [FinBERT Inference] Processing {ticker}...')
    if not os.path.exists(src_file) or not os.path.exists(price_file):
        print(f'  ! Missing source files for {ticker}, skipping.')
        return
        
    df_sent = pd.read_csv(src_file)
    analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert', device=device, framework='pt')
    
    texts = df_sent['Content'].astype(str).str[:512].tolist()
    batch_size = 64
    scores, intensities = [], []
    
    print(f'  > Analyzing sentiment for {len(texts)} news articles...')
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_res = analyzer(texts[i : i + batch_size])
        for res in batch_res:
            lbl = res['label'].lower()
            conf = res['score']
            if 'positive' in lbl:
                score = conf
            elif 'negative' in lbl:
                score = -conf
            else:
                score = 0.0
            scores.append(score)
            intensities.append(conf)
            
    df_sent['sentiment_score'] = scores
    df_sent['intensity'] = intensities
    df_sent = apply_backward_logic(df_sent)
    
    daily_sent = df_sent.groupby(df_sent['Date_Adjusted'].dt.date).agg({
        'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
        'intensity': ['mean', 'max', 'std']
    })
    daily_sent.columns = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    daily_sent = daily_sent.reset_index().rename(columns={'Date_Adjusted': 'Date'}).fillna(0)
    daily_sent['Date'] = pd.to_datetime(daily_sent['Date']).dt.date
    
    df_price = pd.read_csv(price_file)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_price = df_price[price_cols]
    
    final_df = pd.merge(df_price, daily_sent, on='Date', how='left').fillna(0)
    final_df.to_csv(out_file, index=False)
    print(f'  => Saved FinBERT dataset: {out_file} ({len(final_df)} rows)')

def process_phobert(ticker, src_file, price_file, out_file, device=0):
    print(f'\n>>> [PhoBERT Inference] Processing {ticker}...')
    if not os.path.exists(src_file) or not os.path.exists(price_file):
        print(f'  ! Missing source files for {ticker}, skipping.')
        return
        
    df_sent = pd.read_csv(src_file)
    analyzer = pipeline('sentiment-analysis', model='wonrax/phobert-base-vietnamese-sentiment', device=device, framework='pt')
    
    raw_texts = df_sent['Content'].astype(str).str[:512].tolist()
    texts = [ViTokenizer.tokenize(t) for t in raw_texts]
    batch_size = 64
    scores, intensities = [], []
    
    print(f'  > Analyzing sentiment for {len(texts)} Vietnamese news articles...')
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_res = analyzer(texts[i : i + batch_size])
        for res in batch_res:
            lbl = res['label'].upper()
            conf = res['score']
            if 'POS' in lbl:
                score = conf
            elif 'NEG' in lbl:
                score = -conf
            else:
                score = 0.0
            scores.append(score)
            intensities.append(conf)
            
    df_sent['sentiment_score'] = scores
    df_sent['intensity'] = intensities
    df_sent = apply_backward_logic(df_sent)
    
    daily_sent = df_sent.groupby(df_sent['Date_Adjusted'].dt.date).agg({
        'sentiment_score': ['mean', 'max', 'min', 'sum', 'count'],
        'intensity': ['mean', 'max', 'std']
    })
    daily_sent.columns = ['avg_polarity', 'max_polarity', 'min_polarity', 'sum_polarity', 'count_mentions', 'avg_intensity', 'avg_credibility', 'avg_relevance']
    daily_sent = daily_sent.reset_index().rename(columns={'Date_Adjusted': 'Date'}).fillna(0)
    daily_sent['Date'] = pd.to_datetime(daily_sent['Date']).dt.date
    
    df_price = pd.read_csv(price_file)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_price = df_price[price_cols]
    
    final_df = pd.merge(df_price, daily_sent, on='Date', how='left').fillna(0)
    final_df.to_csv(out_file, index=False)
    print(f'  => Saved PhoBERT dataset: {out_file} ({len(final_df)} rows)')

if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else -1
    print('Running on device:', 'GPU cuda:0' if device == 0 else 'CPU')
    
    src_dir = os.path.join('data_src', 'data_thay', 'sentiment')
    proc_dir = os.path.join('data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    # 5 US tickers with FinBERT
    us_mapping = {'APPLE': 'AAPL', 'AMAZON': 'AMZN', 'ALIBABA': 'BABA', 'GOOGLE': 'GOOGL', 'META': 'META'}
    for thay_name, ticker in us_mapping.items():
        s_in = os.path.join(src_dir, f'{thay_name}_sentiment.csv')
        p_clean = os.path.join(proc_dir, f'{ticker}_final_v4_clean.csv')
        out_path = os.path.join(proc_dir, f'{ticker}_finbert_final.csv')
        process_finbert(ticker, s_in, p_clean, out_path, device=device)
        
    # VNM with PhoBERT
    s_in_vnm = os.path.join(src_dir, 'VNM_sentiment.csv')
    p_clean_vnm = os.path.join(proc_dir, 'VNM_final_v4_clean.csv')
    out_path_vnm = os.path.join(proc_dir, 'VNM_phobert_final.csv')
    process_phobert('VNM', s_in_vnm, p_clean_vnm, out_path_vnm, device=device)
    
    print('\n==================================================')
    print('HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG FINBERT & PHOBERT!')
    print('==================================================')
