# Ghi chú làm việc – Chuyên đề 2 (NCS Võ Hoàng Khang)

> File này lưu lại toàn bộ thông tin đã trao đổi với Claude ngày 11/10/2026 để tiếp tục làm việc ở phiên sau.
> Khi mở phiên mới, chỉ cần nói: "Đọc docs/GHI_CHU_CHUYEN_DE_2.md rồi làm tiếp".

## 1. Thông tin chung

- **File gốc:** `ChuyenDe2_Vo_Hoang_Khang_07-10-2026.docx` (người dùng upload, không lưu trong repo — cần upload lại nếu cần đọc nội dung).
- **Tên đề tài:** "Nghiên cứu mô hình học sâu đa phương thức tích hợp Transformer và dữ liệu cảm xúc trong dự báo xu hướng thị trường chứng khoán".
- **Cán bộ hướng dẫn:** PGS. TS. Nguyễn Đình Thuân. Mã NCS: N230201.
- **Mã cổ phiếu thực nghiệm:** VNM (Việt Nam); META, AMZN, AAPL, BABA, GOOGL (Hoa Kỳ).
- **18 cấu hình/cổ phiếu:** Naive, ARIMA, {DT, RF, XGBoost, BiLSTM} × {Pure, Lexicon, PhoBERT/FinBERT, mBERT}.
- **8 đặc trưng cảm xúc:** avg_polarity, max_polarity, min_polarity, sum_polarity, count_mentions, avg_intensity, avg_credibility, avg_relevance; thêm P-N Score, WSI.
- **Cơ chế đề xuất:** Backward Weekend Logic (BWL) xử lý Weekend Information Gap (WIG).

## 2. File đã tạo (thư mục `docs/`)

| File | Nội dung |
|---|---|
| `Danh_muc_tu_viet_tat.docx` | 51 thuật ngữ + 8 mã cổ phiếu + 6 từ thông dụng |
| `Danh_muc_hinh.docx` | 4 hình (Hình 1, 3, 4, 5) |
| `Bang4_VNM_rut_gon.docx` | Bảng 4 rút gọn 11 dòng (bản cũ, nên làm lại theo khung 8 dòng) |
| `Bang5_META_rut_gon.docx` | Bảng 5 rút gọn 11 dòng (bản cũ, nên làm lại theo khung 8 dòng) |

## 3. Quyết định đã thống nhất về văn phong

- Mục "Điểm qua một số nghiên cứu gần đây" → **"Tổng quan các nghiên cứu liên quan"**.
- Mục "Nhận xét tổng hợp" → **"Đánh giá tổng hợp và khoảng trống nghiên cứu"**.
- Tránh khẳng định tuyệt đối ("không có nghiên cứu nào", "hầu hết") → dùng "phần lớn", "còn hạn chế".
- "ví tài sản" → "danh mục đầu tư"; "thị trường đang phát triển" → "thị trường mới nổi".
- Không viết "mBERT tốt nhất" — kết quả không đồng đều giữa các cổ phiếu.

### Ba khoảng trống nghiên cứu (mục 1.1)
1. Tối ưu siêu tham số (PSO/GA) cho mô hình học sâu đa phương thức còn hạn chế.
2. **Thông tin cuối tuần (WIG) chưa được xử lý hệ thống** + cảm xúc còn đơn giản, đơn ngữ → **chuyên đề này giải quyết** (mBERT, đặc trưng đa chiều, BWL).
3. Dự báo xu hướng gắn quyết định đầu tư, quản trị rủi ro danh mục còn ít → hướng tiếp theo.
- Lưu ý: BWL là cơ chế của tác giả → nêu ở phần giải pháp, không viết "các nghiên cứu ít nói về BWL".

### Mục tiêu (mục 1.2) — 5 mục tiêu cụ thể
(1) Bộ dữ liệu đa phương thức → (2) Cảm xúc đa ngôn ngữ mBERT, đặc trưng đa chiều → (3) WIG/BWL → (4) Mô hình hợp nhất (BiLSTM, XGBoost + mBERT) → (5) Đánh giá (MAE, MSE, RMSE, MAPE, R², DA, Wilcoxon, ablation). Mục "định hướng mở rộng" chuyển thành đoạn kết.

## 4. Sai lệch giữa chuyên đề và source code (cần sửa)

1. **Chia dữ liệu:** code chia 80% train / 20% test theo thời gian. Chỉ BiLSTM lấy thêm 10% cuối tập train làm validation (EarlyStopping, patience=7) → ~72/8/20. Mục 4.1 ghi "ba tập train/validation/test" là chưa đúng. Câu đề xuất:
   > "Dữ liệu được chia theo thứ tự thời gian với tỷ lệ 80% cho tập huấn luyện và 20% cho tập kiểm thử nhằm tránh rò rỉ thông tin trong chuỗi thời gian. Đối với mô hình BiLSTM, 10% dữ liệu cuối của tập huấn luyện được tách ra làm tập xác thực, phục vụ cơ chế dừng sớm (early stopping) nhằm hạn chế hiện tượng quá khớp."
2. **Lý do XGBoost không có validation:** siêu tham số cố định (n_estimators=100, lr=0.05, max_depth=5), không có quá trình lặp cần dừng sớm. Đã có câu giải thích; đề xuất tùy chọn sửa code dùng `eval_set` + `early_stopping_rounds` (chưa làm).
3. **"Châu Âu"** trong mục 4.1: không có dữ liệu châu Âu trong source.
4. **Thư viện:** BiLSTM dùng TensorFlow/Keras (không phải PyTorch); PyTorch chỉ dùng cho Transformers.
5. **P-N Score:** chuyên đề ghi mẫu số +1; `finalize_features.py:61-65` tính (P−N)/(P+N) không +1; `analyze_multi_models.py:61` lại tính `sum_polarity/(count_mentions+1)`. Mô tả "tỷ lệ cung và cầu" sai → "tỷ lệ chênh lệch giữa số tin tích cực và tiêu cực".
6. **avg_relevance:** chuyên đề ghi "độ lệch chuẩn"; code tính trung bình `topic_relevance`.
7. **Danh mục hình:** thiếu Hình 2 (đánh số 1, 3, 4, 5); tên Hình 4 và 5 gần trùng.
8. **Tiêu đề Bảng 9** ghi "Alibaba (BABA)" nhưng dữ liệu là **GOOGL**.

## 5. Rút gọn Bảng 4–9 — khung thống nhất

**Nguyên tắc:** (a) gộp Naive và ARIMA nếu trùng; (b) Random Forest đại diện nhóm cây (vượt DT, XGBoost); (c) mỗi mô hình đại diện giữ bộ ba Pure / mô hình chuyên biệt (PhoBERT/FinBERT hoặc Lexicon) / mBERT; (d) loại cấu hình có R² < 0; (e) bảng đầy đủ 18 dòng đưa vào Phụ lục.
Công thức % thay đổi tương đối: |mới − gốc| / gốc × 100. R² và DA dùng chênh lệch tuyệt đối.

### Bảng 4 – VNM (đề xuất 8 dòng)
Naive (1.024 / 1.380 / 1.63% / 0.9011 / 3.77%); ARIMA(0,1,1) (1.014 / 1.378 / 1.62% / 0.9014 / 57.86%); RF Pure (**0.623 / 0.926 / 0.97% / 0.9554** / 81.76%); RF+Lexicon (0.645 / 0.970 / 1.00% / 0.9511 / **82.39%**); RF+mBERT (0.669 / 1.007 / 1.04% / 0.9474 / 81.76%); BiLSTM Pure (1.840 / 2.571 / 2.89% / 0.6567 / 54.72%); BiLSTM+PhoBERT (1.676 / 2.286 / 2.68% / 0.7285 / 54.72%); BiLSTM+mBERT (1.673 / 2.270 / 2.68% / 0.7324 / 51.57%).
Ý chính: cây tốt nhất; BiLSTM+mBERT giảm MAE 9,1%, RMSE 11,7%; mBERT ≈ PhoBERT; DA BiLSTM+mBERT giảm. 4 cấu hình DT trùng nhau (0.663/1.003/1.04%/0.9477/77.36%).

### Bảng 5 – META (đề xuất 8 dòng)
Naive (**9.234 / 13.953 / 1.32% / 0.9221** / 0.00%); ARIMA(0,1,0) (9.262 / 13.970 / 1.32% / 0.9219 / 50.33%); RF Pure (13.374 / 19.748 / 1.82% / 0.8439 / 68.21%); RF+FinBERT (13.268 / 19.601 / 1.81% / 0.8462 / **68.87%**); RF+mBERT (13.224 / 19.612 / 1.80% / 0.8460 / 68.21%); BiLSTM Pure (26.088 / 31.314 / 3.69% / 0.6075 / 49.67%); BiLSTM+FinBERT (20.425 / 26.635 / 2.93% / 0.7160 / 51.66%); BiLSTM+mBERT (33.017 / 37.390 / 4.66% / 0.4403 / 48.34%).
Ý chính: Naive thắng sai số; RF tốt DA (+18 điểm so với ARIMA), MAE cao hơn Naive 44,8% (cây không ngoại suy); FinBERT giúp BiLSTM (−21,7% MAE), mBERT kém nhất (+26,6%). mBERT hợp thị trường đa ngôn ngữ, FinBERT hợp thị trường tiếng Anh.

### Bảng 6 – AMZN (7 dòng; loại BiLSTM+FinBERT vì R² = −0,0852)
Naive/ARIMA trùng (3.220 / 4.514 / 1.42% / 0.8409 / 0.00%); RF Pure (**1.873 / 2.836 / 0.81% / 0.9372** / 79.50%); RF+FinBERT (1.989 / 3.058 / 0.86% / 0.9270 / **80.75%**); RF+mBERT (2.049 / 3.149 / 0.88% / 0.9226 / 78.26%); BiLSTM Pure (8.482 / 10.420 / 3.69% / 0.1525 / 45.34%); BiLSTM+Lexicon (6.443 / 8.869 / 2.88% / 0.3860 / 52.17%); BiLSTM+mBERT (6.689 / 8.774 / 2.94% / 0.3990 / 50.31%).
Ý chính: RF giảm 41,8% MAE so với Naive; cảm xúc tăng sai số RF (+6,2% FinBERT, +9,4% mBERT); BiLSTM+mBERT giảm 21,1% MAE, R² 0.1525→0.3990.

### Bảng 7 – AAPL (7 dòng)
Naive/ARIMA trùng (**2.185 / 3.097 / 0.93% / 0.9881** / 0.00%); RF Pure (6.220 / 10.035 / 2.34% / 0.8755 / 64.90%); RF+FinBERT (6.223 / 10.045 / 2.34% / 0.8753 / **65.56%**); RF+mBERT (6.249 / 10.089 / 2.35% / 0.8742 / **65.56%**); BiLSTM Pure (6.457 / 7.907 / 2.47% / 0.9218 / 49.67%); BiLSTM+FinBERT (8.441 / 10.428 / 3.32% / 0.8654 / 50.99%); BiLSTM+mBERT (9.089 / 11.105 / 3.63% / 0.8477 / 50.99%).
Ý chính: RF MAE ≈ 2,8 lần Naive; cảm xúc gần như không đổi RF; BiLSTM Pure tốt nhất học sâu, cảm xúc làm MAE tăng 30,7% (FinBERT), 40,8% (mBERT) — tin AAPL nhiều, nhiễu, nguy cơ quá khớp.

### Bảng 8 – BABA (5 dòng; loại toàn bộ 12 cấu hình cây vì R² từ −0,1079 đến −0,3106)
Naive/ARIMA trùng (**3.177 / 4.290 / 2.06% / 0.9465** / 0.00%); BiLSTM Pure (8.849 / 11.196 / 5.51% / 0.6356 / **59.63%**); +Lexicon (11.156 / 12.819 / 7.01% / 0.5223 / 57.76%); +FinBERT (11.583 / 13.230 / 7.27% / 0.4912 / 57.76%); +mBERT (11.422 / 13.669 / 7.11% / 0.4568 / 59.01%).
Ý chính: cây thất bại do không ngoại suy (DA ~65% nhưng ít ý nghĩa); BiLSTM duy nhất có ý nghĩa; cảm xúc làm MAE tăng 26,1% / 30,9% / 29,1%. Gắn với mục 4.4 "Khả năng ngoại suy".

### Bảng 9 – GOOGL (5 dòng; loại 12 cấu hình cây vì R² từ −0,3139 đến −0,4336)
Naive/ARIMA trùng (**3.087 / 4.384 / 1.29% / 0.9925** / 0.00%); BiLSTM Pure (23.997 / 30.892 / 9.00% / 0.6254 / 45.03%); +Lexicon (34.935 / 44.543 / 13.08% / 0.2212 / **45.70%**); +FinBERT (24.799 / 32.460 / 9.23% / 0.5864 / 44.37%); +mBERT (26.019 / 33.500 / 9.77% / 0.5595 / 45.03%).
Ý chính: MAPE cây > 11 lần Naive; BiLSTM DA < 50%; cảm xúc làm MAE tăng 3,3% (FinBERT), 8,4% (mBERT), 45,6% (Lexicon) → Transformer ổn định hơn Lexicon.

## 6. Ba quy luật rút ra cho mục 4.5 Thảo luận
1. Naive/ARIMA luôn thắng sai số trên cổ phiếu Hoa Kỳ nhưng DA ≈ 0% → DA là chỉ số cần thiết.
2. Mô hình cây tốt khi giá nằm trong khoảng huấn luyện (VNM, AMZN), thất bại khi phải ngoại suy (BABA, GOOGL).
3. Hiệu quả dữ liệu cảm xúc phụ thuộc cổ phiếu: rõ nhất ở VNM, AMZN; trái chiều/tiêu cực ở META, AAPL, BABA, GOOGL.

## 7. Việc còn dở / có thể làm tiếp
- [ ] Xuất 6 bảng rút gọn (4–9) theo khung thống nhất ra 1 file Word.
- [ ] Làm lại Bảng 4, 5 theo khung 8 dòng (thay bản 11 dòng).
- [ ] Viết mục 4.5 Thảo luận dựa trên 3 quy luật.
- [ ] (Tùy chọn) Sửa code XGBoost dùng tập xác thực + early stopping, chạy lại thực nghiệm.
- [ ] Thống nhất công thức P-N Score giữa chuyên đề và code.
