# Machine Learning — Rossmann

## Mục tiêu
Dự đoán `Sales` theo `Store + Date`. Dataset gốc có **1.017.209** bản ghi lịch sử và **41.088** bản ghi test không có nhãn.

## Pipeline
1. Đọc `train.csv`, `store.csv`; nối theo `Store` (`many_to_one`).
2. Tạo đặc trưng thời gian, khuyến mãi, ngày lễ, cửa hàng, cạnh tranh.
3. Không sử dụng `Customers` hoặc `Sales` trong features (tránh target leakage / inference mismatch).
4. **42 ngày cuối lịch sử** làm validation; chỉ huấn luyện trước thời điểm validation.
5. Fit LightGBM trên `log1p(Sales)` khi cửa hàng mở và doanh thu dương; clamp dự đoán về ≥0, ép 0 khi đóng.
6. So với baseline trung bình nhóm (`Store`, `DayOfWeek`, `Promo`) dựa trên lịch sử **trước** validation.
7. Đánh giá MAE, RMSE, RMSPE (loại Sales=0), WAPE. Sau đó fit lại trên toàn bộ lịch sử và lưu `models/rossmann_sales_lgbm.joblib`.
8. Dự đoán `test.csv` và xuất `results/submission.csv` với đúng Id.

## Lưu ý
- MAE/RMSE phải đọc từ `results/evaluation_metrics.json` sau khi chạy train, không điền đoán.
- `Open` trong test thiếu 11 bản ghi tại Store 622, điền 0 nếu Chủ nhật, khác điền 1: **giả định**.
- Với Open=1, Sales=0 có 54 ngoại lệ lịch sử; không fit 54 hàng này, nhưng vẫn tính điểm nếu có trong holdout.
- Các đặc trưng lịch giả định cho các ngày ngoài test có thể sai với thực tế; không gọi đây là lịch promo tương lai chắc chắn.
- Cách tính tuần Promo2 từ năm ISO gần đúng, không xử lý hoàn hảo năm 53 tuần.
- **Không có kết quả thử nghiệm trên hidden Kaggle test** cho tới khi người dùng nộp CSV hợp lệ.
