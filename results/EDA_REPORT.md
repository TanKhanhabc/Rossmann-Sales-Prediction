# Phân tích dữ liệu Rossmann (tính từ CSV gốc)

- Train: **1,017,209** bản ghi, từ 2013-01-01 đến 2015-07-31.
- Test: **41,088** bản ghi, từ 2015-08-01 đến 2015-09-17.
- Store: **1,115** cửa hàng; test có 856 cửa hàng.
- Doanh thu train: **5,873,180,623 EUR** (tiền tệ dựa trên mô tả bộ dữ liệu Rossmann).
- Doanh thu trung bình khi Open=1: **6,955.51**.
- Test thiếu Open: **11** hàng (cửa hàng 622).

## Thiếu dữ liệu trong store.csv

- CompetitionDistance: 3 (0.3%)
- CompetitionOpenSinceMonth: 354 (31.7%)
- CompetitionOpenSinceYear: 354 (31.7%)
- Promo2SinceWeek: 544 (48.8%)
- Promo2SinceYear: 544 (48.8%)
- PromoInterval: 544 (48.8%)

## Quy ước mô hình

- **Sales** là biến mục tiêu, đơn vị doanh thu cửa hàng theo ngày.
- Không sử dụng **Customers** làm đầu vào vì test.csv không có trường này.
- Dữ liệu validation chia theo thời gian, không xáo trộn ngẫu nhiên.
- Cửa hàng đóng cửa => dự đoán 0; các ngày Open thiếu được nội suy theo thứ trong tuần.
- Dự báo ngoài khoảng test.csv là kịch bản theo giả định, không phải lịch khuyến mại thực.

## Doanh thu theo tháng (5 tháng cuối)

| Month   |     Sales |   Records |
|:--------|----------:|----------:|
| 2015-03 | 205631683 |     34565 |
| 2015-04 | 197918886 |     33450 |
| 2015-05 | 189143897 |     34565 |
| 2015-06 | 207363373 |     33450 |
| 2015-07 | 212322616 |     34565 |
