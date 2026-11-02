# Từ điển dữ liệu Rossmann

## train.csv — 1.017.209 bản ghi, 9 cột
| Cột | Ý nghĩa | Sử dụng |
|---|---|---|
| Store | Mã cửa hàng | Khóa join, đặc trưng |
| DayOfWeek | 1 = Thứ Hai, 7 = Chủ nhật | Đặc trưng (tính lại từ Date) |
| Date | Ngày giao dịch | Đặc trưng thời gian |
| Sales | Doanh thu ngày của cửa hàng, EUR | **Biến mục tiêu** |
| Customers | Số khách ngày đó | **Không dùng** vì test không có |
| Open | 1 mở cửa, 0 đóng | Quy tắc đóng cửa → Sales=0 |
| Promo | Khuyến mãi thông thường | Đặc trưng |
| StateHoliday | 0, a, b, c | Đặc trưng |
| SchoolHoliday | Ngày nghỉ học | Đặc trưng |

## test.csv — 41.088 bản ghi, 8 cột
Chứa `Id`, `Store`, `DayOfWeek`, `Date`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`. **Không có Sales và Customers**. Có 11 dòng thiếu `Open`, đều tại Store 622; script `scripts/data/prepare_data.py` điền theo ngày trong tuần (Chủ nhật đóng) vào `data/processed/test.csv`; file gốc không bị sửa. `Id` chỉ để đối chiếu khi xuất submission.

## store.csv — 1.115 bản ghi, 10 cột
`Store`, `StoreType`, `Assortment`, `CompetitionDistance`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `Promo2`, `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`.

Giá trị NA trong trường thời điểm mở cửa đối thủ hoặc Promo2 là hợp lệ (không áp dụng hoặc chưa biết); script chuẩn bị dữ liệu giữ lại hoặc mã hóa các giá trị này thay vì xóa hàng. Khoảng cách cạnh tranh thiếu được thay bằng `-1` và đánh dấu ở `CompetitionMissing`.

## Tính năng mở rộng (feature engineering)
Thứ, tháng, quý, tuần ISO, vị trí ngày trong năm, tuần hoàn thời gian; khoảng cách và thời gian hoạt động của đối thủ; thời gian Promo2 và tháng Promo2 đang hiệu lực. **Không lấy Sales hoặc Customers để tính đầu vào**, tránh lệch khả năng dự đoán test.
