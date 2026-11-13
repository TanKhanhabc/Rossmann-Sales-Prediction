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
Chứa `Id`, `Store`, `DayOfWeek`, `Date`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`. **Không có Sales và Customers**. Có 11 dòng thiếu `Open`, đều tại Store 622; script `scripts/preprocessing/prepare_data.py` điền theo ngày trong tuần (Chủ nhật đóng) vào `data/processed/test.csv`; file gốc không bị sửa. `Id` chỉ để đối chiếu khi xuất submission. Dữ liệu train/test đã nối thông tin cửa hàng được lưu lần lượt trong `data/merged/train_merged.csv` và `data/merged/test_merged.csv`.

## store.csv — 1.115 bản ghi, 10 cột nguồn
`Store`, `StoreType`, `Assortment`, `CompetitionDistance`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `Promo2`, `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`.

Giá trị NA trong trường thời điểm mở cửa đối thủ hoặc Promo2 là hợp lệ (không áp dụng hoặc chưa biết); script chuẩn bị dữ liệu giữ lại hoặc mã hóa các giá trị này thay vì xóa hàng. Khoảng cách cạnh tranh thiếu được thay bằng `-1` và đánh dấu ở `CompetitionMissing`.

## Dữ liệu thời tiết ngoài
- `data/external/weather.csv`: quan sát thời tiết theo tên bang (`file`) và `Date`.
- `data/external/state_names.csv`: ánh xạ tên bang sang mã `State`.
- `data/external/store_states.csv`: ánh xạ `Store` sang mã bang.
- Script làm sạch ba tệp trên vào `data/processed/external/`, nối theo `Store → State → ngày`, rồi bổ sung vào `data/merged/train_merged.csv` và `data/merged/test_merged.csv`. `State` được thêm vào `data/processed/store.csv`.
- Các cột đo thời tiết bị thiếu được điền bằng trung vị của bang (sau đó dùng trung vị toàn bộ dữ liệu nếu bang không có giá trị). `Events` được mã hóa thành cờ mưa, tuyết, sương mù, mưa đá và giông; hướng gió được chuyển thành sin/cos. `WeatherMissingCount` giữ số phép đo thiếu trước khi điền.
- Với ngày không có quan sát thời tiết tương ứng, lúc dự đoán dùng trung vị lịch sử của bang; đây không phải dự báo thời tiết.

## Tính năng mở rộng (feature engineering)
Thứ, tháng, quý, tuần ISO, vị trí ngày trong năm, tuần hoàn thời gian; khoảng cách và thời gian hoạt động của đối thủ; thời gian Promo2 và tháng Promo2 đang hiệu lực; nhiệt độ, độ ẩm, áp suất, tầm nhìn, gió, lượng mưa, mây và sự kiện thời tiết. **Không lấy Sales hoặc Customers để tính đầu vào**, tránh lệch khả năng dự đoán test.
