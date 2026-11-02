# API Contract

Swagger: `http://127.0.0.1:8000/docs`

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/health` | Kiểm tra |
| GET | `/api/overview` | KPI toàn bộ lịch sử |
| GET | `/api/stores` | Danh sách 1115 cửa hàng |
| GET | `/api/history?store=1&days=45` | Doanh thu lịch sử |
| GET | `/api/data-profile` | Cột, missing, cảnh báo |
| GET | `/api/model-info` | Metrics từ holdout thực |
| POST | `/api/predict` | Dự đoán 1 cửa hàng 1 ngày |
| POST | `/api/predict-batch` | Dự đoán tối đa 500 bản ghi |
| GET | `/api/forecast?store=1&start_date=2015-08-01&days=14` | Dự báo nhiều ngày, dùng lịch test khi có |
| GET | `/api/submission` | Tải Id,Sales cho Kaggle |

Request `/api/predict`:
```json
{"store":1,"date":"2015-08-03","promo":1,"open":1,"state_holiday":"0","school_holiday":0}
```
Response:
```json
{"store":1,"date":"2015-08-03","predicted_sales":5500.0,"open":1,"promo":1,"currency":"EUR"}
```
Con số 5500.0 ở trên chỉ minh họa dạng dữ liệu, không phải kết quả đo.

`/api/forecast`: nếu ngày và cửa hàng có trong test.csv thì lấy Promo, Open, StateHoliday và SchoolHoliday thật từ **lịch test**. Nếu không, API ước tính `Open = 0` vào Chủ nhật và `Promo/SchoolHoliday/StateHoliday` theo giả định do user truyền; Response nêu `schedule_source` cho từng ngày. Đây là **dự đoán theo kịch bản**, không phải lịch khuyến mãi tương lai được xác minh.
