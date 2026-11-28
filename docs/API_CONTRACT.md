# API Contract

Swagger: `http://127.0.0.1:8000/docs`

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/health` | Kiểm tra |
| GET | `/api/overview` | KPI toàn bộ lịch sử |
| GET | `/api/stores` | Danh sách cửa hàng trong store.csv |
| GET | `/api/test-stores` | Danh sách cửa hàng có trong test.csv và ngày bắt đầu/kết thúc của từng cửa hàng |
| GET | `/api/actual-sales?store=1&start_date=2015-07-01&end_date=2015-07-31` | Doanh thu thực tế theo cửa hàng và khoảng ngày trong `train`; từ chối khoảng ngày ngoài biên dữ liệu |
| GET | `/api/data-profile` | Cột, missing, cảnh báo |
| GET | `/api/model-info` | Metrics từ holdout thực |
| POST | `/api/predict` | Dự đoán 1 cửa hàng 1 ngày |
| GET | `/api/test-predictions?store=1&start_date=2015-08-01&end_date=2015-08-31` | Dự đoán theo ngày trên các dòng `test.csv` trong khoảng ngày |

`GET /api/test-stores` trả cửa hàng có bản ghi trong `test.csv`, kèm `start_date` và `end_date` lấy riêng theo từng cửa hàng. `GET /api/test-predictions` yêu cầu `store`, `start_date` và `end_date`; hai ngày được tính bao gồm cả hai đầu mút và phải nằm trong phạm vi của cửa hàng đó theo `test.csv`. API lọc các dòng tương ứng trong file, tự tạo đặc trưng cửa hàng/thời gian/thời tiết bằng pipeline của mô hình đã huấn luyện, rồi trả `predicted_sales` theo ngày và tổng dự đoán. Không cần gửi đặc trưng trong request.

Response:
```json
{
  "store": 1,
  "start_date": "2015-08-01",
  "end_date": "2015-08-02",
  "total_predicted_sales": 6000.12,
  "points": [
    {"date":"2015-08-01","predicted_sales":6000.12,"open":1,"promo":0,"state_holiday":"0","school_holiday":1},
    {"date":"2015-08-02","predicted_sales":0.0,"open":0,"promo":0,"state_holiday":"0","school_holiday":1}
  ]
}
```

Giá trị trong response chỉ minh họa cấu trúc; `test.csv` không có doanh thu thực tế.

Request `/api/predict`:
```json
{"store":1,"date":"2015-08-03","promo":1,"open":1,"state_holiday":"0","school_holiday":0}
```
Response:
```json
{"store":1,"date":"2015-08-03","predicted_sales":5500.0,"open":1,"promo":1,"currency":"EUR"}
```
Con số 5500.0 ở trên chỉ minh họa dạng dữ liệu, không phải kết quả đo.

`/api/actual-sales`: trả về các ngày có bản ghi trong `train` ở khoảng ngày bao gồm cả hai đầu mút. Có thể bỏ `start_date`/`end_date` để dùng toàn bộ thời gian của `train`. API không gọi mô hình dự đoán và không tạo điểm cho ngày không có bản ghi trong `train`.
