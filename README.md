# Rossmann Sales Forecast — Dự đoán doanh thu chuỗi cửa hàng

Một **monorepo hoàn chỉnh** dựa trên cách chia thư mục của [CoffeeInvoicePrediction](https://github.com/TanKhanhabc/CoffeeInvoicePrediction), nhưng thay bài toán **đếm hóa đơn** bằng **dự đoán Sales của từng cửa hàng mỗi ngày**. Dùng ba CSV mà bạn cung cấp, không phải dữ liệu giả.

## Đã có gì?
- **Dataset gốc**: `data/raw/train.csv` (1.017.209 dòng, 9 cột), `data/raw/test.csv` (41.088 dòng, 8 cột), `data/raw/store.csv` (1.115 dòng, 10 cột).
- **EDA**: báo cáo theo tháng, thiếu dữ liệu, kiểm tra thời gian, không nhầm `Customers` với doanh thu.
- **LightGBM**: đặc trưng cửa hàng, lịch và thời tiết theo bang/ngày; validation **42 ngày cuối** (MAE, RMSE, RMSPE, WAPE), baseline dự đoán theo Store/Weekday/Promo. Lưu model tại `models/rossmann_sales_lgbm.joblib`.
- **FastAPI**: overview, stores, lịch sử, đánh giá, dự đoán 1 ngày và dự báo 7/14/30 ngày theo lịch test.
- **React + Vite + TypeScript + Recharts**: Dashboard tiếng Việt, biểu đồ và dự đoán trực tiếp từ ML API (không vẽ dữ liệu mô phỏng).
- Docker Compose, unit tests, notebooks và hướng dẫn.

**LƯU Ý QUAN TRỌNG:** Dữ liệu Rossmann từ **2013–2015**. Khoảng test là **2015-08-01 đến 2015-09-17**. Dự báo ngoài khoảng ngày có lịch test sẽ dùng **giả định kịch bản**, không phải lịch khuyến mãi thực hay kết quả live năm 2026. `Sales` đo bằng EUR.

## 1. Cấu trúc thư mục
```text
RossmannSalesForecast/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/ (data.py, predictions.py)
│   │   ├── services/ (data_service.py)
│   │   └── model/ (features.py, train_model.py, evaluate_model.py, predict_model.py)
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/ (App.tsx, api.ts, styles.css, main.tsx)
│   ├── package.json
│   └── Dockerfile
├── data/raw/ (train.csv, test.csv, store.csv)
├── data/external/ (weather.csv, state_names.csv, store_states.csv)
├── data/processed/ (cleaned CSVs and preprocessing report)
├── data/merged/ (train_merged.csv and test_merged.csv)
├── models/ (rossmann_sales_lgbm.joblib)
├── results/ (evaluation_metrics.json, EDA_REPORT.md, monthly_sales.csv)
├── scripts/
│   ├── preprocessing/prepare_data.py
│   ├── analysis/run_eda.py
│   ├── evaluation/run_evaluation.py
├── notebooks/ (exploratory/eda.ipynb, ML_Experiment.ipynb)
├── docs/ (DATA_DICTIONARY.md, ML_REPORT.md, API_CONTRACT.md, TEAM_ASSIGNMENT.md)
├── docker-compose.yml
└── start_windows.ps1
```

## 2. Chạy trên Windows (VS Code / PowerShell)
Yêu cầu: **Python 3.11–3.12**, **Node.js 20+**. Giải nén, mở VS Code tại thư mục `RossmannSalesForecast`.

**Terminal 1 — API** (ở gốc dự án):
```powershell
py -m pip install -r backend/requirements.txt
cd backend
py -m uvicorn app.main:app --reload --port 8000
```
API Swagger: http://localhost:8000/docs

**Terminal 2 — Dashboard** (ở gốc dự án):
```powershell
cd frontend
npm install
npm run dev
```
Giao diện: **http://localhost:5173**. Vite proxy `/api` sang FastAPI ở cổng 8000.

Hoặc chạy `./start_windows.ps1` khi PowerShell policy cho phép.

## 3. Train lại model / đánh giá
Ở thư mục gốc dự án (không đứng trong backend), chuẩn bị dữ liệu một lần sau khi thêm CSV vào `data/raw/`:
```powershell
py scripts/preprocessing/prepare_data.py
py scripts/analysis/run_eda.py
py scripts/evaluation/run_evaluation.py
py -m pytest backend/tests -q
```
Script lưu CSV đã xử lý và dữ liệu ngoài đã làm sạch trong `data/processed/`, các file `train_merged.csv` và `test_merged.csv` (đã bổ sung thời tiết) trong `data/merged/`, và không sửa các file nguồn. Thời tiết được ánh xạ qua mã bang và ngày; giá trị đo bị thiếu được điền bằng trung vị của bang. Khi dự đoán ngày không có quan sát thời tiết, mô hình dùng trung vị lịch sử của bang. Chạy lại script khi thay đổi CSV nguồn. Khi dùng `start_windows.ps1`, bước chuẩn bị này được chạy tự động; Docker cũng chuẩn bị dữ liệu trong lúc build image.
Mô hình đã được train trên CSV thật và kèm trong ZIP; bạn không cần train để xem giao diện. Tập validation 42 ngày sẽ **không được dùng trong quá trình fit mô hình khi tính metric**. Sau khi chấm điểm, mô hình lưu cho dự báo test được huấn luyện lại trên **toàn bộ lịch sử**.

### Kết quả đánh giá đã đo trên ba dataset gửi kèm

- **LightGBM có đặc trưng thời tiết** (validation 42 ngày cuối): MAE **509,82 EUR**, RMSE **811,59 EUR**, RMSPE **0,12134** (~12,13%), WAPE **8,49%**.
- **Baseline** lịch sử Store × DayOfWeek × Promo: MAE **639,99 EUR**, RMSE **1.028,21 EUR**.
- Mô hình cuối đã được train lại với toàn bộ 844.338 dòng cửa hàng mở có doanh thu > 0. Metrics đo trên holdout chỉ áp dụng cho mô hình trước khi train lại.

## 4. Run Docker
Đã có mô hình và kết quả trong ZIP. Từ thư mục gốc:
```powershell
docker compose up --build
```
- Frontend: http://localhost:5173
- API: http://localhost:8000/docs

Dockerfiles đã được cung cấp và kiểm tra cú pháp; nếu chưa có Docker Engine hoặc không chạy Docker trên máy thì hãy chạy Python/Node như phần 2.

## 5. Ý nghĩa màn hình
- **Tổng quan**: doanh thu thực trong train, số cửa hàng và số bản ghi.
- **Dự báo theo cửa hàng**: chọn mã Store, ngày bắt đầu, 7/14/30 ngày. Nếu lịch nằm trong test.csv thì dùng Promo/Open/... từ chính test; ngoài khoảng đó là giả định.
- **Dự đoán kịch bản**: chọn Store/Date/Open/Promo/StateHoliday/SchoolHoliday và chạy mô hình thực.
- **Hiệu suất**: điểm kiểm định MAE/RMSE/RMSPE và baseline.
- **Dữ liệu**: tình trạng 3 dataset và nguồn dữ liệu dự báo.

## 6. Lưu ý ML
- **Sales** là doanh thu trên một cửa hàng và một ngày (không phải số lượng hóa đơn).
- **Customers** nằm trong train.csv nhưng không nằm trong test.csv: **không đưa vào mô hình** để tránh sai lệch khi dự báo.
- `Open=0` → Sales = 0. Có **11 Open còn thiếu** trong test; dự án điền theo thứ trong tuần (giả định).
- Dùng chronological holdout: tránh chia train/test ngẫu nhiên theo hàng (time leakage).
- **Không khẳng định điểm Kaggle public/private score**; không có nhãn Sales thật trong test để đo.
- Dự báo nhiều ngày trên dashboard là các dự đoán từng ngày, **không tự tăng số lượng theo công thức trang trí**.

Chi tiết: `docs/ML_REPORT.md`, `docs/API_CONTRACT.md`, `docs/DATA_DICTIONARY.md`.
