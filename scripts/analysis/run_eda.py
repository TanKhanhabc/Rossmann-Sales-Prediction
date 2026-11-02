"""Write computed exploratory report and small CSV for plots."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.data_service import get_train, get_test, get_stores, overview

def run():
    train, test, store = get_train(), get_test(), get_stores()
    a = overview()
    month = train.assign(Month=train.Date.dt.to_period("M").astype(str)).groupby("Month").agg(Sales=("Sales", "sum"), Records=("Sales", "size"))
    month.to_csv(ROOT / "results" / "monthly_sales.csv")
    missing = store.isna().sum()
    missing["CompetitionDistance"] = int(store["CompetitionMissing"].sum())
    report = ["# Phân tích dữ liệu Rossmann (tính từ CSV gốc)", "",
        f"- Train: **{len(train):,}** bản ghi, từ {a['date_start']} đến {a['date_end']}.",
        f"- Test: **{len(test):,}** bản ghi, từ {a['test_start']} đến {a['test_end']}.",
        f"- Store: **{len(store):,}** cửa hàng; test có {test.Store.nunique():,} cửa hàng.",
        f"- Doanh thu train: **{int(train.Sales.sum()):,} EUR** (tiền tệ dựa trên mô tả bộ dữ liệu Rossmann).",
        f"- Doanh thu trung bình khi Open=1: **{a['mean_sales_open']:,.2f}**.",
        f"- Test thiếu Open: **{a['missing_test_open']}** hàng (cửa hàng 622).",
        "", "## Thiếu dữ liệu trong store.csv", ""]
    report += [f"- {col}: {int(n):,} ({n/len(store)*100:.1f}%)" for col,n in missing.items() if n]
    report += ["", "## Quy ước mô hình", "",
        "- **Sales** là biến mục tiêu, đơn vị doanh thu cửa hàng theo ngày.",
        "- Không sử dụng **Customers** làm đầu vào vì test.csv không có trường này.",
        "- Dữ liệu validation chia theo thời gian, không xáo trộn ngẫu nhiên.",
        "- Cửa hàng đóng cửa => dự đoán 0; các ngày Open thiếu được nội suy theo thứ trong tuần.",
        "- Dự báo ngoài khoảng test.csv là kịch bản theo giả định, không phải lịch khuyến mại thực.",
        "", "## Doanh thu theo tháng (5 tháng cuối)", "", month.tail().to_markdown(), ""]
    path = ROOT / "results" / "EDA_REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print('Written',path)

if __name__ == "__main__":
    run()
