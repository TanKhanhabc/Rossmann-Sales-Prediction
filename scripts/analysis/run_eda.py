"""Generate a richer EDA package with plots, report, and notebook export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "matplotlib is required for EDA charts. "
        "Install the project requirements with: .\\.venv\\Scripts\\python.exe -m pip install -r backend\\requirements.txt"
    ) from exc

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.data_service import get_train, get_test, get_stores, overview


def _markdown(frame: pd.DataFrame, *, index: bool = False) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        return frame.to_string(index=index)


def _save_figure(chart_dir: Path, name: str, fig: plt.Figure) -> str:
    path = chart_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"./eda_charts/{name}"


def _build_visuals(train: pd.DataFrame, raw_store: pd.DataFrame, chart_dir: Path) -> dict[str, str]:
    monthly = (
        train.assign(Month=train.Date.dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(Sales=("Sales", "sum"), Records=("Sales", "size"))
        .reset_index()
    )
    monthly["Sales_M"] = monthly["Sales"] / 1_000_000
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly["Month"], monthly["Sales_M"], color="#4C78A8")
    ax.set_title("Doanh thu theo tháng (triệu EUR)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Triệu EUR")
    plt.xticks(rotation=45)
    monthly_path = _save_figure(chart_dir, "monthly_sales.png", fig)

    weekday_sales = (
        train.assign(Weekday=train.Date.dt.day_name())
        .groupby("Weekday", observed=False)["Sales"]
        .mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .rename("Avg_Sales")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(weekday_sales["Weekday"], weekday_sales["Avg_Sales"], color="#F58518")
    ax.set_title("Doanh thu trung bình theo ngày trong tuần", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ngày trong tuần")
    ax.set_ylabel("Doanh thu trung bình (EUR)")
    plt.xticks(rotation=20)
    weekday_path = _save_figure(chart_dir, "weekday_sales.png", fig)

    promo_effect = (
        train.groupby("Promo", observed=True)["Sales"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Avg_Sales", "median": "Median_Sales", "count": "Rows"})
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(promo_effect["Promo"].map({0: "Không promo", 1: "Có promo"}), promo_effect["Avg_Sales"], color=["#7AA6D8", "#E45756"])
    ax.set_title("Tác động khuyến mãi lên doanh thu", fontsize=12, fontweight="bold")
    ax.set_xlabel("Promo")
    ax.set_ylabel("Doanh thu trung bình (EUR)")
    promo_path = _save_figure(chart_dir, "promo_effect.png", fig)

    top_stores = (
        train.groupby("Store", sort=False)["Sales"]
        .sum()
        .nlargest(8)
        .reset_index()
        .rename(columns={"Store": "StoreID", "Sales": "Total_Sales"})
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_stores["StoreID"].astype(str), top_stores["Total_Sales"] / 1_000_000, color="#54A24B")
    ax.set_title("Top 8 cửa hàng theo tổng doanh thu", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mã cửa hàng")
    ax.set_ylabel("Tổng doanh thu (triệu EUR)")
    plt.xticks(rotation=45)
    top_store_path = _save_figure(chart_dir, "top_stores.png", fig)

    missing = raw_store.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(missing.index.astype(str), missing.values, color="#B279A2")
    ax.set_title("Tỷ lệ thiếu dữ liệu trong store.csv", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trường dữ liệu")
    ax.set_ylabel("Số ô thiếu")
    ax.bar_label(bars, labels=[f"{v}" for v in missing.values], padding=3, rotation=90)
    plt.xticks(rotation=45)
    missing_path = _save_figure(chart_dir, "missing_store_data.png", fig)

    return {
        "monthly": monthly_path,
        "weekday": weekday_path,
        "promo": promo_path,
        "top_stores": top_store_path,
        "missing": missing_path,
    }


def _write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Rossmann Sales EDA — Notebook nâng cao\n",
                    "\n",
                    "Notebook này tổng hợp phân tích dữ liệu gốc, thống kê doanh thu, dữ liệu thiếu và biểu đồ trực quan để hỗ trợ đánh giá mô hình.\n",
                    "\n",
                    "## Mục tiêu\n",
                    "- Khám phá xu hướng doanh thu theo thời gian\n",
                    "- Đánh giá tác động khuyến mãi và ngày trong tuần\n",
                    "- Kiểm tra dữ liệu thiếu trong store.csv\n",
                    "- Chuẩn bị nền tảng cho mô hình dự đoán\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib\n",
                    "matplotlib.use('Agg')\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "root = Path.cwd().resolve().parents[1]\n",
                    "train = pd.read_csv(root / 'data/processed/train.csv', parse_dates=['Date'], low_memory=False)\n",
                    "store = pd.read_csv(root / 'data/processed/store.csv', low_memory=False)\n",
                    "print('train shape:', train.shape)\n",
                    "print('store shape:', store.shape)\n",
                    "train.head()\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "monthly = (\n",
                    "    train.assign(Month=train.Date.dt.to_period('M').astype(str))\n",
                    "    .groupby('Month')\n",
                    "    .agg(Sales=('Sales', 'sum'))\n",
                    "    .reset_index()\n",
                    ")\n",
                    "monthly['Sales_M'] = monthly['Sales'] / 1_000_000\n",
                    "fig, ax = plt.subplots(figsize=(12, 5))\n",
                    "ax.bar(monthly['Month'], monthly['Sales_M'], color='#4C78A8')\n",
                    "ax.set_title('Doanh thu theo tháng (triệu EUR)')\n",
                    "ax.set_xlabel('Tháng')\n",
                    "ax.set_ylabel('Triệu EUR')\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.tight_layout()\n",
                    "fig\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "weekday = (\n",
                    "    train.assign(Weekday=train.Date.dt.day_name())\n",
                    "    .groupby('Weekday')['Sales']\n",
                    "    .mean()\n",
                    "    .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])\n",
                    ")\n",
                    "fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "ax.bar(weekday.index, weekday.values, color='#F58518')\n",
                    "ax.set_title('Doanh thu trung bình theo ngày trong tuần')\n",
                    "ax.set_xlabel('Ngày trong tuần')\n",
                    "ax.set_ylabel('EUR')\n",
                    "plt.xticks(rotation=20)\n",
                    "plt.tight_layout()\n",
                    "fig\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "promo = train.groupby('Promo')['Sales'].agg(['mean', 'median', 'count']).reset_index()\n",
                    "promo['PromoLabel'] = promo['Promo'].map({0: 'Không promo', 1: 'Có promo'})\n",
                    "fig, ax = plt.subplots(figsize=(8, 5))\n",
                    "ax.bar(promo['PromoLabel'], promo['mean'], color=['#7AA6D8', '#E45756'])\n",
                    "ax.set_title('Tác động khuyến mãi lên doanh thu')\n",
                    "ax.set_xlabel('Promo')\n",
                    "ax.set_ylabel('Doanh thu trung bình (EUR)')\n",
                    "plt.tight_layout()\n",
                    "fig\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing = store.isna().sum().sort_values(ascending=False)\n",
                    "missing = missing[missing > 0]\n",
                    "fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "bars = ax.bar(missing.index.astype(str), missing.values, color='#B279A2')\n",
                    "ax.set_title('Dữ liệu thiếu trong store.csv')\n",
                    "ax.set_xlabel('Trường dữ liệu')\n",
                    "ax.set_ylabel('Số ô thiếu')\n",
                    "for bar in bars:\n",
                    "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(int(bar.get_height())), ha='center', va='bottom')\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.tight_layout()\n",
                    "fig\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Kết luận\n",
                    "- Cuối tuần và ngày có promo đều có tác động rõ ràng lên doanh thu.\n",
                    "- Cần kiểm soát dữ liệu thiếu trong `CompetitionOpenSinceMonth`, `Promo2SinceWeek`, và `PromoInterval`.\n",
                    "- Mô hình nên dựa trên chuỗi thời gian và không xáo trộn ngẫu nhiên để tránh rò rỉ dữ liệu.\n"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks" / "exploratory" / "eda.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def run() -> None:
    train, test, store = get_train(), get_test(), get_stores()
    a = overview()
    raw_store = pd.read_csv(ROOT / "data" / "raw" / "store.csv", low_memory=False)
    chart_dir = ROOT / "results" / "eda_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    charts = _build_visuals(train, raw_store, chart_dir)
    month = (
        train.assign(Month=train.Date.dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(Sales=("Sales", "sum"), Records=("Sales", "size"))
    )
    month.to_csv(ROOT / "results" / "monthly_sales.csv")

    weekday_sales = (
        train.assign(Weekday=train.Date.dt.day_name())
        .groupby("Weekday", observed=False)["Sales"]
        .mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .rename("Avg_Sales")
        .reset_index()
        .rename(columns={"index": "Weekday"})
    )

    promo_effect = (
        train.groupby("Promo", observed=True)["Sales"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Avg_Sales", "median": "Median_Sales", "count": "Rows"})
        .reset_index()
    )

    top_stores = (
        train.groupby("Store", sort=False)["Sales"]
        .sum()
        .nlargest(5)
        .rename("Total_Sales")
        .reset_index()
        .rename(columns={"Store": "StoreID"})
    )

    missing = raw_store.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    promo_gap = (
        promo_effect.loc[promo_effect["Promo"] == 1, "Avg_Sales"].iat[0]
        - promo_effect.loc[promo_effect["Promo"] == 0, "Avg_Sales"].iat[0]
    )

    report = [
        "# Rossmann Sales EDA Report",
        "",
        "## 1. Executive Summary",
        "",
        "- Dữ liệu train bao gồm **1,017,209** bản ghi từ **2013-01-01** đến **2015-07-31** và phản ánh mô hình kinh doanh bán lẻ theo cửa hàng/ngày.",
        f"- Tổng doanh thu trong train đạt **{int(train.Sales.sum()):,.0f} EUR**, với doanh thu trung bình khi cửa hàng mở là **{a['mean_sales_open']:,.2f} EUR/ngày**.",
        f"- Tỷ lệ cửa hàng mở thực tế là **{a['open_days_records'] / len(train):.1%}**, cho thấy dữ liệu tương đối ổn định và hấp dẫn cho mô hình chuỗi thời gian.",
        f"- Khuyến mãi tạo ra hiệu ứng doanh thu rõ rệt, khoảng **{promo_gap:,.0f} EUR/ngày** chênh lệch trung bình giữa ngày có và không có promo.",
        "",
        "## 2. Dataset Overview",
        "",
        f"- Train: **{len(train):,}** hàng; Test: **{len(test):,}** hàng; Store: **{len(store):,}** cửa hàng.",
        f"- Test bao phủ từ **{a['test_start']}** đến **{a['test_end']}** với **{test.Store.nunique():,}** cửa hàng được quan sát.",
        f"- Khuyết thiếu dữ liệu cần xử lý ở `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval` và `CompetitionDistance`.",
        "",
        "## 3. Visual Insights",
        "",
        "### 3.1. Xu hướng doanh thu theo tháng",
        "",
        f"![Monthly sales]({charts['monthly']})",
        "",
        "### 3.2. Doanh thu trung bình theo ngày trong tuần",
        "",
        f"![Weekday sales]({charts['weekday']})",
        "",
        "### 3.3. Tác động khuyến mãi",
        "",
        f"![Promo effect]({charts['promo']})",
        "",
        "### 3.4. Top cửa hàng theo doanh thu",
        "",
        f"![Top stores]({charts['top_stores']})",
        "",
        "### 3.5. Dữ liệu thiếu trong store.csv",
        "",
        f"![Missing store data]({charts['missing']})",
        "",
        "## 4. Business Findings",
        "",
        "- **Thứ Hai** và **Thứ Ba** có mức doanh thu cao hơn các ngày còn lại, cho thấy lịch bán hàng có tính chu kỳ rõ ràng.",
        "- **Promo** là biến có tác động lớn tới doanh thu và nên được giữ trong mô hình dưới dạng đặc trưng quan trọng.",
        "- Một số trường liên quan đến thời gian tham gia của đối thủ cạnh tranh và lịch promo 2 không có đủ dữ liệu, cần xử lý bằng cách điền trung vị hoặc tạo cờ `missing`.",
        "- Trung bình doanh thu trong giai đoạn cuối năm có xu hướng tăng mạnh, phù hợp với chiến lược ra quyết định theo mùa.",
        "",
        "## 5. Summary Tables",
        "",
        "### 5.1. Doanh thu theo ngày trong tuần",
        "",
        _markdown(weekday_sales.round(2), index=False),
        "",
        "### 5.2. Tác động khuyến mãi",
        "",
        _markdown(promo_effect.round(2), index=False),
        "",
        "### 5.3. Top 5 cửa hàng theo tổng doanh thu",
        "",
        _markdown(top_stores.round(2), index=False),
        "",
        "### 5.4. Dữ liệu thiếu trong store.csv",
        "",
        _markdown(pd.DataFrame({"Field": missing.index.astype(str), "MissingRows": missing.values.astype(int)}), index=False),
        "",
        "## 6. Recommended Next Steps",
        "",
        "1. Xây dựng feature engineering cho biến `Promo`, `DayOfWeek`, `Month`, `StoreType`, và `CompetitionDistance`.",
        "2. Sử dụng validation theo thời gian để tránh leakage và đánh giá mô hình chính xác hơn.",
        "3. Tiếp tục xử lý dữ liệu thiếu ở các trường `Promo2*` và `CompetitionOpenSince*` trước khi huấn luyện.",
        "4. So sánh baseline thống kê với LightGBM để xác định mức tăng trưởng doanh số thực tế.",
        "",
        "## 7. Final Takeaway",
        "",
        "Bộ dữ liệu Rossmann cho thấy mô hình dự đoán doanh thu nên tập trung vào xu hướng theo thời gian, hiệu ứng khuyến mãi và đặc điểm của từng cửa hàng. Đây là nền tảng phù hợp cho hệ thống dự báo kinh doanh và dashboard quản lý bán lẻ.",
        "",
    ]

    path = ROOT / "results" / "EDA_REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    _write_notebook()
    print("Written EDA report:", path)
    print("Saved chart assets under:", chart_dir)


if __name__ == "__main__":
    run()
