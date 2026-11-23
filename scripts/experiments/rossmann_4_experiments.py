import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
# CẤU HÌNH HỆ THỐNG
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEMP_MODEL_DIR = PROJECT_ROOT / "temp_models_full"
RESULTS_DIR = PROJECT_ROOT / "results"
TEMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def compute_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính chỉ số RMSPE chuẩn của Rossmann."""
    mask = y_true > 0
    y_t = y_true[mask]
    y_p = np.maximum(y_pred[mask], 0)
    return float(np.sqrt(np.mean(((y_t - y_p) / y_t) ** 2)))


# ==============================================================================
# 1. CHUẨN BỊ DỮ LIỆU & FEATURE ENGINEERING KHÔNG RÒ RỈ
# ==============================================================================
def load_and_prepare_data(data_dir: str):
    print("1. Đang tải và xử lý dữ liệu Rossmann...")
    train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    store_raw = pd.read_csv(os.path.join(data_dir, "store.csv"))

    df = pd.merge(train_raw, store_raw, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["StateHoliday"] = df["StateHoliday"].astype(str)

    # Tính trạng thái liền kề TRƯỚC khi lọc Open == 0
    df["Open"] = df["Open"].fillna(1)
    df["IsOpenYesterday"] = df.groupby("Store")["Open"].shift(1).fillna(1).astype(int)
    df["IsOpenTomorrow"] = df.groupby("Store")["Open"].shift(-1).fillna(1).astype(int)

    # Lọc các ngày mở cửa có phát sinh doanh thu
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Đặc trưng thời gian cơ bản & nâng cao
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Đặc trưng thời gian mở cửa của đối thủ & khuyến mãi dài hạn
    df["CompetitionOpenMonths"] = (
        12 * (df["Year"] - df["CompetitionOpenSinceYear"])
        + (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].apply(
        lambda x: max(x, 0) if pd.notnull(x) else np.nan
    )

    df["Promo2OpenMonths"] = (
        12 * (df["Year"] - df["Promo2SinceYear"])
        + (df["WeekOfYear"] - df["Promo2SinceWeek"]) / 4.345
    )
    df["Promo2OpenMonths"] = np.where(
        (df["Promo2"] == 1) & (df["Promo2OpenMonths"] > 0),
        df["Promo2OpenMonths"],
        0.0,
    )

    # Sắp xếp lại theo thời gian và chia 42 ngày cuối làm Validation
    df = df.sort_values(["Date", "Store"]).reset_index(drop=True)
    split_date = df["Date"].max() - pd.Timedelta(days=42)

    train_df = df[df["Date"] <= split_date].copy()
    val_df = df[df["Date"] > split_date].copy()
    print(f"   -> Train: {len(train_df):,} dòng | Validation: {len(val_df):,} dòng (42 ngày cuối)")
    return train_df, val_df


# ==============================================================================
# 2. HÀM HỖ TRỢ HUẤN LUYỆN & GHI NHẬN KẾT QUẢ
# ==============================================================================
def build_sklearn_preprocessor(num_cols, cat_cols, encoder_type="ordinal"):
    if encoder_type == "target":
        cat_encoder = TargetEncoder(random_state=RANDOM_STATE)
    else:
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", cat_encoder),
                ]),
                cat_cols,
            ),
        ]
    )


def run_all_experiments(train_df: pd.DataFrame, val_df: pd.DataFrame):
    records = []

    base_cat = ["Store", "DayOfWeek", "StateHoliday", "StoreType", "Assortment"]
    base_num = ["Promo", "SchoolHoliday", "CompetitionDistance", "Year", "Month", "Day"]
    base_feats = base_cat + base_num

    adv_cat = base_cat
    adv_num = base_num + [
        "IsOpenYesterday",
        "IsOpenTomorrow",
        "DayOfYear",
        "WeekOfYear",
        "CompetitionOpenMonths",
        "Promo2OpenMonths",
    ]
    adv_feats = adv_cat + adv_num

    y_train_log = np.log1p(train_df["Sales"].values)
    y_val_raw = val_df["Sales"].values
    y_val_log = np.log1p(y_val_raw)

    def evaluate_model(exp_label, model_name, model_obj, X_v, train_t, preds_raw=None):
        t0 = time.perf_counter()
        if preds_raw is None:
            preds_raw = np.expm1(model_obj.predict(X_v))
        infer_ms = (time.perf_counter() - t0) * 1000.0
        score = compute_rmspe(y_val_raw, preds_raw)

        # Lưu tạm để đo dung lượng file
        fname = f"{exp_label[:4]}_{model_name.replace(' ', '_')}.joblib"
        fpath = os.path.join(TEMP_MODEL_DIR, fname)
        joblib.dump(model_obj, fpath, compress=3)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)

        records.append({
            "Thực nghiệm": exp_label,
            "Mô hình": model_name,
            "RMSPE": round(score, 5),
            "Train Time (s)": round(train_t, 2),
            "Inference (ms)": round(infer_ms, 2),
            "Size (MB)": round(size_mb, 3),
        })
        return preds_raw

    # ==========================================================================
    # THỰC NGHIỆM 1: BASELINE BENCHMARK (5 MÔ HÌNH, DỮ LIỆU GỐC, ORDINAL ENCODING)
    # ==========================================================================
    exp1_name = "TN1: Baseline (Dữ liệu gốc)"
    print(f"\n>>> Đang chạy {exp1_name}...")
    prep_exp1 = build_sklearn_preprocessor(base_num, base_cat, encoder_type="ordinal")

    models_exp1 = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }

    for m_name, est in models_exp1.items():
        pipe = Pipeline([("prep", prep_exp1), ("model", est)])
        t0 = time.perf_counter()
        pipe.fit(train_df[base_feats], y_train_log)
        t_train = time.perf_counter() - t0
        evaluate_model(exp1_name, m_name, pipe, val_df[base_feats], t_train)

    # ==========================================================================
    # THỰC NGHIỆM 2: CATEGORICAL HANDLING (TARGET ENCODER VS NATIVE CATEGORY)
    # ==========================================================================
    exp2_name = "TN2: Tối ưu Categorical"
    print(f"\n>>> Đang chạy {exp2_name}...")
    prep_exp2 = build_sklearn_preprocessor(base_num, base_cat, encoder_type="target")

    models_exp2_sk = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    }

    for m_name, est in models_exp2_sk.items():
        pipe = Pipeline([("prep", prep_exp2), ("model", est)])
        t0 = time.perf_counter()
        pipe.fit(train_df[base_feats], y_train_log)
        t_train = time.perf_counter() - t0
        evaluate_model(exp2_name, m_name, pipe, val_df[base_feats], t_train)

    # LightGBM sử dụng Native Categorical Split (Fisher's Exact Test)
    X_tr_lgb2 = train_df[base_feats].copy()
    X_va_lgb2 = val_df[base_feats].copy()
    for c in base_cat:
        X_tr_lgb2[c] = X_tr_lgb2[c].astype("category")
        X_va_lgb2[c] = X_va_lgb2[c].astype("category")

    lgb_exp2 = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    t0 = time.perf_counter()
    lgb_exp2.fit(X_tr_lgb2, y_train_log)
    t_train = time.perf_counter() - t0
    evaluate_model(exp2_name, "LightGBM", lgb_exp2, X_va_lgb2, t_train)

    # ==========================================================================
    # THỰC NGHIỆM 3: FEATURE ENGINEERING BÁN LẺ & TUNING CHO CẢ 5 MÔ HÌNH
    # ==========================================================================
    exp3_name = "TN3: FE Bán lẻ & Tuning"
    print(f"\n>>> Đang chạy {exp3_name}...")
    prep_exp3 = build_sklearn_preprocessor(adv_num, adv_cat, encoder_type="target")

    models_exp3_sk = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=16, min_samples_leaf=20, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE),
    }

    exp3_predictions = {}  # Lưu lại dự đoán của TN3 để chạy tiếp TN4

    for m_name, est in models_exp3_sk.items():
        pipe = Pipeline([("prep", prep_exp3), ("model", est)])
        t0 = time.perf_counter()
        pipe.fit(train_df[adv_feats], y_train_log)
        t_train = time.perf_counter() - t0
        preds = evaluate_model(exp3_name, m_name, pipe, val_df[adv_feats], t_train)
        exp3_predictions[m_name] = (pipe, val_df[adv_feats], preds, t_train)

    # LightGBM với bộ đặc trưng nâng cao + Tuning + Early Stopping
    X_tr_lgb3 = train_df[adv_feats].copy()
    X_va_lgb3 = val_df[adv_feats].copy()
    for c in adv_cat:
        X_tr_lgb3[c] = X_tr_lgb3[c].astype("category")
        X_va_lgb3[c] = X_va_lgb3[c].astype("category")

    lgb_exp3 = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    lgb_exp3.fit(
        X_tr_lgb3,
        y_train_log,
        eval_set=[(X_va_lgb3, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    t_train_lgb3 = time.perf_counter() - t0
    preds_lgb3 = evaluate_model(exp3_name, "LightGBM", lgb_exp3, X_va_lgb3, t_train_lgb3)
    exp3_predictions["LightGBM"] = (lgb_exp3, X_va_lgb3, preds_lgb3, t_train_lgb3)

    # ==========================================================================
    # THỰC NGHIỆM 4: POST-PROCESSING MULTIPLIER (TỐI ƯU HỆ SỐ ALPHA CHO CẢ 5)
    # ==========================================================================
    exp4_name = "TN4: Post-processing (RMSPE)"
    print(f"\n>>> Đang chạy {exp4_name}...")
    alphas = np.arange(0.980, 1.001, 0.002)

    for m_name, (model_obj, X_v, base_preds, t_train) in exp3_predictions.items():
        best_alpha = 1.0
        best_score = compute_rmspe(y_val_raw, base_preds)
        for a in alphas:
            sc = compute_rmspe(y_val_raw, base_preds * a)
            if sc < best_score:
                best_score = sc
                best_alpha = round(float(a), 4)

        print(f"   -> {m_name:15s}: best_alpha = {best_alpha:.4f} | RMSPE = {best_score:.5f}")
        evaluate_model(
            exp4_name,
            m_name,
            model_obj,
            X_v,
            t_train,
            preds_raw=base_preds * best_alpha,
        )

    return pd.DataFrame(records)


# ==============================================================================
# 3. TRỰC QUAN HÓA MA TRẬN 4 THỰC NGHIỆM X 5 MÔ HÌNH
# ==============================================================================
def plot_4x5_comparison(
    df_res: pd.DataFrame,
    save_path: str | Path = RESULTS_DIR / "rossmann_4x5_benchmark.png",
):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Biểu đồ 1: Biểu đồ đường (Line Plot) theo dõi sự giảm RMSPE qua 4 thực nghiệm
    palette = {
        "Dummy (Median)": "#95a5a6",
        "Ridge": "#3498db",
        "Decision Tree": "#e67e22",
        "Random Forest": "#9b59b6",
        "LightGBM": "#2ecc71",
    }

    sns.lineplot(
        data=df_res,
        x="Thực nghiệm",
        y="RMSPE",
        hue="Mô hình",
        style="Mô hình",
        markers=True,
        dashes=False,
        linewidth=2.8,
        markersize=9,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Quỹ đạo giảm sai số RMSPE qua 4 Thực nghiệm (Thấp hơn = Tốt hơn)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RMSPE (Validation)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    # Gắn nhãn giá trị cho riêng đường LightGBM để làm nổi bật
    lgb_rows = df_res[df_res["Mô hình"] == "LightGBM"]
    for idx, row in enumerate(lgb_rows.itertuples()):
        axes[0].text(idx, row.RMSPE - 0.015, f"{row.RMSPE:.4f}", color="#1e8449", fontweight="bold", ha="center", fontsize=9)

    # Biểu đồ 2: So sánh RMSPE cuối cùng ở Thực nghiệm 4 (Kết quả chung cuộc)
    final_df = df_res[df_res["Thực nghiệm"] == "TN4: Post-processing (RMSPE)"].sort_values("RMSPE", ascending=False)
    bar_colors = [palette[m] for m in final_df["Mô hình"]]

    sns.barplot(
        data=final_df,
        x="RMSPE",
        y="Mô hình",
        palette=bar_colors,
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("Kết quả chung cuộc tại Thực nghiệm 4 (RMSPE)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("RMSPE")
    axes[1].set_ylabel("")
    for i, v in enumerate(final_df["RMSPE"]):
        axes[1].text(v + 0.005, i, f"{v:.5f}", va="center", fontweight="bold", fontsize=9.5)
    axes[1].set_xlim(0, final_df["RMSPE"].max() * 1.16)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nĐã lưu biểu đồ tổng hợp tại: {save_path}")
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    train_data, val_data = load_and_prepare_data(str(DATA_DIR))
    df_results = run_all_experiments(train_data, val_data)
    results_csv = RESULTS_DIR / "rossmann_4_experiments_metrics.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nĐã lưu bảng kết quả tại: {results_csv}")

    # In bảng Pivot so sánh chéo RMSPE (4 Thực nghiệm x 5 Mô hình)
    pivot_rmspe = df_results.pivot(index="Mô hình", columns="Thực nghiệm", values="RMSPE")
    pivot_rmspe = pivot_rmspe.sort_values("TN4: Post-processing (RMSPE)", ascending=False)

    print("\n" + "=" * 95)
    print("BẢNG MA TRẬN SO SÁNH RMSPE QUA 4 THỰC NGHIỆM (CÀNG THẤP CÀNG TỐT)")
    print("=" * 95)
    print(pivot_rmspe.to_string())

    print("\n" + "=" * 95)
    print("CHI TIẾT HIỆU NĂNG VẬN HÀNH TẠI THỰC NGHIỆM 4 (CHUNG CUỘC)")
    print("=" * 95)
    final_metrics = df_results[df_results["Thực nghiệm"] == "TN4: Post-processing (RMSPE)"].sort_values("RMSPE")
    print(final_metrics[["Mô hình", "RMSPE", "Train Time (s)", "Inference (ms)", "Size (MB)"]].to_string(index=False))
    print("=" * 95)

    plot_4x5_comparison(df_results)