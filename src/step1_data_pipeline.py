"""
Step 1 – Data Cleaning & Feature Engineering
IMPACT Framework: M – Mastering the Data
"""

import pandas as pd
import numpy as np
import os


# ══════════════════════════════════════════════════════════
# STEP 1 – LOAD
# ══════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  [LOAD] {len(df):,} rows × {df.shape[1]} cols  ← '{os.path.basename(path)}'")
    return df


# ══════════════════════════════════════════════════════════
# STEP 2 – DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════
def quality_report(df: pd.DataFrame, save_path: str):
    lines = ["="*55,
             "  DATA QUALITY REPORT – retail_store_inventory.csv",
             "="*55]

    lines.append(f"\n[1] Shape      : {len(df):,} rows × {df.shape[1]} cols")
    lines.append(f"[2] Duplicates : {int(df.duplicated().sum())} rows")

    lines.append("\n[3] Missing values:")
    for col in df.columns:
        n   = int(df[col].isnull().sum())
        pct = n / len(df) * 100
        lines.append(f"    {col:<25} {'✓ OK' if n==0 else f'⚠ {n} ({pct:.1f}%)'}")

    numeric_cols = df.select_dtypes(include="number").columns
    lines.append("\n[4] Outliers (IQR):")
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = int(((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum())
        lines.append(f"    {col:<25} {n_out:>5} rows ({n_out/len(df)*100:.2f}%)")

    lines.append("\n[5] Logical checks:")
    lines.append(f"    Units Sold > Inventory Level : {int((df['Units Sold']>df['Inventory Level']).sum())} rows")
    lines.append(f"    Negative Units Sold          : {int((df['Units Sold']<0).sum())} rows")
    lines.append(f"    Discount values              : {sorted(df['Discount'].unique().tolist())}")
    lines.append("\n[6] Seasonality column:")
    lines.append("    ⚠ All 4 seasons appear in every calendar month (synthetic data)")
    lines.append("    → Will engineer 'calendar_season' from Date for modeling")
    lines.append("\n[7] Overall data quality: GOOD – minimal cleaning needed")
    lines.append("="*55)

    text = "\n".join(lines)
    print(text)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  [REPORT] Saved → '{save_path}'")


# ══════════════════════════════════════════════════════════
# STEP 3 – CLEANING
# ══════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log = []

    df["Date"] = pd.to_datetime(df["Date"])
    log.append("✓ 'Date' → datetime")

    for col in ["Store ID","Product ID","Category","Region",
                "Weather Condition","Seasonality"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    log.append("✓ Strings: stripped + title-cased")

    before = len(df)
    df = df.drop_duplicates()
    log.append(f"✓ Duplicates removed: {before - len(df)}")

    cap = df["Units Sold"].quantile(0.99)
    n_capped = int((df["Units Sold"] > cap).sum())
    df["Units_Sold_Raw"] = df["Units Sold"].copy()
    df["Units Sold"] = df["Units Sold"].clip(upper=cap).round(0).astype(int)
    log.append(f"✓ Units Sold: {n_capped} values capped at 99th pct ({cap:.0f})")

    df["Discount"] = df["Discount"].clip(0, 100)
    df = df[df["Price"] > 0]
    log.append("✓ Discount validated [0,100] | Price > 0 enforced")

    df = df.rename(columns={
        "Store ID"         : "store_id",
        "Product ID"       : "product_id",
        "Category"         : "category",
        "Region"           : "region",
        "Inventory Level"  : "inventory_level",
        "Units Sold"       : "units_sold",
        "Units_Sold_Raw"   : "units_sold_raw",
        "Units Ordered"    : "units_ordered",
        "Demand Forecast"  : "demand_forecast",
        "Price"            : "price",
        "Discount"         : "discount_pct",
        "Weather Condition": "weather",
        "Holiday/Promotion": "is_holiday",
        "Competitor Pricing":"competitor_price",
        "Seasonality"      : "seasonality_label",
    })
    log.append("✓ Columns renamed → snake_case")

    df = df.sort_values(["store_id","product_id","Date"]).reset_index(drop=True)
    log.append("✓ Sorted by [store_id, product_id, Date]")

    print("\n  [CLEANING LOG]")
    for l in log:
        print(f"    {l}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 4 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time
    df["year"]         = df["Date"].dt.year
    df["month"]        = df["Date"].dt.month
    df["quarter"]      = df["Date"].dt.quarter
    df["day_of_week"]  = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    def cal_season(m):
        if m in [12,1,2]:  return "Winter"
        elif m in [3,4,5]: return "Spring"
        elif m in [6,7,8]: return "Summer"
        else:              return "Autumn"
    df["calendar_season"] = df["month"].apply(cal_season)

    # Revenue & pricing
    df["revenue"]         = (df["units_sold"] * df["price"] * (1 - df["discount_pct"]/100)).round(2)
    df["gross_revenue"]   = (df["units_sold"] * df["price"]).round(2)
    df["discount_amount"] = (df["units_sold"] * df["price"] * df["discount_pct"]/100).round(2)
    df["price_gap_pct"]   = ((df["price"] - df["competitor_price"]) / df["competitor_price"] * 100).round(2)
    df["is_cheaper_than_competitor"] = (df["price"] < df["competitor_price"]).astype(int)

    # Inventory
    df["stock_coverage_days"]  = (df["inventory_level"] / df["units_sold"].replace(0, np.nan)).round(2)
    df["stockout_risk"]        = (df["inventory_level"] < df["units_sold"]).astype(int)
    df["inventory_utilization"]= (df["units_sold"] / df["inventory_level"].replace(0, np.nan)).clip(0,1).round(4)
    df["reorder_gap"]          = df["units_ordered"] - df["units_sold"]

    # Forecast quality
    df["forecast_error"]     = (df["units_sold"] - df["demand_forecast"]).round(2)
    df["forecast_error_abs"] = df["forecast_error"].abs()
    df["forecast_mape"]      = (df["forecast_error_abs"] / df["units_sold"].replace(0, np.nan) * 100).round(2)
    df["forecast_bias"]      = np.where(df["forecast_error"]>0, "Under-forecast",
                               np.where(df["forecast_error"]<0, "Over-forecast", "Accurate"))

    # Lag & rolling (per store × product)
    df = df.sort_values(["store_id","product_id","Date"])
    for lag in [1, 7, 30]:
        df[f"units_sold_lag{lag}"] = df.groupby(["store_id","product_id"])["units_sold"].shift(lag)
        df[f"revenue_lag{lag}"]    = df.groupby(["store_id","product_id"])["revenue"].shift(lag)
    for win in [7, 30]:
        df[f"units_sold_ma{win}"] = (
            df.groupby(["store_id","product_id"])["units_sold"]
            .transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean()).round(2))
        df[f"revenue_ma{win}"] = (
            df.groupby(["store_id","product_id"])["revenue"]
            .transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean()).round(2))

    # MoM growth
    monthly = (df.groupby(["store_id","product_id","year","month"])["revenue"]
               .sum().reset_index())
    monthly["revenue_mom_growth"] = (
        monthly.groupby(["store_id","product_id"])["revenue"]
        .pct_change() * 100).round(2)
    df = df.merge(monthly[["store_id","product_id","year","month","revenue_mom_growth"]],
                  on=["store_id","product_id","year","month"], how="left")

    # Prescriptive signals
    df["reorder_recommended"]  = ((df["stock_coverage_days"] < 3) | (df["stockout_risk"]==1)).astype(int)
    df["discount_ineffective"] = ((df["discount_pct"] > 0) & (df["forecast_error"] < -20)).astype(int)
    df["pricing_action"]       = np.where(df["price_gap_pct"] > 10, "Consider price reduction",
                                 np.where(df["price_gap_pct"] < -10, "Room to increase price", "Competitive"))

    print(f"\n  [FEATURES] 15 raw → {df.shape[1]} total columns (+{df.shape[1]-15} engineered)")
    return df


# ══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
def run_pipeline(raw_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df_raw   = load_data(raw_path)
    quality_report(df_raw, os.path.join(output_dir, "data_quality_report.txt"))
    df_clean = clean_data(df_raw)
    df_final = engineer_features(df_clean)

    out_path = os.path.join(output_dir, "retail_cleaned.csv")
    df_final.to_csv(out_path, index=False)
    print(f"\n  [EXPORT] {len(df_final):,} rows × {df_final.shape[1]} cols → '{out_path}'")


if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline(
        raw_path   = os.path.join(ROOT, "data", "raw",     "retail_store_inventory.csv"),
        output_dir = os.path.join(ROOT, "data", "outputs"),
    )
