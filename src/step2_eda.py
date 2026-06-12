"""
Step 2 – Exploratory Data Analysis (19 charts)
IMPACT Framework: P – Performing the Analysis
Covers: Descriptive · Diagnostic · Predictive · Prescriptive
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

PALETTE   = ["#5B4FD4","#1D9E75","#E85D24","#EF9F27","#D4537E"]
CAT_ORDER = ["Clothing","Electronics","Furniture","Groceries","Toys"]
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"figure.dpi":150,
                     "axes.spines.top":False,"axes.spines.right":False})


def save(fig, chart_dir, name):
    p = os.path.join(chart_dir, f"{name}.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {name}.png")


# ── DESCRIPTIVE ──────────────────────────────────────────
def descriptive(df, chart_dir):
    print("\n  [1/4] Descriptive Analysis")

    # 01 Monthly Revenue Trend
    monthly = df.groupby(["year","month"])["revenue"].sum().reset_index()
    monthly["period"] = pd.to_datetime(monthly["year"].astype(str)+"-"+monthly["month"].astype(str).str.zfill(2))
    monthly = monthly.sort_values("period")
    fig, ax = plt.subplots(figsize=(11,4.5))
    ax.fill_between(monthly["period"], monthly["revenue"]/1e6, alpha=0.12, color="#5B4FD4")
    ax.plot(monthly["period"], monthly["revenue"]/1e6, color="#5B4FD4", lw=2.5, marker="o", ms=4)
    idx_max = monthly["revenue"].idxmax()
    idx_min = monthly["revenue"].idxmin()
    for idx, label, color in [(idx_max,"Peak","green"),(idx_min,"Low","red")]:
        ax.annotate(f'{label}: ${monthly.loc[idx,"revenue"]/1e6:.1f}M',
            xy=(monthly.loc[idx,"period"], monthly.loc[idx,"revenue"]/1e6),
            xytext=(0, 18 if label=="Peak" else -28), textcoords="offset points",
            ha="center", fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1))
    ax.set_title("Monthly Revenue Trend (Jan 2022 – Dec 2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (USD Million)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
    fig.autofmt_xdate(rotation=30)
    save(fig, chart_dir, "01_monthly_revenue_trend")

    # 02 Revenue by Category
    cat_year = df.groupby(["year","category"])["revenue"].sum().reset_index()
    cat_year["revenue_M"] = cat_year["revenue"]/1e6
    pivot = cat_year.pivot(index="category", columns="year", values="revenue_M").reindex(CAT_ORDER)
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot.plot(kind="bar", ax=ax, color=["#5B4FD4","#1D9E75"], width=0.65, edgecolor="white")
    ax.set_title("Annual Revenue by Category (2022 vs 2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(""); ax.set_ylabel("Revenue (USD Million)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
    ax.set_xticklabels(CAT_ORDER, rotation=0); ax.legend(title="Year"); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "02_revenue_by_category")

    # 03 Units Sold Distribution
    fig, ax = plt.subplots(figsize=(10,4.5))
    data_box = [df[df["category"]==c]["units_sold"].values for c in CAT_ORDER]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.55, medianprops=dict(color="white",lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_title("Distribution of Daily Units Sold by Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Units Sold")
    save(fig, chart_dir, "03_units_sold_distribution")

    # 04 Store × Region Heatmap
    store_region = df.groupby(["store_id","region"])["revenue"].sum().reset_index()
    pivot_sr = store_region.pivot(index="store_id", columns="region", values="revenue").fillna(0)/1e6
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(pivot_sr.round(1), annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label":"USD M"})
    ax.set_title("Revenue by Store × Region (USD Million, 2022–2023)", fontsize=13, fontweight="bold", pad=12)
    save(fig, chart_dir, "04_store_region_heatmap")

    # 05 Monthly Units by Category
    cat_monthly = df.groupby(["year","month","category"])["units_sold"].sum().reset_index()
    cat_monthly["period"] = pd.to_datetime(cat_monthly["year"].astype(str)+"-"+cat_monthly["month"].astype(str).str.zfill(2))
    fig, ax = plt.subplots(figsize=(11,4.5))
    for cat, color in zip(CAT_ORDER, PALETTE):
        sub = cat_monthly[cat_monthly["category"]==cat].sort_values("period")
        ax.plot(sub["period"], sub["units_sold"], label=cat, color=color, lw=2)
    ax.set_title("Monthly Units Sold by Category (2022–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Units Sold")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.15))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    fig.autofmt_xdate(rotation=30)
    save(fig, chart_dir, "05_monthly_units_by_category")


# ── DIAGNOSTIC ───────────────────────────────────────────
def diagnostic(df, chart_dir):
    print("\n  [2/4] Diagnostic Analysis")

    # 06 Discount vs Revenue
    sample = df.sample(3000, random_state=42)
    fig, ax = plt.subplots(figsize=(10,4.5))
    for cat, color in zip(CAT_ORDER, PALETTE):
        sub = sample[sample["category"]==cat]
        ax.scatter(sub["discount_pct"], sub["revenue"], alpha=0.25, color=color, s=15, label=cat)
    ax.set_title("Impact of Discount on Revenue by Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Discount (%)"); ax.set_ylabel("Revenue (USD)")
    ax.set_xticks([0,5,10,15,20])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.16))
    save(fig, chart_dir, "06_discount_vs_revenue")

    # 07 Weather Impact
    weather_cat = df.groupby(["weather","category"])["units_sold"].mean().reset_index()
    pivot_w = weather_cat.pivot(index="weather", columns="category", values="units_sold")[CAT_ORDER]
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot_w.plot(kind="bar", ax=ax, color=PALETTE, width=0.72, edgecolor="white")
    ax.set_title("Avg Units Sold by Weather Condition & Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(""); ax.set_ylabel("Avg Units Sold")
    ax.set_xticklabels(pivot_w.index, rotation=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.12)); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "07_weather_impact")

    # 08 Holiday Effect
    holiday = df.groupby(["is_holiday","category"])["revenue"].mean().reset_index()
    holiday["label"] = holiday["is_holiday"].map({0:"No Promotion",1:"Holiday/Promo"})
    pivot_h = holiday.pivot(index="category", columns="label", values="revenue").reindex(CAT_ORDER)
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot_h.plot(kind="bar", ax=ax, color=["#888780","#E85D24"], width=0.65, edgecolor="white")
    ax.set_title("Effect of Holiday/Promotion on Avg Revenue", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels(CAT_ORDER, rotation=0); ax.set_ylabel("Avg Revenue (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "08_holiday_promotion_effect")

    # 09 Stock Coverage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4.5))
    cov_clip = df["stock_coverage_days"].clip(upper=30).dropna()
    ax1.hist(cov_clip, bins=40, color="#5B4FD4", alpha=0.75, edgecolor="white")
    ax1.axvline(2, color="red", ls="--", lw=1.5, label="Critical (<2 days)")
    ax1.axvline(5, color="orange", ls="--", lw=1.5, label="Warning (<5 days)")
    ax1.set_title("Stock Coverage Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Days"); ax1.set_ylabel("Frequency"); ax1.legend(fontsize=9)
    data_cov = [df[(df["category"]==c)&(df["stock_coverage_days"]<=30)]["stock_coverage_days"].dropna().values for c in CAT_ORDER]
    bp2 = ax2.boxplot(data_cov, patch_artist=True, widths=0.55, medianprops=dict(color="white",lw=2))
    for patch, color in zip(bp2["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax2.set_xticklabels(CAT_ORDER, rotation=15, ha="right")
    ax2.set_title("Stock Coverage by Category", fontsize=11, fontweight="bold"); ax2.set_ylabel("Days")
    fig.suptitle("Inventory Stock Coverage Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, chart_dir, "09_stock_coverage_analysis")

    # 10 Price vs Competitor
    price_gap = df.groupby("category").agg(avg_price=("price","mean"), avg_comp=("competitor_price","mean")).reset_index()
    price_gap = price_gap.set_index("category").reindex(CAT_ORDER)
    fig, ax = plt.subplots(figsize=(10,4.5))
    x = np.arange(len(CAT_ORDER)); w=0.35
    ax.bar(x-w/2, price_gap["avg_price"], w, label="Our Price",   color="#5B4FD4", alpha=0.85)
    ax.bar(x+w/2, price_gap["avg_comp"],  w, label="Competitor",  color="#888780", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CAT_ORDER)
    ax.set_title("Average Price vs Competitor Price by Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Avg Price (USD)"); ax.legend(); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "10_price_vs_competitor")

    # 11 Forecast Bias
    bias_cat = df.groupby(["category","forecast_bias"]).size().reset_index(name="count")
    pivot_b  = bias_cat.pivot(index="category", columns="forecast_bias", values="count").reindex(CAT_ORDER).fillna(0)
    cols_ord = [c for c in ["Under-forecast","Accurate","Over-forecast"] if c in pivot_b.columns]
    pivot_b  = pivot_b[cols_ord]
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot_b.plot(kind="bar", ax=ax, color=["#E85D24","#1D9E75","#5B4FD4"][:len(cols_ord)], stacked=True, edgecolor="white")
    ax.set_title("Forecast Bias Distribution by Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(""); ax.set_ylabel("Number of Records")
    ax.set_xticklabels(CAT_ORDER, rotation=0); ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    save(fig, chart_dir, "11_forecast_bias")


# ── PREDICTIVE ───────────────────────────────────────────
def predictive(df, chart_dir):
    print("\n  [3/4] Predictive Analysis")

    # 12 Correlation Matrix
    corr_cols = ["units_sold","revenue","inventory_level","price","discount_pct",
                 "competitor_price","demand_forecast","price_gap_pct",
                 "stock_coverage_days","is_holiday","units_sold_ma7","units_sold_ma30"]
    corr = df[corr_cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(10,8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.4, ax=ax, mask=mask, annot_kws={"size":8},
                cbar_kws={"label":"Pearson r"})
    ax.set_title("Feature Correlation Matrix (Key Variables)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, chart_dir, "12_correlation_matrix")

    # 13 Moving Average Trend
    sub = df[(df["store_id"]=="S001")&(df["product_id"]=="P0001")].sort_values("Date")
    fig, ax = plt.subplots(figsize=(11,4.5))
    ax.plot(sub["Date"], sub["units_sold"],     color="#888780", lw=1,   alpha=0.5, label="Actual")
    ax.plot(sub["Date"], sub["units_sold_ma7"], color="#5B4FD4", lw=2,             label="MA-7")
    ax.plot(sub["Date"], sub["units_sold_ma30"],color="#E85D24", lw=2.5, ls="--",  label="MA-30")
    ax.set_title("Actual Units Sold vs Moving Averages – Store S001 (P0001)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Units Sold"); ax.legend(); fig.autofmt_xdate(rotation=30)
    save(fig, chart_dir, "13_moving_average_trend")

    # 14 MoM Growth Distribution
    growth_clean = df["revenue_mom_growth"].dropna()
    growth_clean = growth_clean[growth_clean.between(-80,80)]
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.hist(growth_clean, bins=50, color="#1D9E75", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.5, label="0% growth")
    ax.axvline(growth_clean.mean(), color="#5B4FD4", ls=":", lw=2, label=f"Mean: {growth_clean.mean():.1f}%")
    ax.set_title("Month-over-Month Revenue Growth Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("MoM Growth (%)"); ax.set_ylabel("Frequency"); ax.legend(); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "14_mom_growth_distribution")

    # 15 Forecast vs Actual
    sample2 = df.sample(3000, random_state=1)
    fig, ax = plt.subplots(figsize=(8,5))
    for cat, color in zip(CAT_ORDER, PALETTE):
        sub = sample2[sample2["category"]==cat]
        ax.scatter(sub["demand_forecast"], sub["units_sold"], alpha=0.2, color=color, s=12, label=cat)
    max_val = max(sample2["demand_forecast"].max(), sample2["units_sold"].max())
    ax.plot([0,max_val],[0,max_val], "r--", lw=1.5, label="Perfect Forecast")
    ax.set_title("Existing Demand Forecast vs Actual Units Sold", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Demand Forecast"); ax.set_ylabel("Actual Units Sold")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    save(fig, chart_dir, "15_forecast_vs_actual")


# ── PRESCRIPTIVE ─────────────────────────────────────────
def prescriptive(df, chart_dir):
    print("\n  [4/4] Prescriptive Analysis")

    # 16 Reorder Flags
    reorder = df[df["reorder_recommended"]==1].groupby(["store_id","category"]).size().reset_index(name="flags")
    pivot_r = reorder.pivot(index="store_id", columns="category", values="flags").fillna(0)[CAT_ORDER]
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot_r.plot(kind="bar", ax=ax, stacked=True, color=PALETTE, edgecolor="white")
    ax.set_title("Reorder Recommendation Flags by Store & Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Store"); ax.set_ylabel("Reorder Flags")
    ax.set_xticklabels(pivot_r.index, rotation=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.14))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    save(fig, chart_dir, "16_reorder_recommendations")

    # 17 Pricing Actions
    pricing = df.groupby(["category","pricing_action"]).size().reset_index(name="count")
    color_map_p = {"Competitive":"#1D9E75","Consider price reduction":"#E85D24","Room to increase price":"#EF9F27"}
    pivot_p = pricing.pivot(index="category", columns="pricing_action", values="count").reindex(CAT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(10,4.5))
    bottom = np.zeros(len(CAT_ORDER))
    for col in pivot_p.columns:
        ax.bar(CAT_ORDER, pivot_p[col], bottom=bottom, label=col,
               color=color_map_p.get(col,"#888780"), edgecolor="white")
        bottom += pivot_p[col].values
    ax.set_title("Pricing Action Recommendations by Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Records"); ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    save(fig, chart_dir, "17_pricing_actions")

    # 18 KPI Summary
    kpis = [
        ("Total Revenue",        f"${df['revenue'].sum()/1e6:.1f}M"),
        ("Avg Daily Units Sold", f"{df['units_sold'].mean():.0f}"),
        ("Avg Stock Coverage",   f"{df['stock_coverage_days'].mean():.1f} days"),
        ("Reorder Flags",        f"{df['reorder_recommended'].sum():,}"),
        ("Avg Forecast MAPE",    f"{df['forecast_mape'].mean():.1f}%"),
        ("Price Competitive",    f"{(df['pricing_action']=='Competitive').mean()*100:.0f}%"),
    ]
    fig, axes = plt.subplots(2,3, figsize=(11,4.5))
    fig.patch.set_facecolor("#F8F8F8")
    for ax_i, (label, value) in zip(axes.flat, kpis):
        ax_i.set_facecolor("white")
        ax_i.text(0.5, 0.62, value, ha="center", va="center", transform=ax_i.transAxes,
                  fontsize=24, fontweight="bold", color="#5B4FD4")
        ax_i.text(0.5, 0.25, label, ha="center", va="center", transform=ax_i.transAxes,
                  fontsize=11, color="#555555")
        for spine in ax_i.spines.values(): spine.set_edgecolor("#E0E0E0")
        ax_i.set_xticks([]); ax_i.set_yticks([])
    fig.suptitle("KPI Summary – Retail Performance (2022–2023)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.5)
    save(fig, chart_dir, "18_kpi_summary")

    # 19 Discount Effectiveness
    disc_eff = df[df["discount_pct"]>0].groupby(["discount_pct","category"])["discount_ineffective"].mean().reset_index()
    disc_eff["pct"] = disc_eff["discount_ineffective"]*100
    pivot_de = disc_eff.pivot(index="discount_pct", columns="category", values="pct")[CAT_ORDER]
    fig, ax = plt.subplots(figsize=(10,4.5))
    pivot_de.plot(kind="bar", ax=ax, color=PALETTE, width=0.72, edgecolor="white")
    ax.set_title("Discount Ineffectiveness Rate by Discount Level & Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Discount (%)"); ax.set_ylabel("Ineffective Rate (%)")
    ax.set_xticklabels([f"{int(x)}%" for x in pivot_de.index], rotation=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.14)); ax.grid(axis="y", alpha=0.4)
    save(fig, chart_dir, "19_discount_effectiveness")


# ══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
def run_eda(clean_path: str, chart_dir: str):
    os.makedirs(chart_dir, exist_ok=True)
    df = pd.read_csv(clean_path, parse_dates=["Date"])
    df = df[~((df["year"]==2024)&(df["month"]==1))]
    print(f"  [LOAD] {len(df):,} rows (2022-01 → 2023-12)")

    descriptive(df, chart_dir)
    diagnostic(df, chart_dir)
    predictive(df, chart_dir)
    prescriptive(df, chart_dir)

    print(f"\n  [EDA] 19 charts saved → '{chart_dir}/'")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_eda(
        clean_path = os.path.join(ROOT, "data", "outputs", "retail_cleaned.csv"),
        chart_dir  = os.path.join(ROOT, "data", "outputs", "charts"),
    )
