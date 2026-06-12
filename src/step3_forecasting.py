"""
Step 3 – Forecasting Models
SARIMA · Prophet · XGBoost
IMPACT Framework: A – Addressing the Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os, json
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product as iproduct

PALETTE  = {"sarima":"#5B4FD4","prophet":"#1D9E75","xgboost":"#E85D24","actual":"#333333"}
N_TEST   = 3
N_FC     = 3
FEATURES = ["month","quarter","year","month_sin","month_cos",
            "lag1","lag2","lag3","lag12","rolling3","rolling6",
            "avg_price","avg_discount","avg_competitor","price_gap","is_holiday"]


# ── helpers ──────────────────────────────────────────────
def metrics(actual, predicted, name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs(actual - predicted) / actual) * 100
    r2   = 1 - np.sum((actual-predicted)**2)/np.sum((actual-np.mean(actual))**2)
    print(f"    {name:<10}  MAE={mae:>8,.0f}  RMSE={rmse:>8,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")
    return {"mae":round(mae,2),"rmse":round(rmse,2),"mape":round(mape,4),"r2":round(r2,4)}

def save(fig, chart_dir, name):
    p = os.path.join(chart_dir, f"{name}.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"    ✓ {name}.png")


# ── data prep ────────────────────────────────────────────
def prepare(clean_path):
    df = pd.read_csv(clean_path, parse_dates=["Date"])
    df = df[~((df["year"]==2024)&(df["month"]==1))]
    base = df.groupby("Date").agg(
        units_sold=("units_sold","sum"), avg_price=("price","mean"),
        avg_discount=("discount_pct","mean"), avg_competitor=("competitor_price","mean"),
        is_holiday=("is_holiday","max"),
    ).reset_index()
    monthly = base.resample("ME", on="Date").agg({
        "units_sold":"sum","avg_price":"mean","avg_discount":"mean",
        "avg_competitor":"mean","is_holiday":"max",
    }).reset_index()
    print(f"  [DATA] {len(monthly)} monthly periods "
          f"({monthly['Date'].min().strftime('%Y-%m')} → {monthly['Date'].max().strftime('%Y-%m')})")
    return monthly


# ── MODEL 1: SARIMA ───────────────────────────────────────
def run_sarima(train, test):
    print("\n  [MODEL 1] SARIMA")
    ts_train = train.set_index("Date")["units_sold"]
    ts_test  = test.set_index("Date")["units_sold"]

    adf_p = adfuller(ts_train)[1]
    print(f"    ADF p={adf_p:.4f} → {'Stationary ✓' if adf_p<0.05 else 'Non-stationary'}")

    best_aic, best_order, best_sorder = np.inf, (1,1,1), (0,1,1,12)
    for p,d,q in iproduct([0,1,2],[0,1],[0,1,2]):
        for P,D,Q in iproduct([0,1],[0,1],[0,1]):
            try:
                m = SARIMAX(ts_train, order=(p,d,q), seasonal_order=(P,D,Q,12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if m.aic < best_aic:
                    best_aic=m.aic; best_order=(p,d,q); best_sorder=(P,D,Q,12)
            except: pass
    print(f"    Best: SARIMA{best_order}×{best_sorder}  AIC={best_aic:.2f}")

    model = SARIMAX(ts_train, order=best_order, seasonal_order=best_sorder,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    test_pred = model.forecast(N_TEST)
    m = metrics(ts_test.values, test_pred.values, "SARIMA")

    ts_all     = pd.concat([ts_train, ts_test])
    model_full = SARIMAX(ts_all, order=best_order, seasonal_order=best_sorder,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    future     = model_full.forecast(N_FC)
    return {"test_pred":test_pred.values,"future_pred":future.values,
            "future_dates":future.index,"metrics":m}


# ── MODEL 2: PROPHET ──────────────────────────────────────
def run_prophet(train, test, monthly):
    print("\n  [MODEL 2] PROPHET")
    train_p = train[["Date","units_sold"]].rename(columns={"Date":"ds","units_sold":"y"})
    all_p   = monthly[["Date","units_sold"]].rename(columns={"Date":"ds","units_sold":"y"})

    def fit_prophet(data):
        m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10)
        m.fit(data)
        return m

    model = fit_prophet(train_p)
    fc_test = model.predict(model.make_future_dataframe(periods=N_TEST, freq="ME"))
    test_pred = fc_test[fc_test["ds"].isin(test["Date"])]["yhat"].values
    m = metrics(test["units_sold"].values, test_pred, "Prophet")

    model_full = fit_prophet(all_p)
    fc_all = model_full.predict(model_full.make_future_dataframe(periods=N_FC, freq="ME"))
    fc_future = fc_all.tail(N_FC)
    return {"test_pred":test_pred,
            "future_pred":fc_future["yhat"].values,
            "future_pred_lower":fc_future["yhat_lower"].values,
            "future_pred_upper":fc_future["yhat_upper"].values,
            "future_dates":fc_future["ds"].values,"metrics":m}


# ── MODEL 3: XGBOOST ─────────────────────────────────────
def build_xgb_features(monthly):
    df = monthly.copy().sort_values("Date").reset_index(drop=True)
    df["month"]     = df["Date"].dt.month
    df["quarter"]   = df["Date"].dt.quarter
    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["lag1"]      = df["units_sold"].shift(1)
    df["lag2"]      = df["units_sold"].shift(2)
    df["lag3"]      = df["units_sold"].shift(3)
    df["lag12"]     = df["units_sold"].shift(12)
    df["rolling3"]  = df["units_sold"].shift(1).rolling(3).mean()
    df["rolling6"]  = df["units_sold"].shift(1).rolling(6).mean()
    df["price_gap"] = df["avg_price"] - df["avg_competitor"]
    return df.dropna()


def run_xgboost(train, test, monthly):
    print("\n  [MODEL 3] XGBOOST")
    feat  = build_xgb_features(monthly)
    tr    = feat[feat["Date"].isin(train["Date"])]
    te    = feat[feat["Date"].isin(test["Date"])]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(tr[FEATURES], tr["units_sold"], eval_set=[(te[FEATURES], te["units_sold"])], verbose=False)
    test_pred = model.predict(te[FEATURES])
    m = metrics(te["units_sold"].values, test_pred, "XGBoost")

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    model_full = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model_full.fit(feat[FEATURES], feat["units_sold"])

    hist = monthly.copy(); future_preds=[]; future_dates=[]
    last_date = monthly["Date"].max()
    for i in range(1, N_FC+1):
        next_date = (last_date + pd.DateOffset(months=i)) + pd.offsets.MonthEnd(0)
        hs = hist["units_sold"].values
        row = {"month":next_date.month,"quarter":next_date.quarter,"year":next_date.year,
               "month_sin":np.sin(2*np.pi*next_date.month/12),
               "month_cos":np.cos(2*np.pi*next_date.month/12),
               "lag1":hs[-1],"lag2":hs[-2],"lag3":hs[-3],
               "lag12":hs[-12] if len(hs)>=12 else hs.mean(),
               "rolling3":np.mean(hs[-3:]),"rolling6":np.mean(hs[-6:]),
               "avg_price":hist["avg_price"].mean(),"avg_discount":hist["avg_discount"].mean(),
               "avg_competitor":hist["avg_competitor"].mean(),
               "price_gap":hist["avg_price"].mean()-hist["avg_competitor"].mean(),"is_holiday":0}
        pred = model_full.predict(pd.DataFrame([row])[FEATURES])[0]
        future_preds.append(pred); future_dates.append(next_date)
        hist = pd.concat([hist, pd.DataFrame([{"Date":next_date,"units_sold":pred,
            "avg_price":row["avg_price"],"avg_discount":row["avg_discount"],
            "avg_competitor":row["avg_competitor"],"is_holiday":0}])], ignore_index=True)
    return {"test_pred":test_pred,"future_pred":np.array(future_preds),
            "future_dates":future_dates,"metrics":m,"feat_importance":fi}


# ── CHARTS ────────────────────────────────────────────────
def make_charts(monthly, train, test, sr, pr, xr, best, chart_dir):
    # Chart 20: test comparison
    fig, axes = plt.subplots(1,3, figsize=(15,5), sharey=True)
    fig.suptitle("Forecast vs Actual – Test Period (Oct–Dec 2023)", fontsize=14, fontweight="bold", y=1.01)
    pairs = [("SARIMA",sr["test_pred"],PALETTE["sarima"]),
             ("Prophet",pr["test_pred"],PALETTE["prophet"]),
             ("XGBoost",xr["test_pred"],PALETTE["xgboost"])]
    td = test["Date"].values; ta = test["units_sold"].values
    for ax, (name, preds, color) in zip(axes, pairs):
        mape = np.mean(np.abs(ta-preds)/ta)*100
        ax.plot(td, ta, "o-", color=PALETTE["actual"], lw=2, ms=7, label="Actual", zorder=5)
        ax.plot(td, preds, "s--", color=color, lw=2, ms=7, label=f"{name}\nMAPE={mape:.2f}%")
        ax.fill_between(td, ta, preds, alpha=0.12, color=color)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.set_ylabel("Units Sold" if ax==axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
        if name==best: ax.set_facecolor("#F5FFF8")
    plt.tight_layout()
    save(fig, chart_dir, "20_model_comparison_test")

    # Chart 21: full history + forecast
    fig, ax = plt.subplots(figsize=(13,5))
    ax.fill_between(monthly["Date"], monthly["units_sold"], alpha=0.08, color="#5B4FD4")
    ax.plot(monthly["Date"], monthly["units_sold"], color=PALETTE["actual"], lw=2, label="Historical")
    ax.axvline(test["Date"].iloc[0], color="gray", ls=":", lw=1.2, alpha=0.7)
    for name, dates, preds, color in [
        ("SARIMA", sr["future_dates"], sr["future_pred"], PALETTE["sarima"]),
        ("Prophet",pr["future_dates"],pr["future_pred"],  PALETTE["prophet"]),
        ("XGBoost",xr["future_dates"],xr["future_pred"],  PALETTE["xgboost"]),
    ]:
        lw = 3 if name==best else 1.8; ls = "-" if name==best else "--"
        mape_val = sr["metrics"]["mape"] if name=="SARIMA" else pr["metrics"]["mape"] if name=="Prophet" else xr["metrics"]["mape"]
        ax.plot(dates, preds, marker="o", ms=6, lw=lw, ls=ls, color=color,
                alpha=1.0 if name==best else 0.65, label=f"{name} (MAPE={mape_val:.2f}%)")
        last_d = monthly["Date"].iloc[-1]; last_v = monthly["units_sold"].iloc[-1]
        ax.plot([last_d, dates[0]], [last_v, preds[0]], color=color, lw=lw, ls=ls,
                alpha=1.0 if name==best else 0.65)
    ax.fill_between(pr["future_dates"], pr["future_pred_lower"], pr["future_pred_upper"],
                    alpha=0.08, color=PALETTE["prophet"], label="Prophet 95% CI")
    ax.set_title("Retail Units Sold – Historical & 3-Month Forecast (Jan–Mar 2024)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Total Units Sold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))
    ax.legend(ncol=2, loc="lower left", fontsize=10); ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30); plt.tight_layout()
    save(fig, chart_dir, "21_full_forecast")

    # Chart 22: metrics table
    results_dict = {"SARIMA":sr["metrics"],"Prophet":pr["metrics"],"XGBoost":xr["metrics"]}
    fig, ax = plt.subplots(figsize=(10,3.2)); ax.axis("off")
    headers = ["Model","MAE","RMSE","MAPE (%)","R²","Winner"]
    rows = [[name, f"{m['mae']:,.0f}", f"{m['rmse']:,.0f}", f"{m['mape']:.2f}%", f"{m['r2']:.4f}",
             "⭐ Best" if name==best else ""] for name, m in results_dict.items()]
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 2.2)
    for j in range(len(headers)):
        table[0,j].set_facecolor("#5B4FD4"); table[0,j].set_text_props(color="white", fontweight="bold")
    best_row = list(results_dict.keys()).index(best)+1
    for j in range(len(headers)):
        table[best_row,j].set_facecolor("#E8F5E9"); table[best_row,j].set_text_props(fontweight="bold")
    ax.set_title("Model Performance Comparison – Test Period", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    save(fig, chart_dir, "22_model_metrics_table")

    # Chart 23: XGBoost feature importance
    fi = xr["feat_importance"].head(10)
    fig, ax = plt.subplots(figsize=(9,4.5))
    colors = [PALETTE["xgboost"] if i<3 else "#AAAAAA" for i in range(len(fi))]
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
    ax.set_title("XGBoost – Top 10 Feature Importance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score"); ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, fi.values.max()*1.2)
    plt.tight_layout()
    save(fig, chart_dir, "23_xgboost_feature_importance")

    # Chart 24: forecast summary
    months = ["Jan 2024","Feb 2024","Mar 2024"]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(13,4.5), gridspec_kw={"width_ratios":[1.3,1]})
    x = np.arange(3); w=0.25
    ax1.bar(x-w, sr["future_pred"]/1e3, w, label="SARIMA",  color=PALETTE["sarima"],  alpha=0.85)
    ax1.bar(x,   pr["future_pred"]/1e3, w, label="Prophet", color=PALETTE["prophet"], alpha=0.85)
    ax1.bar(x+w, xr["future_pred"]/1e3, w, label="XGBoost", color=PALETTE["xgboost"], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(months)
    ax1.set_ylabel("Units Sold (K)"); ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("3-Month Forecast by Model", fontsize=12, fontweight="bold"); ax1.legend()
    ax2.axis("off")
    last_actual = monthly["units_sold"].iloc[-1]
    rows_t = []
    for name, fc in [("SARIMA",sr["future_pred"]),("Prophet",pr["future_pred"]),("XGBoost",xr["future_pred"])]:
        chg = (fc.mean()-last_actual)/last_actual*100
        rows_t.append([name, f"{fc[0]:,.0f}", f"{fc[1]:,.0f}", f"{fc[2]:,.0f}", f"{chg:+.1f}%"])
    table2 = ax2.table(cellText=rows_t, colLabels=["Model","Jan","Feb","Mar","Trend"],
                       cellLoc="center", loc="center")
    table2.auto_set_font_size(False); table2.set_fontsize(11); table2.scale(1.1, 2.0)
    for j in range(5):
        table2[0,j].set_facecolor("#5B4FD4"); table2[0,j].set_text_props(color="white", fontweight="bold")
    best_row2 = ["SARIMA","Prophet","XGBoost"].index(best)+1
    for j in range(5):
        table2[best_row2,j].set_facecolor("#E8F5E9"); table2[best_row2,j].set_text_props(fontweight="bold")
    ax2.set_title("Forecast Values & Trend", fontsize=12, fontweight="bold")
    fig.suptitle("Demand Forecast Summary – Jan–Mar 2024", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, chart_dir, "24_forecast_summary")


# ══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
def run_forecasting(clean_path: str, chart_dir: str, result_path: str,
                    skip_prophet: bool = False):
    os.makedirs(chart_dir, exist_ok=True)
    monthly = prepare(clean_path)
    train = monthly.iloc[:-N_TEST].copy()
    test  = monthly.iloc[-N_TEST:].copy()
    print(f"  Train: {len(train)} months | Test: {len(test)} months")
    print(f"\n  {'Model':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'R²':>8}")
    print("  " + "-"*50)

    sr = run_sarima(train, test)
    xr = run_xgboost(train, test, monthly)

    if skip_prophet:
        print("\n  [MODEL 2] PROPHET — ⏭  skipped (use --skip-prophet to suppress this)")
        # Create a dummy prophet result so charts still render
        pr = {
            "test_pred"        : sr["test_pred"],      # fallback = SARIMA
            "future_pred"      : sr["future_pred"],
            "future_pred_lower": sr["future_pred"] * 0.97,
            "future_pred_upper": sr["future_pred"] * 1.03,
            "future_dates"     : sr["future_dates"],
            "metrics"          : {"mae":0,"rmse":0,"mape":999.0,"r2":0},
        }
        results = {"SARIMA":sr["metrics"],"XGBoost":xr["metrics"]}
    else:
        pr = run_prophet(train, test, monthly)
        results = {"SARIMA":sr["metrics"],"Prophet":pr["metrics"],"XGBoost":xr["metrics"]}

    best = min(results, key=lambda k: results[k]["mape"])
    print(f"\n  ✅ Best model: {best} (MAPE={results[best]['mape']:.2f}%)")

    print("\n  [CHARTS]")
    make_charts(monthly, train, test, sr, pr, xr, best, chart_dir)

    fc_dict = {
        "SARIMA" : [round(v,0) for v in sr["future_pred"].tolist()],
        "XGBoost": [round(v,0) for v in xr["future_pred"].tolist()],
    }
    if not skip_prophet:
        fc_dict["Prophet"] = [round(v,0) for v in pr["future_pred"].tolist()]

    output = {
        "best_model"     : best,
        "evaluation"     : results,
        "future_forecast": fc_dict,
        "forecast_months": ["Jan 2024","Feb 2024","Mar 2024"],
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [RESULTS] Saved → '{result_path}'")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_forecasting(
        clean_path  = os.path.join(ROOT, "data", "outputs", "retail_cleaned.csv"),
        chart_dir   = os.path.join(ROOT, "data", "outputs", "charts"),
        result_path = os.path.join(ROOT, "data", "outputs", "model_results.json"),
    )
