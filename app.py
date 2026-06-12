"""
RetailCast – FAA4023  |  Streamlit Dashboard v3
Fixes: dark bg full, import CSV tab, checkbox filters,
       equal KPI cards, richer charts, real-time data flow
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, json, os, io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RetailCast", page_icon="▣",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* BASE */
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#06080f!important;color:#c8dde8;}
.main,.main>div,.block-container{background:#06080f!important;}
.main .block-container{padding:2rem 2.5rem;max-width:1400px;}
section[data-testid="stSidebar"]>div{background:linear-gradient(180deg,#090c16,#06080f)!important;}

/* SIDEBAR */
[data-testid="stSidebar"]{border-right:1px solid #122030;}
[data-testid="stSidebar"] *{color:#c8dde8!important;}
[data-testid="stSidebar"] .stRadio>label{font-size:10px!important;font-weight:700!important;
  letter-spacing:2px!important;text-transform:uppercase!important;color:#4a8fa8!important;margin-bottom:8px;display:block;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{font-size:12px!important;
  font-weight:700!important;letter-spacing:1px!important;text-transform:uppercase!important;
  color:#7ab8cc!important;padding:9px 14px;border-radius:4px;margin-bottom:2px;
  border-left:2px solid transparent;transition:all .15s;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{
  background:rgba(0,200,220,.07);color:#00c8dc!important;border-left-color:#00c8dc;}

/* SECTION LABEL */
.sec-label{font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#00c8dc;margin-bottom:6px;}

/* PAGE TITLE */
.page-title{font-size:26px;font-weight:800;
  background:linear-gradient(90deg,#00c8dc,#9b5de5);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  letter-spacing:-0.5px;margin-bottom:4px;}

/* KPI CARD — fixed height so all equal */
.kpi{background:linear-gradient(135deg,#0d1828,#0a1320);border:1px solid #1a3048;
  border-top:2px solid #00c8dc;border-radius:6px;
  padding:18px 12px;text-align:center;
  min-height:110px;display:flex;flex-direction:column;
  justify-content:center;align-items:center;}
.kpi-val{font-size:24px;font-weight:800;color:#00c8dc;line-height:1.1;}
.kpi-lbl{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  color:#5a8fa8;margin-top:7px;}
.kpi-sub{font-size:11px;font-weight:600;color:#00e5a0;margin-top:4px;}
.kpi-sub.warn{color:#ff3d6b;}
.kpi-sub.neutral{color:#5a8fa8;}

/* DIVIDER */
hr.divider{border:none;border-top:1px solid #122030;margin:20px 0;}

/* ALERTS */
.alert{border-radius:4px;padding:12px 16px;font-size:13px;margin:8px 0;border-left:3px solid;}
.alert-blue {background:rgba(0,200,220,.06);border-color:#00c8dc;color:#7ac8d8;}
.alert-green{background:rgba(0,229,160,.06);border-color:#00e5a0;color:#7ac8d8;}
.alert-red  {background:rgba(255,61,107,.06);border-color:#ff3d6b;color:#7ac8d8;}
.alert-amber{background:rgba(255,180,0,.06);border-color:#ffb400;color:#7ac8d8;}

/* MODEL CARD */
.model-card{background:linear-gradient(135deg,#0d1828,#0a1320);border:1px solid #1a3048;
  border-radius:6px;padding:20px;text-align:center;height:100%;}
.model-card.best{border-color:#00c8dc;box-shadow:0 0 18px rgba(0,200,220,.15);}
.model-name{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#5a8fa8;}
.model-mape{font-size:30px;font-weight:800;color:#00c8dc;margin:6px 0 2px;}
.model-mape.good{color:#00e5a0;}.model-mape.bad{color:#ff3d6b;}
.model-meta{font-size:11px;color:#6aaabb;line-height:2;}
.best-tag{display:inline-block;background:rgba(0,200,220,.12);color:#00c8dc;
  border:1px solid rgba(0,200,220,.3);font-size:9px;font-weight:700;
  letter-spacing:1px;padding:2px 8px;border-radius:20px;text-transform:uppercase;
  vertical-align:middle;margin-left:8px;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #122030;gap:2px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#5a8fa8;font-size:10px;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;padding:10px 18px;border-radius:4px 4px 0 0;transition:all .15s;}
.stTabs [data-baseweb="tab"]:hover{color:#00c8dc;}
.stTabs [aria-selected="true"]{background:rgba(0,200,220,.06)!important;color:#00c8dc!important;
  border-bottom:2px solid #00c8dc!important;}

/* INPUTS */
.stSelectbox>div>div,.stMultiSelect>div>div{background:#0d1828!important;
  border-color:#1a3048!important;color:#c8dde8!important;}
div[data-baseweb="select"]>div{background:#0d1828!important;border-color:#1a3048!important;}
.stCheckbox label p{font-size:13px;font-weight:600;color:#c8dde8!important;}
.stCheckbox{margin-bottom:2px;}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{background:#0d1828;border:1px dashed #1a3048;
  border-radius:6px;padding:8px;}
[data-testid="stFileUploader"] *{color:#4a8fa8!important;}

/* BUTTON */
.stButton>button{background:transparent;color:#00c8dc;font-weight:700;font-size:11px;
  letter-spacing:1.5px;text-transform:uppercase;border:1px solid #00c8dc;
  border-radius:4px;padding:10px 28px;transition:all .2s;}
.stButton>button:hover{background:rgba(0,200,220,.1);box-shadow:0 0 14px rgba(0,200,220,.25);}

/* SUCCESS / INFO / WARNING overrides */
.stSuccess,.stInfo,.stWarning,.stError{background:#0d1828!important;border-radius:6px;}

/* ── DataFrame / Table — FULL dark override ── */
.stDataFrame{border-radius:8px;overflow:hidden;border:1px solid #1a3048!important;}
.stDataFrame > div{background:#0d1828!important;}
[data-testid="stDataFrameResizable"]{background:#0d1828!important;}
.stDataFrame iframe{background:#0d1828!important;color-scheme:dark;}

/* st.table */
.stTable table{width:100%;border-collapse:collapse;background:#0d1828;
  border-radius:8px;overflow:hidden;font-size:13px;}
.stTable table thead tr th{background:#0a1320!important;color:#00c8dc!important;
  font-size:10px!important;font-weight:700!important;letter-spacing:1.5px!important;
  text-transform:uppercase!important;padding:10px 14px!important;border-bottom:1px solid #1a3048!important;}
.stTable table tbody tr{background:#0d1828!important;}
.stTable table tbody tr:nth-child(even){background:#0a1320!important;}
.stTable table tbody tr:hover{background:#122030!important;}
.stTable table tbody tr td{color:#c8dde8!important;padding:9px 14px!important;
  border-bottom:1px solid #0f1e2e!important;font-size:13px!important;}

/* Custom dark HTML table */
.rc-table{width:100%;border-collapse:collapse;font-size:12.5px;font-family:'Inter',sans-serif;}
.rc-table thead tr th{background:#0a1320;color:#00c8dc;font-size:10px;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;padding:11px 14px;
  border-bottom:1px solid #1a3048;white-space:nowrap;}
.rc-table tbody tr{background:#0d1828;transition:background .12s;}
.rc-table tbody tr:nth-child(even){background:#090f1c;}
.rc-table tbody tr:hover{background:#122030;}
.rc-table tbody tr td{color:#c8dde8;padding:8px 14px;border-bottom:1px solid #0f1e2e;white-space:nowrap;}
.rc-table tbody tr td.num{color:#7ab8cc;text-align:right;font-variant-numeric:tabular-nums;}
.rc-table tbody tr td.hi{color:#00e5a0;font-weight:600;}
.rc-table tbody tr td.warn{color:#ff3d6b;font-weight:600;}
.rc-table tbody tr td.mid{color:#ffb400;font-weight:600;}
.rc-table-wrap{background:#0d1828;border:1px solid #1a3048;border-radius:8px;
  overflow:auto;max-height:400px;}

/* Expander */
.streamlit-expanderHeader{background:#0d1828;border:1px solid #1a3048;border-radius:4px;
  font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#5a8fa8!important;}

/* Scrollbar */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:#06080f;}
::-webkit-scrollbar-thumb{background:#1a2a3a;border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:rgba(0,200,220,.3);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CHART THEME
# ══════════════════════════════════════════════════════
BG = "#06080f"; AX = "#0d1828"
TC = "#8ab8cc"   # tick / label color — bright enough on dark bg
BLUE="#00c8dc"; GREEN="#00e5a0"; RED="#ff3d6b"; AMBER="#ffb400"; PURPLE="#9b5de5"
PALETTE=[BLUE,GREEN,RED,AMBER,PURPLE]
CAT_ORDER=["Clothing","Electronics","Furniture","Groceries","Toys"]

plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":AX,"axes.edgecolor":"#1a3048",
    "axes.labelcolor":TC,"xtick.color":TC,"ytick.color":TC,"text.color":TC,
    "grid.color":"#0d1e2e","grid.linewidth":.5,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.spines.left":False,"axes.spines.bottom":False,
    "figure.dpi":130,"font.family":"DejaVu Sans",
    "legend.facecolor":AX,"legend.edgecolor":"#1a3048","legend.labelcolor":"#a0c8d8",
    "axes.titlecolor":TC,"axes.titlesize":10,
})

# ══════════════════════════════════════════════════════
# PATHS & SESSION STATE
# ══════════════════════════════════════════════════════
ROOT        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(ROOT,"data","outputs")
RESULT_PATH = os.path.join(OUTPUT_DIR,"model_results.json")
CHART_DIR   = os.path.join(OUTPUT_DIR,"charts")

# Session state for uploaded data
if "df" not in st.session_state:
    st.session_state.df = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
REQUIRED_COLS = {
    "Date","Store ID","Product ID","Category","Region",
    "Inventory Level","Units Sold","Units Ordered","Demand Forecast",
    "Price","Discount","Weather Condition","Holiday/Promotion",
    "Competitor Pricing","Seasonality"
}
COL_ALIASES = {
    # common variants users might name columns
    "date":"Date","store id":"Store ID","storeid":"Store ID",
    "product id":"Product ID","productid":"Product ID",
    "category":"Category","region":"Region",
    "inventory level":"Inventory Level","inventorylevel":"Inventory Level",
    "units sold":"Units Sold","unitssold":"Units Sold",
    "units ordered":"Units Ordered","unitsordered":"Units Ordered",
    "demand forecast":"Demand Forecast","demandforecast":"Demand Forecast",
    "price":"Price","discount":"Discount",
    "weather condition":"Weather Condition","weather":"Weather Condition",
    "holiday/promotion":"Holiday/Promotion","holiday":"Holiday/Promotion","promotion":"Holiday/Promotion",
    "competitor pricing":"Competitor Pricing","competitor price":"Competitor Pricing",
    "seasonality":"Seasonality",
    "season":"Seasonality",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-rename columns to expected names regardless of case/spacing."""
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace("_"," ")
        if normalized in COL_ALIASES:
            rename_map[col] = COL_ALIASES[normalized]
    return df.rename(columns=rename_map)

def fill_missing_columns(df: pd.DataFrame) -> tuple:
    """
    Fill missing required columns with sensible defaults.
    Returns (df_filled, list_of_filled_cols).
    
    Strategy per column:
    - Units Sold     → if missing, can't proceed (core metric). Try to infer from similar cols, else raise.
    - Date           → if missing, create synthetic daily dates.
    - Store ID       → 'S001' (single store)
    - Product ID     → 'P001' (single product)
    - Category       → 'General'
    - Region         → 'Unknown'
    - Inventory Level → 3× Units Sold (reasonable coverage assumption)
    - Units Ordered   → same as Units Sold
    - Demand Forecast → same as Units Sold (no error assumption)
    - Price          → median of existing 'price'-like cols, else 50.0
    - Discount       → 0 (no discount)
    - Weather Condition → 'Sunny'
    - Holiday/Promotion → 0
    - Competitor Pricing → same as Price
    - Seasonality    → infer from month if Date present, else 'Summer'
    """
    df = df.copy()
    filled = []

    # ── Core: Units Sold ──
    if "Units Sold" not in df.columns:
        # Try common alternatives
        for alt in ["quantity","qty","sales","quantity_sold","amount_sold","sold"]:
            match = [c for c in df.columns if c.strip().lower().replace(" ","_") == alt]
            if match:
                df["Units Sold"] = pd.to_numeric(df[match[0]], errors="coerce").fillna(1).astype(int)
                filled.append(f"Units Sold ← '{match[0]}'")
                break
        else:
            # Last resort: fill with 1
            df["Units Sold"] = 1
            filled.append("Units Sold = 1 (không tìm thấy cột tương đương)")

    # ── Date ──
    if "Date" not in df.columns:
        date_like = [c for c in df.columns if "date" in c.strip().lower() or "time" in c.strip().lower()]
        if date_like:
            df["Date"] = pd.to_datetime(df[date_like[0]], errors="coerce")
            filled.append(f"Date ← '{date_like[0]}'")
        else:
            df["Date"] = pd.date_range(start="2022-01-01", periods=len(df), freq="D")
            filled.append("Date = synthetic 2022-01-01 + index days")

    # ── Identifiers ──
    if "Store ID" not in df.columns:
        store_like = [c for c in df.columns if "store" in c.strip().lower()]
        if store_like:
            df["Store ID"] = df[store_like[0]].astype(str)
            filled.append(f"Store ID ← '{store_like[0]}'")
        else:
            df["Store ID"] = "S001"
            filled.append("Store ID = 'S001'")

    if "Product ID" not in df.columns:
        prod_like = [c for c in df.columns if "product" in c.strip().lower() or "item" in c.strip().lower() or "sku" in c.strip().lower()]
        if prod_like:
            df["Product ID"] = df[prod_like[0]].astype(str)
            filled.append(f"Product ID ← '{prod_like[0]}'")
        else:
            df["Product ID"] = "P001"
            filled.append("Product ID = 'P001'")

    if "Category" not in df.columns:
        cat_like = [c for c in df.columns if "categ" in c.strip().lower() or "type" in c.strip().lower() or "group" in c.strip().lower()]
        if cat_like:
            df["Category"] = df[cat_like[0]].astype(str)
            filled.append(f"Category ← '{cat_like[0]}'")
        else:
            df["Category"] = "General"
            filled.append("Category = 'General'")

    if "Region" not in df.columns:
        reg_like = [c for c in df.columns if "region" in c.strip().lower() or "location" in c.strip().lower() or "area" in c.strip().lower() or "city" in c.strip().lower()]
        if reg_like:
            df["Region"] = df[reg_like[0]].astype(str)
            filled.append(f"Region ← '{reg_like[0]}'")
        else:
            df["Region"] = "Unknown"
            filled.append("Region = 'Unknown'")

    # ── Price ──
    if "Price" not in df.columns:
        price_like = [c for c in df.columns if "price" in c.strip().lower() or "cost" in c.strip().lower() or "amount" in c.strip().lower() or "value" in c.strip().lower()]
        if price_like:
            vals = pd.to_numeric(df[price_like[0]], errors="coerce").fillna(50.0)
            df["Price"] = vals
            filled.append(f"Price ← '{price_like[0]}'")
        else:
            df["Price"] = 50.0
            filled.append("Price = 50.0 (mặc định)")

    # ── Derived numerics ──
    if "Inventory Level" not in df.columns:
        df["Inventory Level"] = (pd.to_numeric(df["Units Sold"], errors="coerce").fillna(1) * 3).astype(int)
        filled.append("Inventory Level = Units Sold × 3")

    if "Units Ordered" not in df.columns:
        df["Units Ordered"] = df["Units Sold"]
        filled.append("Units Ordered = Units Sold")

    if "Demand Forecast" not in df.columns:
        df["Demand Forecast"] = df["Units Sold"]
        filled.append("Demand Forecast = Units Sold (MAPE = 0%)")

    if "Competitor Pricing" not in df.columns:
        df["Competitor Pricing"] = df["Price"]
        filled.append("Competitor Pricing = Price (gap = 0%)")

    # ── Categorical / flags ──
    if "Discount" not in df.columns:
        disc_like = [c for c in df.columns if "discount" in c.strip().lower() or "promo" in c.strip().lower()]
        if disc_like:
            df["Discount"] = pd.to_numeric(df[disc_like[0]], errors="coerce").fillna(0)
            filled.append(f"Discount ← '{disc_like[0]}'")
        else:
            df["Discount"] = 0
            filled.append("Discount = 0%")

    if "Holiday/Promotion" not in df.columns:
        hol_like = [c for c in df.columns if "holiday" in c.strip().lower() or "promotion" in c.strip().lower()]
        if hol_like:
            df["Holiday/Promotion"] = pd.to_numeric(df[hol_like[0]], errors="coerce").fillna(0).astype(int)
            filled.append(f"Holiday/Promotion ← '{hol_like[0]}'")
        else:
            df["Holiday/Promotion"] = 0
            filled.append("Holiday/Promotion = 0")

    if "Weather Condition" not in df.columns:
        df["Weather Condition"] = "Sunny"
        filled.append("Weather Condition = 'Sunny'")

    if "Seasonality" not in df.columns:
        # Infer from month if Date exists
        try:
            months = pd.to_datetime(df["Date"]).dt.month
            season_map = {12:"Winter",1:"Winter",2:"Winter",
                          3:"Spring",4:"Spring",5:"Spring",
                          6:"Summer",7:"Summer",8:"Summer",
                          9:"Fall",10:"Fall",11:"Fall"}
            df["Seasonality"] = months.map(season_map).fillna("Summer")
            filled.append("Seasonality = tự suy từ tháng (Date)")
        except Exception:
            df["Seasonality"] = "Summer"
            filled.append("Seasonality = 'Summer'")

    return df, filled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Store ID","Product ID","Category","Region","Weather Condition","Seasonality"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    # Cap outliers
    cap = df["Units Sold"].quantile(0.99)
    df["Units Sold"] = df["Units Sold"].clip(upper=cap).round(0).astype(int)
    # Rename
    df = df.rename(columns={
        "Store ID":"store_id","Product ID":"product_id","Category":"category",
        "Region":"region","Inventory Level":"inventory_level","Units Sold":"units_sold",
        "Units Ordered":"units_ordered","Demand Forecast":"demand_forecast",
        "Price":"price","Discount":"discount_pct","Weather Condition":"weather",
        "Holiday/Promotion":"is_holiday","Competitor Pricing":"competitor_price",
        "Seasonality":"seasonality_label",
    })
    df = df.sort_values(["store_id","product_id","Date"]).reset_index(drop=True)
    # Time
    df["year"]    = df["Date"].dt.year
    df["month"]   = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    # Revenue
    df["revenue"]       = (df["units_sold"]*df["price"]*(1-df["discount_pct"]/100)).round(2)
    df["gross_revenue"] = (df["units_sold"]*df["price"]).round(2)
    df["price_gap_pct"] = ((df["price"]-df["competitor_price"])/df["competitor_price"]*100).round(2)
    # Inventory
    df["stock_coverage_days"] = (df["inventory_level"]/df["units_sold"].replace(0,np.nan)).round(2)
    df["stockout_risk"]       = (df["inventory_level"]<df["units_sold"]).astype(int)
    df["reorder_gap"]         = df["units_ordered"]-df["units_sold"]
    # Forecast quality
    df["forecast_error"] = (df["units_sold"]-df["demand_forecast"]).round(2)
    df["forecast_mape"]  = (df["forecast_error"].abs()/df["units_sold"].replace(0,np.nan)*100).round(2)
    df["forecast_bias"]  = np.where(df["forecast_error"]>0,"Under-forecast",
                           np.where(df["forecast_error"]<0,"Over-forecast","Accurate"))
    # Lags & rolling
    df = df.sort_values(["store_id","product_id","Date"])
    for lag in [1,7,30]:
        df[f"units_sold_lag{lag}"] = df.groupby(["store_id","product_id"])["units_sold"].shift(lag)
    for win in [7,30]:
        df[f"units_sold_ma{win}"] = (
            df.groupby(["store_id","product_id"])["units_sold"]
            .transform(lambda x: x.shift(1).rolling(win,min_periods=1).mean()).round(2))
    # Growth
    monthly = (df.groupby(["store_id","product_id","year","month"])["revenue"]
               .sum().reset_index())
    monthly["revenue_mom_growth"] = (
        monthly.groupby(["store_id","product_id"])["revenue"].pct_change()*100).round(2)
    df = df.merge(monthly[["store_id","product_id","year","month","revenue_mom_growth"]],
                  on=["store_id","product_id","year","month"],how="left")
    # Prescriptive
    df["reorder_recommended"]  = ((df["stock_coverage_days"]<3)|(df["stockout_risk"]==1)).astype(int)
    df["discount_ineffective"] = ((df["discount_pct"]>0)&(df["forecast_error"]<-20)).astype(int)
    df["pricing_action"]       = np.where(df["price_gap_pct"]>10,"Consider price reduction",
                                 np.where(df["price_gap_pct"]<-10,"Room to increase price","Competitive"))
    return df

@st.cache_data(show_spinner=False)
def load_default():
    path = os.path.join(ROOT,"data","raw","retail_store_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = normalize_columns(df)
        df = engineer_features(df)
        df = df[~((df["year"]==2024)&(df["month"]==1))]
        return df
    return None

@st.cache_data(show_spinner=False)
def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return None

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight",facecolor=BG,dpi=130)
    buf.seek(0)
    st.image(buf, use_container_width="always")
    plt.close(fig)

def dark_table(df_in, num_cols=None, hi_map=None, max_rows=50, height=400):
    """Render a pandas DataFrame as a fully dark-themed HTML table."""
    import numpy as np
    df_show = df_in.head(max_rows).copy()
    num_cols = num_cols or []
    hi_map   = hi_map or {}
    rows_html = ""
    for _, row in df_show.iterrows():
        cells = ""
        for col in df_show.columns:
            val = row[col]
            css = "num" if col in num_cols else ""
            if col in hi_map:
                fn, cls = hi_map[col]
                try:
                    if fn(val): css = cls
                except: pass
            if isinstance(val, float):
                display = f"{val:,.2f}"
            elif isinstance(val, (int,)) or (hasattr(val,"item") and isinstance(val.item(), int)):
                display = f"{int(val):,}"
            else:
                display = str(val)
            cells += f'<td class="{css}">{display}</td>'
        rows_html += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df_show.columns)
    html = f"""
    <div class="rc-table-wrap" style="max-height:{height}px">
      <table class="rc-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def kpi(val,label,sub=None,sub_cls=""):
    sub_html = f'<div class="kpi-sub {sub_cls}">{sub}</div>' if sub else '<div class="kpi-sub neutral">&nbsp;</div>'
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-val">{val}</div>
        <div class="kpi-lbl">{label}</div>
        {sub_html}
    </div>""",unsafe_allow_html=True)

def section(tag,title,sub=""):
    st.markdown(f'<div class="sec-label">{tag}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>',unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p style="color:#6aaabb;font-size:14px;margin-bottom:20px">{sub}</p>',unsafe_allow_html=True)

def get_df():
    """Return only the uploaded dataframe — never fall back to default file."""
    return st.session_state.df  # None if not uploaded yet

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px">
        <div style="font-size:15px;font-weight:800;letter-spacing:3px;text-transform:uppercase;
            background:linear-gradient(90deg,#00c8dc,#9b5de5);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
            RETAILCAST</div>
        <div style="font-size:9px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
            color:#4a8aa8;margin-top:5px">DEMAND FORECASTING</div>
    </div>""",unsafe_allow_html=True)

    page = st.radio("NAVIGATION",[
        "DATA IMPORT",
        "OVERVIEW",
        "DESCRIPTIVE",
        "DIAGNOSTIC",
        "PREDICTIVE",
        "PRESCRIPTIVE",
        "FORECAST",
    ])
    st.markdown("<hr style='border-color:#122030;margin:16px 0'>",unsafe_allow_html=True)
    df_check = get_df()
    if df_check is not None:
        st.markdown(f"""
        <div style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a8aa8;line-height:2.4">
        ACTIVE DATASET<br>
        <span style="color:#7ab8cc;font-weight:400;text-transform:none;letter-spacing:0">
        {len(df_check):,} rows · {df_check.shape[1]} features<br>
        {df_check['Date'].min().strftime('%b %Y')} – {df_check['Date'].max().strftime('%b %Y')}<br>
        {df_check['store_id'].nunique()} stores · {df_check['category'].nunique()} categories
        </span></div>""",unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert alert-amber" style="font-size:11px">
        No data loaded.<br>Go to DATA IMPORT first.
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: DATA IMPORT
# ══════════════════════════════════════════════════════
if page == "DATA IMPORT":
    section("STEP 1","DATA IMPORT & VALIDATION",
            "Upload your CSV file. The system auto-detects and standardizes column names.")

    col_up, col_info = st.columns([1.4, 1])

    with col_up:
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">UPLOAD FILE</div>',unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv"],
                                    help="CSV with retail inventory/sales data")

        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                st.markdown(f'<div class="alert alert-blue">Loaded <strong>{len(raw):,} rows × {raw.shape[1]} columns</strong> from <code>{uploaded.name}</code></div>',unsafe_allow_html=True)

                # Normalize columns
                raw_norm = normalize_columns(raw)
                found    = set(raw_norm.columns)
                missing  = REQUIRED_COLS - found

                if missing:
                    # Auto-fill missing columns with smart defaults
                    raw_filled, filled_list = fill_missing_columns(raw_norm)

                    # Show what was filled
                    st.markdown(f'<div class="alert alert-amber">⚠ <strong>{len(missing)} cột không tìm thấy</strong> — đã tự động tạo giá trị mặc định để vẫn chạy đầy đủ.</div>', unsafe_allow_html=True)

                    fill_rows = []
                    for item in filled_list:
                        fill_rows.append({"Cột được tạo": item.split("=")[0].split("←")[0].strip(),
                                          "Giá trị / Nguồn": item})
                    if fill_rows:
                        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin:10px 0 6px">GIÁ TRỊ MẶC ĐỊNH ĐÃ ĐƯỢC TẠO</div>', unsafe_allow_html=True)
                        dark_table(pd.DataFrame(fill_rows), height=min(40 + len(fill_rows)*38, 360))

                    raw_norm = raw_filled

                # Engineer features
                with st.spinner("Processing data..."):
                    df_processed = engineer_features(raw_norm)
                    # Remove partial months at edges
                    month_counts = df_processed.groupby(["year","month"]).size()
                    median_count = month_counts.median()
                    valid_months = month_counts[month_counts >= median_count * 0.5].reset_index()[["year","month"]]
                    df_processed = df_processed.merge(valid_months, on=["year","month"])

                st.session_state.df = df_processed
                st.session_state.data_source = uploaded.name
                n_filled = len(missing) if missing else 0
                note = f" · {n_filled} cột được tự động điền" if n_filled else ""
                st.markdown(f'<div class="alert alert-green">✓ Data processed successfully — <strong>{len(df_processed):,} rows</strong> ready for analysis{note}.</div>',unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="alert alert-red">❌ Error reading file: <strong>{e}</strong></div>',unsafe_allow_html=True)

        else:
            st.markdown('<div class="alert alert-amber">📂 Chưa có dữ liệu. Vui lòng upload file CSV ở trên để bắt đầu.</div>',unsafe_allow_html=True)

    with col_info:
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:6px">COLUMNS & GIÁ TRỊ MẶC ĐỊNH</div>',unsafe_allow_html=True)
        st.markdown('<div class="alert alert-green" style="font-size:11px;margin-bottom:10px">✅ Cột thiếu sẽ tự động được điền — không cần đủ tất cả.</div>', unsafe_allow_html=True)
        req_rows = [
            {"Cột":"Date","Bắt buộc":"✓","Mặc định nếu thiếu":"synthetic 2022-01-01+"},
            {"Cột":"Units Sold","Bắt buộc":"✓","Mặc định nếu thiếu":"tìm qty/sales/sold"},
            {"Cột":"Category","Bắt buộc":"","Mặc định nếu thiếu":"'General'"},
            {"Cột":"Store ID","Bắt buộc":"","Mặc định nếu thiếu":"'S001'"},
            {"Cột":"Product ID","Bắt buộc":"","Mặc định nếu thiếu":"'P001'"},
            {"Cột":"Region","Bắt buộc":"","Mặc định nếu thiếu":"'Unknown'"},
            {"Cột":"Price","Bắt buộc":"","Mặc định nếu thiếu":"tìm price/cost, else 50"},
            {"Cột":"Inventory Level","Bắt buộc":"","Mặc định nếu thiếu":"Units Sold × 3"},
            {"Cột":"Units Ordered","Bắt buộc":"","Mặc định nếu thiếu":"= Units Sold"},
            {"Cột":"Demand Forecast","Bắt buộc":"","Mặc định nếu thiếu":"= Units Sold"},
            {"Cột":"Competitor Pricing","Bắt buộc":"","Mặc định nếu thiếu":"= Price"},
            {"Cột":"Discount","Bắt buộc":"","Mặc định nếu thiếu":"0%"},
            {"Cột":"Holiday/Promotion","Bắt buộc":"","Mặc định nếu thiếu":"0"},
            {"Cột":"Weather Condition","Bắt buộc":"","Mặc định nếu thiếu":"'Sunny'"},
            {"Cột":"Seasonality","Bắt buộc":"","Mặc định nếu thiếu":"suy từ tháng"},
        ]
        dark_table(pd.DataFrame(req_rows), height=520)

    # Show data preview if loaded
    df = get_df()
    if df is not None:
        st.markdown('<hr class="divider">',unsafe_allow_html=True)

        # ── VALIDATION KPIs ──
        raw_cols = ["store_id","product_id","category","region","units_sold",
                    "price","discount_pct","inventory_level","demand_forecast",
                    "weather","is_holiday","competitor_price"]
        raw_nulls = df[[c for c in raw_cols if c in df.columns]].isnull().sum().sum()
        cats      = sorted(df["category"].unique())
        cat_label = ", ".join(cats[:3]) + (f" +{len(cats)-3}" if len(cats)>3 else "")
        d_min = df["Date"].min().strftime("%b'%y")
        d_max = df["Date"].max().strftime("%b'%y")
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi(f"{len(df):,}","TOTAL ROWS",f"{df.shape[1]} cột sau xử lý")
        with c2: kpi("✓ Sạch" if raw_nulls==0 else f"{raw_nulls:,}","GIÁ TRỊ NULL",
                     "Không có lỗi" if raw_nulls==0 else "Cần kiểm tra",
                     "" if raw_nulls==0 else "warn")
        with c3: kpi(f"{len(cats)}","DANH MỤC", cat_label)
        with c4: kpi(f"{d_min}–{d_max}","KHOẢNG NGÀY",f"{df['Date'].nunique():,} ngày")

        # ── QUICK STATS CHARTS ──
        st.markdown('<hr class="divider">',unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:12px">TỔNG QUAN NHANH</div>',unsafe_allow_html=True)
        qa1,qa2,qa3 = st.columns(3)
        with qa1:
            cat_rev = df.groupby("category")["units_sold"].sum().sort_values()
            fig,ax = plt.subplots(figsize=(4,2.8))
            ax.barh(cat_rev.index, cat_rev.values/1e3, color=PALETTE[:len(cat_rev)], edgecolor="none", height=0.55)
            ax.set_xlabel("Units Sold (K)"); ax.grid(axis="x")
            ax.set_title("UNITS SOLD BY CATEGORY",fontsize=9,fontweight="bold",pad=6)
            fig.tight_layout(); fig_to_img(fig)
        with qa2:
            store_rev = df.groupby("store_id")["revenue"].sum().sort_values()
            fig,ax = plt.subplots(figsize=(4,2.8))
            ax.barh(store_rev.index, store_rev.values/1e6, color=BLUE, edgecolor="none", height=0.55)
            ax.set_xlabel("Revenue (USD M)"); ax.grid(axis="x")
            ax.set_title("REVENUE BY STORE",fontsize=9,fontweight="bold",pad=6)
            fig.tight_layout(); fig_to_img(fig)
        with qa3:
            mo = df.groupby(["year","month"])["units_sold"].sum().reset_index()
            mo["period"] = pd.to_datetime(mo["year"].astype(str)+"-"+mo["month"].astype(str).str.zfill(2))
            mo = mo.sort_values("period")
            fig,ax = plt.subplots(figsize=(4,2.8))
            ax.fill_between(mo["period"], mo["units_sold"]/1e3, alpha=0.15, color=GREEN)
            ax.plot(mo["period"], mo["units_sold"]/1e3, color=GREEN, lw=1.8)
            ax.set_ylabel("Units (K)"); ax.grid(axis="y")
            ax.set_title("TREND HÀNG THÁNG",fontsize=9,fontweight="bold",pad=6)
            fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

        # ── DATA PREVIEW TABLE ──
        st.markdown('<hr class="divider">',unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">XEM TRƯỚC DỮ LIỆU (50 hàng đầu)</div>',unsafe_allow_html=True)
        preview_cols = ["Date","store_id","category","region","units_sold","revenue",
                        "price","discount_pct","inventory_level","stock_coverage_days",
                        "demand_forecast","forecast_error","reorder_recommended"]
        available = [c for c in preview_cols if c in df.columns]
        prev_df = df[available].head(50).copy()
        prev_df["Date"] = prev_df["Date"].dt.strftime("%Y-%m-%d")
        dark_table(
            prev_df,
            num_cols=["units_sold","revenue","price","discount_pct","inventory_level",
                      "stock_coverage_days","demand_forecast","forecast_error"],
            hi_map={
                "reorder_recommended": (lambda v: v==1, "warn"),
                "stock_coverage_days": (lambda v: isinstance(v,(int,float)) and v<2, "warn"),
            },
            max_rows=50, height=380
        )

# Guard: require data for all other pages
else:
    df = get_df()
    if df is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px">
            <div style="font-size:48px;margin-bottom:16px">▣</div>
            <div class="page-title" style="text-align:center">NO DATA LOADED</div>
            <p style="color:#6aaabb;font-size:15px;margin-top:8px">
                Go to <strong style="color:#00c8dc">DATA IMPORT</strong> in the sidebar to upload your dataset first.
            </p>
        </div>""",unsafe_allow_html=True)
        st.stop()

    # ══════════════════════════════════════════════════
    # OVERVIEW
    # ══════════════════════════════════════════════════
    if page == "OVERVIEW":
        section("RETAILCAST · FAA4023","RETAIL DEMAND FORECASTING",
                "Automated pipeline: data cleaning → 4-type EDA → 3 forecasting models → inventory alerts")

        st.markdown('<hr class="divider">',unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:12px">KEY PERFORMANCE INDICATORS</div>',unsafe_allow_html=True)

        rev_first = df[df["year"]==df["year"].min()]["revenue"].sum()
        rev_last  = df[df["year"]==df["year"].max()]["revenue"].sum()
        yoy = (rev_last-rev_first)/rev_first*100 if rev_first>0 else 0
        avg_forecast_mape = df["forecast_mape"].mean() if "forecast_mape" in df.columns else 0

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: kpi(f"${df['revenue'].sum()/1e6:.0f}M","TOTAL REVENUE","All periods")
        with c2: kpi(f"{df['units_sold'].sum()/1e6:.1f}M","UNITS SOLD","All stores")
        with c3: kpi(f"{df['store_id'].nunique()}","STORES",f"{df['category'].nunique()} categories")
        with c4: kpi(f"{df['reorder_recommended'].sum():,}","REORDER FLAGS","Urgent action","warn")
        with c5: kpi(f"{avg_forecast_mape:.0f}%","AVG FORECAST MAPE","From imported data","warn")
        with c6: kpi(f"{yoy:+.1f}%","YOY GROWTH",f"{df['year'].min()}→{df['year'].max()}","neutral")

        st.markdown('<hr class="divider">',unsafe_allow_html=True)

        # Hero chart
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">MONTHLY REVENUE TREND</div>',unsafe_allow_html=True)
        monthly = df.groupby(["year","month"])["revenue"].sum().reset_index()
        monthly["period"] = pd.to_datetime(monthly["year"].astype(str)+"-"+monthly["month"].astype(str).str.zfill(2))
        monthly = monthly.sort_values("period")

        fig,ax = plt.subplots(figsize=(12,3.5))
        ax.fill_between(monthly["period"],monthly["revenue"]/1e6,alpha=0.12,color=BLUE)
        ax.plot(monthly["period"],monthly["revenue"]/1e6,color=BLUE,lw=2.2,marker="o",ms=4)
        # Year separators
        for yr in df["year"].unique()[1:]:
            sep = pd.Timestamp(f"{yr}-01-01")
            ax.axvline(sep,color="#1a2a3a",ls="--",lw=1)
            ax.text(sep,monthly["revenue"].max()/1e6*0.97,f"  {yr}",fontsize=9,color=TC)
        ax.set_ylabel("USD Million",fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
        ax.grid(axis="y"); fig.autofmt_xdate(rotation=30); fig.tight_layout()
        fig_to_img(fig)

        col_a,col_b = st.columns(2)
        with col_a:
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">REVENUE BY CATEGORY</div>',unsafe_allow_html=True)
            cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
            fig,ax = plt.subplots(figsize=(6,3.2))
            colors_ = [BLUE if i==len(cat_rev)-1 else "#0d2535" for i in range(len(cat_rev))]
            bars = ax.barh(cat_rev.index,cat_rev.values/1e6,color=colors_,edgecolor="none",height=0.55)
            for bar in bars:
                ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                        f"${bar.get_width():.0f}M",va="center",fontsize=9,color="#5ba0b8")
            ax.set_xlabel("Revenue (USD M)",fontsize=9); ax.grid(axis="x")
            ax.set_xlim(0,cat_rev.max()/1e6*1.22); fig.tight_layout(); fig_to_img(fig)

        with col_b:
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">DATA SUMMARY</div>',unsafe_allow_html=True)
            # Live stats from imported data
            top_cat   = df.groupby("category")["revenue"].sum().idxmax()
            top_store = df.groupby("store_id")["revenue"].sum().idxmax()
            stockout_rate = df["stockout_risk"].mean()*100
            avg_disc  = df["discount_pct"].mean()
            reorder_rate = df["reorder_recommended"].mean()*100
            date_range = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"
            stats_rows = [
                ("📅 Khoảng thời gian", date_range),
                ("🏆 Category doanh thu cao nhất", top_cat),
                ("🏪 Store doanh thu cao nhất", top_store),
                ("⚠ Tỷ lệ stockout risk", f"{stockout_rate:.1f}%"),
                ("💸 Chiết khấu trung bình", f"{avg_disc:.1f}%"),
                ("🔔 Tỷ lệ cần reorder", f"{reorder_rate:.1f}%"),
                ("📦 Tổng sản phẩm", f"{df['product_id'].nunique():,}"),
                ("📍 Số vùng", f"{df['region'].nunique()}"),
            ]
            rows_html = ""
            for label, val in stats_rows:
                rows_html += f"""<div style="display:flex;align-items:center;justify-content:space-between;
                    padding:8px 14px;background:#0d1828;border:1px solid #1a3048;border-radius:6px;margin-bottom:5px">
                    <span style="font-size:11px;font-weight:600;color:#7ab8cc">{label}</span>
                    <span style="font-size:12px;font-weight:700;color:#00c8dc">{val}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:10px;color:#5a9ab8;margin-top:4px">Dữ liệu từ file đã import · Chạy SARIMA → trang <strong style="color:#00c8dc">PREDICTIVE / FORECAST</strong></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # DESCRIPTIVE
    # ══════════════════════════════════════════════════
    elif page == "DESCRIPTIVE":
        section("DESCRIPTIVE ANALYSIS","WHAT HAPPENED?",
                "Revenue trends, category performance, store and regional breakdown.")

        # FILTERS — checkboxes
        with st.expander("FILTERS", expanded=True):
            fc1,fc2,fc3 = st.columns(3)
            with fc1:
                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:#5a9ab8;margin-bottom:6px">CATEGORY</div>',unsafe_allow_html=True)
                all_cats = sorted(df["category"].unique())
                sel_cat = [c for c in all_cats if st.checkbox(c, value=True, key=f"cat_{c}")]
            with fc2:
                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:#5a9ab8;margin-bottom:6px">STORE</div>',unsafe_allow_html=True)
                all_stores = sorted(df["store_id"].unique())
                sel_store = [s for s in all_stores if st.checkbox(s, value=True, key=f"store_{s}")]
            with fc3:
                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:#5a9ab8;margin-bottom:6px">YEAR</div>',unsafe_allow_html=True)
                all_years = sorted(df["year"].unique())
                sel_year = [y for y in all_years if st.checkbox(str(y), value=True, key=f"yr_{y}")]

        if not sel_cat or not sel_store or not sel_year:
            st.markdown('<div class="alert alert-amber">Select at least one option per filter.</div>',unsafe_allow_html=True)
            st.stop()

        dff = df[df["category"].isin(sel_cat)&df["store_id"].isin(sel_store)&df["year"].isin(sel_year)]

        # KPIs — equal height via kpi()
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi(f"${dff['revenue'].sum()/1e6:.1f}M","TOTAL REVENUE")
        with c2: kpi(f"{dff['units_sold'].sum()/1e6:.1f}M","UNITS SOLD")
        with c3: kpi(f"${dff['price'].mean():.0f}","AVG PRICE USD")
        with c4: kpi(f"{dff['discount_pct'].mean():.1f}%","AVG DISCOUNT")
        with c5: kpi(f"${dff['revenue'].mean():.0f}","AVG DAILY REV")

        st.markdown('<hr class="divider">',unsafe_allow_html=True)
        t1,t2,t3 = st.tabs(["REVENUE TREND","BY CATEGORY","BY STORE & REGION"])

        with t1:
            monthly = dff.groupby(["year","month"])["revenue"].sum().reset_index()
            monthly["period"] = pd.to_datetime(monthly["year"].astype(str)+"-"+monthly["month"].astype(str).str.zfill(2))
            monthly = monthly.sort_values("period")
            fig,ax = plt.subplots(figsize=(12,4))
            ax.fill_between(monthly["period"],monthly["revenue"]/1e6,alpha=0.1,color=BLUE)
            ax.plot(monthly["period"],monthly["revenue"]/1e6,color=BLUE,lw=2.2,marker="o",ms=5)
            idx_max = monthly["revenue"].idxmax()
            ax.annotate(f'${monthly.loc[idx_max,"revenue"]/1e6:.1f}M',
                xy=(monthly.loc[idx_max,"period"],monthly.loc[idx_max,"revenue"]/1e6),
                xytext=(0,14),textcoords="offset points",ha="center",fontsize=9,color=GREEN,
                fontweight="bold",arrowprops=dict(arrowstyle="->",color=GREEN,lw=1))
            ax.set_ylabel("USD Million"); ax.grid(axis="y")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
            fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

            # Quarterly bar
            q = dff.groupby(["year","quarter"])["revenue"].sum().reset_index()
            q["label"] = q["year"].astype(str)+" Q"+q["quarter"].astype(str)
            fig,ax = plt.subplots(figsize=(12,3))
            colors_ = [BLUE if i%2==0 else PURPLE for i in range(len(q))]
            bars = ax.bar(q["label"],q["revenue"]/1e6,color=colors_,edgecolor="none",width=0.6)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                        f"${bar.get_height():.0f}M",ha="center",fontsize=8,color="#5ba0b8")
            ax.set_ylabel("USD Million"); ax.grid(axis="y")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
            plt.xticks(rotation=30,ha="right"); fig.tight_layout(); fig_to_img(fig)

        with t2:
            col_a,col_b = st.columns(2)
            with col_a:
                years = sorted(dff["year"].unique())
                cat_year = dff.groupby(["year","category"])["revenue"].sum().reset_index()
                cat_year["rev_M"] = cat_year["revenue"]/1e6
                pivot = cat_year.pivot(index="category",columns="year",values="rev_M").fillna(0)
                avail_cats = [c for c in CAT_ORDER if c in pivot.index]
                pivot = pivot.reindex(avail_cats)
                fig,ax = plt.subplots(figsize=(6,4))
                x = np.arange(len(avail_cats)); w=0.35/(max(len(years)-1,1)+1)*2
                colors_yr=[BLUE,GREEN,AMBER,RED,PURPLE]
                for j,(yr,color) in enumerate(zip(sorted(pivot.columns),colors_yr)):
                    ax.bar(x+(j-(len(years)-1)/2)*w,pivot[yr],w,label=str(yr),color=color,edgecolor="none",alpha=0.9)
                ax.set_xticks(x); ax.set_xticklabels(avail_cats,rotation=15,ha="right",fontsize=9)
                ax.set_ylabel("Revenue (USD M)"); ax.legend(fontsize=9); ax.grid(axis="y")
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
                ax.set_title("REVENUE BY CATEGORY",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_b:
                cat_monthly = dff.groupby(["year","month","category"])["units_sold"].sum().reset_index()
                cat_monthly["period"] = pd.to_datetime(cat_monthly["year"].astype(str)+"-"+cat_monthly["month"].astype(str).str.zfill(2))
                fig,ax = plt.subplots(figsize=(6,4))
                for cat,color in zip([c for c in CAT_ORDER if c in sel_cat],PALETTE):
                    sub = cat_monthly[cat_monthly["category"]==cat].sort_values("period")
                    if not sub.empty:
                        ax.plot(sub["period"],sub["units_sold"],label=cat,color=color,lw=1.8)
                ax.set_ylabel("Units Sold"); ax.legend(fontsize=8,ncol=2)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}"))
                ax.set_title("UNITS SOLD BY CATEGORY",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

            # Category share pie
            col_c,col_d = st.columns(2)
            with col_c:
                cat_share = dff.groupby("category")["revenue"].sum().reindex(
                    [c for c in CAT_ORDER if c in dff["category"].unique()])
                fig,ax = plt.subplots(figsize=(5,4),facecolor=BG)
                ax.set_facecolor(BG)
                wedges,texts,autotexts = ax.pie(
                    cat_share.values,labels=cat_share.index,colors=PALETTE[:len(cat_share)],
                    autopct="%1.1f%%",startangle=140,
                    wedgeprops=dict(linewidth=2,edgecolor=BG))
                for t in texts: t.set_color("#5ba0b8"); t.set_fontsize(10)
                for at in autotexts: at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
                ax.set_title("REVENUE SHARE",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_d:
                # YoY growth per category
                if len(years) >= 2:
                    y1,y2 = sorted(years)[-2],sorted(years)[-1]
                    g1 = dff[dff["year"]==y1].groupby("category")["revenue"].sum()
                    g2 = dff[dff["year"]==y2].groupby("category")["revenue"].sum()
                    growth = ((g2-g1)/g1*100).dropna().sort_values()
                    fig,ax = plt.subplots(figsize=(5,4))
                    colors_ = [GREEN if v>=0 else RED for v in growth.values]
                    bars = ax.barh(growth.index,growth.values,color=colors_,edgecolor="none",height=0.5)
                    ax.axvline(0,color="#1a3048",lw=1)
                    for bar,val in zip(bars,growth.values):
                        ax.text(val+(1 if val>=0 else -1),bar.get_y()+bar.get_height()/2,
                                f"{val:+.1f}%",va="center",fontsize=9,color="#5ba0b8")
                    ax.set_xlabel("YoY Growth (%)")
                    ax.set_title(f"YOY GROWTH {y1}→{y2}",fontsize=10,fontweight="bold",color=TC,pad=8)
                    ax.grid(axis="x"); fig.tight_layout(); fig_to_img(fig)

        with t3:
            col_a,col_b = st.columns(2)
            with col_a:
                store_rev = dff.groupby("store_id")["revenue"].sum().sort_values(ascending=True)
                fig,ax = plt.subplots(figsize=(5,3.5))
                colors_ = [BLUE if i==len(store_rev)-1 else "#0d2535" for i in range(len(store_rev))]
                ax.barh(store_rev.index,store_rev.values/1e6,color=colors_,edgecolor="none")
                for i,(idx,val) in enumerate(store_rev.items()):
                    ax.text(val/1e6+0.2,i,f"${val/1e6:.0f}M",va="center",fontsize=9,color="#5ba0b8")
                ax.set_xlabel("Revenue (USD M)"); ax.grid(axis="x")
                ax.set_title("REVENUE BY STORE",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_b:
                sr = dff.groupby(["store_id","region"])["revenue"].sum().reset_index()
                pivot_sr = sr.pivot(index="store_id",columns="region",values="revenue").fillna(0)/1e6
                fig,ax = plt.subplots(figsize=(6,3.5))
                sns.heatmap(pivot_sr.round(1),annot=True,fmt=".1f",
                            cmap=sns.light_palette(BLUE,as_cmap=True),
                            linewidths=0.5,linecolor=BG,ax=ax,cbar_kws={"label":"USD M"})
                ax.set_title("STORE × REGION (USD M)",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

    # ══════════════════════════════════════════════════
    # DIAGNOSTIC
    # ══════════════════════════════════════════════════
    elif page == "DIAGNOSTIC":
        section("DIAGNOSTIC ANALYSIS","WHY DID IT HAPPEN?",
                "Root causes: pricing vs competition, inventory gaps, promotion effects, forecast bias.")

        t1,t2,t3 = st.tabs(["PRICING & PROMOTIONS","INVENTORY HEALTH","FORECAST ACCURACY"])

        with t1:
            col_a,col_b = st.columns(2)
            with col_a:
                pg = df.groupby("category").agg(our=("price","mean"),comp=("competitor_price","mean"))
                pg = pg.reindex([c for c in CAT_ORDER if c in pg.index])
                pg["gap_pct"] = ((pg["our"]-pg["comp"])/pg["comp"]*100).round(2)
                fig,ax = plt.subplots(figsize=(6,4))
                x=np.arange(len(pg)); w=0.35
                ax.bar(x-w/2,pg["our"],w,label="OUR PRICE",color=BLUE,edgecolor="none",alpha=0.9)
                ax.bar(x+w/2,pg["comp"],w,label="COMPETITOR",color="#0d2535",edgecolor=BLUE,linewidth=0.7)
                for i,(idx,row) in enumerate(pg.iterrows()):
                    color = RED if row["gap_pct"]>5 else (GREEN if row["gap_pct"]<-5 else AMBER)
                    ax.text(i,max(row["our"],row["comp"])+1,f"{row['gap_pct']:+.1f}%",
                            ha="center",fontsize=8,color=color,fontweight="bold")
                ax.set_xticks(x); ax.set_xticklabels(pg.index,rotation=15,ha="right",fontsize=9)
                ax.set_ylabel("Avg Price (USD)"); ax.legend(fontsize=9); ax.grid(axis="y")
                ax.set_title("OUR PRICE VS COMPETITOR",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_b:
                # Discount effectiveness: discount_pct vs units_sold scatter
                sample = df.sample(min(3000,len(df)),random_state=42)
                fig,ax = plt.subplots(figsize=(6,4))
                for cat,color in zip([c for c in CAT_ORDER if c in df["category"].unique()],PALETTE):
                    sub = sample[sample["category"]==cat]
                    ax.scatter(sub["discount_pct"],sub["revenue"],alpha=0.2,color=color,s=12,label=cat)
                ax.set_xlabel("Discount (%)"); ax.set_ylabel("Revenue (USD)")
                ax.set_title("DISCOUNT vs REVENUE",fontsize=10,fontweight="bold",color=TC,pad=8)
                ax.legend(fontsize=8,ncol=2); ax.grid(axis="both")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
                fig.tight_layout(); fig_to_img(fig)

            col_c,col_d = st.columns(2)
            with col_c:
                hol = df.groupby(["is_holiday","category"])["revenue"].mean().reset_index()
                hol["label"] = hol["is_holiday"].map({0:"NO PROMO",1:"PROMO"})
                pivot_h = hol.pivot(index="category",columns="label",values="revenue").fillna(0)
                pivot_h = pivot_h.reindex([c for c in CAT_ORDER if c in pivot_h.index])
                fig,ax = plt.subplots(figsize=(6,3.5))
                x=np.arange(len(pivot_h)); w=0.35
                for j,(col_name,color) in enumerate(zip(["NO PROMO","PROMO"],[BLUE,GREEN])):
                    if col_name in pivot_h.columns:
                        ax.bar(x+(j-0.5)*w,pivot_h[col_name],w,label=col_name,color=color,edgecolor="none",alpha=0.9)
                ax.set_xticks(x); ax.set_xticklabels(pivot_h.index,rotation=15,ha="right",fontsize=9)
                ax.set_ylabel("Avg Revenue (USD)"); ax.legend(fontsize=9); ax.grid(axis="y")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
                ax.set_title("PROMOTION EFFECT ON REVENUE",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_d:
                weather_cat = df.groupby(["weather","category"])["units_sold"].mean().reset_index()
                pivot_w = weather_cat.pivot(index="weather",columns="category",values="units_sold").fillna(0)
                avail = [c for c in CAT_ORDER if c in pivot_w.columns]
                fig,ax = plt.subplots(figsize=(6,3.5))
                pivot_w[avail].plot(kind="bar",ax=ax,color=PALETTE[:len(avail)],width=0.7,edgecolor="none")
                ax.set_xlabel(""); ax.set_ylabel("Avg Units Sold")
                ax.set_xticklabels(pivot_w.index,rotation=0,fontsize=9)
                ax.legend(fontsize=8,ncol=3); ax.grid(axis="y")
                ax.set_title("WEATHER IMPACT ON SALES",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

        with t2:
            crit = int((df["stock_coverage_days"]<2).sum())
            warn_ = int(((df["stock_coverage_days"]>=2)&(df["stock_coverage_days"]<5)).sum())
            safe = int((df["stock_coverage_days"]>=5).sum())
            c1,c2,c3,c4 = st.columns(4)
            with c1: kpi(f"{crit:,}","CRITICAL","< 2 days stock","warn")
            with c2: kpi(f"{warn_:,}","WARNING","2–5 days stock","warn")
            with c3: kpi(f"{safe:,}","SAFE","≥ 5 days stock")
            with c4: kpi(f"{df['stockout_risk'].mean()*100:.1f}%","STOCKOUT RISK RATE","overall","warn")

            col_a,col_b = st.columns(2)
            with col_a:
                cov = df["stock_coverage_days"].clip(upper=30).dropna()
                fig,ax = plt.subplots(figsize=(6,3.5))
                ax.hist(cov,bins=35,color=BLUE,alpha=0.7,edgecolor="none")
                ax.axvline(2,color=RED,ls="--",lw=1.5,label="CRITICAL (<2d)")
                ax.axvline(5,color=AMBER,ls="--",lw=1.5,label="WARNING (<5d)")
                ax.set_xlabel("Days"); ax.set_ylabel("Frequency")
                ax.legend(fontsize=9); ax.grid(axis="y")
                ax.set_title("STOCK COVERAGE DISTRIBUTION",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_b:
                reorder_cat = df.groupby("category")["reorder_recommended"].sum().reindex(
                    [c for c in CAT_ORDER if c in df["category"].unique()])
                stockout_cat = df.groupby("category")["stockout_risk"].mean()*100
                fig,ax = plt.subplots(figsize=(6,3.5))
                x=np.arange(len(reorder_cat)); w=0.35
                ax.bar(x-w/2,reorder_cat.values,w,label="REORDER FLAGS",color=AMBER,edgecolor="none",alpha=0.9)
                ax2=ax.twinx()
                ax2.plot(x,stockout_cat.reindex(reorder_cat.index).values,"o-",color=RED,lw=2,ms=6,label="STOCKOUT %")
                ax2.set_ylabel("Stockout Risk %",color=RED); ax2.tick_params(axis="y",colors=RED)
                ax.set_xticks(x); ax.set_xticklabels(reorder_cat.index,rotation=15,ha="right",fontsize=9)
                ax.set_ylabel("Reorder Flags"); ax.grid(axis="y")
                ax.set_title("REORDER FLAGS & STOCKOUT RISK",fontsize=10,fontweight="bold",color=TC,pad=8)
                lines1,labels1 = ax.get_legend_handles_labels()
                lines2,labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1+lines2,labels1+labels2,fontsize=8)
                fig.tight_layout(); fig_to_img(fig)

        with t3:
            c1,c2 = st.columns(2)
            with c1:
                mape_cat = df.groupby("category")["forecast_mape"].mean().reindex(
                    [c for c in CAT_ORDER if c in df["category"].unique()])
                fig,ax = plt.subplots(figsize=(6,3.5))
                colors_ = [RED if v==mape_cat.max() else BLUE for v in mape_cat.values]
                bars = ax.bar(mape_cat.index,mape_cat.values,color=colors_,edgecolor="none")
                ax.axhline(25,color=AMBER,ls="--",lw=1,label="25% THRESHOLD")
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                            f"{bar.get_height():.1f}%",ha="center",fontsize=9,color="#5ba0b8")
                ax.set_ylabel("MAPE (%)"); ax.legend(fontsize=9); ax.grid(axis="y")
                ax.set_xticklabels(mape_cat.index,rotation=15,ha="right",fontsize=9)
                ax.set_title("EXISTING FORECAST MAPE",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with c2:
                bias = df.groupby(["category","forecast_bias"]).size().reset_index(name="cnt")
                pivot_b = bias.pivot(index="category",columns="forecast_bias",values="cnt").fillna(0)
                avail_cats2 = [c for c in CAT_ORDER if c in pivot_b.index]
                pivot_b = pivot_b.reindex(avail_cats2)
                cols_b = [c for c in ["Under-forecast","Accurate","Over-forecast"] if c in pivot_b.columns]
                fig,ax = plt.subplots(figsize=(6,3.5))
                bottom = np.zeros(len(avail_cats2))
                for col_b,color in zip(cols_b,[RED,GREEN,BLUE]):
                    ax.bar(avail_cats2,pivot_b[col_b],bottom=bottom,label=col_b,color=color,edgecolor="none",alpha=0.85)
                    bottom += pivot_b[col_b].values
                ax.set_ylabel("Records"); ax.legend(fontsize=9)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}"))
                ax.set_xticklabels(avail_cats2,rotation=15,ha="right",fontsize=9)
                ax.set_title("FORECAST BIAS BY CATEGORY",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            # Forecast error time series
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin:16px 0 8px">FORECAST ERROR OVER TIME (ALL STORES)</div>',unsafe_allow_html=True)
            err_monthly = df.groupby(["year","month"])["forecast_error"].mean().reset_index()
            err_monthly["period"] = pd.to_datetime(err_monthly["year"].astype(str)+"-"+err_monthly["month"].astype(str).str.zfill(2))
            fig,ax = plt.subplots(figsize=(12,3))
            ax.fill_between(err_monthly["period"],err_monthly["forecast_error"],
                            where=err_monthly["forecast_error"]>0,alpha=0.3,color=RED,label="Under-forecast")
            ax.fill_between(err_monthly["period"],err_monthly["forecast_error"],
                            where=err_monthly["forecast_error"]<0,alpha=0.3,color=BLUE,label="Over-forecast")
            ax.plot(err_monthly["period"],err_monthly["forecast_error"],color="#5ba0b8",lw=1.5)
            ax.axhline(0,color="#1a3048",lw=1)
            ax.set_ylabel("Avg Forecast Error"); ax.legend(fontsize=9); ax.grid(axis="y")
            fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

            st.markdown('<div class="alert alert-red">Existing demand forecast has consistent <strong>under-forecast bias</strong> (actual > forecast). Run SARIMA trên tab PREDICTIVE để so sánh với dữ liệu thực tế bạn đã import.</div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # PREDICTIVE
    # ══════════════════════════════════════════════════
    elif page == "PREDICTIVE":
        section("PREDICTIVE ANALYSIS","MODEL RESULTS",
                "SARIMA trained on historical data from imported file, evaluated on hold-out test period.")

        # Live info from imported data — no dependency on pipeline JSON
        n_months_pred = df.groupby(["year","month"]).ngroups
        date_min_pred = df["Date"].min().strftime("%b %Y")
        date_max_pred = df["Date"].max().strftime("%b %Y")
        st.markdown(f"""
        <div style="display:flex;gap:0;margin-bottom:20px">
            <div style="flex:1;padding:16px 18px;background:#0d1828;border:1px solid #00c8dc;
                        border-top:2px solid #00c8dc;border-radius:7px;margin:0 5px;text-align:center">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#7ab8cc">MÔ HÌNH</div>
                <div style="font-size:32px;font-weight:800;color:#00c8dc;margin:8px 0 2px">SARIMA</div>
                <div style="font-size:9px;color:#00c8dc;font-weight:700;margin-bottom:8px">SEASONAL ARIMA</div>
                <div style="font-size:11px;color:#5a9ab8;line-height:1.9">
                    (0,0,0)×(0,1,1,12)<br>Seasonal differencing<br>Monthly aggregation
                </div>
            </div>
            <div style="flex:1;padding:16px 18px;background:#0d1828;border:1px solid #1a3048;
                        border-top:2px solid #00e5a0;border-radius:7px;margin:0 5px;text-align:center">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#7ab8cc">DỮ LIỆU ĐÃ IMPORT</div>
                <div style="font-size:32px;font-weight:800;color:#00e5a0;margin:8px 0 2px">{n_months_pred}</div>
                <div style="font-size:9px;color:#00e5a0;font-weight:700;margin-bottom:8px">THÁNG DỮ LIỆU</div>
                <div style="font-size:11px;color:#5a9ab8;line-height:1.9">
                    {date_min_pred} → {date_max_pred}<br>
                    {df['store_id'].nunique()} stores · {df['category'].nunique()} cats<br>
                    {df['product_id'].nunique():,} products
                </div>
            </div>
            <div style="flex:1;padding:16px 18px;background:#0d1828;border:1px solid #1a3048;
                        border-top:2px solid #ffb400;border-radius:7px;margin:0 5px;text-align:center">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#7ab8cc">ĐÁNH GIÁ</div>
                <div style="font-size:32px;font-weight:800;color:#ffb400;margin:8px 0 2px">LIVE</div>
                <div style="font-size:9px;color:#ffb400;font-weight:700;margin-bottom:8px">REAL-TIME</div>
                <div style="font-size:11px;color:#5a9ab8;line-height:1.9">
                    MAPE tính trực tiếp<br>từ data import<br>Chọn test split bên dưới
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider">',unsafe_allow_html=True)

        t1,t2,t3 = st.tabs(["FORECAST CHARTS","FEATURE INSIGHTS","TREND SIGNALS"])

        with t1:
            # ══ FORECAST CHARTS — 100% từ data thật người dùng upload ══
            # Không có model giả, không có data cứng định sẵn
            # Tất cả tính toán dựa trên df (session state từ file upload)

            ts_full = df.groupby("Date")["units_sold"].sum().resample("ME").sum().sort_index()
            n_total = len(ts_full)

            if n_total < 6:
                st.markdown('<div class="alert alert-red">❌ Cần ít nhất 6 tháng dữ liệu để vẽ biểu đồ forecast.</div>', unsafe_allow_html=True)
            else:
                # ── Cho người dùng chọn test split ──
                date_min = ts_full.index[0].strftime("%b %Y")
                date_max = ts_full.index[-1].strftime("%b %Y")

                st.markdown(f'''
                <div style="background:#0d1828;border:1px solid #1a3048;border-left:3px solid #00c8dc;
                            border-radius:6px;padding:12px 18px;margin-bottom:16px">
                    <div style="font-size:11px;font-weight:700;color:#00c8dc;margin-bottom:5px">
                        📊 DỮ LIỆU: {date_min} → {date_max} · {n_total} tháng
                    </div>
                    <div style="font-size:12px;color:#a0c8d8">
                        Tất cả biểu đồ được tính toán trực tiếp từ file bạn đã upload.
                        Chọn số tháng cuối làm tập test để đánh giá độ chính xác.
                    </div>
                </div>''', unsafe_allow_html=True)

                n_test = st.slider("Số tháng test (cuối dataset)", min_value=2,
                                   max_value=min(6, n_total-6), value=min(3, n_total-6),
                                   key="pred_test_split")

                train_ts = ts_full.iloc[:-n_test]
                test_ts  = ts_full.iloc[-n_test:]
                t_start  = test_ts.index[0].strftime("%b %Y")
                t_end    = test_ts.index[-1].strftime("%b %Y")

                # ── Chạy SARIMA thật trên train data ──
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    with st.spinner(f"Đang fit SARIMA trên {len(train_ts)} tháng train data..."):
                        m_sar = SARIMAX(train_ts, order=(1,1,1),
                                        seasonal_order=(0,1,1,min(12,len(train_ts)//2)),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit(disp=False)
                    fc_test_sar = m_sar.forecast(n_test)
                    mape_sar = float(np.mean(np.abs((test_ts.values - fc_test_sar.values) / test_ts.values)) * 100)

                    # ── Chart 1: Actual vs SARIMA Forecast — test period ──
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">① SO SÁNH DỰ BÁO VS THỰC TẾ ({t_start} – {t_end})</div>', unsafe_allow_html=True)

                    # context = 4 tháng cuối train + toàn bộ test
                    ctx = train_ts.iloc[-4:]
                    all_x   = list(ctx.index) + list(test_ts.index)
                    actual_y = list(ctx.values) + list(test_ts.values)

                    fig, ax = plt.subplots(figsize=(12, 4.5))
                    # Actual line
                    ax.plot(all_x, [v/1e3 for v in actual_y], "o-",
                            color="#c8dde8", lw=2, ms=6, label="Thực tế (Actual)", zorder=3)
                    # Forecast line
                    ax.plot(test_ts.index, fc_test_sar.values/1e3, "s--",
                            color=BLUE, lw=2.5, ms=8, label=f"SARIMA Forecast", zorder=4)
                    # Error fill
                    ax.fill_between(test_ts.index,
                                    fc_test_sar.values/1e3,
                                    test_ts.values/1e3,
                                    alpha=0.2, color=RED, label="Sai số dự báo")
                    # Annotate forecast points
                    for d, v_fc, v_ac in zip(test_ts.index, fc_test_sar.values, test_ts.values):
                        err = abs(v_fc - v_ac)/v_ac*100
                        ax.annotate(f"{v_fc/1e3:.1f}K\n(err {err:.1f}%)",
                                    (d, v_fc/1e3), textcoords="offset points", xytext=(0, 14),
                                    ha="center", fontsize=8, color=BLUE, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1320", edgecolor=BLUE, alpha=0.85))
                    # Train/test split line
                    ax.axvline(train_ts.index[-1], color=AMBER, ls="--", lw=1.5,
                               label=f"Train / Test split ({train_ts.index[-1].strftime('%b %Y')})")
                    ax.text(0.02, 0.96,
                            f"TRAIN: {len(train_ts)} tháng  |  TEST: {n_test} tháng",
                            transform=ax.transAxes, fontsize=9, color="#5a9ab8", va="top")
                    ax.text(0.98, 0.96,
                            f"MAPE = {mape_sar:.2f}%",
                            transform=ax.transAxes, fontsize=11, fontweight="bold",
                            color=GREEN if mape_sar<2 else AMBER, va="top", ha="right",
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a1320", edgecolor=GREEN if mape_sar<2 else AMBER, alpha=0.9))
                    ax.set_ylabel("Units Sold (K)", fontsize=10)
                    ax.set_title(f"SARIMA — Forecast vs Actual  |  Test period: {t_start} → {t_end}",
                                 fontsize=12, fontweight="bold", color="#c8dde8", pad=10)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}K"))
                    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
                    fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

                    # ── Chart 2: Full history + future forecast ──
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    n_future = st.slider("Số tháng dự báo tương lai", 1, 6, 3, key="pred_future")
                    st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">② LỊCH SỬ ĐẦY ĐỦ + DỰ BÁO {n_future} THÁNG TỚI</div>', unsafe_allow_html=True)

                    with st.spinner("Fit SARIMA trên toàn bộ data để dự báo tương lai..."):
                        m_full = SARIMAX(ts_full, order=(1,1,1),
                                         seasonal_order=(0,1,1,min(12,n_total//2)),
                                         enforce_stationarity=False,
                                         enforce_invertibility=False).fit(disp=False)
                    fc_future = m_full.forecast(n_future)
                    fc_ci     = m_full.get_forecast(n_future).conf_int(alpha=0.2)

                    fig, ax = plt.subplots(figsize=(13, 4.5))
                    ax.fill_between(ts_full.index, ts_full.values/1e3, alpha=0.07, color=BLUE)
                    ax.plot(ts_full.index, ts_full.values/1e3, color="#8ab8cc", lw=1.6,
                            label=f"Lịch sử thực tế ({date_min} – {date_max})")
                    ax.fill_between(fc_future.index,
                                    fc_ci.iloc[:,0]/1e3, fc_ci.iloc[:,1]/1e3,
                                    alpha=0.25, color=BLUE, label="Khoảng tin cậy 80%")
                    ax.plot(fc_future.index, fc_future.values/1e3, "o-",
                            color=BLUE, lw=2.5, ms=9, label=f"Forecast {n_future} tháng tới", zorder=4)
                    for d, v in zip(fc_future.index, fc_future.values):
                        ax.annotate(f"{v/1e3:.1f}K", (d, v/1e3), textcoords="offset points",
                                    xytext=(0,14), ha="center", fontsize=9, color=BLUE, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1320", edgecolor=BLUE, alpha=0.85))
                    ax.axvline(ts_full.index[-1], color=AMBER, ls="--", lw=1.5, alpha=0.8, label="Điểm bắt đầu forecast")
                    # Year separators
                    for yr in sorted(df["year"].unique())[1:]:
                        sep = pd.Timestamp(f"{yr}-01-01")
                        if sep <= ts_full.index[-1]:
                            ax.axvline(sep, color="#1a3048", ls="--", lw=0.8)
                            ax.text(sep, ts_full.max()/1e3*0.99, f"  {yr}", fontsize=8, color="#5a9ab8")
                    ax.set_ylabel("Units Sold (K)", fontsize=10)
                    ax.set_title(f"Toàn bộ lịch sử + Dự báo {n_future} tháng tới (SARIMA)",
                                 fontsize=12, fontweight="bold", color="#c8dde8", pad=10)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}K"))
                    ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y", alpha=0.4)
                    fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

                    # ── Chart 3: Forecast summary table ──
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">③ BẢNG CHI TIẾT DỰ BÁO TƯƠNG LAI</div>', unsafe_allow_html=True)
                    sc1, sc2 = st.columns([1.3, 1])
                    with sc1:
                        fig, ax = plt.subplots(figsize=(7, 3.6))
                        mlabels = [d.strftime("%b %Y") for d in fc_future.index]
                        bars = ax.bar(mlabels, fc_future.values/1e3, color=BLUE, edgecolor="none", alpha=0.88, width=0.5)
                        ax.errorbar(mlabels, fc_future.values/1e3,
                                    yerr=[(fc_future.values - fc_ci.iloc[:,0])/1e3,
                                          (fc_ci.iloc[:,1] - fc_future.values)/1e3],
                                    fmt="none", color=AMBER, capsize=5, lw=2, label="CI 80%")
                        for bar, v in zip(bars, fc_future.values):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                                    f"{v/1e3:.1f}K", ha="center", fontsize=9, color=BLUE, fontweight="bold")
                        ax.set_ylabel("Units Sold (K)"); ax.legend(fontsize=9)
                        ax.set_title(f"SARIMA Forecast — {n_future} tháng tới", fontsize=11, fontweight="bold", color="#c8dde8", pad=8)
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}K"))
                        ax.grid(axis="y", alpha=0.4); fig.tight_layout(); fig_to_img(fig)
                    with sc2:
                        tbl = []
                        last_actual = ts_full.iloc[-1]
                        for d, v, lo, hi in zip(fc_future.index, fc_future.values,
                                                 fc_ci.iloc[:,0], fc_ci.iloc[:,1]):
                            chg = (v - last_actual)/last_actual*100
                            tbl.append({
                                "Tháng": d.strftime("%b %Y"),
                                "Dự báo": f"{v:,.0f}",
                                "CI thấp": f"{lo:,.0f}",
                                "CI cao":  f"{hi:,.0f}",
                                "Thay đổi": f"{chg:+.1f}%",
                            })
                        dark_table(pd.DataFrame(tbl), height=240)
                        st.markdown(f'<div class="alert alert-green" style="margin-top:8px">✅ SARIMA fit trên <strong>{n_total} tháng</strong> data thật · Test MAPE <strong>{mape_sar:.2f}%</strong></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<div class="alert alert-red">❌ Lỗi khi chạy SARIMA: <strong>{e}</strong></div>', unsafe_allow_html=True)
                    import traceback
                    st.code(traceback.format_exc(), language="python")


        with t2:
            # ── Feature importance computed from data ──
            col_a,col_b = st.columns([1,1])
            with col_a:
                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">FEATURE IMPORTANCE (TƯƠNG QUAN VỚI UNITS_SOLD)</div>', unsafe_allow_html=True)
                feat_cols = ["price","discount_pct","competitor_price","price_gap_pct",
                             "inventory_level","is_holiday","units_sold_ma7","units_sold_ma30",
                             "revenue_mom_growth","demand_forecast"]
                avail_f = [c for c in feat_cols if c in df.columns]
                importances = df[avail_f].corrwith(df["units_sold"]).abs().sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                colors_ = [BLUE if v >= importances.quantile(0.7) else "#1a3a52" for v in importances.values]
                bars = ax.barh(importances.index, importances.values, color=colors_, edgecolor="none", height=0.6)
                for bar, val in zip(bars, importances.values):
                    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                            f"{val:.3f}", va="center", fontsize=8.5, color=TC)
                ax.set_xlabel("|Correlation| with units_sold", fontsize=9)
                ax.set_title("FEATURE IMPORTANCE (Pearson |r|)", fontsize=10, fontweight="bold", color="#c8dde8", pad=8)
                ax.set_xlim(0, importances.max()*1.25)
                ax.grid(axis="x", alpha=0.4); fig.tight_layout(); fig_to_img(fig)

            with col_b:
                corr_cols = ["units_sold","revenue","price","discount_pct",
                             "competitor_price","demand_forecast","price_gap_pct",
                             "units_sold_ma7","units_sold_ma30","is_holiday"]
                avail_c = [c for c in corr_cols if c in df.columns]
                corr = df[avail_c].corr().round(2)
                mask = np.zeros_like(corr, dtype=bool)
                mask[np.triu_indices_from(mask, k=1)] = True
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(corr, annot=True, fmt=".2f",
                            cmap=sns.diverging_palette(220, 20, as_cmap=True),
                            center=0, linewidths=0.3, linecolor=BG, ax=ax, mask=mask,
                            annot_kws={"size":7.5, "color":"#c8dde8"},
                            cbar_kws={"label":"r", "shrink":0.8})
                ax.set_title("CORRELATION MATRIX", fontsize=10, fontweight="bold", color="#c8dde8", pad=8)
                ax.tick_params(axis="x", rotation=40, labelsize=8)
                ax.tick_params(axis="y", rotation=0, labelsize=8)
                fig.tight_layout(); fig_to_img(fig)

            st.markdown('<div class="alert alert-green">💡 Các biến <strong>units_sold_ma30, units_sold_ma7</strong> có tương quan cao nhất — mô hình hưởng lợi lớn từ lag features. <strong>price_gap_pct</strong> và <strong>competitor_price</strong> phản ánh độ nhạy cảm giá.</div>', unsafe_allow_html=True)

        with t3:
            # ── Store selector for trend signals ──
            all_stores_t3 = sorted(df["store_id"].unique())
            sel_store_t3 = st.selectbox("CHỌN STORE ĐỂ PHÂN TÍCH", all_stores_t3, key="t3_store")
            all_prods = sorted(df[df["store_id"]==sel_store_t3]["product_id"].unique())
            sel_prod_t3 = st.selectbox("CHỌN SẢN PHẨM", all_prods[:10], key="t3_prod")

            col_a, col_b = st.columns(2)
            with col_a:
                sub = df[(df["store_id"]==sel_store_t3)&(df["product_id"]==sel_prod_t3)].sort_values("Date")
                if "units_sold_ma7" in sub.columns and len(sub) > 10:
                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    ax.fill_between(sub["Date"], sub["units_sold"], alpha=0.06, color=BLUE)
                    ax.plot(sub["Date"], sub["units_sold"], color="#4a7a96", lw=0.9, alpha=0.7, label="Actual (daily)")
                    ax.plot(sub["Date"], sub["units_sold_ma7"], color=BLUE, lw=2, label="MA-7")
                    ax.plot(sub["Date"], sub["units_sold_ma30"], color=PURPLE, lw=2.2, ls="--", label="MA-30")
                    ax.set_ylabel("Units Sold")
                    ax.set_title(f"MOVING AVERAGES — {sel_store_t3} / {sel_prod_t3}",
                                 fontsize=10, fontweight="bold", color="#c8dde8", pad=8)
                    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)
                    fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

            with col_b:
                if "revenue_mom_growth" in df.columns:
                    g = df["revenue_mom_growth"].dropna()
                    g = g[g.between(-80, 80)]
                    mean_g = g.mean(); median_g = g.median()
                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    n, bins, patches = ax.hist(g, bins=50, edgecolor="none")
                    for patch, left in zip(patches, bins[:-1]):
                        patch.set_facecolor(GREEN if left >= 0 else RED)
                        patch.set_alpha(0.75)
                    ax.axvline(0, color="#c8dde8", ls="-", lw=1.2, alpha=0.5)
                    ax.axvline(mean_g, color=AMBER, ls="--", lw=2, label=f"Mean {mean_g:.1f}%")
                    ax.axvline(median_g, color=BLUE, ls=":", lw=2, label=f"Median {median_g:.1f}%")
                    ax.set_xlabel("MoM Revenue Growth (%)"); ax.set_ylabel("Frequency")
                    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
                    ax.set_title("PHÂN PHỐI TĂNG TRƯỞNG DOANH THU MoM",
                                 fontsize=10, fontweight="bold", color="#c8dde8", pad=8)
                    fig.tight_layout(); fig_to_img(fig)

            # ── Seasonal decomposition per store ──
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:10px">DOANH SỐ THEO MÙA VỤ — TẤT CẢ STORES</div>', unsafe_allow_html=True)
            monthly_store = df.groupby(["year","month","store_id"])["units_sold"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(13, 3.8))
            for sid, sc in zip(sorted(df["store_id"].unique()), [BLUE,GREEN,AMBER,PURPLE,RED]):
                ms = monthly_store[monthly_store["store_id"]==sid].sort_values(["year","month"])
                ms["period"] = pd.to_datetime(ms["year"].astype(str)+"-"+ms["month"].astype(str).str.zfill(2))
                ax.plot(ms["period"], ms["units_sold"]/1e3, lw=1.8, color=sc, label=sid, alpha=0.85)
            ax.set_ylabel("Units Sold (K)"); ax.legend(fontsize=9, ncol=5)
            ax.set_title("Monthly Units Sold per Store — Seasonal Pattern",
                         fontsize=11, fontweight="bold", color="#c8dde8", pad=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}K"))
            ax.grid(axis="y", alpha=0.4)
            fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

    # ══════════════════════════════════════════════════
    # PRESCRIPTIVE
    # ══════════════════════════════════════════════════
    elif page == "PRESCRIPTIVE":
        section("PRESCRIPTIVE ANALYSIS","WHAT SHOULD WE DO?",
                "Automated alerts: inventory reorder, pricing actions, discount optimization.")

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi(f"{df['reorder_recommended'].sum():,}","REORDER FLAGS","Immediate action","warn")
        with c2: kpi(f"{(df['pricing_action']=='Consider price reduction').sum():,}","PRICE TOO HIGH",">10% above comp","warn")
        with c3: kpi(f"{(df['pricing_action']=='Room to increase price').sum():,}","PRICING HEADROOM",">10% below comp")
        with c4: kpi(f"{(df['pricing_action']=='Competitive').mean()*100:.0f}%","COMPETITIVE","Within ±10%")

        st.markdown('<hr class="divider">',unsafe_allow_html=True)
        t1,t2 = st.tabs(["INVENTORY ALERTS","PRICING & DISCOUNTS"])

        with t1:
            st.markdown(f'<div class="alert alert-red">⚠ <strong>{int((df["stock_coverage_days"]<2).sum()):,} records</strong> have stock coverage below 2 days — immediate reorder required.</div>',unsafe_allow_html=True)
            col_a,col_b = st.columns(2)
            with col_a:
                rg = df[df["reorder_recommended"]==1].groupby(["store_id","category"]).size().reset_index(name="flags")
                avail_cats3 = [c for c in CAT_ORDER if c in df["category"].unique()]
                if not rg.empty:
                    pivot_r = rg.pivot(index="store_id",columns="category",values="flags").fillna(0)
                    for c in avail_cats3:
                        if c not in pivot_r.columns: pivot_r[c]=0
                    pivot_r = pivot_r[avail_cats3]
                    fig,ax = plt.subplots(figsize=(6,3.5))
                    bottom=np.zeros(len(pivot_r))
                    for cat,color in zip(avail_cats3,PALETTE):
                        ax.bar(pivot_r.index,pivot_r[cat],bottom=bottom,label=cat,color=color,edgecolor="none",alpha=0.85)
                        bottom+=pivot_r[cat].values
                    ax.set_xlabel("Store"); ax.set_ylabel("Reorder Flags")
                    ax.legend(fontsize=8,ncol=3); ax.grid(axis="y")
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}"))
                    ax.set_title("REORDER FLAGS BY STORE & CATEGORY",fontsize=10,fontweight="bold",color=TC,pad=8)
                    fig.tight_layout(); fig_to_img(fig)

            with col_b:
                critical = df[df["stock_coverage_days"]<2][
                    ["store_id","category","units_sold","inventory_level","stock_coverage_days"]
                ].sort_values("stock_coverage_days").head(20)
                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:#5a9ab8;margin-bottom:8px">CRITICAL RECORDS (LOWEST COVERAGE)</div>',unsafe_allow_html=True)
                dark_table(critical.reset_index(drop=True), num_cols=['units_sold','inventory_level','stock_coverage_days'], hi_map={'stock_coverage_days':(lambda v: isinstance(v,(int,float)) and v<2,'warn')}, height=280)

        with t2:
            col_a,col_b = st.columns(2)
            with col_a:
                pg2 = df.groupby(["category","pricing_action"]).size().reset_index(name="cnt")
                color_map={"Competitive":BLUE,"Consider price reduction":RED,"Room to increase price":AMBER}
                pivot_p = pg2.pivot(index="category",columns="pricing_action",values="cnt").fillna(0)
                avail_cats4 = [c for c in CAT_ORDER if c in pivot_p.index]
                fig,ax = plt.subplots(figsize=(6,3.5))
                bottom=np.zeros(len(avail_cats4))
                for act in ["Competitive","Consider price reduction","Room to increase price"]:
                    if act in pivot_p.columns:
                        ax.bar(avail_cats4,pivot_p.reindex(avail_cats4)[act],bottom=bottom,
                               label=act.upper(),color=color_map[act],edgecolor="none",alpha=0.85)
                        bottom+=pivot_p.reindex(avail_cats4)[act].values
                ax.set_ylabel("Records"); ax.legend(fontsize=8)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}"))
                ax.set_xticklabels(avail_cats4,rotation=15,ha="right",fontsize=9)
                ax.set_title("PRICING RECOMMENDATIONS",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

            with col_b:
                de = df[df["discount_pct"]>0].groupby(["discount_pct","category"])["discount_ineffective"].mean().reset_index()
                de["pct"]=de["discount_ineffective"]*100
                avail_cats5=[c for c in CAT_ORDER if c in df["category"].unique()]
                pivot_de = de.pivot(index="discount_pct",columns="category",values="pct").fillna(0)
                for c in avail_cats5:
                    if c not in pivot_de.columns: pivot_de[c]=0
                pivot_de = pivot_de[avail_cats5]
                fig,ax = plt.subplots(figsize=(6,3.5))
                pivot_de.plot(kind="bar",ax=ax,color=PALETTE[:len(avail_cats5)],width=0.7,edgecolor="none")
                ax.set_xticklabels([f"{int(x)}%" for x in pivot_de.index],rotation=0)
                ax.set_xlabel("Discount Level"); ax.set_ylabel("Ineffective Rate (%)")
                ax.legend(fontsize=8,ncol=3); ax.grid(axis="y")
                ax.set_title("DISCOUNT INEFFECTIVENESS RATE",fontsize=10,fontweight="bold",color=TC,pad=8)
                fig.tight_layout(); fig_to_img(fig)

    # ══════════════════════════════════════════════════
    # FORECAST
    # ══════════════════════════════════════════════════
    elif page == "FORECAST":
        section("LIVE FORECASTING ENGINE","DEMAND FORECAST",
                "Dự báo số đơn vị hàng hóa bán ra (units_sold) theo tháng — lọc theo store hoặc category.")

        st.markdown("""
        <div style="background:#0d1828;border:1px solid #1a3048;border-left:3px solid #00c8dc;
                    border-radius:6px;padding:14px 18px;margin-bottom:18px">
            <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#00c8dc;margin-bottom:6px">
                📊 MÔ HÌNH DỰ BÁO: SARIMA (Seasonal ARIMA)
            </div>
            <div style="font-size:13px;color:#a0c8d8;line-height:1.9">
                Mô hình học từ dữ liệu lịch sử để phát hiện <strong style="color:#c8dde8">xu hướng + chu kỳ mùa vụ hàng năm</strong>.
                Kết quả dự báo là <strong style="color:#00e5a0">tổng units_sold (đơn vị sản phẩm bán ra)</strong> mỗi tháng.
                Bạn có thể chọn dự báo cho toàn dataset, riêng từng store, hoặc riêng từng danh mục sản phẩm.
            </div>
        </div>""", unsafe_allow_html=True)

        col_cfg, col_out = st.columns([1, 2.5])
        with col_cfg:
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:12px">⚙ CẤU HÌNH DỰ BÁO</div>', unsafe_allow_html=True)
            all_stores = sorted(df["store_id"].unique())
            all_cats   = sorted(df["category"].unique())
            scope = st.selectbox("PHẠM VI DỰ BÁO",
                ["Toàn bộ dataset"] + [f"Store: {s}" for s in all_stores] + [f"Category: {c}" for c in all_cats])
            n_months = st.slider("SỐ THÁNG DỰ BÁO", 1, 6, 3)
            run_btn  = st.button("▶  CHẠY DỰ BÁO", type="primary", use_container_width=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin-bottom:8px">THÔNG TIN DỮ LIỆU</div>', unsafe_allow_html=True)
            # Show live data info instead of pipeline JSON results
            n_months_data = df.groupby(["year","month"]).ngroups
            date_min_fc = df["Date"].min().strftime("%b %Y")
            date_max_fc = df["Date"].max().strftime("%b %Y")
            st.markdown(f"""
            <div style="padding:12px 14px;background:#0d1828;border:1px solid #1a3048;border-top:2px solid #00c8dc;border-radius:6px">
                <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;color:#7ab8cc">DỮ LIỆU ĐÃ IMPORT</div>
                <div style="font-size:22px;font-weight:800;color:#00c8dc;margin:4px 0">{n_months_data} tháng</div>
                <div style="font-size:10px;color:#5a9ab8">{date_min_fc} → {date_max_fc}</div>
                <div style="font-size:10px;color:#5a9ab8;margin-top:4px">{df['store_id'].nunique()} stores · {len(all_cats)} categories</div>
            </div>""", unsafe_allow_html=True)
            if n_months_data < 12:
                st.markdown('<div class="alert alert-amber" style="margin-top:8px">⚠ Cần ít nhất 12 tháng để SARIMA hoạt động tốt.</div>', unsafe_allow_html=True)

        with col_out:
            if run_btn:
                if scope == "Toàn bộ dataset":
                    ts_df = df; scope_label = "Toàn bộ dataset"
                elif scope.startswith("Store:"):
                    sid = scope.replace("Store: ", "")
                    ts_df = df[df["store_id"] == sid]; scope_label = f"Store {sid}"
                else:
                    cat = scope.replace("Category: ", "")
                    ts_df = df[df["category"] == cat]; scope_label = f"Category: {cat}"

                with st.spinner(f"Đang chạy SARIMA cho {scope_label}..."):
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    ts = ts_df.groupby("Date")["units_sold"].sum().resample("ME").sum()
                    if len(ts) < 12:
                        st.markdown('<div class="alert alert-red">❌ Cần ít nhất 12 tháng dữ liệu để chạy SARIMA.</div>', unsafe_allow_html=True)
                        st.stop()
                    model = SARIMAX(ts, order=(0,0,0), seasonal_order=(0,1,1,12),
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    fc = model.forecast(n_months)

                st.markdown(f"""
                <div style="background:#0d1828;border:1px solid #1a3048;border-radius:6px;padding:10px 16px;margin-bottom:14px">
                    <span style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8">DỰ BÁO: </span>
                    <span style="font-size:14px;font-weight:700;color:#00c8dc">{scope_label}</span>
                    <span style="font-size:11px;color:#6aaabb;margin-left:12px">· Chỉ số: Tổng Units Sold mỗi tháng</span>
                </div>""", unsafe_allow_html=True)

                last_val = ts.iloc[-1]
                fc_cols = st.columns(n_months)
                for i, (date, val) in enumerate(zip(fc.index, fc.values)):
                    chg = (val - last_val) / last_val * 100 if last_val > 0 else 0
                    with fc_cols[i]:
                        kpi(f"{val/1e3:.1f}K", date.strftime("%b %Y").upper(),
                            f"{chg:+.1f}% so với kỳ trước", "" if chg >= 0 else "warn")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.fill_between(ts.index, ts.values, alpha=0.08, color=BLUE)
                ax.plot(ts.index, ts.values, color=TC, lw=1.5, label="📈 Lịch sử thực tế")
                ax.plot(fc.index, fc.values, "o-", color=BLUE, lw=2.5, ms=8, label=f"🔮 Dự báo {n_months} tháng (SARIMA)")
                ax.fill_between(fc.index, fc.values*0.97, fc.values*1.03, alpha=0.15, color=BLUE, label="Khoảng tin cậy ±3%")
                ax.axvline(ts.index[-1], color=AMBER, ls="--", lw=1.5, alpha=0.7, label="Điểm bắt đầu dự báo")
                for d, v in zip(fc.index, fc.values):
                    ax.annotate(f"{v/1e3:.1f}K", (d, v), textcoords="offset points", xytext=(0, 12),
                                ha="center", fontsize=9, color=BLUE, fontweight="bold")
                ax.set_ylabel("Đơn vị bán ra / tháng", fontsize=10)
                ax.set_title(f"DỰ BÁO NHU CẦU — {scope_label.upper()}  |  {n_months} THÁNG TỚI",
                             fontsize=11, fontweight="bold", pad=10)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
                ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y")
                fig.autofmt_xdate(rotation=30); fig.tight_layout(); fig_to_img(fig)

                st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#5a9ab8;margin:14px 0 8px">CHI TIẾT DỰ BÁO THEO THÁNG</div>', unsafe_allow_html=True)
                tbl = [{"Tháng": d.strftime("%B %Y"),
                        "Units Sold (Dự báo)": f"{v:,.0f}",
                        "Thay đổi so kỳ trước": f"{(v-last_val)/last_val*100:+.1f}%",
                        "Trạng thái": "🔺 Tăng" if v > last_val*1.01 else ("🔻 Giảm" if v < last_val*0.99 else "➡ Ổn định")}
                       for d, v in zip(fc.index, fc.values)]
                dark_table(pd.DataFrame(tbl), height=240)
                st.markdown(f'<div class="alert alert-green">✅ SARIMA(0,0,0)×(0,1,1,12) · Dữ liệu import: <strong>{len(ts)} tháng</strong> · Phạm vi: <strong>{scope_label}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="margin-top:16px;padding:32px;background:#0d1828;border:1px dashed #1a3048;
                            border-radius:6px;text-align:center">
                    <div style="font-size:32px;margin-bottom:12px">🔮</div>
                    <div style="color:#a0c8d8;font-size:14px;margin-bottom:8px">
                        Chọn <strong style="color:#00c8dc">phạm vi</strong> (toàn bộ / theo store / theo category),<br>
                        số tháng muốn dự báo, rồi nhấn <strong style="color:#00c8dc">▶ CHẠY DỰ BÁO</strong>
                    </div>
                    <div style="color:#5a9ab8;font-size:11px">
                        SARIMA phân tích xu hướng + mùa vụ → dự báo tổng units_sold theo tháng từ dữ liệu đã import
                    </div>
                </div>""", unsafe_allow_html=True)
