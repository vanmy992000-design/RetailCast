"""
=============================================================
RetailCast – FAA4023 Master Pipeline Runner
=============================================================
Chạy toàn bộ dự án theo thứ tự:
  Step 1 → Data Cleaning & Feature Engineering
  Step 2 → EDA (19 charts)
  Step 3 → Forecasting Models (SARIMA · Prophet · XGBoost)

Usage:
  python run_pipeline.py                  # chạy tất cả
  python run_pipeline.py --step 1         # chỉ chạy step 1
  python run_pipeline.py --step 2         # chỉ chạy step 2
  python run_pipeline.py --step 3         # chỉ chạy step 3
  python run_pipeline.py --skip-prophet   # bỏ qua Prophet nếu chưa cài
=============================================================
"""

import argparse, sys, time, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def banner(text):
    print("\n" + "═"*60)
    print(f"  {text}")
    print("═"*60)


def check_dependencies():
    """Kiểm tra thư viện trước khi chạy và hướng dẫn cài nếu thiếu."""
    missing = []
    optional_missing = []

    required = {
        "pandas"      : "pip install pandas",
        "numpy"       : "pip install numpy",
        "matplotlib"  : "pip install matplotlib",
        "seaborn"     : "pip install seaborn",
        "sklearn"     : "pip install scikit-learn",
        "xgboost"     : "pip install xgboost",
        "statsmodels" : "pip install statsmodels",
        "streamlit"   : "pip install streamlit",
    }
    optional = {
        "prophet"     : "pip install prophet  (hoặc: conda install -c conda-forge prophet)",
    }

    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append((pkg, install_cmd))

    for pkg, install_cmd in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            optional_missing.append((pkg, install_cmd))

    if missing:
        print("\n❌ Thiếu thư viện bắt buộc. Chạy lệnh sau để cài:")
        for pkg, cmd in missing:
            print(f"   {cmd}")
        sys.exit(1)

    if optional_missing:
        print("\n⚠️  Prophet chưa được cài.")
        print("   Cài bằng: pip install prophet")
        print("   Hoặc:     conda install -c conda-forge prophet")
        print("   Nếu muốn bỏ qua Prophet: python run_pipeline.py --skip-prophet\n")
        return False   # prophet missing
    return True        # all ok


def run_step1():
    banner("STEP 1 — Data Cleaning & Feature Engineering")
    t0 = time.time()
    from src.step1_data_pipeline import run_pipeline
    run_pipeline(
        raw_path   = os.path.join(ROOT, "data", "raw",     "retail_store_inventory.csv"),
        output_dir = os.path.join(ROOT, "data", "outputs"),
    )
    print(f"\n  ✅ Step 1 done in {time.time()-t0:.1f}s")


def run_step2():
    banner("STEP 2 — Exploratory Data Analysis (19 charts)")
    t0 = time.time()
    from src.step2_eda import run_eda
    run_eda(
        clean_path = os.path.join(ROOT, "data", "outputs", "retail_cleaned.csv"),
        chart_dir  = os.path.join(ROOT, "data", "outputs", "charts"),
    )
    print(f"\n  ✅ Step 2 done in {time.time()-t0:.1f}s")


def run_step3(skip_prophet=False):
    banner("STEP 3 — Forecasting Models (SARIMA · Prophet · XGBoost)")
    t0 = time.time()
    from src.step3_forecasting import run_forecasting
    run_forecasting(
        clean_path   = os.path.join(ROOT, "data", "outputs", "retail_cleaned.csv"),
        chart_dir    = os.path.join(ROOT, "data", "outputs", "charts"),
        result_path  = os.path.join(ROOT, "data", "outputs", "model_results.json"),
        skip_prophet = skip_prophet,
    )
    print(f"\n  ✅ Step 3 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailCast Pipeline Runner")
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                        help="Run specific step only (1, 2, or 3)")
    parser.add_argument("--skip-prophet", action="store_true",
                        help="Skip Prophet model (use if prophet is not installed)")
    args = parser.parse_args()

    total_start = time.time()
    print("\n📦 RetailCast – FAA4023 Demand Forecasting Pipeline")
    print(f"   Working directory: {ROOT}")

    # Check dependencies
    prophet_ok = check_dependencies()
    skip_prophet = args.skip_prophet or (not prophet_ok)
    if skip_prophet and not args.skip_prophet:
        print("   ℹ️  Tự động bật --skip-prophet vì prophet chưa cài.\n")

    # Run
    if args.step == 1:
        run_step1()
    elif args.step == 2:
        run_step2()
    elif args.step == 3:
        run_step3(skip_prophet=skip_prophet)
    else:
        run_step1()
        run_step2()
        run_step3(skip_prophet=skip_prophet)
        banner("ALL STEPS COMPLETE")
        print(f"  Total time : {time.time()-total_start:.1f}s")
        print(f"  Outputs    : data/outputs/")
        print(f"  Charts     : data/outputs/charts/  ({19 + (5 if not skip_prophet else 4)} files)")
        print(f"  Results    : data/outputs/model_results.json")
        if skip_prophet:
            print(f"\n  ⚠️  Prophet bị bỏ qua — kết quả chỉ có SARIMA + XGBoost")
            print(f"     Cài prophet sau: pip install prophet  rồi chạy lại")
        print(f"\n  ▶  Launch dashboard: streamlit run app.py")
        print("═"*60)
