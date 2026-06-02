from pathlib import Path

from clean_existing_data import clean_dataset
from rsi_calculator import add_rsi
from random_forest import main as run_random_forest_model
from light_gbm import main as run_lightgbm_model

# Directory Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

# File Paths
RAW_DATA = DATA_DIR / "psx_historical_data_clean.csv"
CLEAN_DATA = DATA_DIR / "psx_historical_data_clean_no_hyphen.csv"
RSI_DATA = DATA_DIR / "psx_with_rsi.csv"

# Hyperparameters / Constants
INITIAL_CAPITAL = 100000.0
MAX_POSITIONS = 2
RF_ESTIMATORS = 100
LIGHTGBM_ESTIMATORS = 300


def require_raw_data() -> None:
    """Ensures the initial raw dataset exists before running the pipeline."""
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {RAW_DATA}\n"
            "Put your original cleaned PSX CSV at data/psx_historical_data_clean.csv first."
        )


def print_step_header(step_info: str) -> None:
    """Helper to print clean, standardized section dividers."""
    print("\n" + "=" * 70)
    print(step_info)
    print("=" * 70)


if __name__ == "__main__":
    # Setup data directory and validate input
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    require_raw_data()

    # STEP 1: Data Cleaning
    print_step_header("STEP 1/4 - CLEAN EXISTING DATA")
    print(f"Input : {RAW_DATA}")
    print(f"Output: {CLEAN_DATA}")
    clean_dataset(str(RAW_DATA), str(CLEAN_DATA))

    # STEP 2: RSI Calculation
    print_step_header("STEP 2/4 - CALCULATE RSI")
    print(f"Input : {CLEAN_DATA}")
    print(f"Output: {RSI_DATA}")
    add_rsi(str(CLEAN_DATA), str(RSI_DATA))

    # STEP 3: Random Forest Model
    print_step_header("STEP 3/4 - RUN RANDOM FOREST FINAL CODE")
    run_random_forest_model(
        input_file=str(RSI_DATA),
        output_dir=str(BASE_DIR),  # Saves directly to main directory
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        n_estimators=RF_ESTIMATORS,
    )

    # STEP 4: LightGBM Model
    print_step_header("STEP 4/4 - RUN LIGHTGBM MODEL")
    run_lightgbm_model(
        input_file=str(RSI_DATA),
        output_dir=str(BASE_DIR),  # Saves directly to main directory
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        test_years=1,
        n_estimators=LIGHTGBM_ESTIMATORS,
    )

    # Pipeline Wrap-up
    print_step_header("PIPELINE COMPLETE")
    print("Data files created:")
    print(f"- {CLEAN_DATA}")
    print(f"- {RSI_DATA}")
    print(f"\nModel files created directly in main directory ({BASE_DIR}):")
    print("- rf_hedge_fund_signals.csv")
    print("- rf_hedge_fund_trade_log.csv")
    print("- rf_hedge_fund_portfolio_history.csv")
    print("- lgbm_signals.csv")
    print("- lgbm_trade_log.csv")
    print("- lgbm_portfolio_history.csv")