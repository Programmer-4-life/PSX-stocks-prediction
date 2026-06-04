# PSX Stock Buy/Sell Signal Prediction System

> An end-to-end Machine Learning pipeline that generates **Buy / Sell** trading signals for
> **Pakistan Stock Exchange (PSX)** equities — from automated data scraping and technical
> feature engineering to risk-managed backtesting and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/Random%20Forest-scikit--learn-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-success)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Purpose-Educational-lightgrey)

---

## 📌 Overview

This project predicts short-term upward price moves for PSX stocks and converts those
predictions into explainable **BUY / STRONG BUY / HOLD** signals, which are then evaluated by a
risk-managed portfolio backtest. Two complementary models — a **Random Forest** and a
**LightGBM** gradient-boosting classifier — are trained on technical indicators engineered from
~5 years of daily market data, validated strictly out-of-sample, and compared side-by-side in an
interactive dashboard.

| | |
|---|---|
| **Data source** | PSX Data Portal — `dps.psx.com.pk/historical` (scraped) |
| **Coverage** | 15 Mar 2021 → 13 Mar 2026 (≈ 5 years, business days) |
| **Raw records** | 749,460 daily rows across 3,768 tickers |
| **Clean records** | 463,378 rows after filtering |
| **Models** | Random Forest (scikit-learn) · LightGBM |
| **Out-of-sample ROI** | Random Forest **+139.3%** · LightGBM **+14.8%** (max 2 positions) |

> ⚠️ **Disclaimer:** This is an **educational / research** project and a **paper-trading**
> simulator. It is **not financial advice** and must not be used for real trading decisions.

---

## ✨ Features

- 🕸️ **Automated data scraping** of PSX historical prices (Selenium + BeautifulSoup) with
  pagination handling and strict de-duplication.
- 🧹 **Data cleaning** — removes derivative (hyphenated) symbols, fixes numeric formatting, drops
  incomplete rows.
- 📈 **Technical feature engineering** — RSI (1/7/30-day), MACD, ATR, Bollinger Bands, EMA/SMA
  distances and volume ratios.
- 🤖 **Two ML models** trained on a leakage-free, time-based train/test split and scored with
  ROC-AUC.
- 🎯 **Signal generation** combining model probability with quantitative confirmations, a
  liquidity/volatility regime filter and an overbought reject rule.
- 💰 **Risk-managed backtester** — ATR trailing stops, hard loss caps, reward targets, time stops
  and position-size sensitivity analysis.
- 📊 **Interactive Streamlit dashboard** — 7 pages including a model comparison view and a
  two-stock paper-trading simulator.

---

## 👥 Team

| Member | Role | Responsibilities |
|---|---|---|
| **Muhammad Abdullah** | Team Lead | PSX data scraping, RSI calculation, Random Forest & LightGBM models, pipeline integration |
| **Sheeraz** | Data Engineer | Data cleaning, preprocessing & transformation |
| **Rabia** | Dashboard Developer | Streamlit dashboard, visualization, UI & deployment |

---

## 🗂️ Project Structure

```
ml_project/
├── psx_scraper.py                  # Step 0: scrape PSX historical data (Selenium + BeautifulSoup)
├── clean_existing_data.py          # Step 1: clean & de-duplicate the raw dataset
├── rsi_calculator.py               # Step 2: compute RSI_1D / RSI_7D / RSI_30D
├── random_forest.py                # Step 3: Random Forest model + signals + backtest
├── light_gbm.py                    # Step 4: LightGBM model + signals + backtest
├── run_pipeline.py                 # Orchestrates Steps 1 → 4 with one command
├── app.py                          # Streamlit dashboard (7 pages)
├── requirements.txt                # Python dependencies
│
├── psx_historical_data_clean.csv           # Raw scraped data
├── psx_historical_data_clean_no_hyphen.csv # After cleaning
├── psx_with_rsi.csv                         # After RSI feature engineering
│
├── rf_hedge_fund_signals.csv               # Random Forest outputs
├── rf_hedge_fund_trade_log.csv
├── rf_hedge_fund_portfolio_history.csv
├── rf_position_sensitivity.csv
│
├── lgbm_signals.csv                        # LightGBM outputs
├── lgbm_trade_log.csv
├── lgbm_portfolio_history.csv
└── lgbm_position_sensitivity.csv
```

---

## 🔄 Pipeline / Workflow

```
PSX Data Source
      ↓   psx_scraper.py
Raw market data  (psx_historical_data_clean.csv)
      ↓   clean_existing_data.py
Cleaned data     (psx_historical_data_clean_no_hyphen.csv)
      ↓   rsi_calculator.py
Feature data     (psx_with_rsi.csv)
      ↓   random_forest.py  +  light_gbm.py
Buy/Sell signals, trade logs, portfolio history, sensitivity (CSV)
      ↓   app.py
Interactive Streamlit Dashboard
```

The pipeline is **file-decoupled**: each stage writes a CSV that the next stage reads, so any
step can be re-run independently.

---

## ⚙️ Installation

**Requirements:** Python 3.10+ and Google Chrome (only needed for live scraping).

```bash
# 1. (optional) create a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
streamlit>=1.32.0
plotly>=5.18.0
```

> Scraping additionally uses `selenium`, `webdriver-manager`, `beautifulsoup4` and `lxml`.
> Install them with: `pip install selenium webdriver-manager beautifulsoup4 lxml`

---

## ▶️ Usage

### 1. (Optional) Scrape fresh data
Regenerates `psx_historical_data_clean.csv`. Requires Chrome and internet access.
```bash
python psx_scraper.py
```

### 2. Run the full ML pipeline
Cleans the data, computes RSI, then trains and backtests **both** models, writing all output CSVs.
```bash
python run_pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Then open the URL shown (typically `http://localhost:8501`).

> If you only have the raw CSV, you can run individual stages directly, e.g.
> `python clean_existing_data.py`, `python rsi_calculator.py`,
> `python random_forest.py`, `python light_gbm.py`.

---

## 🧠 Machine Learning Models

| Parameter | Random Forest | LightGBM |
|---|---|---|
| Algorithm | Bagged decision trees | Gradient-boosted trees |
| Estimators | 100 | 300 |
| Max depth | 6 | 6 |
| Learning rate | — | 0.05 |
| Num leaves | — | 31 |
| Regularization | `class_weight='balanced'` | L1 = 0.1, L2 = 0.1, subsample = 0.8 |
| Prediction target | +1.5% within 5 days (`Target_5D`) | +3% within 10 days (`Target_10D`) |
| Random state | 42 | 42 |

**Shared features (8):** `RSI_7D`, `RSI_30D`, `Vol_Ratio`, `MACD_Hist`, `ATR_Pct`,
`Dist_EMA20`, `Dist_SMA200`, `BB_Width`.

**Validation:** chronological split — train on the earliest ~4 years, test on the most recent
12 months (out-of-sample), scored with ROC-AUC.

---

## 🎯 Signal Logic

A model probability is converted to a trade only after passing a funnel of filters:

1. **Probability gate** — RF ≥ 0.55 (0.65 for STRONG BUY); LightGBM uses dynamic 90th–95th
   percentile thresholds.
2. **Quant confirmations** — trend (Close > EMA-50), momentum (MACD histogram > 0), volume
   (`Vol_Ratio` ≥ 1.2). ≥ 2 → **BUY**, ≥ 3 → **STRONG BUY**.
3. **Regime filter** — minimum liquidity (`Vol_MA20`) and volatility (`ATR%`).
4. **Overbought reject** — skip if `RSI_7D` is too high (> 75 RF / > 80 LightGBM).
5. Otherwise → **HOLD**.

**Risk management (backtest):** Rs. 100,000 starting capital, max 2 concurrent positions
(configurable), ATR trailing stops, hard loss cap (10% RF / 8% LightGBM), reward targets and a
time stop.

---

## 📊 Results (Out-of-Sample, Mar 2025 – Mar 2026)

| Metric | Random Forest | LightGBM |
|---|---|---|
| Final equity | Rs. 239,322 | Rs. 114,829 |
| Net profit | Rs. 139,322 | Rs. 14,829 |
| Portfolio ROI | **+139.3%** | **+14.8%** |
| Trades executed | 39 | 51 |
| Win rate | 46.2% | 43.1% |
| Average trade ROI | +5.3% | +1.9% |
| Best / worst trade | +48.2% / −18.2% | +46.4% / −10.0% |

*Backtested with max 2 positions on Rs. 100,000 starting capital. Results are historical and
exclude transaction costs and slippage.*

---

## 🖥️ Dashboard

The Streamlit app (`app.py`) reads the generated CSV outputs (cached) and provides 7 pages:

| Page | Function |
|---|---|
| **Dashboard** | KPI cards, equity curve, model scoreboard, signal mix |
| **Model Comparison** | Side-by-side metrics, bar charts, probability distributions |
| **Position Sensitivity** | ROI / win-rate vs. the max-positions limit |
| **Recommendations** | Filterable BUY calls per date + cross-model overlap |
| **Simulator** | Two-stock paper-trading account (10% stop / 8% target) |
| **Trade History** | Filterable trade log + exit-reason distribution |
| **Search** | Free-text search across historical signals & trades |

---

## 📁 Output Files

| File | Produced by | Contents |
|---|---|---|
| `*_signals.csv` | model scripts | Per-stock daily signal, probability and reason |
| `*_trade_log.csv` | model scripts | Entry/exit, return %, profit, exit reason per trade |
| `*_portfolio_history.csv` | model scripts | Daily total equity curve |
| `*_position_sensitivity.csv` | model scripts | ROI/win-rate sweep over 2–10 positions |

*(`*` = `rf_hedge_fund` for Random Forest, `lgbm` for LightGBM.)*

---

## 🚀 Future Enhancements

- Walk-forward (rolling-origin) validation instead of a single hold-out year.
- Transaction costs, slippage and liquidity modeling.
- Deep-learning sequence models (LSTM / Transformer).
- Live data feed with scheduled refresh and signal alerts.
- Fundamental and news-sentiment features.

---

## 📜 License & Disclaimer

This project is provided for **educational and research purposes only**. It does **not**
constitute financial advice, and the authors accept no liability for any use of the code,
signals or results. Trade at your own risk.
