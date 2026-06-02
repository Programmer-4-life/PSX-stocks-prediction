from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PSX Model Comparison Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INITIAL_CAPITAL = 100000.0
MAX_STOCK_RULE = 2

def get_fallback_paths(filename: str) -> list[Path]:
    """Generates fallback paths for the main directory."""
    return [BASE_DIR / filename, Path(filename)]

MODEL_CONFIGS = {
    "Random Forest": {
        "short": "RF",
        "signals": get_fallback_paths("rf_hedge_fund_signals.csv"),
        "trades": get_fallback_paths("rf_hedge_fund_trade_log.csv"),
        "portfolio": get_fallback_paths("rf_hedge_fund_portfolio_history.csv"),
        "sensitivity": get_fallback_paths("rf_position_sensitivity.csv"),
        "accent": "#00D4FF",
    },
    "LightGBM": {
        "short": "LGBM",
        "signals": get_fallback_paths("lgbm_signals.csv"),
        "trades": get_fallback_paths("lgbm_trade_log.csv"),
        "portfolio": get_fallback_paths("lgbm_portfolio_history.csv"),
        "sensitivity": get_fallback_paths("lgbm_position_sensitivity.csv"),
        "accent": "#FFB703",
    },
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(255, 0, 128, 0.20), transparent 30%),
            radial-gradient(circle at 86% 6%, rgba(0, 212, 255, 0.20), transparent 30%),
            radial-gradient(circle at 55% 88%, rgba(132, 80, 255, 0.20), transparent 35%),
            linear-gradient(135deg, #070817 0%, #11172f 48%, #080a18 100%);
        color: #f8fbff;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 16, 38, 0.98), rgba(14, 19, 43, 0.92));
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        position: relative;
        padding: 2.2rem 2rem;
        border-radius: 32px;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.05)),
            linear-gradient(90deg, rgba(255,0,128,0.40), rgba(0,212,255,0.32), rgba(255,183,3,0.32), rgba(132,80,255,0.40));
        border: 1px solid rgba(255,255,255,0.20);
        box-shadow: 0 24px 70px rgba(0,0,0,0.35);
        animation: floatIn 0.8s ease-out both;
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: -80px;
        background: conic-gradient(from 90deg, transparent, rgba(255,255,255,0.24), transparent, rgba(255,255,255,0.10), transparent);
        animation: spinGlow 9s linear infinite;
        opacity: 0.45;
    }

    .hero-content { position: relative; z-index: 1; }
    .hero h1 {
        margin: 0;
        font-size: 3rem;
        line-height: 1.02;
        letter-spacing: -0.08em;
        font-weight: 900;
        color: white;
    }
    .hero p { margin-top: 0.9rem; max-width: 900px; color: rgba(255,255,255,0.83); font-size: 1.04rem; }

    .badge-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 1.2rem; }
    .glow-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.20);
        color: #ffffff;
        font-weight: 700;
        backdrop-filter: blur(14px);
    }

    .metric-card {
        min-height: 142px;
        padding: 1.2rem;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,0.16);
        background: linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.055));
        box-shadow: 0 18px 48px rgba(0,0,0,0.25);
        backdrop-filter: blur(18px);
        position: relative;
        overflow: hidden;
        animation: floatIn 0.75s ease-out both;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover { transform: translateY(-6px) scale(1.01); border-color: rgba(0, 212, 255, 0.55); }
    .metric-label { color: rgba(255,255,255,0.68); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; }
    .metric-value { color: #ffffff; font-size: 1.85rem; font-weight: 900; letter-spacing: -0.04em; margin-top: 0.4rem; }
    .metric-help { color: rgba(255,255,255,0.58); font-size: 0.82rem; margin-top: 0.3rem; }

    .glass-panel {
        padding: 1.2rem;
        border-radius: 28px;
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 55px rgba(0,0,0,0.23);
        backdrop-filter: blur(18px);
        animation: floatIn 0.65s ease-out both;
    }

    .section-title { font-size: 1.6rem; font-weight: 900; letter-spacing: -0.05em; color: white; margin: 0.4rem 0 1rem; }
    .subtle { color: rgba(255,255,255,0.66); font-size: 0.94rem; }
    .danger-note {
        padding: 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255,0,128,0.20), rgba(255,255,255,0.07));
        border: 1px solid rgba(255,0,128,0.28);
        color: rgba(255,255,255,0.88);
    }
    .success-note {
        padding: 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(132,80,255,0.14));
        border: 1px solid rgba(0,212,255,0.30);
        color: rgba(255,255,255,0.88);
    }

    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    label, .stMarkdown, .stTextInput, .stSelectbox, .stMultiSelect, .stSlider { color: #f8fbff !important; }
    div[data-testid="stDataFrame"] { border-radius: 20px; overflow: hidden; border: 1px solid rgba(255,255,255,0.12); }

    @keyframes floatIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes spinGlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def money(value):
    if pd.isna(value):
        return "Rs. 0"
    return f"Rs. {value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "0.00%"
    return f"{value:.2f}%"


def safe_to_datetime(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def display_table(df, cols=None, height=430):
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    show = df.copy()
    if cols:
        cols = [c for c in cols if c in show.columns]
        show = show[cols]
    st.dataframe(show, use_container_width=True, height=height)


def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <h1>PSX Model Comparison Dashboard</h1>
                <p>Compare Random Forest and LightGBM signals, portfolio equity, returns, trade quality, latest recommendations, and paper-simulator decisions in one Streamlit dashboard.</p>
                <div class="badge-row">
                    <span class="glow-badge">Random Forest vs LightGBM</span>
                    <span class="glow-badge">Side-by-side equity curves</span>
                    <span class="glow-badge">RF 68% win-rate + LightGBM comparison</span>
                    <span class="glow-badge">Model overlap analysis</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_model_outputs():
    models = {}
    missing = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        signal_path = first_existing(cfg["signals"])
        trade_path = first_existing(cfg["trades"])
        portfolio_path = first_existing(cfg["portfolio"])
        sensitivity_path = first_existing(cfg.get("sensitivity", []))

        missing_parts = []
        if signal_path is None:
            missing_parts.append("signals")
        if trade_path is None:
            missing_parts.append("trade log")
        if portfolio_path is None:
            missing_parts.append("portfolio history")
        if missing_parts:
            missing[model_name] = missing_parts
            continue

        signals = pd.read_csv(signal_path)
        trades = pd.read_csv(trade_path)
        portfolio = pd.read_csv(portfolio_path)
        sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path is not None else pd.DataFrame()

        signals = safe_to_datetime(signals, ["Date"])
        trades = safe_to_datetime(trades, ["Entry_Date", "Exit_Date"])
        portfolio = safe_to_datetime(portfolio, ["Date"])

        signals["Model"] = model_name
        trades["Model"] = model_name
        portfolio["Model"] = model_name
        if not sensitivity.empty:
            sensitivity["Model"] = model_name

        if "Signal" not in signals.columns:
            signals["Signal"] = "HOLD"
        if "Reason" not in signals.columns:
            signals["Reason"] = ""
        if "ML_Prob" not in signals.columns:
            signals["ML_Prob"] = np.nan
        if "Total_Equity" not in portfolio.columns:
            portfolio["Total_Equity"] = INITIAL_CAPITAL
        if "Profit_Rs" not in trades.columns:
            trades["Profit_Rs"] = 0.0
        if "Return_%" not in trades.columns:
            trades["Return_%"] = 0.0
        if "Exit_Reason" not in trades.columns:
            trades["Exit_Reason"] = ""

        models[model_name] = {
            "signals": signals,
            "trades": trades,
            "portfolio": portfolio,
            "sensitivity": sensitivity,
            "paths": {
                "signals": str(signal_path),
                "trades": str(trade_path),
                "portfolio": str(portfolio_path),
                "sensitivity": str(sensitivity_path) if sensitivity_path is not None else "not generated yet",
            },
        }
    return models, missing


def metrics_for_model(bundle):
    signals = bundle["signals"]
    trades = bundle["trades"]
    portfolio = bundle["portfolio"]
    final_equity = portfolio.dropna(subset=["Total_Equity"]).iloc[-1]["Total_Equity"] if not portfolio.empty else INITIAL_CAPITAL
    net_profit = final_equity - INITIAL_CAPITAL
    roi = (net_profit / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL else 0.0
    trade_count = len(trades)
    wins = trades[trades["Return_%"] > 0] if not trades.empty else pd.DataFrame()
    losses = trades[trades["Return_%"] <= 0] if not trades.empty else pd.DataFrame()
    win_rate = len(wins) / trade_count * 100 if trade_count else 0.0
    avg_return = trades["Return_%"].mean() if trade_count else 0.0
    best_trade = trades["Return_%"].max() if trade_count else 0.0
    worst_trade = trades["Return_%"].min() if trade_count else 0.0
    gross_profit = trades.loc[trades["Profit_Rs"] > 0, "Profit_Rs"].sum() if trade_count else 0.0
    gross_loss = abs(trades.loc[trades["Profit_Rs"] <= 0, "Profit_Rs"].sum()) if trade_count else 0.0
    profit_factor = gross_profit / max(gross_loss, 1e-9) if trade_count else 0.0
    latest_date = signals["Date"].max().date() if not signals.empty else None
    latest_calls = 0
    if latest_date is not None:
        latest_calls = len(signals[(signals["Date"].dt.date == latest_date) & (signals["Signal"].isin(["BUY", "STRONG BUY"]))])
    return {
        "Final Equity": final_equity,
        "Net Profit": net_profit,
        "ROI %": roi,
        "Trades": trade_count,
        "Wins": len(wins),
        "Losses": len(losses),
        "Win Rate %": win_rate,
        "Avg Trade ROI %": avg_return,
        "Best Trade %": best_trade,
        "Worst Trade %": worst_trade,
        "Profit Factor": profit_factor,
        "Latest Date": latest_date,
        "Latest Buy Calls": latest_calls,
    }


def build_metrics_df(models):
    rows = []
    for model_name, bundle in models.items():
        row = {"Model": model_name}
        row.update(metrics_for_model(bundle))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ROI %", ascending=False)
    return df


def all_frames(models, key):
    frames = [bundle[key] for bundle in models.values() if not bundle[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_equity_chart(portfolio):
    fig = px.line(
        portfolio.sort_values("Date"),
        x="Date",
        y="Total_Equity",
        color="Model",
        title="Portfolio Equity Curve by Model",
        template="plotly_dark",
        markers=False,
    )
    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=460,
        font=dict(family="Inter", color="white"),
        legend_title_text="Model",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def make_metric_bar(metrics_df, metric, title):
    fig = px.bar(metrics_df, x="Model", y=metric, color="Model", text=metric, title=title, template="plotly_dark")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=380,
        font=dict(family="Inter", color="white"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def make_position_sensitivity_chart(sensitivity_df, metric):
    fig = px.line(
        sensitivity_df.sort_values(["Model", "Max_Positions"]),
        x="Max_Positions",
        y=metric,
        color="Model",
        markers=True,
        title=f"{metric} by Max Positions",
        template="plotly_dark",
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=9))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=430,
        font=dict(family="Inter", color="white"),
        legend_title_text="Model",
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def make_signal_mix_chart(signals, selected_date):
    day = signals[signals["Date"].dt.date == selected_date].copy()
    counts = day.groupby(["Model", "Signal"], as_index=False).size().rename(columns={"size": "Count"})
    fig = px.bar(counts, x="Signal", y="Count", color="Model", barmode="group", title="Signal Mix on Selected Date", template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=380,
        font=dict(family="Inter", color="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def make_probability_chart(signals):
    fig = px.histogram(
        signals.dropna(subset=["ML_Prob"]),
        x="ML_Prob",
        color="Model",
        nbins=40,
        barmode="overlay",
        opacity=0.68,
        title="ML Probability Distribution by Model",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=390,
        font=dict(family="Inter", color="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def make_profit_chart(trades):
    if trades.empty:
        return None
    top = trades.groupby(["Model", "SYMBOL"], as_index=False)["Profit_Rs"].sum()
    top = top.sort_values("Profit_Rs", ascending=False).head(20)
    fig = px.bar(top, x="SYMBOL", y="Profit_Rs", color="Model", title="Top Profit Contributors by Model", template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=420,
        font=dict(family="Inter", color="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.09)")
    return fig


def model_recommendations(signals, selected_models, selected_date, selected_signals, min_prob, search_text, top_n):
    filtered = signals[
        (signals["Model"].isin(selected_models)) &
        (signals["Date"].dt.date == selected_date) &
        (signals["Signal"].isin(selected_signals)) &
        (signals["ML_Prob"] >= min_prob)
    ].copy()
    if search_text:
        filtered = filtered[filtered["SYMBOL"].astype(str).str.contains(search_text, case=False, na=False)]
    rank_signal = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2}
    filtered["Signal_Rank"] = filtered["Signal"].map(rank_signal).fillna(99)
    return filtered.sort_values(["Model", "Signal_Rank", "ML_Prob"], ascending=[True, True, False]).drop(columns=["Signal_Rank"], errors="ignore").head(top_n)


def recommendation_overlap(signals, selected_date, selected_signals, min_prob):
    sets = {}
    for model_name in signals["Model"].dropna().unique():
        subset = signals[
            (signals["Model"] == model_name) &
            (signals["Date"].dt.date == selected_date) &
            (signals["Signal"].isin(selected_signals)) &
            (signals["ML_Prob"] >= min_prob)
        ]
        sets[model_name] = set(subset["SYMBOL"].dropna().astype(str))
    if len(sets) < 2:
        return pd.DataFrame()
    names = list(sets.keys())
    common = sorted(set.intersection(*sets.values())) if all(sets.values()) else []
    only_rows = []
    for name in names:
        others = set.union(*[s for n, s in sets.items() if n != name]) if len(sets) > 1 else set()
        only_rows.append({"Bucket": f"Only {name}", "Count": len(sets[name] - others), "Symbols": ", ".join(sorted(sets[name] - others)[:50])})
    rows = [{"Bucket": "Common to all selected models", "Count": len(common), "Symbols": ", ".join(common[:50])}] + only_rows
    return pd.DataFrame(rows)


# Paper simulator helpers

def init_paper_account():
    if "paper_initial_capital" not in st.session_state:
        st.session_state.paper_initial_capital = INITIAL_CAPITAL
    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash = INITIAL_CAPITAL
    if "paper_holdings" not in st.session_state:
        st.session_state.paper_holdings = {}
    if "paper_transactions" not in st.session_state:
        st.session_state.paper_transactions = []


def reset_paper_account(capital):
    st.session_state.paper_initial_capital = float(capital)
    st.session_state.paper_cash = float(capital)
    st.session_state.paper_holdings = {}
    st.session_state.paper_transactions = []


def record_transaction(action, model, symbol, qty, price, date, note="", profit_rs=0.0):
    st.session_state.paper_transactions.append({
        "Date": str(date),
        "Model": model,
        "Action": action,
        "SYMBOL": symbol,
        "Quantity": float(qty),
        "Price": float(price),
        "Amount": float(qty) * float(price),
        "Profit_Rs": float(profit_rs),
        "Note": note,
    })


def build_holdings_df(price_map, valuation_date):
    rows = []
    for symbol, h in st.session_state.paper_holdings.items():
        qty = float(h.get("qty", 0))
        avg_price = float(h.get("avg_price", 0))
        current_price = float(price_map.get(symbol, avg_price))
        invested = qty * avg_price
        current_value = qty * current_price
        pnl = current_value - invested
        pnl_pct = (pnl / invested * 100) if invested else 0
        stop_loss = float(h.get("stop_loss", avg_price * 0.90))
        target_price = float(h.get("target_price", avg_price * 1.08))
        if current_price <= stop_loss:
            system_status = "SELL - 10% stop hit"
        elif current_price >= target_price:
            system_status = "SELL - target hit"
        else:
            system_status = "HOLD"
        rows.append({
            "Model": h.get("model", ""),
            "SYMBOL": symbol,
            "Entry_Date": h.get("entry_date"),
            "Quantity": qty,
            "Avg_Buy_Price": avg_price,
            "Current_Date": valuation_date,
            "Current_Price": current_price,
            "Invested": invested,
            "Current_Value": current_value,
            "Unrealized_PnL": pnl,
            "Unrealized_PnL_%": pnl_pct,
            "Stop_Loss": stop_loss,
            "Target_Price": target_price,
            "System_Status": system_status,
        })
    return pd.DataFrame(rows)


models, missing = load_model_outputs()

with st.sidebar:
    st.markdown("### PSX Simulator")
    st.caption("Compare Random Forest and LightGBM")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Model Comparison", "Position Sensitivity", "Recommendations", "Simulator", "Trade History", "Search"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Loaded models")
    if models:
        for model_name, bundle in models.items():
            st.success(f"{model_name} loaded")
            with st.expander(f"{model_name} files", expanded=False):
                st.caption(bundle["paths"]["signals"])
                st.caption(bundle["paths"]["trades"])
                st.caption(bundle["paths"]["portfolio"])
                st.caption(bundle["paths"].get("sensitivity", "not generated yet"))
    if missing:
        for model_name, parts in missing.items():
            st.warning(f"{model_name} missing: {', '.join(parts)}")
    st.divider()
    st.markdown("#### Project Rules")
    st.markdown(f"- Paper simulator max **{MAX_STOCK_RULE} stocks**")
    st.markdown("- Sell is rule-controlled")
    st.markdown("- Educational simulator only")

hero()
st.write("")

if not models:
    st.markdown('<div class="danger-note">No model output CSV files were found. Generate outputs first, then launch the dashboard.</div>', unsafe_allow_html=True)
    st.code("pip install -r requirements.txt\npython run_pipeline.py\nstreamlit run app.py", language="bash")
    st.stop()

signals_all = all_frames(models, "signals")
trades_all = all_frames(models, "trades")
portfolio_all = all_frames(models, "portfolio")
sensitivity_all = all_frames(models, "sensitivity")
metrics_df = build_metrics_df(models)
available_models = list(models.keys())
all_dates = sorted(signals_all["Date"].dt.date.dropna().unique().tolist())
min_date = min(all_dates)
latest_date = max(all_dates)

if page == "Dashboard":
    selected_model = st.selectbox("Primary model for dashboard cards", available_models, index=0)
    main_metrics = metrics_for_model(models[selected_model])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{selected_model} Final Equity", money(main_metrics["Final Equity"]), "Backtested value")
    with c2:
        metric_card("Net Profit", money(main_metrics["Net Profit"]), f"ROI {pct(main_metrics['ROI %'])}")
    with c3:
        metric_card("Win Rate", pct(main_metrics["Win Rate %"]), f"{main_metrics['Trades']:,} total trades")
    with c4:
        metric_card("Latest Buy Calls", f"{main_metrics['Latest Buy Calls']:,}", f"Latest date: {main_metrics['Latest Date']}")

    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.plotly_chart(make_equity_chart(portfolio_all), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Scoreboard</div>', unsafe_allow_html=True)
        display_table(metrics_df[["Model", "Final Equity", "Net Profit", "ROI %", "Trades", "Win Rate %", "Avg Trade ROI %", "Latest Buy Calls"]], height=390)
        st.markdown('</div>', unsafe_allow_html=True)

    left2, right2 = st.columns([1, 1])
    with left2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        fig = make_profit_chart(trades_all)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade history available.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.plotly_chart(make_signal_mix_chart(signals_all, latest_date), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model Comparison":
    st.markdown('<div class="section-title">Side-by-side Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="success-note">This page compares both models using the same starting capital baseline and the generated portfolio, trade, and signal CSV outputs.</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    if len(metrics_df) >= 2:
        winner_roi = metrics_df.sort_values("ROI %", ascending=False).iloc[0]
        winner_win = metrics_df.sort_values("Win Rate %", ascending=False).iloc[0]
        winner_profit_factor = metrics_df.sort_values("Profit Factor", ascending=False).iloc[0]
        with c1:
            metric_card("Best ROI", str(winner_roi["Model"]), pct(winner_roi["ROI %"]))
        with c2:
            metric_card("Best Win Rate", str(winner_win["Model"]), pct(winner_win["Win Rate %"]))
        with c3:
            metric_card("Best Profit Factor", str(winner_profit_factor["Model"]), f"{winner_profit_factor['Profit Factor']:.2f}x")
    else:
        with c1:
            metric_card("Loaded Models", str(len(metrics_df)), "Add the second model output to compare")

    st.write("")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    display_table(metrics_df[["Model", "Final Equity", "Net Profit", "ROI %", "Trades", "Wins", "Losses", "Win Rate %", "Avg Trade ROI %", "Best Trade %", "Worst Trade %", "Profit Factor", "Latest Buy Calls"]], height=300)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.plotly_chart(make_equity_chart(portfolio_all), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        metric_choice = st.selectbox("Compare metric", ["ROI %", "Final Equity", "Net Profit", "Win Rate %", "Avg Trade ROI %", "Trades", "Profit Factor"])
        st.plotly_chart(make_metric_bar(metrics_df, metric_choice, f"{metric_choice} Comparison"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    left2, right2 = st.columns([1, 1])
    with left2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        selected_date = st.date_input("Signal mix date", value=latest_date, min_value=min_date, max_value=latest_date, key="comparison_signal_date")
        st.plotly_chart(make_signal_mix_chart(signals_all, selected_date), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.plotly_chart(make_probability_chart(signals_all), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Position Sensitivity":
    st.markdown('<div class="section-title">Max Positions Sensitivity</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="success-note">This tests whether allowing more simultaneous orders improves the strategy. The model is trained once, then only the portfolio slot limit is changed, so you can isolate the effect of order capacity.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if sensitivity_all.empty:
        st.markdown('<div class="danger-note">Sensitivity CSVs were not generated yet. Run <b>python run_pipeline.py</b> again; it will create rf_position_sensitivity.csv and lgbm_position_sensitivity.csv.</div>', unsafe_allow_html=True)
    else:
        metric_options = [col for col in ["Win_Rate_%", "ROI_%", "Final_Equity", "Net_Profit", "Trades", "Avg_Trade_ROI_%", "Profit_Factor"] if col in sensitivity_all.columns]
        selected_metric = st.selectbox("Metric to chart", metric_options, index=0 if "Win_Rate_%" in metric_options else 0)

        best_win = sensitivity_all.sort_values("Win_Rate_%", ascending=False).iloc[0] if "Win_Rate_%" in sensitivity_all.columns else None
        best_roi = sensitivity_all.sort_values("ROI_%", ascending=False).iloc[0] if "ROI_%" in sensitivity_all.columns else None
        best_pf = sensitivity_all.sort_values("Profit_Factor", ascending=False).iloc[0] if "Profit_Factor" in sensitivity_all.columns else None

        c1, c2, c3 = st.columns(3)
        with c1:
            if best_win is not None:
                metric_card("Best Win Rate", f"{best_win['Model']} / {int(best_win['Max_Positions'])} slots", pct(best_win["Win_Rate_%"]))
        with c2:
            if best_roi is not None:
                metric_card("Best ROI", f"{best_roi['Model']} / {int(best_roi['Max_Positions'])} slots", pct(best_roi["ROI_%"]))
        with c3:
            if best_pf is not None:
                metric_card("Best Profit Factor", f"{best_pf['Model']} / {int(best_pf['Max_Positions'])} slots", f"{best_pf['Profit_Factor']:.2f}x")

        st.write("")
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.plotly_chart(make_position_sensitivity_chart(sensitivity_all, selected_metric), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Reading this correctly</div>', unsafe_allow_html=True)
            st.markdown(
                """
                More slots can increase the number of trades, but it can also force the portfolio into weaker signals.

                For this system, compare **Win_Rate_%** with **ROI_%** and **Profit_Factor**. A higher win rate is not automatically better if the average win/loss quality gets worse.
                """
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        display_table(
            sensitivity_all.sort_values(["Model", "Max_Positions"]),
            ["Model", "Max_Positions", "Final_Equity", "Net_Profit", "ROI_%", "Trades", "Wins", "Losses", "Win_Rate_%", "Avg_Trade_ROI_%", "Profit_Factor"],
            430,
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Recommendations":
    st.markdown('<div class="section-title">Compare Stock Recommendations</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1, 1.35, 1, 1])
    with f1:
        selected_date = st.date_input("Recommendation Date", value=latest_date, min_value=min_date, max_value=latest_date)
    with f2:
        selected_models = st.multiselect("Models", available_models, default=available_models)
    with f3:
        selected_signals = st.multiselect("Signal", ["STRONG BUY", "BUY", "HOLD"], default=["STRONG BUY", "BUY"])
    with f4:
        min_prob = st.slider("Minimum ML Probability", 0.0, 1.0, 0.50, 0.01)

    s1, s2 = st.columns([1.4, 1])
    with s1:
        q = st.text_input("Search symbol", placeholder="Example: TRG, MARI, UNITY")
    with s2:
        top_n = st.slider("Rows", 10, 300, 80, 10)

    recs = model_recommendations(signals_all, selected_models, selected_date, selected_signals, min_prob, q, top_n)
    overlap = recommendation_overlap(signals_all[signals_all["Model"].isin(selected_models)], selected_date, selected_signals, min_prob)

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        display_table(
            recs,
            ["Model", "Date", "SYMBOL", "CLOSE", "ML_Prob", "Signal", "Reason", "RSI_7D", "RSI_30D", "Vol_Ratio", "ATR_Pct"],
            560,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommendation Overlap</div>', unsafe_allow_html=True)
        if overlap.empty:
            st.info("Load at least two models to see overlap.")
        else:
            display_table(overlap, None, 260)
        st.plotly_chart(make_probability_chart(recs if not recs.empty else signals_all.head(1)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Simulator":
    st.markdown('<div class="section-title">Interactive Buy / Sell Paper Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="success-note"><b>Model-aware simulator:</b> select which model should supply BUY / STRONG BUY candidates, then paper-trade up to three active holdings.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    init_paper_account()

    sim_model = st.selectbox("Model used for simulator buy calls", available_models)
    sim_signals = models[sim_model]["signals"].copy()
    available_dates = sorted(sim_signals["Date"].dt.date.dropna().unique().tolist())
    latest_index = len(available_dates) - 1

    top_controls = st.columns([1, 1, 1])
    with top_controls[0]:
        starting_capital = st.number_input("Paper Account Capital", min_value=1000.0, value=float(st.session_state.paper_initial_capital), step=5000.0)
    with top_controls[1]:
        buy_date = st.selectbox("Buy Signal Date", available_dates, index=latest_index)
    with top_controls[2]:
        valuation_date = st.selectbox("Current / Sell Date", available_dates, index=latest_index)

    if st.button("Reset Paper Account", type="secondary", use_container_width=True):
        reset_paper_account(starting_capital)
        st.success("Paper account reset.")
        st.rerun()

    buy_day = sim_signals[sim_signals["Date"].dt.date == buy_date].copy()
    valuation_day = sim_signals[sim_signals["Date"].dt.date == valuation_date].copy()
    price_map = dict(zip(valuation_day["SYMBOL"].astype(str), valuation_day["CLOSE"].astype(float))) if "CLOSE" in valuation_day.columns else {}

    holdings_df = build_holdings_df(price_map, valuation_date)
    holdings_value = holdings_df["Current_Value"].sum() if not holdings_df.empty else 0.0
    total_equity_live = st.session_state.paper_cash + holdings_value
    live_pnl = total_equity_live - st.session_state.paper_initial_capital

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        metric_card("Cash Balance", money(st.session_state.paper_cash), "Available for new buys")
    with a2:
        metric_card("Holdings Value", money(holdings_value), "Marked to selected sell date")
    with a3:
        metric_card("Paper Equity", money(total_equity_live), "Cash + holdings")
    with a4:
        metric_card("Paper P/L", money(live_pnl), "Current simulator profit/loss")

    st.write("")
    buy_candidates = buy_day[buy_day["Signal"].isin(["BUY", "STRONG BUY"])].copy()
    if not buy_candidates.empty:
        buy_candidates = buy_candidates.sort_values(["Signal", "ML_Prob"], ascending=[True, False])

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Step 1 - Buy Recommended Stocks</div>', unsafe_allow_html=True)
        if buy_candidates.empty:
            st.warning("No BUY or STRONG BUY calls are available on the selected buy date. Pick another date or model.")
        else:
            symbol_options = buy_candidates["SYMBOL"].dropna().astype(str).unique().tolist()
            selected_symbols = st.multiselect("Choose stocks to buy, maximum 2 active holdings", symbol_options, max_selections=MAX_STOCK_RULE)
            selected_orders = []
            if selected_symbols:
                st.markdown("##### Enter quantity")
                for symbol in selected_symbols:
                    row = buy_candidates[buy_candidates["SYMBOL"].astype(str) == symbol].iloc[0]
                    price = float(row["CLOSE"])
                    signal = str(row["Signal"])
                    prob = float(row["ML_Prob"])
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
                    c1.markdown(f"**{symbol}**")
                    c2.markdown(f"Rs. {price:,.2f}")
                    c3.markdown(f"{signal}")
                    qty = c4.number_input(f"Qty {symbol}", min_value=1, value=1, step=1, label_visibility="collapsed", key=f"buy_qty_{sim_model}_{symbol}_{buy_date}")
                    selected_orders.append({"Model": sim_model, "SYMBOL": symbol, "Quantity": int(qty), "Buy_Price": price, "Signal": signal, "ML_Prob": prob, "Amount": int(qty) * price})

                order_df = pd.DataFrame(selected_orders)
                total_cost = order_df["Amount"].sum() if not order_df.empty else 0.0
                st.markdown("##### Order Preview")
                display_table(order_df, None, 210)
                b1, b2 = st.columns([1, 1])
                with b1:
                    metric_card("Order Cost", money(total_cost), "Quantity x buy price")
                with b2:
                    metric_card("Cash After Buy", money(st.session_state.paper_cash - total_cost), "Must be positive")

                if st.button("Buy Selected Stocks", type="primary", use_container_width=True):
                    current_symbols = set(st.session_state.paper_holdings.keys())
                    if len(current_symbols | set(order_df["SYMBOL"])) > MAX_STOCK_RULE:
                        st.error(f"You can hold only {MAX_STOCK_RULE} stocks at a time. Sell one first or reduce your selection.")
                    elif total_cost <= 0:
                        st.error("Please enter a valid quantity.")
                    elif total_cost > st.session_state.paper_cash:
                        st.error("Not enough paper cash for this order.")
                    else:
                        for _, order in order_df.iterrows():
                            symbol = str(order["SYMBOL"])
                            qty = float(order["Quantity"])
                            price = float(order["Buy_Price"])
                            amount = qty * price
                            if symbol in st.session_state.paper_holdings:
                                old = st.session_state.paper_holdings[symbol]
                                old_qty = float(old["qty"])
                                old_avg = float(old["avg_price"])
                                new_qty = old_qty + qty
                                new_avg = ((old_qty * old_avg) + amount) / new_qty
                                old["qty"] = new_qty
                                old["avg_price"] = new_avg
                                old["stop_loss"] = new_avg * 0.90
                                old["target_price"] = new_avg * 1.08
                                old["model"] = sim_model
                            else:
                                st.session_state.paper_holdings[symbol] = {
                                    "model": sim_model,
                                    "qty": qty,
                                    "avg_price": price,
                                    "entry_date": str(buy_date),
                                    "entry_signal": str(order["Signal"]),
                                    "ml_prob": float(order["ML_Prob"]),
                                    "stop_loss": price * 0.90,
                                    "target_price": price * 1.08,
                                }
                            st.session_state.paper_cash -= amount
                            record_transaction("BUY", sim_model, symbol, qty, price, buy_date, note=f"Bought from {order['Signal']} call, ML probability {float(order['ML_Prob']):.2f}")
                        st.success("Buy order executed in the paper simulator.")
                        st.rerun()
            else:
                st.info("Select one or two recommended stocks, enter quantity, then press Buy.")
                display_table(buy_candidates.head(30), ["Model", "Date", "SYMBOL", "CLOSE", "ML_Prob", "Signal", "Reason"], 340)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How Buy/Sell Works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            **Model selector:** chooses which model supplies recommendations.  
            **Buy button:** records a simulated purchase and reduces paper cash.  
            **System Sell Check:** sells only holdings where the rule says SELL.  
            **Manual Sell:** kept for UI demonstration/testing.

            Demo sell rules:
            - sell if price hits the **10% stop loss**
            - sell if price reaches the **8% target**
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Step 2 - Holdings & Sell Controls</div>', unsafe_allow_html=True)
    holdings_df = build_holdings_df(price_map, valuation_date)
    if holdings_df.empty:
        st.info("No active paper holdings yet. Buy a recommended stock first.")
    else:
        display_table(holdings_df, ["Model", "SYMBOL", "Entry_Date", "Quantity", "Avg_Buy_Price", "Current_Date", "Current_Price", "Invested", "Current_Value", "Unrealized_PnL", "Unrealized_PnL_%", "Stop_Loss", "Target_Price", "System_Status"], 320)
        sell_ready = holdings_df[holdings_df["System_Status"].astype(str).str.startswith("SELL")]
        if st.button("Run System Sell Check", type="primary", use_container_width=True):
            if sell_ready.empty:
                st.warning("No system sell condition is currently triggered. Status is HOLD.")
            else:
                for _, row in sell_ready.iterrows():
                    symbol = str(row["SYMBOL"])
                    holding = st.session_state.paper_holdings.get(symbol)
                    if not holding:
                        continue
                    qty = float(holding["qty"])
                    avg_price = float(holding["avg_price"])
                    sell_price = float(row["Current_Price"])
                    revenue = qty * sell_price
                    profit = (sell_price - avg_price) * qty
                    st.session_state.paper_cash += revenue
                    model_name = holding.get("model", sim_model)
                    del st.session_state.paper_holdings[symbol]
                    record_transaction("SELL", model_name, symbol, qty, sell_price, valuation_date, note=str(row["System_Status"]), profit_rs=profit)
                st.success("System sell orders executed.")
                st.rerun()

        st.markdown("##### Manual sell controls for demo")
        for _, row in holdings_df.iterrows():
            symbol = str(row["SYMBOL"])
            holding = st.session_state.paper_holdings[symbol]
            max_qty = float(holding["qty"])
            sell_price = float(row["Current_Price"])
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
            c1.markdown(f"**{symbol}**")
            c2.markdown(f"Price: Rs. {sell_price:,.2f}")
            c3.markdown(f"Held: {max_qty:g}")
            sell_qty = c4.number_input(f"Sell Qty {symbol}", min_value=1.0, max_value=max_qty, value=max_qty, step=1.0, label_visibility="collapsed", key=f"sell_qty_{symbol}_{valuation_date}")
            if c5.button(f"Sell {symbol}", key=f"manual_sell_{symbol}_{valuation_date}", use_container_width=True):
                avg_price = float(holding["avg_price"])
                revenue = float(sell_qty) * sell_price
                profit = (sell_price - avg_price) * float(sell_qty)
                st.session_state.paper_cash += revenue
                remaining_qty = max_qty - float(sell_qty)
                model_name = holding.get("model", sim_model)
                if remaining_qty <= 0:
                    del st.session_state.paper_holdings[symbol]
                else:
                    st.session_state.paper_holdings[symbol]["qty"] = remaining_qty
                record_transaction("SELL", model_name, symbol, sell_qty, sell_price, valuation_date, note="Manual demo sell", profit_rs=profit)
                st.success(f"Sold {sell_qty:g} shares of {symbol} in the paper simulator.")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Paper Transaction History</div>', unsafe_allow_html=True)
    tx_df = pd.DataFrame(st.session_state.paper_transactions)
    if tx_df.empty:
        st.info("No paper transactions yet.")
    else:
        display_table(tx_df.iloc[::-1], None, 320)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Trade History":
    st.markdown('<div class="section-title">Trade History & Exit Reasons</div>', unsafe_allow_html=True)
    if trades_all.empty:
        st.warning("No trades were generated.")
    else:
        f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.2, 1])
        with f1:
            selected_models = st.multiselect("Models", available_models, default=available_models, key="trade_models")
        with f2:
            symbols = sorted(trades_all["SYMBOL"].dropna().astype(str).unique().tolist())
            selected_symbols = st.multiselect("Symbols", symbols, default=[])
        with f3:
            reasons = sorted(trades_all["Exit_Reason"].dropna().astype(str).unique().tolist())
            selected_reasons = st.multiselect("Exit Reason", reasons, default=[])
        with f4:
            profitable_only = st.toggle("Profitable trades only", value=False)

        hist = trades_all[trades_all["Model"].isin(selected_models)].copy()
        if selected_symbols:
            hist = hist[hist["SYMBOL"].isin(selected_symbols)]
        if selected_reasons:
            hist = hist[hist["Exit_Reason"].isin(selected_reasons)]
        if profitable_only:
            hist = hist[hist["Profit_Rs"] > 0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Filtered Trades", f"{len(hist):,}", "After filters")
        with c2:
            metric_card("Filtered Profit", money(hist["Profit_Rs"].sum() if not hist.empty else 0), "Sum of P/L")
        with c3:
            metric_card("Avg Return", pct(hist["Return_%"].mean() if not hist.empty else 0), "Per trade")
        with c4:
            metric_card("Best Trade", pct(hist["Return_%"].max() if not hist.empty else 0), "Max return")

        left, right = st.columns([1.25, 1])
        with left:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            sort_col = "Exit_Date" if "Exit_Date" in hist.columns else "SYMBOL"
            display_table(hist.sort_values(sort_col, ascending=False), None, 500)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            reason_counts = hist.groupby(["Model", "Exit_Reason"], as_index=False).size().rename(columns={"size": "Count"}) if not hist.empty else pd.DataFrame(columns=["Model", "Exit_Reason", "Count"])
            fig = px.bar(reason_counts, y="Exit_Reason", x="Count", color="Model", orientation="h", title="Exit Reason Distribution", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(l=20, r=20, t=55, b=20), font=dict(family="Inter", color="white"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Search":
    st.markdown('<div class="section-title">Search Historical Signals & Trades</div>', unsafe_allow_html=True)
    query = st.text_input("Search symbol, signal, reason, model, or exit reason", placeholder="Example: BUY, Reward Target, TRG, LightGBM")
    d1, d2, d3, d4 = st.columns([1, 1, 1, 1.2])
    with d1:
        start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=latest_date)
    with d2:
        end_date = st.date_input("End", value=latest_date, min_value=min_date, max_value=latest_date)
    with d3:
        min_prob_search = st.slider("Min ML Probability", 0.0, 1.0, 0.0, 0.01, key="search_prob")
    with d4:
        selected_models = st.multiselect("Models", available_models, default=available_models, key="search_models")

    sig_result = signals_all[
        (signals_all["Model"].isin(selected_models)) &
        (signals_all["Date"].dt.date >= start_date) &
        (signals_all["Date"].dt.date <= end_date) &
        (signals_all["ML_Prob"] >= min_prob_search)
    ].copy()

    trade_result = trades_all[trades_all["Model"].isin(selected_models)].copy()
    if not trade_result.empty and "Entry_Date" in trade_result.columns and "Exit_Date" in trade_result.columns:
        trade_result = trade_result[(trade_result["Entry_Date"].dt.date <= end_date) & (trade_result["Exit_Date"].dt.date >= start_date)].copy()

    if query:
        q = query.lower()
        sig_result = sig_result[
            sig_result["Model"].astype(str).str.lower().str.contains(q, na=False) |
            sig_result["SYMBOL"].astype(str).str.lower().str.contains(q, na=False) |
            sig_result["Signal"].astype(str).str.lower().str.contains(q, na=False) |
            sig_result["Reason"].astype(str).str.lower().str.contains(q, na=False)
        ]
        if not trade_result.empty:
            trade_result = trade_result[
                trade_result["Model"].astype(str).str.lower().str.contains(q, na=False) |
                trade_result["SYMBOL"].astype(str).str.lower().str.contains(q, na=False) |
                trade_result["Exit_Reason"].astype(str).str.lower().str.contains(q, na=False)
            ]

    tab1, tab2 = st.tabs(["Historical Signals", "Matching Trades"])
    with tab1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        display_table(sig_result.sort_values(["Date", "Model", "ML_Prob"], ascending=[False, True, False]), ["Model", "Date", "SYMBOL", "CLOSE", "ML_Prob", "Signal", "Reason", "RSI_7D", "RSI_30D"], 560)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        if trade_result.empty:
            st.info("No matching trades found.")
        else:
            sort_col = "Exit_Date" if "Exit_Date" in trade_result.columns else "SYMBOL"
            display_table(trade_result.sort_values(sort_col, ascending=False), None, 560)
        st.markdown('</div>', unsafe_allow_html=True)
