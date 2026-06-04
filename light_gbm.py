from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load and Clean Data
# ==========================================
def load_and_clean_data(filepath):
    print(f"Loading and cleaning market data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_1D', 'RSI_7D', 'RSI_30D']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['SYMBOL', 'Date', 'CLOSE']).copy()
    df = df.sort_values(by=['SYMBOL', 'Date']).reset_index(drop=True)
    return df


# ==========================================
# 2. Compute Technicals & ML Target
# ==========================================
def compute_features(group):
    close = group['CLOSE']
    high  = group['HIGH']
    low   = group['LOW']

    # OPTIMIZATION: Extended horizon to 10 days, target 3% to match swing trading execution
    group['Target_10D'] = (group['CLOSE'].shift(-10) > (group['CLOSE'] * 1.03)).astype(int)

    group['Vol_MA20']  = group['VOLUME'].rolling(window=20).mean()
    #Mathematical Calculation Errors can occur if the 20-day average volume is zero.
    #Only divide the numbers if the 20-day average is greater than zero. 
    # If it is zero, don't try to divide it; just assign the ratio a default value of 1.0 (meaning normal volume).
    group['Vol_Ratio'] = np.where(group['Vol_MA20'] > 0,
                                  group['VOLUME'] / group['Vol_MA20'], 1.0)

    group['EMA_20']      = close.ewm(span=20, adjust=False).mean()
    group['EMA_50']      = close.ewm(span=50, adjust=False).mean()
    group['SMA_200']     = close.rolling(window=200).mean()
    group['Dist_EMA20']  = (close - group['EMA_20'])  / group['EMA_20']
    group['Dist_SMA200'] = (close - group['SMA_200']) / group['SMA_200']

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    group['MACD']        = ema_12 - ema_26
    group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
    group['MACD_Hist']   = group['MACD'] - group['MACD_Signal']

    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    group['BB_Upper'] = sma_20 + (2 * std_20)
    group['BB_Lower'] = sma_20 - (2 * std_20)
    group['BB_Width'] = (group['BB_Upper'] - group['BB_Lower']) / sma_20

    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    abs(high - prev_close),
                    abs(low  - prev_close)], axis=1).max(axis=1)
    group['ATR_14']  = tr.rolling(window=14).mean()
    group['ATR_Pct'] = group['ATR_14'] / close

    return group


# ==========================================
# 3. LightGBM Training
# ==========================================
def train_and_predict_lgbm(df, test_years=1, n_estimators=300):
    print("\nPreparing LightGBM ML Pipeline...")

    features = ['RSI_7D', 'RSI_30D', 'Vol_Ratio', 'MACD_Hist',
                'ATR_Pct', 'Dist_EMA20', 'Dist_SMA200', 'BB_Width']

    #Just in case any bizarre math calculation sneaks through and creates an "infinity" value, 
    # this line hunts down all infinities and converts them to NaN (Not a Number), 
    # which LightGBM is uniquely designed to handle safely without crashing.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ml_df = df.dropna(subset=features + ['Target_10D']).copy()

    max_date    = ml_df['Date'].max()
    cutoff_date = max_date - pd.DateOffset(years=test_years)
    train_data  = ml_df[ml_df['Date'] <= cutoff_date].copy()
    test_data   = ml_df[ml_df['Date'] >  cutoff_date].copy()

    X_train = train_data[features]
    y_train = train_data['Target_10D']
    X_test  = test_data[features]
    y_test  = test_data['Target_10D']

    # Dynamically adjusted n_estimators via the pipeline parameter
    lgbm_model = lgb.LGBMClassifier(
        n_estimators     = n_estimators,
        learning_rate    = 0.05,
        num_leaves       = 31,
        max_depth        = 6,
        min_child_samples= 20,
        subsample        = 0.8, #If subsample = 0.8, it means that when LightGBM is getting ready to build a new decision tree, it randomly throws out 20% of the historical days and trains only on the remaining 80%. When it moves on to build the next tree, it reshuffles the deck and picks a different random 80%.
        subsample_freq   = 1, 
        colsample_bytree = 0.8, #If colsample_bytree = 0.8, LightGBM will randomly hide 2 of those indicators from every single tree it builds. One tree might be forced to make predictions without seeing MACD_Hist. The next tree might have to make predictions without seeing RSI_7D.
        class_weight     = 'balanced',
        reg_alpha        = 0.1, #Mathematical Penalties (Regularization): If an indicator (like RSI_30D) isn't consistently helping the model, L1 regularization will aggressively shrink its mathematical importance to exactly $0$. It essentially acts as an automatic cleanup crew, deleting useless noise from the equation.
        reg_lambda       = 0.1, #L2 Regularization (reg_lambda): If the model becomes overly confident and assigns massive importance to a single indicator, L2 mathematically penalizes it. This forces the model to distribute its decision-making power smoothly across multiple indicators rather than blindly trusting just one.
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1
    )

    print(f"  Training LightGBM with {n_estimators} estimators...")
    lgbm_model.fit(X_train, y_train)

    test_data            = test_data.copy()
    test_data['ML_Prob'] = lgbm_model.predict_proba(X_test)[:, 1]
    
    probs = test_data['ML_Prob']
    auc   = roc_auc_score(y_test, probs)
    print(f"  ROC-AUC (out-of-sample) : {auc:.4f}")

    return test_data, lgbm_model, auc


# ==========================================
# 4. Signal Generation
# ==========================================
def generate_signals(df):
    print("\nApplying Quant Logic to LightGBM Predictions...")
    df = df.copy()
    df['Signal'] = 'HOLD'
    df['Reason'] = ''

    p90 = df['ML_Prob'].quantile(0.90)
    p95 = df['ML_Prob'].quantile(0.95)

    buy_thresh        = max(p90, 0.45)
    strong_buy_thresh = max(p95, 0.50)

    regime_pass    = (df['Vol_MA20'] > 50000) & (df['ATR_Pct'] > 0.015) 
    trend_bullish  = (df['CLOSE'] > df['EMA_50']) & (df['EMA_50'] > df['SMA_200']) 
    mom_macd_cross = (df['MACD_Hist'] > 0)
    mom_reject     = (df['RSI_7D'] > 80) 

    conf_count = (trend_bullish.astype(int) +
                  mom_macd_cross.astype(int) +
                  (df['Vol_Ratio'] >= 1.2).astype(int))

    buy_mask        = (df['ML_Prob'] >= buy_thresh)        & (conf_count >= 2) & regime_pass & ~mom_reject
    strong_buy_mask = (df['ML_Prob'] >= strong_buy_thresh) & (conf_count >= 3) & regime_pass & ~mom_reject

    df.loc[buy_mask,        'Signal'] = 'BUY'
    df.loc[buy_mask,        'Reason'] = f'LGBM>={buy_thresh:.3f} + 2 confirms'
    df.loc[strong_buy_mask, 'Signal'] = 'STRONG BUY'
    df.loc[strong_buy_mask, 'Reason'] = f'LGBM>={strong_buy_thresh:.3f} + 3 confirms'

    return df


# ==========================================
# 5. Portfolio Backtesting (Risk-Based Sizing)
# ==========================================
RISK_PER_TRADE_PCT = 0.02
MAX_POSITION_PCT   = 0.12 
MAX_LOSS_PCT       = 0.08 

def backtest_portfolio_system(test_df, initial_capital=100_000.0, max_positions=8):
    print(f"\nRunning Portfolio Simulation...")

    cash         = initial_capital
    positions    = {}
    trade_log    = []
    port_history = []

    test_df     = test_df.sort_values(by=['Date', 'SYMBOL'])
    date_groups = dict(tuple(test_df.groupby('Date')))

    for date in sorted(date_groups.keys()):
        day_data = date_groups[date]
        prices   = dict(zip(day_data['SYMBOL'], day_data['CLOSE']))
        signals  = dict(zip(day_data['SYMBOL'], day_data['Signal']))
        atrs     = dict(zip(day_data['SYMBOL'], day_data['ATR_14']))

        sv_now           = sum(pos['shares'] * prices.get(s, pos['entry_price'])
                               for s, pos in positions.items())
        portfolio_equity = cash + sv_now

        symbols_to_sell = []

        for sym, pos in positions.items():
            cur = prices.get(sym, pos['entry_price'])
            pos['days_held'] += 1
            roi = (cur - pos['entry_price']) / pos['entry_price']

            if cur > pos['highest_price']:
                pos['highest_price'] = cur
                
            if roi >= 0.04 and pos['stop_loss'] < pos['entry_price']:
                pos['stop_loss'] = pos['entry_price'] * 1.01
                
            trail_dist = min(2.0 * pos['entry_atr'], cur * MAX_LOSS_PCT)
            new_stop = pos['highest_price'] - trail_dist
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop

            if cur <= pos['stop_loss']:
                if cur < (pos['entry_price'] * 0.90):
                    exec_price = pos['entry_price'] * 0.90 
                    reason = "Stop Hit (Strict 10% Loss Cap)"
                else:
                    exec_price = cur
                    reason = "Stop Loss Hit" if roi <= 0 else "Trailing Stop / Break-Even Hit"
                
                symbols_to_sell.append((sym, reason, exec_price))
            
            elif cur >= pos['entry_price'] + 3 * pos['entry_atr']:
                symbols_to_sell.append((sym, "Reward Target Reached (3x ATR)", cur))
            elif pos['days_held'] >= 15 and roi < 0.015:
                symbols_to_sell.append((sym, "Time Stop (Capital Reallocation)", cur))

        for sym, exit_reason, final_exit_price in symbols_to_sell:
            sh    = positions[sym]['shares']
            entp  = positions[sym]['entry_price']
            
            cash += sh * final_exit_price
            trade_log.append({
                'SYMBOL'     : sym,
                'Entry_Date' : positions[sym]['entry_date'],
                'Entry_Price': entp,
                'Exit_Date'  : date,
                'Exit_Price' : final_exit_price,
                'Return_%'   : (final_exit_price - entp) / entp * 100,
                'Profit_Rs'  : (sh * final_exit_price) - (sh * entp),
                'Exit_Reason': exit_reason
            })
            del positions[sym]

        to_buy = [(s, sig) for s, sig in signals.items()
                  if sig in ['BUY', 'STRONG BUY'] and s not in positions]
        to_buy.sort(key=lambda x: 1 if x[1] == 'STRONG BUY' else 0, reverse=True)

        for sym, sig in to_buy:
            if len(positions) >= max_positions:
                break
            ep  = prices.get(sym, 0)
            atr = atrs.get(sym, 0)
            if ep <= 0 or atr <= 0:
                continue

            dollar_risk   = portfolio_equity * RISK_PER_TRADE_PCT
            stop_distance = min(2.0 * atr, ep * MAX_LOSS_PCT)
            if stop_distance <= 0:
                continue

            shares        = dollar_risk / stop_distance
            shares        = min(shares, (portfolio_equity * MAX_POSITION_PCT) / ep)
            position_val  = shares * ep

            if position_val > cash:
                shares       = cash / ep
                position_val = cash

            if shares <= 0:
                continue

            cash -= position_val
            positions[sym] = {
                'shares'       : shares,
                'entry_price'  : ep,
                'entry_date'   : date,
                'highest_price': ep,
                'days_held'    : 0,
                'entry_atr'    : atr,
                'stop_loss'    : ep - stop_distance
            }

        sv = sum(pos['shares'] * prices.get(s, pos['entry_price'])
                 for s, pos in positions.items())
        port_history.append({'Date': date, 'Total_Equity': cash + sv})

    return pd.DataFrame(port_history), pd.DataFrame(trade_log)


# ==========================================
# 6. Performance Report
# ==========================================
def print_performance_metrics(trade_log, portfolio_df, initial_capital=100_000.0):
    print("\n" + "=" * 55)
    print("  LIGHTGBM HYBRID PERFORMANCE REPORT")
    print("=" * 55)

    final_equity = portfolio_df.iloc[-1]['Total_Equity'] if not portfolio_df.empty else initial_capital
    total_profit = final_equity - initial_capital
    roi_pct      = total_profit / initial_capital * 100

    print(f"  Starting Capital     : Rs. {initial_capital:>12,.2f}")
    print(f"  Final Capital        : Rs. {final_equity:>12,.2f}")
    print(f"  Net Profit           : Rs. {total_profit:>12,.2f}")
    print(f"  Total Portfolio ROI  :     {roi_pct:>+10.2f}%")
    print("-" * 55)

    if trade_log.empty:
        print("  No trades triggered.")
        return

    total_trades  = len(trade_log)
    wins          = (trade_log['Return_%'] > 0).sum()
    losses        = (trade_log['Return_%'] <= 0).sum()
    win_rate      = wins / total_trades * 100
    avg_return    = trade_log['Return_%'].mean()
    avg_win       = trade_log.loc[trade_log['Return_%'] > 0,  'Return_%'].mean()
    avg_loss      = trade_log.loc[trade_log['Return_%'] <= 0, 'Return_%'].mean()
    gross_profit  = trade_log.loc[trade_log['Return_%'] > 0,  'Profit_Rs'].sum()
    gross_loss    = abs(trade_log.loc[trade_log['Return_%'] <= 0, 'Profit_Rs'].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    print(f"  Total Trades         :     {total_trades:>8}")
    print(f"  Wins / Losses        :     {wins:>4} / {losses:<4}")
    print(f"  Win Rate             :     {win_rate:>8.2f}%")
    print(f"  Avg Trade ROI        :     {avg_return:>+8.2f}%")
    print(f"  Avg Win              :     {avg_win:>+8.2f}%")
    print(f"  Avg Loss             :     {avg_loss:>+8.2f}%")
    print(f"  Profit Factor        :     {profit_factor:>8.2f}x")
    print(f"  Best Trade           :     {trade_log['Return_%'].max():>+8.2f}%")
    print(f"  Worst Trade          :     {trade_log['Return_%'].min():>+8.2f}%")

    print("\n  [Exit Reason Distribution]")
    for reason, count in trade_log['Exit_Reason'].value_counts().items():
        pct = count / total_trades * 100
        print(f"    {reason:<35} {count:>4}  ({pct:.1f}%)")

    print("=" * 55)


# ==========================================
# 7. EXPORTABLE ENTRY POINT (MAIN PIPELINE)
# ==========================================
def main(input_file: str, output_dir: str, initial_capital: float, max_positions: int, test_years: int, n_estimators: int) -> None:
    """
    Exposes the LightGBM execution context to external pipelines.
    Runs the base model AND generates the position sensitivity analysis CSV.
    """
    out_path = Path(output_dir)

    df = load_and_clean_data(input_file)
    print("Computing technical features per symbol...")
    df = df.groupby('SYMBOL', group_keys=False).apply(compute_features)

    test_df, model, auc = train_and_predict_lgbm(df, test_years=test_years, n_estimators=n_estimators)
    test_df              = generate_signals(test_df)
    
    # --- Run Base Portfolio Simulation ---
    portfolio_df, trade_log_df = backtest_portfolio_system(
        test_df, initial_capital=initial_capital, max_positions=max_positions
    )
    print_performance_metrics(trade_log_df, portfolio_df, initial_capital=initial_capital)

    # Save standard outputs
    output_cols = ['Date', 'SYMBOL', 'CLOSE', 'ML_Prob', 'Signal', 'Reason']
    test_df[output_cols].to_csv(out_path / "lgbm_signals.csv",   index=False)
    trade_log_df.to_csv(out_path / "lgbm_trade_log.csv",         index=False)
    portfolio_df.to_csv(out_path / "lgbm_portfolio_history.csv", index=False)

    # --- NEW: Position Sensitivity Analysis Loop ---
    print("\nGenerating Position Size Sensitivity Analysis...")
    sensitivity_results = []
    
    # Test different max position limits (e.g., from 2 positions up to 10 positions)
    for pos_limit in range(2, 11):
        print(f"  Testing pipeline with MAX_POSITIONS = {pos_limit}...")
        p_df, t_log = backtest_portfolio_system(
            test_df, initial_capital=initial_capital, max_positions=pos_limit
        )
        
        if not p_df.empty and not t_log.empty:
            final_equity = p_df.iloc[-1]['Total_Equity']
            net_profit = final_equity - initial_capital
            roi_pct = (net_profit / initial_capital) * 100
            total_trades = len(t_log)
            win_rate = ((t_log['Return_%'] > 0).sum() / total_trades) * 100
            
            sensitivity_results.append({
                'Max_Positions': pos_limit,
                'Final_Capital': final_equity,
                'Net_Profit': net_profit,
                'Portfolio_ROI_Pct': roi_pct,
                'Total_Trades': total_trades,
                'Win_Rate_Pct': win_rate
            })
            
    # Save the sensitivity results to CSV
    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df.to_csv(out_path / "lgbm_position_sensitivity.csv", index=False)
    print(f"--> Sensitivity analysis saved: {out_path / 'lgbm_position_sensitivity.csv'}")

# Retain standalone manual execution option
if __name__ == "__main__":
    main(
        input_file="psx_with_rsi.csv",
        output_dir=".",
        initial_capital=100000.0,
        max_positions=8,
        test_years=1,
        n_estimators=300
    )