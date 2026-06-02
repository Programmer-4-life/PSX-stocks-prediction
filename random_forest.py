import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load and Clean Data
# ==========================================
def load_and_clean_data(filepath):
    print("Loading and cleaning market data...")
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
# 2. Compute Technicals & STRICTER ML Target
# ==========================================
def compute_features(group):
    close = group['CLOSE']
    high = group['HIGH']
    low = group['LOW']
    
    # --- UPGRADED ML TARGET ---
    # Does the price go up by at least 1.5% 5 days from now?
    group['Target_5D'] = (group['CLOSE'].shift(-5) > (group['CLOSE'] * 1.015)).astype(int)
    
    # --- VOLUME ---
    group['Vol_MA20'] = group['VOLUME'].rolling(window=20).mean()
    group['Vol_Ratio'] = np.where(group['Vol_MA20'] > 0, group['VOLUME'] / group['Vol_MA20'], 1.0)
    
    # --- TREND & DISTANCE ---
    group['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    group['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    group['SMA_200'] = close.rolling(window=200).mean()
    
    group['Dist_EMA20'] = (close - group['EMA_20']) / group['EMA_20']
    group['Dist_SMA200'] = (close - group['SMA_200']) / group['SMA_200']
    
    # --- MOMENTUM (MACD) ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    group['MACD'] = ema_12 - ema_26
    group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
    group['MACD_Hist'] = group['MACD'] - group['MACD_Signal']
    
    # --- VOLATILITY ---
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    group['BB_Upper'] = sma_20 + (2 * std_20)
    group['BB_Lower'] = sma_20 - (2 * std_20)
    group['BB_Width'] = (group['BB_Upper'] - group['BB_Lower']) / sma_20
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    group['ATR_14'] = tr.rolling(window=14).mean()
    group['ATR_Pct'] = group['ATR_14'] / close 
    
    return group

# ==========================================
# 3. RANDOM FOREST TRAINING LOOP
# ==========================================
def train_and_predict_ml(df, test_years=1):
    print("\nPreparing Random Forest ML Pipeline...")
    
    features = ['RSI_7D', 'RSI_30D', 'Vol_Ratio', 'MACD_Hist', 
                'ATR_Pct', 'Dist_EMA20', 'Dist_SMA200', 'BB_Width']
    
    # Clean anomalies safely
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ml_df = df.dropna(subset=features + ['Target_5D']).copy()
    
    # Time-based Train/Test Split
    max_date = ml_df['Date'].max()
    cutoff_date = max_date - pd.DateOffset(years=test_years)
    
    train_data = ml_df[ml_df['Date'] <= cutoff_date].copy()
    test_data = ml_df[ml_df['Date'] > cutoff_date].copy()
    
    print(f"Training Model on {len(train_data)} rows (Past 4 Years)...")
    print(f"Testing Model on {len(test_data)} rows (Last 1 Year)...")
    
    X_train = train_data[features]
    y_train = train_data['Target_5D']
    X_test = test_data[features]
    
    # Train Random Forest (No scaling needed for trees, better with noisy data)
    print("Training Random Forest Classifier (Building trees)...")
    # max_depth=6 prevents overfitting. class_weight='balanced' helps if the 1.5% target is rare.
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, 
                                      class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Generating Out-Of-Sample Predictions...")
    test_data['ML_Prob'] = rf_model.predict_proba(X_test)[:, 1]
    
    return test_data

# ==========================================
# 4. BALANCED DECISION FRAMEWORK
# ==========================================
def generate_signals(df):
    print("\nApplying Quant Logic to Random Forest Predictions...")
    df['Signal'] = 'HOLD'
    df['Reason'] = ''
    
    df['Prev_CLOSE'] = df.groupby('SYMBOL')['CLOSE'].shift(1)
    df['Prev_EMA_20'] = df.groupby('SYMBOL')['EMA_20'].shift(1)
    df['Prev_MACD_Hist'] = df.groupby('SYMBOL')['MACD_Hist'].shift(1)
    
    # REGIME: Avoid completely dead stocks
    regime_pass = (df['Vol_MA20'] > 25000) & (df['ATR_Pct'] > 0.01)
    
    # TREND & MOMENTUM
    trend_bullish = (df['CLOSE'] > df['EMA_50'])
    mom_macd_cross = (df['MACD_Hist'] > 0)
    mom_reject = (df['RSI_7D'] > 75)
    
    # SCORING ALIGNMENT
    conf_count = trend_bullish.astype(int) + mom_macd_cross.astype(int) + (df['Vol_Ratio'] >= 1.2).astype(int)
    
    # DECISION RULES (Relaxed ML threshold because RF is more conservative with probs)
    buy_mask = (df['ML_Prob'] >= 0.55) & (conf_count >= 2) & regime_pass & ~mom_reject
    strong_buy_mask = (df['ML_Prob'] >= 0.65) & (conf_count >= 3) & regime_pass & ~mom_reject
    
    df.loc[buy_mask, 'Signal'] = 'BUY'
    df.loc[buy_mask, 'Reason'] = 'RF Prob > 0.55 + 2 Quant Confirmations'
    
    df.loc[strong_buy_mask, 'Signal'] = 'STRONG BUY'
    df.loc[strong_buy_mask, 'Reason'] = 'RF Prob > 0.65 + 3 Quant Confirmations'
    
    return df

# ==========================================
# 5. WIDENED RISK-ADJUSTED BACKTESTING (WITH 10% HARD CAP)
# ==========================================
def backtest_portfolio_system(test_df, initial_capital=100000.0, max_positions=4):
    print("Running Portfolio Simulation with Strict 10% Risk Ceiling...")
    
    cash = initial_capital
    positions = {} 
    trade_log = []
    portfolio_history = []
    
    # HARD RISK PARAMETERS
    MAX_LOSS_PCT = 0.10  # Never lose more than 10% on a trade
    
    test_df = test_df.sort_values(by=['Date', 'SYMBOL'])
    unique_dates = test_df['Date'].unique()
    date_groups = dict(tuple(test_df.groupby('Date')))
    
    for date in unique_dates:
        day_data = date_groups[date]
        prices = dict(zip(day_data['SYMBOL'], day_data['CLOSE']))
        signals = dict(zip(day_data['SYMBOL'], day_data['Signal']))
        atrs = dict(zip(day_data['SYMBOL'], day_data['ATR_14']))
        
        symbols_to_sell = []
        
        # EVALUATE EXITS
        for sym, pos_data in positions.items():
            current_price = prices.get(sym, pos_data['entry_price'])
            pos_data['days_held'] += 1
            
            if current_price > pos_data['highest_price']:
                pos_data['highest_price'] = current_price
                
                # Calculate Trailing Stop: 2.5x ATR, but NEVER more than 10% away from the highest price
                atr_drop = 2.5 * pos_data['entry_atr']
                max_allowed_drop = current_price * MAX_LOSS_PCT
                actual_drop = min(atr_drop, max_allowed_drop) # Take the smaller drop (tighter stop)
                
                new_stop = current_price - actual_drop
                
                if new_stop > pos_data['stop_loss']:
                    pos_data['stop_loss'] = new_stop
                    
            roi = (current_price - pos_data['entry_price']) / pos_data['entry_price']
            
            # TRIGGER EXITS
            if current_price <= pos_data['stop_loss']:
                if roi <= -0.095: # If it's near the 10% limit
                    symbols_to_sell.append((sym, "Hard 10% Stop Loss Hit"))
                else:
                    symbols_to_sell.append((sym, "ATR Trailing Stop Hit"))
                    
            elif current_price >= pos_data['entry_price'] + (4 * pos_data['entry_atr']):
                symbols_to_sell.append((sym, "Reward Target Reached (4x ATR)"))
                
            elif pos_data['days_held'] >= 30 and roi < 0.02:
                symbols_to_sell.append((sym, "Time Stop (Capital Reallocation)"))

        # PROCESS SELLS
        for sym, exit_reason in symbols_to_sell:
            exit_price = prices.get(sym, positions[sym]['entry_price'])
            shares = positions[sym]['shares']
            entry_price = positions[sym]['entry_price']
            
            trade_value = shares * exit_price
            cash += trade_value
            profit_rs = trade_value - (shares * entry_price)
            
            trade_log.append({
                'SYMBOL': sym, 'Entry_Date': positions[sym]['entry_date'], 'Entry_Price': entry_price,
                'Exit_Date': date, 'Exit_Price': exit_price, 'Return_%': ((exit_price - entry_price) / entry_price) * 100,
                'Profit_Rs': profit_rs, 'Exit_Reason': exit_reason
            })
            del positions[sym]
            
        # PROCESS BUYS
        symbols_to_buy = [(sym, sig) for sym, sig in signals.items() if sig in ['BUY', 'STRONG BUY'] and sym not in positions]
        symbols_to_buy.sort(key=lambda x: 1 if x[1] == 'STRONG BUY' else 0, reverse=True)
        
        for sym, sig in symbols_to_buy:
            if len(positions) < max_positions:
                allocation = cash / (max_positions - len(positions))
                entry_price = prices.get(sym, 0)
                entry_atr = atrs.get(sym, 0)
                
                if entry_price > 0 and entry_atr > 0 and allocation > 0:
                    shares = allocation / entry_price
                    cash -= allocation
                    
                    # INITIAL STOP LOSS: 2.5x ATR, strictly capped at 10% loss
                    atr_drop = 2.5 * entry_atr
                    max_allowed_drop = entry_price * MAX_LOSS_PCT
                    actual_drop = min(atr_drop, max_allowed_drop)
                    
                    initial_stop = entry_price - actual_drop
                    
                    positions[sym] = {
                        'shares': shares, 'entry_price': entry_price, 'entry_date': date,
                        'highest_price': entry_price, 'days_held': 0, 'entry_atr': entry_atr,
                        'stop_loss': initial_stop
                    }
                    
        # RECORD DAILY EQUITY
        stock_value = sum((pos['shares'] * prices.get(sym, pos['entry_price'])) for sym, pos in positions.items())
        portfolio_history.append({'Date': date, 'Total_Equity': cash + stock_value})

    last_date = portfolio_history[-1]['Date'] if portfolio_history else None
    for sym, pos_data in positions.items():
        exit_price = prices.get(sym, pos_data['entry_price'])
        shares = pos_data['shares']
        entry_price = pos_data['entry_price']
        profit_rs = (shares * exit_price) - (shares * entry_price)
        trade_log.append({
            'SYMBOL': sym, 'Entry_Date': pos_data['entry_date'], 'Entry_Price': entry_price,
            'Exit_Date': last_date, 'Exit_Price': exit_price, 'Return_%': ((exit_price - entry_price) / entry_price) * 100,
            'Profit_Rs': profit_rs, 'Exit_Reason': 'End of Backtest Liquidation'
        })

    return pd.DataFrame(portfolio_history), pd.DataFrame(trade_log)

# ==========================================
# 6. Final Outputs
# ==========================================
def print_performance_metrics(trade_log, portfolio_df, initial_capital=100000.0):
    print("\n" + "="*50)
    print("RANDOM FOREST HYBRID PERFORMANCE REPORT")
    print("="*50)
    
    final_equity = portfolio_df.iloc[-1]['Total_Equity'] if not portfolio_df.empty else initial_capital
    total_profit = final_equity - initial_capital
    roi_pct = (total_profit / initial_capital) * 100
    
    print(f"Starting Capital     : Rs. {initial_capital:,.2f}")
    print(f"Final Capital        : Rs. {final_equity:,.2f}")
    print(f"Net Profit           : Rs. {total_profit:,.2f}")
    print(f"Total Portfolio ROI  : {roi_pct:.2f}%")
    print("-" * 50)
    
    if trade_log.empty:
        print("No trades triggered.")
        return
        
    total_trades = len(trade_log)
    wins = trade_log[trade_log['Return_%'] > 0]
    win_rate = (len(wins) / total_trades) * 100
    avg_return = trade_log['Return_%'].mean()
    
    print(f"Total Trades Executed: {total_trades}")
    print(f"System Win Rate      : {win_rate:.2f}%")
    print(f"Average Trade ROI    : {avg_return:.2f}%")
    
    print("\n[Exit Reason Distribution]")
    print(trade_log['Exit_Reason'].value_counts().to_string())

if __name__ == "__main__":
    file_name = "psx_with_rsi.csv"
    
    df = load_and_clean_data(file_name)
    print("Computing technical features...")
    df = df.groupby('SYMBOL', group_keys=False).apply(compute_features)
    
    test_df = train_and_predict_ml(df, test_years=1)
    test_df = generate_signals(test_df)
    portfolio_df, trade_log_df = backtest_portfolio_system(test_df, initial_capital=100000.0, max_positions=4)
    
    print_performance_metrics(trade_log_df, portfolio_df, initial_capital=100000.0)
    
    output_cols = ['Date', 'SYMBOL', 'CLOSE', 'ML_Prob', 'Signal', 'Reason']
    test_df[output_cols].to_csv("rf_hedge_fund_signals.csv", index=False)
    trade_log_df.to_csv("rf_hedge_fund_trade_log.csv", index=False)
    
    print("\nDONE! Output saved to CSVs.")