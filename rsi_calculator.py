import pandas as pd
import numpy as np
import warnings

# Suppress pandas FutureWarnings for clean output
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_rsi_for_group(group, price_col='CLOSE', periods=[1, 7, 30]):
    """
    Calculates the Relative Strength Index (RSI) for a single stock symbol.
    """
    # Sort chronologically
    group = group.sort_values(by='Date')
    
    # Handle missing values: Forward fill
    group[price_col] = group[price_col].ffill()
    
    # 1. Calculate the daily price changes
    delta = group[price_col].diff()
    
    # 2. Separate the gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # 3. Calculate RSI for each specified period
    for period in periods:
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        
        # Handle cases where avg_loss is 0
        rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
        
        group[f'RSI_{period}D'] = rsi
        
    return group

# ==========================================
# Execution & Display
# ==========================================

if __name__ == "__main__":
    file_name = "psx_historical_data_clean_no_hyphen.csv"
    
    try:
        # 1. Load data. low_memory=False resolves the DtypeWarning during initial load.
        print("Loading dataset...")
        df = pd.read_csv(file_name, low_memory=False)
        
        # 2. Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 3. Clean the 'CLOSE' column (remove commas, force to numeric float)
        # errors='coerce' will turn completely unparseable junk into NaN so we can forward-fill it later
        df['CLOSE'] = pd.to_numeric(df['CLOSE'].astype(str).str.replace(',', ''), errors='coerce')
        
        # 4. Calculate RSI PER SYMBOL
        # We must group by SYMBOL so prices from different stocks don't mix!
        print("Calculating RSI per symbol. This may take a moment depending on dataset size...")
        final_df = df.groupby('SYMBOL', group_keys=False).apply(
            lambda x: calculate_rsi_for_group(x, price_col='CLOSE', periods=[1, 7, 30])
        )
        
        # Sort the final output nicely by Symbol and Date
        final_df = final_df.sort_values(['SYMBOL', 'Date']).reset_index(drop=True)
        
        print("\nDataset successfully enhanced with RSI indicators:")
        print("-" * 75)
        # Displaying SYMBOL alongside Date and RSI to verify it worked correctly
        print(final_df[['SYMBOL', 'Date', 'CLOSE', 'RSI_1D', 'RSI_7D', 'RSI_30D']].tail(15).to_string(index=False))
        
        # Optional: Save it to a new CSV
        final_df.to_csv("psx_with_rsi.csv", index=False)
        print("\nSaved to psx_with_rsi.csv")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in the current directory.")