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


def add_rsi(input_file: str, output_file: str) -> None:
    """
    Exposes the RSI calculation logic to external scripts (like run_pipeline.py).
    Accepts dynamic input and output file paths.
    """
    # 1. Load data
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    
    # 2. Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Clean the 'CLOSE' column
    df['CLOSE'] = pd.to_numeric(df['CLOSE'].astype(str).str.replace(',', ''), errors='coerce')
    
    # 4. Calculate RSI PER SYMBOL
    print("Calculating RSI per symbol. This may take a moment depending on dataset size...")
    final_df = df.groupby('SYMBOL', group_keys=False).apply(
        lambda x: calculate_rsi_for_group(x, price_col='CLOSE', periods=[1, 7, 30])
    )
    
    # Sort the final output nicely by Symbol and Date
    final_df = final_df.sort_values(['SYMBOL', 'Date']).reset_index(drop=True)
    
    print("\nDataset successfully enhanced with RSI indicators:")
    print("-" * 75)
    print(final_df[['SYMBOL', 'Date', 'CLOSE', 'RSI_1D', 'RSI_7D', 'RSI_30D']].tail(15).to_string(index=False))
    
    # 5. Save it to the designated output path
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


# Allows you to still run this script directly on its own if needed
if __name__ == "__main__":
    file_name = "psx_historical_data_clean_no_hyphen.csv"
    output_name = "psx_with_rsi.csv"
    
    try:
        add_rsi(file_name, output_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in the current directory.")