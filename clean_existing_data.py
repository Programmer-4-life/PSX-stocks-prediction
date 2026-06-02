import pandas as pd

def clean_dataset(input_file, output_file):
    """Remove rows where SYMBOL contains a hyphen (e.g., TRG-MAR)"""
    
    # Load the dataset
    dataset = pd.read_csv(input_file)
    rows_before = len(dataset)
    
    print(f"\n{'='*60}")
    print(f"DATASET CLEANING")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Rows BEFORE cleaning: {rows_before}")
    
    # Remove rows where SYMBOL contains a hyphen
    dataset_cleaned = dataset[~dataset['SYMBOL'].str.contains('-', na=False)].copy()
    
    rows_after = len(dataset_cleaned)
    rows_removed = rows_before - rows_after
    
    print(f"Rows AFTER cleaning:  {rows_after}")
    print(f"Rows REMOVED:         {rows_removed}")
    
    # Save the cleaned dataset
    dataset_cleaned.to_csv(output_file, index=False)
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    input_file = "psx_historical_data_clean.csv"
    output_file = "psx_historical_data_clean_no_hyphen.csv"
    
    clean_dataset(input_file, output_file)
    print("✓ Cleaning complete!")
