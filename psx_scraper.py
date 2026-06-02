import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment later to run silently
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_psx_historical(start_date, end_date):
    driver = setup_driver()
    url = "https://dps.psx.com.pk/historical"
    master_df = pd.DataFrame()
    
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    driver.get(url)
    print("Loading PSX Portal...")
    time.sleep(4) 
    
    for current_date in date_range:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nScraping data for: {date_str}")
        
        try:
            # 1. Enter the date and click search
            date_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='date' or @name='date']"))
            )
            driver.execute_script(f"arguments[0].value = '{date_str}';", date_input)
            
            search_button = date_input.find_element(
                By.XPATH, 
                "./following::button[contains(translate(text(), 'search', 'SEARCH'), 'SEARCH')][1]"
            )
            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(3) # Wait for initial table to load
            
            # 2. Loop through the pagination safely
            daily_dfs = []
            page_num = 1
            
            while True:
                soup = BeautifulSoup(driver.page_source, 'lxml')
                table = soup.find('table')
                
                if not table:
                    break # No table found
                    
                df = pd.read_html(str(table))[0]
                
                if df.empty:
                    break
                    
                daily_dfs.append(df)
                print(f"  - Scraped page {page_num} ({len(df)} rows)")
                
                # Memorize the very first value in the table (usually the stock Symbol/Code)
                # We will use this to verify when the NEXT page has actually loaded
                first_row_value = df.iloc[0, 0] 
                
                try:
                    next_btn = driver.find_element(By.CSS_SELECTOR, ".paginate_button.next, .page-item.next")
                    
                    if "disabled" in next_btn.get_attribute("class"):
                        break # We are on the last page
                    
                    # Click "Next"
                    driver.execute_script("arguments[0].click();", next_btn)
                    page_num += 1
                    
                    # SMART WAIT: Wait until the first row's value changes
                    wait_time = 0
                    while wait_time < 5: # Max wait 5 seconds per page
                        time.sleep(0.5)
                        temp_soup = BeautifulSoup(driver.page_source, 'lxml')
                        temp_table = temp_soup.find('table')
                        if temp_table:
                            temp_df = pd.read_html(str(temp_table))[0]
                            # If the new table's first symbol is different from the old one, the page has loaded!
                            if not temp_df.empty and temp_df.iloc[0, 0] != first_row_value:
                                break 
                        wait_time += 0.5
                        
                except Exception:
                    break # Next button not found

            # 3. Combine all pages and strictly remove duplicates
            if daily_dfs:
                final_daily_df = pd.concat(daily_dfs, ignore_index=True)
                final_daily_df['Date'] = date_str 
                
                # STRICT DUPLICATE REMOVAL: Drop duplicates based strictly on the Symbol column (1st column)
                first_column_name = final_daily_df.columns[0]
                final_daily_df = final_daily_df.drop_duplicates(subset=[first_column_name], keep='first')
                
                master_df = pd.concat([master_df, final_daily_df], ignore_index=True)
                print(f"--> Total unique rows collected for {date_str}: {len(final_daily_df)}")
            else:
                print(f"--> No data found for {date_str} (market closed/holiday).")
                
        except Exception as e:
            print(f"--> Failed on {date_str}. Error: {type(e).__name__}")
            
    driver.quit()
    return master_df

def clean_dataset(dataset):
    """Remove rows where SYMBOL contains a hyphen (e.g., TRG-MAR)"""
    rows_before = len(dataset)
    print(f"\n{'='*60}")
    print(f"DATASET CLEANING")
    print(f"{'='*60}")
    print(f"Rows BEFORE cleaning: {rows_before}")
    
    # Remove rows where SYMBOL contains a hyphen
    dataset_cleaned = dataset[~dataset['SYMBOL'].str.contains('-', na=False)].copy()
    
    rows_after = len(dataset_cleaned)
    rows_removed = rows_before - rows_after
    
    print(f"Rows AFTER cleaning:  {rows_after}")
    print(f"Rows REMOVED:         {rows_removed}")
    print(f"{'='*60}\n")
    
    return dataset_cleaned

if __name__ == "__main__":
    END_DATE = datetime.today()
    START_DATE = END_DATE - timedelta(days=5*365) # Still on 3-day test mode
    
    print(f"Starting scrape from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    
    dataset = scrape_psx_historical(START_DATE, END_DATE)
    
    if not dataset.empty:
        # Clean the dataset before saving
        dataset_cleaned = clean_dataset(dataset)
        
        output_filename = "psx_historical_data_clean.csv"
        dataset_cleaned.to_csv(output_filename, index=False)
        print(f"Success! Clean dataset saved to {output_filename}")
        print(f"Total unique records collected across all days: {len(dataset)}")
    else:
        print("\nFailed to collect any data.")