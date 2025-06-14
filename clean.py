import pandas as pd
from glob import glob
import os
import argparse

print("Running clean.py...")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csvs', type=int, help='End number for the glob index filter')
args = parser.parse_args()

# example data/Market.csv
    # Index,Date,Open,High,Low,Close,Adj Close,Volume
    # NYA,12/31/1965,528.690002,528.690002,528.690002,528.690002,528.690002,0
    # NYA,1/3/1966,527.210022,527.210022,527.210022,527.210022,527.210022,0

# example data/Stocks/aaba.us.txt
    # Date,Open,High,Low,Close,Volume,OpenInt
    # 1996-04-12,1.05,1.79,1.02,1.38,408720000,0
    # 1996-04-15,1.49,1.5,1.25,1.34,79231200,0

# example data/Stocks/A.csv
    # Date,Open,High,Low,Close,Adj Close,Volume
    # 1999-11-18,32.54649353027344,35.765380859375,28.612302780151367,31.473533630371094,27.06866455078125,62546300
    # 1999-11-19,30.713520050048828,30.75822639465332,28.47818374633789,28.880542755126953,24.838577270507812,15234100

skipped = 0
def load_and_standardize_csv(file_path, columns_mapping, date_format, ticker=None):
    global skipped
    try:
        # print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        df.rename(columns=columns_mapping, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
        df = df[df['Date'] >= '2000-01-01'] #filter out dates before 2000


        # if the minimal close < 1 or maximum close > 1000, then return an empty DataFrame
        if df['Close'].min() < 1 or df['Close'].max() > 1000 or df['Open'].min() < 1 or df['Open'].max() > 1000:
            skipped += 1
            # print(f"Warning: {file_path} has an invalid price range and will be skipped.")
            return pd.DataFrame()
        
        df.drop(columns=['Volume', 'Adj Close', 'OpenInt'], errors='ignore', inplace=True)

        if ticker:
            df['Ticker'] = ticker#.upper()
        return df
    except pd.errors.EmptyDataError:
        skipped += 1
        # print(f"Warning: {file_path} is empty and will be skipped.")
        return pd.DataFrame()

columns_mapping_market = {
    'Index': 'Ticker',
    'Date': 'Date',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Adj Close': 'Adj Close',
    'Volume': 'Volume'
}
columns_mapping_stock_txt = {
    'Date': 'Date',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Adj Close': 'Adj Close',
    'Volume': 'Volume',
    'OpenInt': 'OpenInt'
}
columns_mapping_stock_csv = {
    'Date': 'Date',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Adj Close': 'Adj Close',
    'Volume': 'Volume'
}

# market_file = 'data/Market.csv'
stock_files_txt = glob('data/Stocks/*.us.txt')
stock_files_csv = glob('data/Stocks/*.csv')
if args.csvs:
    stock_files_txt = glob('data/Stocks/*.us.txt')[:min(args.csvs, len(stock_files_txt))]
    stock_files_csv = glob('data/Stocks/*.csv')[:min(args.csvs, len(stock_files_csv))]

# filter out filenames that start with a number
stock_files_txt = [file for file in stock_files_txt if not os.path.basename(file).split('.')[0][0].isdigit()]
stock_files_csv = [file for file in stock_files_csv if not os.path.basename(file).split('.')[0][0].isdigit()]

# market_df = load_and_standardize_csv(market_file, columns_mapping_market, "%m/%d/%Y")
stock_txt_dfs = [load_and_standardize_csv(file, columns_mapping_stock_txt, "%Y-%m-%d", os.path.basename(file).split('.')[0]) for file in stock_files_txt]
stock_csv_dfs = [load_and_standardize_csv(file, columns_mapping_stock_csv, "%Y-%m-%d", os.path.basename(file).split('.')[0]) for file in stock_files_csv]

print(f"Skipped {skipped} files.")

# Filter out empty DataFrames
stock_txt_dfs = [df for df in stock_txt_dfs if not df.empty]
stock_csv_dfs = [df for df in stock_csv_dfs if not df.empty]

print("Combining data...")
# combined_df = pd.concat([market_df] + stock_txt_dfs + stock_csv_dfs, ignore_index=True)
combined_df = pd.concat(stock_txt_dfs + stock_csv_dfs, ignore_index=True)

del stock_files_txt, stock_files_csv
del stock_txt_dfs, stock_csv_dfs

print(f'Got {len(combined_df.index)} rows of data.')

# Remove rows with any missing data
print("Cleaning data...")
combined_df.replace(['', '-', None, 'NaN'], pd.NA, inplace=True)
combined_df.dropna(inplace=True)
print(f'Got {len(combined_df.index)} rows of data.')

# Create an enum column for the ticker strings
print("Creating ticker enums...")
combined_df['TickerEnum'] = combined_df['Ticker'].astype('category').cat.codes

# Group and sort the DataFrame by 'TickerEnum' and then by 'Date'
print("Grouping and sorting data...")
combined_df = combined_df.sort_values(by=['TickerEnum', 'Date'])

# Export the combined DataFrame to individual CSV files per ticker
print("Exporting combined data to individual CSV files...")
os.makedirs('data/days', exist_ok=True)
unique_enums = combined_df['TickerEnum'].unique()
for ticker_enum in unique_enums:
    ticker_df = combined_df[combined_df['TickerEnum'] == ticker_enum]
    ticker = ticker_df['Ticker'].iloc[0]
    ticker_df = ticker_df.drop(columns=['TickerEnum', 'Ticker']).drop_duplicates(subset=['Date'])
    ticker_df.to_csv(f'data/days/{ticker_enum}_{ticker}.csv', index=False)

# Export unique ticker strings and their enums to another CSV file
print("Exporting unique tickers to CSV...")
unique_tickers = combined_df[['TickerEnum', 'Ticker']].drop_duplicates().sort_values(by='TickerEnum')
unique_tickers.to_csv('data/tickers.csv', index=False)

print("Process completed.")
print(combined_df.head())