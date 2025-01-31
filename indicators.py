# example csv:
    # Open,High,Low,Close,Week_sin
    # 1368.692993,1522.824951,1361.213989,1516.604004,0.12053668025532305
    # 1531.712036,1547.708008,1401.706055,1408.848022,0.23931566428755774
    # 1436.887939,1476.456055,1421.641968,1465.084961,0.3546048870425356

# and in new columns calculates the following indicators:
# 50-day moving average
# stohastic RSI
# MACD
# Parabolic SAR
# SuperTrend

# If you encounter a ModuleNotFoundError for 'pkg_resources', install setuptools:
# pip install setuptools


import pandas_ta as ta
import pandas as pd
from glob import glob
import os
import numpy as np

print("Running indicators.py...")

timestamp_2000 = pd.Timestamp('2000-01-01')
scales = ['days', 'weeks', 'months']
processed_files = {scale: set() for scale in scales}

for j, time_scale in enumerate(scales):
    csvs = glob(f'data/{time_scale}/*.csv')[:100]
    count = len(csvs)
    print(f"Processing {time_scale}... {count} csvs")
    for i, csv in enumerate(csvs):
        file_id = os.path.basename(csv)
        if j > 0 and file_id not in processed_files[scales[j-1]]:
            continue  # Skip processing if the file was not processed in the previous scale

        df = pd.read_csv(csv)
        try:
            df['50_EMA'] = ta.ema(df['Close'], length=50)
            stoch_rsi = ta.stochrsi(df['Close'], length=14)
            df['Stoch_RSI'] = stoch_rsi['STOCHRSIk_14_14_3_3']
            df['Stoch_RSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['MACD_Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
            df['MACD_Hist'] = ta.macd(df['Close'])['MACDh_12_26_9']
            df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
            df['SuperTrend'] = ta.supertrend(df['High'], df['Low'], df['Close'])['SUPERT_7_3.0']

            if time_scale == 'days':
                # Convert the date column to a cyclical year 1-364 day representation
                df['Date'] = pd.to_datetime(df['Date'])
                df['Day'] = df['Date'].dt.dayofyear
                df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
                df['Date'] = (df['Date'] - timestamp_2000).dt.days
                df.drop(columns=['Day'], inplace=True)

            df.to_csv(f'./data/indicators/{time_scale}/{file_id}', index=False)
            processed_files[time_scale].add(file_id)
        except Exception as e: pass

        if i % 10 == 0:
            print(f"Processed {round(((i+1)/count) * 100) }%")

        # break
    # exit(0)

# Remove files that are not present in all time scales
all_files = set.union(*[processed_files[scale] for scale in scales])
common_files = set.intersection(*[processed_files[scale] for scale in scales])

for id in all_files - common_files:
    for time_scale in scales:
        if os.path.exists(f'data/indicators/{time_scale}/{id}'):
            os.remove(f'data/indicators/{time_scale}/{id}')

print('Common files:', len(common_files))